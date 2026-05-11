//! Filesystem-based data loader for pre-extracted audio datasets.
//!
//! # Directory layout expected
//! ```text
//! root/
//!   speaker_001/
//!     utterance_001.wav
//!     utterance_001.txt
//!     utterance_002.wav
//!     utterance_002.txt
//!   speaker_002/
//!     ...
//! ```
//!
//! The loader scans the root directory for `(wav, txt)` pairs and provides
//! indexed access plus a batch iterator.  Matching transcript files are
//! optional: if no `.txt` file exists for a WAV, the transcript defaults to
//! an empty string.
//!
//! Speaker IDs are derived from the immediate parent directory name.
//! A directory named `speaker_042` or `2086` maps to the numeric ID `42` /
//! `2086`; non-numeric suffixes are hashed to a `u32`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::batch::AudioBatch;
use crate::error::DataError;
use crate::sample::AudioSample;
use opensesame_audio::Resampler;
use opensesame_audio::WavReader;

/// Filesystem-based data loader for pre-extracted audio datasets.
pub struct DataLoader {
    /// Root directory of the dataset.
    pub root: PathBuf,
    /// Target sample rate after resampling (default `24_000`).
    pub target_sr: u32,
    /// Skip samples longer than this duration in seconds (default `30.0`).
    pub max_duration_secs: f32,
    /// Skip samples shorter than this duration in seconds (default `0.5`).
    pub min_duration_secs: f32,
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Optional RNG seed for sample shuffling.
    pub shuffle_seed: Option<u64>,
    /// Scanned (wav_path, txt_path, speaker_id) triples.
    pub samples: Vec<(PathBuf, PathBuf, u32)>,
}

impl DataLoader {
    /// Create a new `DataLoader` rooted at `root` with the given `batch_size`.
    ///
    /// Scans `root` for `(wav, txt)` pairs immediately.  Returns an error if
    /// `root` does not exist or is not readable.
    pub fn new(root: impl Into<PathBuf>, batch_size: usize) -> Result<Self, DataError> {
        let root = root.into();
        if !root.exists() {
            return Err(DataError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("dataset root not found: {}", root.display()),
            )));
        }
        let samples = scan_dataset(&root);
        Ok(Self {
            root,
            target_sr: 24_000,
            max_duration_secs: 30.0,
            min_duration_secs: 0.5,
            batch_size,
            shuffle_seed: None,
            samples,
        })
    }

    /// Override the resampling target (default `24_000` Hz).
    pub fn with_target_sr(mut self, sr: u32) -> Self {
        self.target_sr = sr;
        self
    }

    /// Skip samples whose duration exceeds `secs`.
    pub fn with_max_duration(mut self, secs: f32) -> Self {
        self.max_duration_secs = secs;
        self
    }

    /// Skip samples whose duration is below `secs`.
    pub fn with_min_duration(mut self, secs: f32) -> Self {
        self.min_duration_secs = secs;
        self
    }

    /// Shuffle the sample list deterministically using `seed`.
    ///
    /// Uses a simple linear-congruential permutation so no external RNG crate
    /// is required.
    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self.samples = lcg_shuffle(self.samples, seed);
        self
    }

    /// Total number of scanned samples (before duration filtering).
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Number of complete batches (last partial batch is included).
    ///
    /// Returns `0` if there are no samples.
    pub fn num_batches(&self) -> usize {
        if self.samples.is_empty() || self.batch_size == 0 {
            return 0;
        }
        (self.samples.len() + self.batch_size - 1) / self.batch_size
    }

    /// Load one sample from disk at scanned index `idx`.
    ///
    /// Reads the WAV file, resamples to `target_sr`, reads the transcript,
    /// and constructs an [`AudioSample`].  Duration filters are **not** applied
    /// here — use [`load_batch`](Self::load_batch) for filtered iteration.
    pub fn load_sample(&self, idx: usize) -> Result<AudioSample, DataError> {
        let (wav_path, txt_path, speaker_id) = &self.samples[idx];

        // Load WAV
        let buf = WavReader::open(wav_path.to_str().unwrap_or(""))
            .map_err(|e| DataError::InvalidWav(e))?;

        // Convert to mono
        let mono = buf.to_mono();

        // Resample to target_sr
        let pcm = if mono.sample_rate == self.target_sr {
            mono.samples.clone()
        } else {
            Resampler::resample(&mono.samples, mono.sample_rate, self.target_sr)
        };

        // Read transcript
        let text = if txt_path.exists() {
            std::fs::read_to_string(txt_path).map_err(|e| {
                DataError::InvalidTranscript(format!("{}: {}", txt_path.display(), e))
            })?
        } else {
            String::new()
        };
        let text = text.trim().to_string();

        let mut sample = AudioSample::new(pcm, self.target_sr, text, *speaker_id);
        // text_tokens left empty — filled by training loop via tokeniser
        sample.text_tokens = Vec::new();
        Ok(sample)
    }

    /// Load and collate one batch of samples.
    ///
    /// Samples that fall outside the `[min_duration_secs, max_duration_secs]`
    /// window are silently skipped.  If the batch falls entirely outside the
    /// scan range, a [`DataError::BatchOutOfRange`] is returned.
    pub fn load_batch(&self, batch_idx: usize) -> Result<AudioBatch, DataError> {
        let nb = self.num_batches();
        if batch_idx >= nb {
            return Err(DataError::BatchOutOfRange {
                batch_idx,
                num_batches: nb,
            });
        }

        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(self.samples.len());

        let mut batch_samples = Vec::with_capacity(end - start);
        for i in start..end {
            match self.load_sample(i) {
                Ok(s) => {
                    if s.duration_secs >= self.min_duration_secs
                        && s.duration_secs <= self.max_duration_secs
                    {
                        batch_samples.push(s);
                    }
                }
                Err(_) => {} // skip unreadable files silently
            }
        }

        Ok(AudioBatch::new(batch_samples))
    }

    /// Iterate over all batches.
    pub fn iter_batches(&self) -> BatchIterator<'_> {
        BatchIterator {
            loader: self,
            current: 0,
        }
    }
}

/// Iterator over all batches produced by a [`DataLoader`].
pub struct BatchIterator<'a> {
    loader: &'a DataLoader,
    current: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Result<AudioBatch, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.loader.num_batches() {
            return None;
        }
        let result = self.loader.load_batch(self.current);
        self.current += 1;
        Some(result)
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Recursively scan `root` for `(wav_path, txt_path, speaker_id)` triples.
///
/// The immediate parent directory name is used to derive the speaker ID.
/// Directory layout can be flat (`root/*.wav`) or one level deep
/// (`root/SPEAKER/*.wav`).
fn scan_dataset(root: &Path) -> Vec<(PathBuf, PathBuf, u32)> {
    let mut results = Vec::new();
    scan_dir(root, root, &mut results);
    results
}

/// Recursively visit `dir`, collecting WAV+TXT pairs.
fn scan_dir(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(PathBuf, PathBuf, u32)>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut files: Vec<PathBuf> = Vec::new();
    let mut subdirs: Vec<PathBuf> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
        } else if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("wav") {
                files.push(path);
            }
        }
    }

    // Determine speaker_id from the current directory name
    let speaker_id = if dir == root {
        // Files at root level have speaker 0
        0u32
    } else {
        dir_to_speaker_id(dir)
    };

    // Pair each WAV with its optional TXT
    for wav in files {
        let txt = wav.with_extension("txt");
        out.push((wav, txt, speaker_id));
    }

    // Recurse into subdirectories
    for sub in subdirs {
        scan_dir(root, &sub, out);
    }
}

/// Derive a numeric speaker ID from a directory name.
///
/// Strips a leading non-numeric prefix (e.g. `"speaker_"`) and parses the
/// remaining digits.  If no numeric suffix exists, the name is hashed to a
/// `u32`.
///
/// Examples: `"speaker_042"` → `42`, `"2086"` → `2086`, `"alice"` → hash.
fn dir_to_speaker_id(dir: &Path) -> u32 {
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("0");

    // Try pure numeric first
    if let Ok(n) = name.parse::<u32>() {
        return n;
    }

    // Strip non-digit prefix and try again
    let digits: String = name.chars().skip_while(|c| !c.is_ascii_digit()).collect();
    if let Ok(n) = digits.parse::<u32>() {
        return n;
    }

    // Fall back to hash
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    (h.finish() & 0xFFFF_FFFF) as u32
}

/// Simple LCG-based Fisher-Yates shuffle (no external RNG needed).
fn lcg_shuffle<T>(mut v: Vec<T>, seed: u64) -> Vec<T> {
    let n = v.len();
    if n < 2 {
        return v;
    }
    let mut state = seed.wrapping_add(1);
    for i in (1..n).rev() {
        // LCG: multiplier / increment from Knuth
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
    v
}
