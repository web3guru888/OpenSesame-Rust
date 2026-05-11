//! One audio+text sample for training.

/// A single training sample consisting of raw PCM audio and its transcript.
///
/// Created by [`crate::loader::DataLoader::load_sample`] and consumed by
/// [`crate::batch::AudioBatch`].
#[derive(Debug, Clone)]
pub struct AudioSample {
    /// Normalised PCM samples in `[-1.0, 1.0]` at `sample_rate` Hz.
    pub pcm: Vec<f32>,
    /// Sample rate after resampling — should be `24_000` for Mimi.
    pub sample_rate: u32,
    /// Raw UTF-8 transcript.
    pub text: String,
    /// BPE token IDs produced by the tokeniser (empty if no tokeniser loaded).
    pub text_tokens: Vec<u32>,
    /// Numeric speaker identifier (parsed from the directory name).
    pub speaker_id: u32,
    /// Audio duration in seconds (`pcm.len() / sample_rate`).
    pub duration_secs: f32,
}

impl AudioSample {
    /// Construct a new `AudioSample`, computing `duration_secs` automatically.
    ///
    /// `text_tokens` defaults to an empty `Vec`; fill it later via the
    /// tokeniser if available.
    pub fn new(pcm: Vec<f32>, sample_rate: u32, text: String, speaker_id: u32) -> Self {
        let duration_secs = if sample_rate == 0 {
            0.0
        } else {
            pcm.len() as f32 / sample_rate as f32
        };
        Self {
            pcm,
            sample_rate,
            text,
            text_tokens: Vec::new(),
            speaker_id,
            duration_secs,
        }
    }

    /// Number of codec frames at the given frame rate (e.g. `12.5` or `25.0`).
    ///
    /// Calculated as `(duration_secs * fps).floor() as usize`.
    pub fn num_frames(&self, fps: f32) -> usize {
        (self.duration_secs * fps) as usize
    }

    /// Returns `true` if the sample has content worth training on.
    ///
    /// A sample is considered valid when it is longer than 0.1 seconds and
    /// contains at least one PCM sample.
    pub fn is_valid(&self) -> bool {
        self.duration_secs > 0.1 && !self.pcm.is_empty()
    }
}
