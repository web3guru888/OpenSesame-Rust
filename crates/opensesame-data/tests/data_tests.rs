//! Integration tests for opensesame-data.
//!
//! Tests use synthetic in-memory WAV files and temporary directories so they
//! run offline with no real dataset required.

use opensesame_data::{AudioBatch, AudioSample, CodeCache, DataError, DataLoader};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// Build a minimal RIFF/PCM-16 WAV in memory.
///
/// `num_samples` zero-samples at `sample_rate` Hz, mono.
fn make_test_wav(num_samples: u32, sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample: u32 = 2; // PCM-16
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let audio_format: u16 = 1; // PCM
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let byte_rate: u32 = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let data_size: u32 = num_samples * bytes_per_sample;
    let file_size: u32 = 36 + data_size; // everything after the "RIFF" + size field

    let mut buf = Vec::with_capacity((44 + data_size) as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk (16 bytes)
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&audio_format.to_le_bytes());
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    // silence (all zeros)
    buf.resize(buf.len() + data_size as usize, 0u8);

    buf
}

/// Write a WAV + transcript pair under `dir/<name>/`.
/// Returns (wav_path, txt_path).
fn write_sample(dir: &Path, speaker: &str, name: &str, num_samples: u32, sr: u32, text: &str)
    -> (PathBuf, PathBuf)
{
    let speaker_dir = dir.join(speaker);
    fs::create_dir_all(&speaker_dir).unwrap();

    let wav_path = speaker_dir.join(format!("{}.wav", name));
    let txt_path = speaker_dir.join(format!("{}.txt", name));

    let wav_bytes = make_test_wav(num_samples, sr);
    fs::write(&wav_path, &wav_bytes).unwrap();
    fs::write(&txt_path, text).unwrap();

    (wav_path, txt_path)
}

/// Create a unique temporary directory under the system temp directory.
fn temp_dir(prefix: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "opensesame_test_{}_{}", prefix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    fs::create_dir_all(&base).unwrap();
    base
}

// ══════════════════════════════════════════════════════════════════════════════
// AudioSample tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_audio_sample_new() {
    let pcm: Vec<f32> = vec![0.0f32; 24_000]; // 1 second
    let s = AudioSample::new(pcm.clone(), 24_000, "Hello".to_string(), 42);
    assert_eq!(s.sample_rate, 24_000);
    assert_eq!(s.text, "Hello");
    assert_eq!(s.speaker_id, 42);
    assert!((s.duration_secs - 1.0).abs() < 1e-5);
    assert!(s.text_tokens.is_empty());
}

#[test]
fn test_audio_sample_num_frames_25fps() {
    // 2 seconds × 25 fps = 50 frames
    let pcm = vec![0.0f32; 48_000];
    let s = AudioSample::new(pcm, 24_000, String::new(), 0);
    assert_eq!(s.num_frames(25.0), 50);
}

#[test]
fn test_audio_sample_num_frames_125fps() {
    // 4 seconds × 12.5 fps = 50 frames
    let pcm = vec![0.0f32; 96_000];
    let s = AudioSample::new(pcm, 24_000, String::new(), 0);
    assert_eq!(s.num_frames(12.5), 50);
}

#[test]
fn test_audio_sample_is_valid_short_rejected() {
    // Only 0.05 s — shorter than 0.1 s threshold
    let pcm = vec![0.0f32; 1_200]; // 0.05 s at 24 kHz
    let s = AudioSample::new(pcm, 24_000, String::new(), 0);
    assert!(!s.is_valid(), "sample shorter than 0.1 s should be invalid");
}

#[test]
fn test_audio_sample_is_valid_normal() {
    let pcm = vec![0.0f32; 24_000]; // 1 second
    let s = AudioSample::new(pcm, 24_000, "Hi".to_string(), 1);
    assert!(s.is_valid());
}

// ══════════════════════════════════════════════════════════════════════════════
// AudioBatch tests
// ══════════════════════════════════════════════════════════════════════════════

fn make_sample(sr: u32, secs: f32, text: &str) -> AudioSample {
    let n = (sr as f32 * secs) as usize;
    let mut s = AudioSample::new(vec![0.0f32; n], sr, text.to_string(), 0);
    // Set 3 fake tokens
    s.text_tokens = vec![1, 2, 3];
    s
}

#[test]
fn test_batch_from_samples() {
    let samples = vec![
        make_sample(24_000, 1.0, "one"),
        make_sample(24_000, 2.0, "two"),
    ];
    let batch = AudioBatch::new(samples);
    assert_eq!(batch.batch_size(), 2);
}

#[test]
fn test_batch_size_correct() {
    let samples = (0..5).map(|_| make_sample(24_000, 1.0, "x")).collect();
    let batch = AudioBatch::new(samples);
    assert_eq!(batch.batch_size(), 5);
}

#[test]
fn test_batch_max_audio_len() {
    let s1 = AudioSample::new(vec![0.0; 24_000], 24_000, String::new(), 0);
    let s2 = AudioSample::new(vec![0.0; 48_000], 24_000, String::new(), 0);
    let batch = AudioBatch::new(vec![s1, s2]);
    assert_eq!(batch.max_audio_len(), 48_000);
}

#[test]
fn test_batch_empty_samples() {
    let batch = AudioBatch::new(vec![]);
    assert_eq!(batch.batch_size(), 0);
    assert_eq!(batch.max_audio_len(), 0);
    assert_eq!(batch.max_token_len(), 0);
}

// ══════════════════════════════════════════════════════════════════════════════
// DataLoader tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_loader_new_empty_dir() {
    let dir = temp_dir("empty");
    let loader = DataLoader::new(&dir, 4).unwrap();
    assert_eq!(loader.num_samples(), 0);
    assert_eq!(loader.num_batches(), 0);
}

#[test]
fn test_loader_scan_flat_structure() {
    let dir = temp_dir("flat");
    // Write 2 WAV files directly in root (speaker=0)
    let wav1 = make_test_wav(24_000, 24_000);
    let wav2 = make_test_wav(24_000, 24_000);
    fs::write(dir.join("a.wav"), &wav1).unwrap();
    fs::write(dir.join("a.txt"), "hello").unwrap();
    fs::write(dir.join("b.wav"), &wav2).unwrap();
    fs::write(dir.join("b.txt"), "world").unwrap();

    let loader = DataLoader::new(&dir, 4).unwrap();
    assert_eq!(loader.num_samples(), 2);
}

#[test]
fn test_loader_scan_speaker_dirs() {
    let dir = temp_dir("speakers");
    write_sample(&dir, "speaker_001", "utt1", 24_000, 24_000, "Hello");
    write_sample(&dir, "speaker_001", "utt2", 24_000, 24_000, "World");
    write_sample(&dir, "speaker_002", "utt1", 24_000, 24_000, "Foo");

    let loader = DataLoader::new(&dir, 4).unwrap();
    assert_eq!(loader.num_samples(), 3);
}

#[test]
fn test_loader_num_samples() {
    let dir = temp_dir("numsamples");
    for i in 0..7 {
        let wav = make_test_wav(24_000, 24_000);
        let spk_dir = dir.join("spk");
        fs::create_dir_all(&spk_dir).unwrap();
        fs::write(spk_dir.join(format!("u{}.wav", i)), &wav).unwrap();
        fs::write(spk_dir.join(format!("u{}.txt", i)), "text").unwrap();
    }
    let loader = DataLoader::new(&dir, 4).unwrap();
    assert_eq!(loader.num_samples(), 7);
}

#[test]
fn test_loader_num_batches_exact() {
    let dir = temp_dir("batches_exact");
    // 8 samples, batch_size=4 → 2 batches
    for i in 0..8 {
        let spk_dir = dir.join("spk");
        fs::create_dir_all(&spk_dir).unwrap();
        let wav = make_test_wav(24_000, 24_000);
        fs::write(spk_dir.join(format!("u{}.wav", i)), &wav).unwrap();
        fs::write(spk_dir.join(format!("u{}.txt", i)), "text").unwrap();
    }
    let loader = DataLoader::new(&dir, 4).unwrap();
    assert_eq!(loader.num_batches(), 2);
}

#[test]
fn test_loader_num_batches_remainder() {
    let dir = temp_dir("batches_rem");
    // 5 samples, batch_size=4 → ceil(5/4) = 2 batches
    for i in 0..5 {
        let spk_dir = dir.join("spk");
        fs::create_dir_all(&spk_dir).unwrap();
        let wav = make_test_wav(24_000, 24_000);
        fs::write(spk_dir.join(format!("u{}.wav", i)), &wav).unwrap();
        fs::write(spk_dir.join(format!("u{}.txt", i)), "text").unwrap();
    }
    let loader = DataLoader::new(&dir, 4).unwrap();
    assert_eq!(loader.num_batches(), 2);
}

#[test]
fn test_loader_load_sample_shape() {
    let dir = temp_dir("load_shape");
    // 1 second at 24 kHz → 24000 samples after load
    write_sample(&dir, "spk", "utt", 24_000, 24_000, "Hi there");

    let loader = DataLoader::new(&dir, 4).unwrap();
    let s = loader.load_sample(0).unwrap();
    assert_eq!(s.sample_rate, 24_000);
    assert_eq!(s.pcm.len(), 24_000);
    assert_eq!(s.text, "Hi there");
}

#[test]
fn test_loader_load_sample_target_sr() {
    let dir = temp_dir("resample");
    // Write a 16 kHz WAV — loader should resample to 24 kHz
    write_sample(&dir, "spk", "utt", 16_000, 16_000, "hello");

    let loader = DataLoader::new(&dir, 4)
        .unwrap()
        .with_target_sr(24_000);
    let s = loader.load_sample(0).unwrap();
    assert_eq!(s.sample_rate, 24_000);
    // 16000 samples at 16kHz → ~24000 samples at 24kHz (within 5%)
    let expected = 24_000usize;
    let got = s.pcm.len();
    let diff = (got as isize - expected as isize).unsigned_abs();
    assert!(diff < 1200, "resampled len={} expected≈{}", got, expected);
}

#[test]
fn test_loader_load_batch_size() {
    let dir = temp_dir("batch_load");
    for i in 0..6 {
        write_sample(&dir, "spk", &format!("u{}", i), 24_000, 24_000, "text");
    }
    let loader = DataLoader::new(&dir, 4).unwrap();
    // batch 0 should have 4 samples (all pass duration filter)
    let batch = loader.load_batch(0).unwrap();
    assert_eq!(batch.batch_size(), 4);
    // batch 1 should have 2 samples
    let batch1 = loader.load_batch(1).unwrap();
    assert_eq!(batch1.batch_size(), 2);
}

#[test]
fn test_loader_shuffle_different_order() {
    let dir = temp_dir("shuffle");
    // Create enough samples that a shuffle is likely to change order
    for i in 0..10 {
        write_sample(&dir, "spk", &format!("u{}", i), 24_000, 24_000, "text");
    }
    let loader_a = DataLoader::new(&dir, 16).unwrap();
    let loader_b = DataLoader::new(&dir, 16).unwrap().with_shuffle(42);

    // Collect paths from each loader
    let paths_a: Vec<_> = loader_a
        .samples
        .iter()
        .map(|(p, _, _)| p.clone())
        .collect();
    let paths_b: Vec<_> = loader_b
        .samples
        .iter()
        .map(|(p, _, _)| p.clone())
        .collect();

    // They should contain the same items but (very likely) in different order
    let mut sorted_a = paths_a.clone();
    let mut sorted_b = paths_b.clone();
    sorted_a.sort();
    sorted_b.sort();
    assert_eq!(sorted_a, sorted_b, "same samples after sort");
    // With 10 elements and a good shuffle, at least one position should differ
    let differs = paths_a.iter().zip(paths_b.iter()).any(|(a, b)| a != b);
    assert!(differs, "shuffle did not change order");
}

#[test]
fn test_loader_max_duration_filter() {
    let dir = temp_dir("maxdur");
    // 5 s sample → should be filtered out (max=3 s)
    write_sample(&dir, "spk", "long", 5 * 24_000, 24_000, "long");
    // 1 s sample → should pass
    write_sample(&dir, "spk", "short", 24_000, 24_000, "short");

    let loader = DataLoader::new(&dir, 8)
        .unwrap()
        .with_max_duration(3.0);
    let batch = loader.load_batch(0).unwrap();
    // Only the 1 s sample should appear
    assert_eq!(batch.batch_size(), 1);
}

#[test]
fn test_loader_min_duration_filter() {
    let dir = temp_dir("mindur");
    // 0.1 s sample → below min=0.5 s
    write_sample(&dir, "spk", "tiny", 2_400, 24_000, "tiny");
    // 1 s sample → passes
    write_sample(&dir, "spk", "normal", 24_000, 24_000, "normal");

    let loader = DataLoader::new(&dir, 8)
        .unwrap()
        .with_min_duration(0.5);
    let batch = loader.load_batch(0).unwrap();
    assert_eq!(batch.batch_size(), 1);
}

#[test]
fn test_loader_invalid_dir_error() {
    let result = DataLoader::new("/nonexistent/path/that/does/not/exist", 4);
    assert!(result.is_err(), "should error on nonexistent directory");
}

// ══════════════════════════════════════════════════════════════════════════════
// CodeCache tests
// ══════════════════════════════════════════════════════════════════════════════

fn temp_cache_path(name: &str) -> PathBuf {
    temp_dir("cache").join(format!("{}.bin", name))
}

#[test]
fn test_cache_write_read_roundtrip() {
    let path = temp_cache_path("roundtrip");
    let mut cache = CodeCache::new(&path);

    // 4 codebooks × 10 frames, values = cb * 10 + t
    let codes: Vec<Vec<u32>> = (0..4)
        .map(|cb| (0..10u32).map(|t| cb * 10 + t).collect())
        .collect();

    cache.write_entry(0, &codes).unwrap();
    let read_back = cache.read_entry(0).unwrap();

    assert_eq!(read_back.len(), 4);
    for (cb, row) in read_back.iter().enumerate() {
        for (t, &val) in row.iter().enumerate() {
            assert_eq!(val, cb as u32 * 10 + t as u32);
        }
    }
}

#[test]
fn test_cache_contains_after_write() {
    let path = temp_cache_path("contains");
    let mut cache = CodeCache::new(&path);
    let codes = vec![vec![1u32, 2, 3], vec![4, 5, 6]];
    cache.write_entry(7, &codes).unwrap();
    assert!(cache.contains(7));
}

#[test]
fn test_cache_not_contains_before_write() {
    let path = temp_cache_path("notcontains");
    let cache = CodeCache::new(&path);
    assert!(!cache.contains(999));
}

#[test]
fn test_cache_multiple_entries() {
    let path = temp_cache_path("multi");
    let mut cache = CodeCache::new(&path);

    for i in 0..5u32 {
        let codes = vec![vec![i * 100, i * 100 + 1], vec![i * 200, i * 200 + 1]];
        cache.write_entry(i, &codes).unwrap();
    }

    // All 5 entries should be readable
    for i in 0..5u32 {
        let r = cache.read_entry(i).unwrap();
        assert_eq!(r[0][0], i * 100);
        assert_eq!(r[1][0], i * 200);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// DataError tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_error_display() {
    let e = DataError::InvalidWav("bad header".to_string());
    assert!(e.to_string().contains("bad header"));

    let e2 = DataError::BatchOutOfRange {
        batch_idx: 5,
        num_batches: 3,
    };
    let s = e2.to_string();
    assert!(s.contains('5') && s.contains('3'));

    let e3 = DataError::EmptySample;
    assert!(!e3.to_string().is_empty());
}

#[test]
fn test_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let data_err: DataError = io_err.into();
    match data_err {
        DataError::Io(_) => {}
        other => panic!("expected DataError::Io, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Bonus: persistence — reload from disk
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_roundtrip_persist_reload() {
    let path = temp_cache_path("persist");
    {
        let mut cache = CodeCache::new(&path);
        let codes = vec![vec![10u32, 20, 30]];
        cache.write_entry(42, &codes).unwrap();
    }
    // Reload from disk in a new instance
    let cache2 = CodeCache::new(&path);
    assert!(cache2.contains(42));
    let r = cache2.read_entry(42).unwrap();
    assert_eq!(r[0], vec![10, 20, 30]);
}

#[test]
fn test_loader_speaker_id_numeric() {
    let dir = temp_dir("spkid");
    write_sample(&dir, "2086", "utt", 24_000, 24_000, "text");
    let loader = DataLoader::new(&dir, 4).unwrap();
    let s = loader.load_sample(0).unwrap();
    assert_eq!(s.speaker_id, 2086, "speaker_id should parse '2086' → 2086");
}

#[test]
fn test_loader_speaker_id_prefixed() {
    let dir = temp_dir("spkidpfx");
    write_sample(&dir, "speaker_099", "utt", 24_000, 24_000, "text");
    let loader = DataLoader::new(&dir, 4).unwrap();
    let s = loader.load_sample(0).unwrap();
    assert_eq!(s.speaker_id, 99, "speaker_001 → 99");
}

#[test]
fn test_batch_max_token_len() {
    let mut s1 = AudioSample::new(vec![0.0; 24_000], 24_000, String::new(), 0);
    s1.text_tokens = vec![1, 2, 3];
    let mut s2 = AudioSample::new(vec![0.0; 24_000], 24_000, String::new(), 0);
    s2.text_tokens = vec![10, 20, 30, 40, 50];
    let batch = AudioBatch::new(vec![s1, s2]);
    assert_eq!(batch.max_token_len(), 5);
}
