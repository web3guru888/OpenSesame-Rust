//! Integration tests for `opensesame-audio`.
//!
//! Target: 30+ tests covering WAV I/O, AudioBuffer, Resampler,
//! RingBuffer, VadState, and DSP utilities.

use opensesame_audio::{AudioBuffer, Resampler, RingBuffer, VadState, WavReader, WavWriter};
use opensesame_audio::dsp;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Generate a mono sine wave.
fn sine_wave(freq: f32, sample_rate: u32, num_frames: usize) -> Vec<f32> {
    (0..num_frames)
        .map(|i| {
            (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
        })
        .collect()
}

/// Compute SNR of `signal` vs `reference` in dB.
fn snr_db(reference: &[f32], signal: &[f32]) -> f32 {
    let n = reference.len().min(signal.len());
    let signal_power: f32 = reference[..n].iter().map(|&x| x * x).sum::<f32>() / n as f32;
    let noise_power: f32 = reference[..n]
        .iter()
        .zip(signal[..n].iter())
        .map(|(&r, &s)| (r - s) * (r - s))
        .sum::<f32>()
        / n as f32;
    if noise_power < 1e-12 {
        return 100.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Write a WAV and read it back; return (original buffer, round-tripped buffer).
fn wav_roundtrip_f32(samples: Vec<f32>, rate: u32, ch: u8) -> (AudioBuffer, AudioBuffer) {
    let buf = AudioBuffer::new(samples, rate, ch);
    let path = format!("/tmp/test_os_audio_{}.wav", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos());
    WavWriter::write(&buf, &path).expect("write failed");
    let readback = WavReader::open(&path).expect("read failed");
    std::fs::remove_file(&path).ok();
    (buf, readback)
}

/// Write a PCM16 WAV and read it back.
fn wav_roundtrip_pcm16(samples: Vec<f32>, rate: u32, ch: u8) -> (AudioBuffer, AudioBuffer) {
    let buf = AudioBuffer::new(samples, rate, ch);
    let path = format!("/tmp/test_os_pcm16_{}.wav", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos());
    WavWriter::write_pcm16(&buf, &path).expect("write_pcm16 failed");
    let readback = WavReader::open(&path).expect("read_pcm16 failed");
    std::fs::remove_file(&path).ok();
    (buf, readback)
}

// ── WAV I/O tests ────────────────────────────────────────────────────────────

#[test]
fn test_wav_roundtrip_f32() {
    let samples: Vec<f32> = (0..512).map(|i| (i as f32 / 256.0 - 1.0) * 0.5).collect();
    let (orig, back) = wav_roundtrip_f32(samples, 44100, 1);
    assert_eq!(orig.samples.len(), back.samples.len());
    assert_eq!(orig.sample_rate, back.sample_rate);
    assert_eq!(orig.channels, back.channels);
    for (o, b) in orig.samples.iter().zip(back.samples.iter()) {
        assert!((o - b).abs() < 1e-6, "f32 roundtrip: {} vs {}", o, b);
    }
}

#[test]
fn test_wav_roundtrip_pcm16() {
    // PCM16 quantises to 1/32768 precision.
    let samples: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0 - 1.0) * 0.8).collect();
    let (orig, back) = wav_roundtrip_pcm16(samples, 24000, 1);
    assert_eq!(orig.samples.len(), back.samples.len());
    for (o, b) in orig.samples.iter().zip(back.samples.iter()) {
        assert!((o - b).abs() < 1e-4, "PCM16 roundtrip: {} vs {}", o, b);
    }
}

#[test]
fn test_wav_stereo_read() {
    // Build a stereo buffer and write/read it back.
    let samples: Vec<f32> = (0..400).map(|i| (i as f32 / 200.0 - 1.0) * 0.5).collect();
    let buf = AudioBuffer::new(samples, 48000, 2);
    let path = "/tmp/test_os_stereo.wav";
    WavWriter::write(&buf, path).unwrap();
    let back = WavReader::open(path).unwrap();
    std::fs::remove_file(path).ok();
    assert_eq!(back.channels, 2, "channels");
    assert_eq!(back.num_frames(), buf.num_frames(), "num_frames");
    assert_eq!(back.sample_rate, 48000, "sample_rate");
}

#[test]
fn test_wav_missing_file() {
    let result = WavReader::open("nonexistent_file_xyz.wav");
    assert!(result.is_err(), "expected Err for missing file");
}

// ── AudioBuffer tests ────────────────────────────────────────────────────────

#[test]
fn test_audio_buffer_to_mono() {
    // Stereo: L = 0.5, R = -0.5 → mono = 0.0
    let samples = vec![0.5_f32, -0.5, 0.5, -0.5, 0.5, -0.5];
    let stereo = AudioBuffer::new(samples, 44100, 2);
    let mono = stereo.to_mono();
    assert_eq!(mono.channels, 1);
    assert_eq!(mono.num_frames(), stereo.num_frames());
    for &s in &mono.samples {
        assert!(s.abs() < 1e-6, "to_mono should give 0: got {}", s);
    }
}

#[test]
fn test_audio_buffer_to_mono_asymmetric() {
    // L = 0.8, R = 0.4 → mono = 0.6
    let samples = vec![0.8_f32, 0.4];
    let stereo = AudioBuffer::new(samples, 44100, 2);
    let mono = stereo.to_mono();
    assert!((mono.samples[0] - 0.6).abs() < 1e-6, "expected 0.6, got {}", mono.samples[0]);
}

#[test]
fn test_audio_buffer_normalize() {
    let samples = vec![0.1_f32, -0.5, 0.3, 0.2];
    let mut buf = AudioBuffer::new(samples, 44100, 1);
    buf.normalize(0.9);
    let p = buf.samples.iter().cloned().fold(0.0_f32, |a, x| a.max(x.abs()));
    assert!((p - 0.9).abs() < 1e-5, "peak after normalize should be 0.9, got {}", p);
}

#[test]
fn test_audio_buffer_segment() {
    // 1 second at 1000 Hz → 1000 samples; segment [0.25, 0.75] → 500 samples
    let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
    let buf = AudioBuffer::new(samples, 1000, 1);
    let seg = buf.segment(0.25, 0.75);
    assert_eq!(seg.num_frames(), 500, "segment length");
    // First sample should be index 250 = 0.25
    assert!((seg.samples[0] - 0.25).abs() < 1e-6);
}

#[test]
fn test_audio_buffer_duration() {
    // 1000 samples at 16kHz = 0.0625s
    let buf = AudioBuffer::new(vec![0.0; 1000], 16000, 1);
    let d = buf.duration_secs();
    assert!((d - 0.0625).abs() < 1e-6, "duration: {}", d);
}

#[test]
fn test_audio_buffer_empty() {
    let buf = AudioBuffer::new(vec![], 44100, 1);
    assert_eq!(buf.duration_secs(), 0.0);
    assert_eq!(buf.num_frames(), 0);
}

#[test]
fn test_audio_buffer_segment_clamping() {
    let buf = AudioBuffer::new(vec![0.0; 100], 100, 1);
    // End beyond buffer → clamp to end
    let seg = buf.segment(0.0, 5.0);
    assert_eq!(seg.num_frames(), 100);
    // Start beyond buffer → empty
    let seg2 = buf.segment(2.0, 3.0);
    assert_eq!(seg2.num_frames(), 0);
}

// ── Resampler tests ──────────────────────────────────────────────────────────

#[test]
fn test_resample_44100_to_24000() {
    let n_in = 44100_usize; // 1 second
    let input = sine_wave(440.0, 44100, n_in);
    let out = Resampler::resample(&input, 44100, 24000);
    let expected_len = ((n_in as f64 * 24000.0 / 44100.0).ceil()) as usize;
    // Allow ±2 samples tolerance due to rounding
    assert!(
        (out.len() as isize - expected_len as isize).abs() <= 2,
        "44100→24000 len: {} vs expected {}", out.len(), expected_len
    );
}

#[test]
fn test_resample_16000_to_24000() {
    let n_in = 16000_usize;
    let input = sine_wave(300.0, 16000, n_in);
    let out = Resampler::resample(&input, 16000, 24000);
    let expected_len = ((n_in as f64 * 24000.0 / 16000.0).ceil()) as usize;
    assert!(
        (out.len() as isize - expected_len as isize).abs() <= 2,
        "16000→24000 len: {} vs expected {}", out.len(), expected_len
    );
}

#[test]
fn test_resample_48000_to_24000() {
    let n_in = 48000_usize;
    let input = sine_wave(500.0, 48000, n_in);
    let out = Resampler::resample(&input, 48000, 24000);
    let expected_len = ((n_in as f64 * 24000.0 / 48000.0).ceil()) as usize;
    assert!(
        (out.len() as isize - expected_len as isize).abs() <= 2,
        "48000→24000 len: {} vs expected {}", out.len(), expected_len
    );
}

#[test]
fn test_resample_silence() {
    let input = vec![0.0_f32; 1000];
    let out = Resampler::resample(&input, 44100, 24000);
    assert!(!out.is_empty());
    for &s in &out {
        assert!(s.abs() < 1e-6, "silence should stay silent, got {}", s);
    }
}

#[test]
fn test_resample_identity_rate() {
    let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let out = Resampler::resample(&input, 44100, 44100);
    assert_eq!(out.len(), input.len());
    for (i, (a, b)) in input.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "identity mismatch at {}: {} vs {}", i, a, b);
    }
}

#[test]
fn test_resample_snr_pure_sine() {
    // Generate 1s of 440Hz at 44100, resample to 24000, then regenerate 440Hz at 24000.
    // Compare the two signals after alignment — SNR should be > 40 dB.
    let n_in = 44100;
    let input = sine_wave(440.0, 44100, n_in);
    let out = Resampler::resample(&input, 44100, 24000);

    // Reference: ideal 440 Hz at 24000 Hz, same length.
    let n_out = out.len();
    let reference = sine_wave(440.0, 24000, n_out);

    // Skip the first and last 64 samples (filter ramp-up/ramp-down).
    let trim = 64;
    if n_out > trim * 2 {
        let snr = snr_db(&reference[trim..n_out - trim], &out[trim..n_out - trim]);
        assert!(snr > 40.0, "SNR too low: {:.1} dB (expected > 40 dB)", snr);
    }
}

// ── RingBuffer tests ─────────────────────────────────────────────────────────

#[test]
fn test_ring_buffer_push_read() {
    let mut rb: RingBuffer<f32> = RingBuffer::new(256);
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let written = rb.push_slice(&data);
    assert_eq!(written, 100);

    let mut out = vec![0.0_f32; 100];
    let read = rb.read_slice(&mut out);
    assert_eq!(read, 100);
    for (i, (&a, &b)) in data.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "mismatch at {}: {} vs {}", i, a, b);
    }
}

#[test]
fn test_ring_buffer_overflow() {
    // Capacity 8 (rounds to 8, usable = 7), push 9 — oldest should be overwritten.
    let mut rb: RingBuffer<i32> = RingBuffer::new(8);
    let data: Vec<i32> = (0..9).collect();
    rb.push_slice(&data);

    // Read back — should get the last 7 elements: [2,3,4,5,6,7,8]
    let mut out = vec![0_i32; 8];
    let n = rb.read_slice(&mut out);
    assert!(n > 0);
    // The last element written (8) must be present.
    assert!(out[..n].contains(&8), "overflow: last elem missing, got {:?}", &out[..n]);
}

#[test]
fn test_ring_buffer_available() {
    let mut rb: RingBuffer<f32> = RingBuffer::new(64);
    assert_eq!(rb.available(), 0);
    rb.push_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(rb.available(), 3);
    let mut out = [0.0_f32; 1];
    rb.read_slice(&mut out);
    assert_eq!(rb.available(), 2);
}

#[test]
fn test_ring_buffer_free_space() {
    let mut rb: RingBuffer<u8> = RingBuffer::new(16);
    let cap_minus_1 = rb.free_space(); // Should be capacity-1
    rb.push_slice(&[0u8; 4]);
    assert_eq!(rb.free_space(), cap_minus_1 - 4);
}

#[test]
fn test_ring_buffer_empty_read() {
    let mut rb: RingBuffer<f32> = RingBuffer::new(32);
    let mut out = vec![0.0_f32; 10];
    let n = rb.read_slice(&mut out);
    assert_eq!(n, 0, "read from empty buffer should return 0");
}

#[test]
fn test_ring_buffer_repeated_cycles() {
    // Push and read in cycles to verify wrap-around.
    let mut rb: RingBuffer<f32> = RingBuffer::new(8);
    for cycle in 0..10_u32 {
        let data = vec![cycle as f32; 4];
        rb.push_slice(&data);
        let mut out = vec![0.0_f32; 4];
        let n = rb.read_slice(&mut out);
        assert_eq!(n, 4);
        assert!((out[0] - cycle as f32).abs() < 1e-6);
    }
}

// ── VAD tests ────────────────────────────────────────────────────────────────

#[test]
fn test_vad_silence() {
    let mut vad = VadState::new();
    let frame = vec![0.0_f32; 256];
    assert!(!vad.is_speech(&frame), "silence should be false");
}

#[test]
fn test_vad_loud_tone() {
    let mut vad = VadState::new();
    let frame = sine_wave(440.0, 16000, 256); // amplitude ≈ 1.0, energy ≈ 0.5
    assert!(vad.is_speech(&frame), "loud tone should be true");
}

#[test]
fn test_vad_hangover() {
    let mut vad = VadState::new();
    // 1 frame of loud audio to trigger speech.
    let speech_frame = sine_wave(440.0, 16000, 256);
    let silence_frame = vec![0.0_f32; 256];

    assert!(vad.is_speech(&speech_frame)); // trigger
    // Next 8 frames should still return true (hangover).
    for _ in 0..vad.hangover_frames {
        assert!(vad.is_speech(&silence_frame), "hangover should keep true");
    }
    // After hangover, silence again.
    assert!(!vad.is_speech(&silence_frame), "after hangover should be false");
}

#[test]
fn test_vad_below_threshold() {
    let mut vad = VadState::with_thresholds(0.5, 0.3);
    // Very quiet signal (energy ≈ 1e-4).
    let frame: Vec<f32> = (0..256).map(|i| 0.01 * (i as f32 / 256.0)).collect();
    assert!(!vad.is_speech(&frame), "very quiet signal should be false");
}

#[test]
fn test_vad_reset() {
    let mut vad = VadState::new();
    let speech = sine_wave(440.0, 16000, 256);
    vad.is_speech(&speech); // arms hangover
    vad.reset();
    let silence = vec![0.0_f32; 256];
    assert!(!vad.is_speech(&silence), "after reset hangover should be cleared");
}

// ── DSP tests ────────────────────────────────────────────────────────────────

#[test]
fn test_dsp_rms_known() {
    // RMS([0, 1, 0, -1]) = sqrt(0.5)
    let samples = [0.0_f32, 1.0, 0.0, -1.0];
    let r = dsp::rms(&samples);
    assert!((r - 0.5_f32.sqrt()).abs() < 1e-6, "rms: {}", r);
}

#[test]
fn test_dsp_peak() {
    let samples = [-0.5_f32, 0.3, 0.7];
    assert!((dsp::peak(&samples) - 0.7).abs() < 1e-6);
}

#[test]
fn test_dsp_peak_negative_dominant() {
    let samples = [0.3_f32, -0.8, 0.1];
    assert!((dsp::peak(&samples) - 0.8).abs() < 1e-6);
}

#[test]
fn test_dsp_hann_window_ends_zero() {
    let w = dsp::hann_window(512);
    assert_eq!(w.len(), 512);
    assert!(w[0].abs() < 1e-5, "hann[0] should be ~0, got {}", w[0]);
    assert!(w[511].abs() < 1e-5, "hann[511] should be ~0, got {}", w[511]);
}

#[test]
fn test_dsp_hann_window_peak() {
    // Peak of Hann window is at the centre.
    let w = dsp::hann_window(512);
    let max = w.iter().cloned().fold(0.0_f32, f32::max);
    assert!((max - 1.0).abs() < 1e-5, "hann peak should be 1.0, got {}", max);
}

#[test]
fn test_dsp_apply_gain() {
    let mut samples = vec![0.1_f32, -0.2, 0.3];
    dsp::apply_gain(&mut samples, 2.0);
    assert!((samples[0] - 0.2).abs() < 1e-6);
    assert!((samples[1] - (-0.4)).abs() < 1e-6);
    assert!((samples[2] - 0.6).abs() < 1e-6);
}

#[test]
fn test_dsp_zero_pad() {
    let samples = vec![1.0_f32, 2.0, 3.0, 4.0];
    let padded = dsp::zero_pad(&samples, 8);
    assert_eq!(padded.len(), 8);
    assert_eq!(&padded[..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&padded[4..], &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_dsp_zero_pad_no_change() {
    let samples = vec![1.0_f32; 8];
    let padded = dsp::zero_pad(&samples, 4);
    // target_len < actual len → no truncation, returns full slice
    assert_eq!(padded.len(), 8);
}

#[test]
fn test_dsp_clip() {
    let mut samples = vec![-1.5_f32, 0.5, 1.2, -0.8];
    dsp::clip(&mut samples, 1.0);
    assert!((samples[0] - (-1.0)).abs() < 1e-6, "clipped to -1");
    assert!((samples[1] - 0.5).abs() < 1e-6, "unchanged");
    assert!((samples[2] - 1.0).abs() < 1e-6, "clipped to 1");
    assert!((samples[3] - (-0.8)).abs() < 1e-6, "unchanged");
}

#[test]
fn test_dsp_dc_offset() {
    let samples = vec![1.0_f32, 1.0, 1.0, 1.0];
    assert!((dsp::dc_offset(&samples) - 1.0).abs() < 1e-6);
    let zero = vec![0.0_f32; 4];
    assert!(dsp::dc_offset(&zero).abs() < 1e-6);
}

#[test]
fn test_dsp_apply_window() {
    let mut samples = vec![1.0_f32; 512];
    let window = dsp::hann_window(512);
    dsp::apply_window(&mut samples, &window);
    // Ends should be near zero.
    assert!(samples[0].abs() < 1e-5);
    assert!(samples[511].abs() < 1e-5);
}

// ── AudioBuffer resample integration ─────────────────────────────────────────

#[test]
fn test_audio_buffer_resample_44100_to_24000() {
    let samples = sine_wave(440.0, 44100, 44100);
    let buf = AudioBuffer::new(samples, 44100, 1);
    let out = buf.resample(24000);
    assert_eq!(out.sample_rate, 24000);
    assert_eq!(out.channels, 1);
    let expected = ((44100_f64 * 24000.0 / 44100.0).ceil()) as usize;
    assert!(
        (out.num_frames() as isize - expected as isize).abs() <= 2,
        "resampled frames: {} vs expected {}", out.num_frames(), expected
    );
}

#[test]
fn test_audio_buffer_resample_identity() {
    let samples = sine_wave(220.0, 24000, 1000);
    let buf = AudioBuffer::new(samples.clone(), 24000, 1);
    let out = buf.resample(24000);
    assert_eq!(out.sample_rate, 24000);
    assert_eq!(out.samples, samples);
}
