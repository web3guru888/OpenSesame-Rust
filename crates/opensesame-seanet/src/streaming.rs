//! Streaming encoder: chunk-by-chunk processing with causal correctness.
//!
//! Uses an accumulator approach: frame t in the output depends only on input
//! samples 0..t*960, so we can re-encode only the samples needed for new frames.
//! This is identical to batch processing the same total audio (causal guarantee).

use crate::encoder::SEANetEncoder;

/// Hop length: samples per latent frame = 8 × 6 × 5 × 4.
const HOP: usize = 960;

/// Streaming encoder that processes audio chunk by chunk.
///
/// Minimum useful chunk: 960 samples (1 output frame at 24 kHz).
/// Streaming output is identical to batch processing the same total audio.
pub struct StreamingEncoder {
    /// The underlying full-batch encoder.
    pub encoder: SEANetEncoder,
    /// All raw PCM samples accumulated so far.
    accumulated: Vec<f32>,
    /// Number of latent frames already returned to the caller.
    frames_returned: usize,
}

impl StreamingEncoder {
    /// Construct a StreamingEncoder with a fresh SEANetEncoder.
    pub fn new() -> Self {
        StreamingEncoder {
            encoder: SEANetEncoder::new(),
            accumulated: Vec::new(),
            frames_returned: 0,
        }
    }

    /// Reset streaming state for a new conversation / utterance.
    pub fn reset(&mut self) {
        self.accumulated.clear();
        self.frames_returned = 0;
    }

    /// Push a chunk of mono 24 kHz PCM samples and return any new latent frames.
    ///
    /// Returns a flat [512 × n_new_frames] vector in channel-major layout [512, T_new],
    /// matching the layout of `SEANetEncoder::forward()`.
    ///
    /// For n_new_frames=1, this is simply [ch0, ch1, ..., ch511].
    pub fn push_chunk(&mut self, chunk: &[f32]) -> Vec<f32> {
        self.accumulated.extend_from_slice(chunk);

        let n_complete = self.accumulated.len() / HOP;
        if n_complete <= self.frames_returned {
            return Vec::new();
        }

        let to_process = n_complete * HOP;
        // full_out is [B=1, 512, t_out] channel-major: full_out[ch * t_out + t].
        let (full_out, t_out) =
            self.encoder.forward(&self.accumulated[..to_process], 1, to_process);

        let new_frames = n_complete - self.frames_returned;
        let start_frame = self.frames_returned;

        // Extract frames [start_frame..n_complete] from channel-major full_out.
        // Output in same channel-major layout: [512, new_frames].
        let mut output = vec![0.0f32; 512 * new_frames];
        for ch in 0..512usize {
            for (i, t) in (start_frame..n_complete).enumerate() {
                output[ch * new_frames + i] = full_out[ch * t_out + t];
            }
        }

        self.frames_returned = n_complete;
        output
    }
}

impl Default for StreamingEncoder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::SEANetEncoder;

    fn sine(n: usize, freq: f32) -> Vec<f32> {
        let sr = 24000.0f32;
        (0..n).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin()).collect()
    }

    #[test]
    fn test_streaming_minimum_chunk() {
        // Exactly 960 samples → 1 output frame.
        let mut se = StreamingEncoder::new();
        let chunk = sine(HOP, 440.0);
        let out = se.push_chunk(&chunk);
        assert_eq!(out.len(), 512, "expected 1 frame × 512 channels");
    }

    #[test]
    fn test_streaming_matches_batch() {
        // 4 chunks of 960 = 3840 samples; streaming frame t must equal batch frame t.
        // Both use channel-major [512, T] layout. For each chunk (1 frame), compare per-channel.
        let n_chunks = 4usize;
        let audio = sine(n_chunks * HOP, 220.0);
        let enc = SEANetEncoder::new();
        let (batch_out, _t_out) = enc.forward(&audio, 1, audio.len());
        // batch_out: [512, n_chunks], batch_out[ch * n_chunks + t] = value at ch, time t.

        let mut se = StreamingEncoder::new();
        for (t, chunk) in audio.chunks(HOP).enumerate() {
            let frame = se.push_chunk(chunk); // 512 elements: [ch0_at_t, ch1_at_t, ...]
            assert_eq!(frame.len(), 512, "expected 1 frame of 512 channels");
            for ch in 0..512 {
                let batch_val = batch_out[ch * n_chunks + t];
                let stream_val = frame[ch];
                assert!(
                    (stream_val - batch_val).abs() < 1e-4,
                    "t={} ch={}: streaming={} batch={}",
                    t, ch, stream_val, batch_val
                );
            }
        }
    }

    #[test]
    fn test_streaming_reset() {
        // After reset(), state is fresh.
        let audio = sine(HOP * 2, 440.0);
        let mut se = StreamingEncoder::new();
        se.push_chunk(&audio[..HOP]);
        se.push_chunk(&audio[HOP..]);
        se.reset();
        assert_eq!(se.frames_returned, 0);
        assert!(se.accumulated.is_empty());

        // Same result as brand-new encoder after reset.
        let out1 = se.push_chunk(&audio[..HOP]);
        let mut se2 = StreamingEncoder::new();
        let out2 = se2.push_chunk(&audio[..HOP]);
        assert_eq!(out1, out2, "reset didn't produce fresh state");
    }

    #[test]
    fn test_streaming_causal_verified() {
        // State from chunk N propagates to chunk N+1.
        // Encoding chunk2 with context (after chunk1) differs from encoding chunk2 alone.
        let audio = sine(HOP * 2, 880.0);
        let mut se = StreamingEncoder::new();
        let _first = se.push_chunk(&audio[..HOP]);
        let second = se.push_chunk(&audio[HOP..]);

        let mut se2 = StreamingEncoder::new();
        let second_alone = se2.push_chunk(&audio[HOP..]);

        let any_diff = second.iter().zip(second_alone.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "streaming state not propagating across chunks");
    }

    #[test]
    fn test_streaming_two_chunks_of_960() {
        // 2 × 960 streaming: compare each frame against batch encoding.
        let audio = sine(HOP * 2, 330.0);
        let enc = SEANetEncoder::new();
        let (batch_out, _t_out) = enc.forward(&audio, 1, audio.len());
        // batch_out: [512, 2], batch_out[ch * 2 + t] = value.

        let mut se = StreamingEncoder::new();
        let frame0 = se.push_chunk(&audio[..HOP]);  // 512 elements
        let frame1 = se.push_chunk(&audio[HOP..]);  // 512 elements
        assert_eq!(frame0.len(), 512);
        assert_eq!(frame1.len(), 512);

        for ch in 0..512 {
            assert!((frame0[ch] - batch_out[ch * 2 + 0]).abs() < 1e-4,
                "frame0 ch={}: {} vs {}", ch, frame0[ch], batch_out[ch * 2 + 0]);
            assert!((frame1[ch] - batch_out[ch * 2 + 1]).abs() < 1e-4,
                "frame1 ch={}: {} vs {}", ch, frame1[ch], batch_out[ch * 2 + 1]);
        }
    }

    #[test]
    fn test_streaming_sub_hop_chunk_no_output() {
        // Less than HOP samples → no output yet.
        let mut se = StreamingEncoder::new();
        let chunk = sine(480, 440.0);
        let out = se.push_chunk(&chunk);
        assert!(out.is_empty(), "sub-hop chunk should not emit frames");
    }

    #[test]
    fn test_streaming_accumulates_correctly() {
        // Two 480-sample chunks → 1 complete frame after second.
        let audio = sine(HOP, 660.0);
        let mut se = StreamingEncoder::new();
        let out1 = se.push_chunk(&audio[..480]);
        let out2 = se.push_chunk(&audio[480..]);
        assert!(out1.is_empty(), "no output after 480 samples");
        assert_eq!(out2.len(), 512, "1 frame after completing 960 samples");
    }
}
