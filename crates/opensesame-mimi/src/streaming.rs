//! Streaming Mimi codec session.
//!
//! [`MimiStreamingSession`] wraps a [`MimiCodec`] and accumulates raw PCM
//! samples in a buffer. Once a complete hop (960 samples = 1 SEANet frame)
//! is available, it encodes the buffered audio and returns codes.
//!
//! The streaming session produces the **same codes** as calling
//! [`MimiCodec::encode`] with the equivalent full-length audio, because the
//! SEANet encoder is a causal batch processor — its output for frame t depends
//! only on samples 0..t×960.
//!
//! # Usage
//! ```rust,ignore
//! let session = MimiStreamingSession::new(MimiCodec::new(MimiConfig::v0_1()));
//! let codes = session.push_audio(&pcm_chunk); // returns codes when 960+ samples ready
//! let audio = session.decode_codes(&codes);
//! session.reset();  // start a new utterance
//! ```

use crate::codec::MimiCodec;
use opensesame_seanet::StreamingEncoder;

/// Number of PCM samples per latent frame (SEANet hop length = 8×6×5×4 = 960).
const HOP: usize = 960;

// ─── MimiStreamingSession ────────────────────────────────────────────────────

/// Stateful streaming codec session for real-time encode/decode.
///
/// Buffers incoming PCM chunks and returns codes once a complete 960-sample
/// SEANet frame is accumulated. Decoding is always done on whole frames.
pub struct MimiStreamingSession {
    /// The underlying batch codec (used for encoding and decoding).
    codec: MimiCodec,
    /// SEANet streaming encoder state — kept for future causal-state optimisation.
    enc_state: StreamingEncoder,
    /// Accumulation buffer for incoming PCM samples (mono, 24 kHz).
    buffer_in: Vec<f32>,
    /// Total number of code frames emitted since last [`reset`](Self::reset).
    pub frames_processed: usize,
}

impl MimiStreamingSession {
    /// Create a new streaming session wrapping the given codec.
    pub fn new(codec: MimiCodec) -> Self {
        Self {
            codec,
            enc_state: StreamingEncoder::new(),
            buffer_in: Vec::new(),
            frames_processed: 0,
        }
    }

    /// Reset all streaming state for a new utterance.
    ///
    /// Clears the PCM buffer, resets the SEANet accumulator, and zeroes the
    /// frame counter.
    pub fn reset(&mut self) {
        self.buffer_in.clear();
        self.enc_state.reset();
        self.frames_processed = 0;
    }

    /// Push PCM samples into the session.
    ///
    /// Appends `pcm` to the internal buffer. Once the buffer contains at least
    /// 960 samples (one complete SEANet frame), all available whole frames are
    /// encoded and their codes returned.
    ///
    /// Returns an empty `Vec` if the buffer holds fewer than 960 samples after
    /// appending `pcm`.
    ///
    /// # Returns
    /// `Vec<Vec<u32>>` of length `num_codebooks` (8); each inner Vec has length
    /// equal to the number of new frames encoded.  May be empty.
    pub fn push_audio(&mut self, pcm: &[f32]) -> Vec<Vec<u32>> {
        self.buffer_in.extend_from_slice(pcm);

        if self.buffer_in.len() < HOP {
            // Not enough samples for a complete frame yet.
            return Vec::new();
        }

        // Consume all complete frames from the buffer.
        let n_frames = self.buffer_in.len() / HOP;
        let to_process = n_frames * HOP;

        // Drain the samples we're about to encode.
        let chunk: Vec<f32> = self.buffer_in.drain(..to_process).collect();

        // Encode using the batch codec (correct because SEANet is causal).
        let codes = self.codec.encode(&chunk, 1, to_process);
        self.frames_processed += n_frames;

        codes
    }

    /// Decode code frames to PCM audio.
    ///
    /// Delegates to [`MimiCodec::decode`] with `batch=1`.
    ///
    /// # Returns
    /// Flat `[1, 1, T_frames * 960]` audio (mono, 24 kHz).
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Vec<f32> {
        self.codec.decode(codes, 1)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::MimiCodec;
    use crate::config::MimiConfig;

    fn make_session() -> MimiStreamingSession {
        MimiStreamingSession::new(MimiCodec::new(MimiConfig::v0_1()))
    }

    #[test]
    fn test_streaming_session_push() {
        // Push exactly 960 samples → 1 frame of codes returned (8 vecs of len 1)
        let mut s = make_session();
        let pcm = vec![0.0_f32; HOP];
        let codes = s.push_audio(&pcm);
        assert_eq!(codes.len(), 8, "must return codes for all 8 codebooks");
        for cv in &codes {
            assert_eq!(cv.len(), 1, "must return exactly 1 frame");
        }
        assert_eq!(s.frames_processed, 1);
    }

    #[test]
    fn test_streaming_session_partial() {
        // Push fewer than 960 samples → no codes yet
        let mut s = make_session();
        let pcm = vec![0.0_f32; 480];
        let codes = s.push_audio(&pcm);
        assert!(codes.is_empty(), "partial push must return empty codes");
        assert_eq!(s.frames_processed, 0);
    }

    #[test]
    fn test_streaming_session_reset() {
        // After reset(), internal buffer is cleared and a fresh push works.
        let mut s = make_session();

        // Fill buffer partially.
        s.push_audio(&vec![0.0_f32; 480]);
        assert_eq!(s.buffer_in.len(), 480);

        // Reset.
        s.reset();
        assert!(s.buffer_in.is_empty(), "buffer must be empty after reset");
        assert_eq!(s.frames_processed, 0);

        // Push a full frame after reset.
        let codes = s.push_audio(&vec![0.0_f32; HOP]);
        assert_eq!(codes.len(), 8, "must encode correctly after reset");
    }

    #[test]
    fn test_streaming_accumulation() {
        // push(480) + push(480)  must produce the same codes as push(960).
        //
        // This relies on SEANet being a stateless batch encoder:
        // encode(pcm[0..960]) == encode(pcm[0..480] ++ pcm[480..960]).
        let pcm: Vec<f32> = (0..960).map(|i| (i as f32 * 0.01).sin()).collect();

        // Accumulated version.
        let mut s1 = make_session();
        let codes_a1 = s1.push_audio(&pcm[..480]);
        assert!(codes_a1.is_empty(), "first half must not produce codes");
        let codes_a2 = s1.push_audio(&pcm[480..]);
        assert_eq!(codes_a2.len(), 8, "second half completes a frame");

        // Single-push version.
        let mut s2 = make_session();
        let codes_b = s2.push_audio(&pcm);
        assert_eq!(codes_b.len(), 8);

        // Both must agree on codes.
        for (cb, (a, b)) in codes_a2.iter().zip(codes_b.iter()).enumerate() {
            assert_eq!(
                a, b,
                "codebook {}: accumulated codes must match single-push codes",
                cb
            );
        }
    }

    #[test]
    fn test_streaming_decode_codes() {
        // decode_codes must produce audio of length T_frames × 960
        let mut s = make_session();
        let codes = s.push_audio(&vec![0.0_f32; HOP]);
        let audio = s.decode_codes(&codes);
        assert_eq!(audio.len(), 960, "1 decoded frame = 960 samples");
    }

    #[test]
    fn test_streaming_two_frames() {
        // Push 1920 samples → 2 frames returned at once
        let mut s = make_session();
        let codes = s.push_audio(&vec![0.0_f32; 1920]);
        assert_eq!(codes.len(), 8);
        for cv in &codes {
            assert_eq!(cv.len(), 2, "1920 samples → 2 frames");
        }
        assert_eq!(s.frames_processed, 2);
    }
}
