//! CsmSession — streaming speech generation session.
//!
//! Wraps a `CsmModel` and manages a PCM sample buffer, allowing callers to
//! push audio samples incrementally and generate new frames as context
//! accumulates.
//!
//! # Usage
//! ```ignore
//! let mut session = CsmSession::new(CsmModel::new_tiny());
//! session.push_audio(&samples);
//! let new_pcm = session.generate_next_frame(Some(42)); // with text token
//! let all_pcm = session.flush_audio();
//! session.reset();
//! ```

use crate::model::CsmModel;

/// Streaming CSM generation session.
///
/// Buffers incoming PCM samples, queues complete Mimi frames, and generates
/// new audio frames on demand using the underlying `CsmModel`.
pub struct CsmSession {
    /// The underlying CSM model.
    pub model: CsmModel,
    /// Accumulation buffer for incoming PCM samples (< one full frame).
    frame_buffer: Vec<f32>,
    /// Number of PCM samples in one Mimi frame (typically 1920 at 24 kHz).
    frame_size: usize,
    /// All PCM samples generated so far (appended by `generate_next_frame`).
    generated_pcm: Vec<f32>,
    /// All RVQ codes generated so far: `[n_codebooks][n_frames_so_far]`.
    generated_codes: Vec<Vec<u32>>,
    /// Number of complete context frames queued (ready for backbone priming).
    queued_frames: usize,
}

impl CsmSession {
    /// Create a new session wrapping the given model.
    ///
    /// The `frame_size` is taken from `model.config.frame_samples`.
    pub fn new(model: CsmModel) -> Self {
        let frame_size = model.config.frame_samples;
        let n_cb = model.config.n_codebooks;
        Self {
            model,
            frame_buffer: Vec::new(),
            frame_size,
            generated_pcm: Vec::new(),
            generated_codes: vec![Vec::new(); n_cb],
            queued_frames: 0,
        }
    }

    /// Push raw PCM samples into the session buffer.
    ///
    /// Returns the number of **complete frames** now queued (ready to be used
    /// as context for the next `generate_next_frame` call).  Partial frames
    /// remain in the internal buffer.
    pub fn push_audio(&mut self, samples: &[f32]) -> usize {
        self.frame_buffer.extend_from_slice(samples);
        let new_full_frames = self.frame_buffer.len() / self.frame_size;
        self.queued_frames += new_full_frames;
        // Retain the remainder
        let used = new_full_frames * self.frame_size;
        self.frame_buffer.drain(..used);
        new_full_frames
    }

    /// Generate one new audio frame and return the decoded PCM.
    ///
    /// - `text_token`: optional BPE token for the current text position.
    ///
    /// The generated PCM is appended to the internal buffer and also returned.
    /// The returned slice has length ≈ `frame_size` (exact count depends on
    /// Mimi's convolutional padding).
    pub fn generate_next_frame(&mut self, text_token: Option<u32>) -> Vec<f32> {
        let text_tokens: Vec<u32> = text_token.map(|t| vec![t]).unwrap_or_default();
        let out = self.model.generate(
            &text_tokens,
            &[], // no raw PCM context — we already primed the KV cache externally
            1,   // one new frame
            self.model.config.temperature,
            self.model.config.topk,
        );

        // Accumulate generated codes
        let n_cb = self.model.config.n_codebooks;
        for cb in 0..n_cb {
            if cb < out.codes.len() && !out.codes[cb].is_empty() {
                self.generated_codes[cb].extend(&out.codes[cb]);
            }
        }

        // Accumulate PCM
        self.generated_pcm.extend(&out.pcm);
        out.pcm
    }

    /// Return all PCM samples generated during this session.
    pub fn flush_audio(&self) -> Vec<f32> {
        self.generated_pcm.clone()
    }

    /// Return the number of complete Mimi frames currently queued in the buffer.
    pub fn queued_frame_count(&self) -> usize {
        self.queued_frames
    }

    /// Reset the session, clearing all buffers, codes, and the backbone KV cache.
    ///
    /// The model weights are preserved; only runtime state is cleared.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.generated_pcm.clear();
        let n_cb = self.model.config.n_codebooks;
        self.generated_codes = vec![Vec::new(); n_cb];
        self.queued_frames = 0;
        self.model.backbone.reset();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CsmModel;

    #[test]
    fn test_session_new() {
        let model = CsmModel::new_tiny();
        let _s = CsmSession::new(model);
    }

    #[test]
    fn test_session_push_audio_partial() {
        let model = CsmModel::new_tiny();
        let frame_size = model.config.frame_samples;
        let mut s = CsmSession::new(model);
        // Push half a frame — no complete frame queued
        let half = frame_size / 2;
        let queued = s.push_audio(&vec![0.01f32; half]);
        assert_eq!(queued, 0, "partial push → 0 frames queued");
        assert_eq!(s.queued_frame_count(), 0);
    }

    #[test]
    fn test_session_push_audio_full() {
        let model = CsmModel::new_tiny();
        let frame_size = model.config.frame_samples;
        let mut s = CsmSession::new(model);
        // Push exactly one frame
        let queued = s.push_audio(&vec![0.01f32; frame_size]);
        assert_eq!(queued, 1, "exactly one frame → 1 frame queued");
        assert_eq!(s.queued_frame_count(), 1);
    }

    #[test]
    fn test_session_push_audio_multiple() {
        let model = CsmModel::new_tiny();
        let frame_size = model.config.frame_samples;
        let mut s = CsmSession::new(model);
        // Push 2.5 frames
        let samples = vec![0.01f32; frame_size * 5 / 2];
        let queued = s.push_audio(&samples);
        assert_eq!(queued, 2, "2.5 frames → 2 complete frames");
    }

    #[test]
    fn test_session_generate_next_frame() {
        let model = CsmModel::new_tiny();
        let mut s = CsmSession::new(model);
        // Generate without any context
        let pcm = s.generate_next_frame(None);
        // May or may not produce PCM depending on Mimi (shape test)
        let _ = pcm; // just ensure no panic
    }

    #[test]
    fn test_session_generate_with_text_token() {
        let model = CsmModel::new_tiny();
        let mut s = CsmSession::new(model);
        let pcm = s.generate_next_frame(Some(10));
        let _ = pcm;
    }

    #[test]
    fn test_session_flush_audio() {
        let model = CsmModel::new_tiny();
        let mut s = CsmSession::new(model);
        // Generate two frames and check flush accumulates them
        let pcm1 = s.generate_next_frame(None);
        let pcm2 = s.generate_next_frame(None);
        let flushed = s.flush_audio();
        assert_eq!(flushed.len(), pcm1.len() + pcm2.len(), "flush returns all generated PCM");
    }

    #[test]
    fn test_session_reset() {
        let model = CsmModel::new_tiny();
        let frame_size = model.config.frame_samples;
        let mut s = CsmSession::new(model);
        s.push_audio(&vec![0.01f32; frame_size]);
        s.generate_next_frame(None);
        s.reset();
        assert_eq!(s.queued_frame_count(), 0, "reset clears frame queue");
        assert!(s.flush_audio().is_empty(), "reset clears generated PCM");
        assert!(s.frame_buffer.is_empty(), "reset clears partial frame buffer");
    }
}
