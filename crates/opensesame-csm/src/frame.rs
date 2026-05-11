//! Frame representation for the CSM model.
//!
//! A **frame** is one time-step in the CSM backbone's input sequence.  Each
//! frame holds tokens for all `FRAME_WIDTH = 33` columns:
//! - Columns 0..31: one token per audio codebook (or 0 / masked if not present).
//! - Column 32:     the corresponding text token (or 0 / masked if not present).
//!
//! The `mask` array has the same shape and indicates which entries are valid.
//! Masked tokens (`mask = false`) contribute zero to the masked-sum embedding.
//!
//! # Example
//! ```
//! # use opensesame_csm::Frame;
//! // Build a text-only prefix (3 tokens)
//! let text_frame = Frame::from_text_tokens(&[10, 20, 30]);
//! assert_eq!(text_frame.tokens.len(), 3);
//! assert!(text_frame.mask[0][32]); // col 32 = text, is valid
//! assert!(!text_frame.mask[0][0]); // col 0  = CB0,  is not valid
//! ```

/// One or more consecutive CSM backbone time-steps, each of width `FRAME_WIDTH`.
///
/// `tokens[t][c]` is the token ID for time-step `t`, column `c`.
/// `mask[t][c]` is `true` iff the token at `[t][c]` is valid (participates in embedding sum).
///
/// The conventional `FRAME_WIDTH` is `n_codebooks + 1 = 33` for CSM-1B, but
/// `Frame` does **not** enforce a fixed width — callers must use a consistent width
/// matching their [`CsmConfig`].
///
/// [`CsmConfig`]: crate::CsmConfig
#[derive(Debug, Clone)]
pub struct Frame {
    /// Token IDs per time-step per column.  Shape: `[seq_len][frame_width]`.
    pub tokens: Vec<Vec<i64>>,
    /// Validity mask.  Shape: `[seq_len][frame_width]`.
    pub mask: Vec<Vec<bool>>,
}

impl Frame {
    /// Create a frame segment containing only **text** tokens.
    ///
    /// Each time-step `t` has:
    /// - `tokens[t][32] = text_token_ids[t]`, `mask[t][32] = true`
    /// - All audio columns (0..31): token = 0, mask = `false`
    ///
    /// The `frame_width` is fixed at `33` (CSM-1B convention).
    ///
    /// # Panics
    /// Does not panic; an empty `text_token_ids` slice produces an empty `Frame`.
    pub fn from_text_tokens(text_token_ids: &[i64]) -> Self {
        const FRAME_WIDTH: usize = 33;
        const TEXT_COL: usize = FRAME_WIDTH - 1; // 32

        let seq_len = text_token_ids.len();
        let mut tokens = vec![vec![0i64; FRAME_WIDTH]; seq_len];
        let mut mask   = vec![vec![false;  FRAME_WIDTH]; seq_len];

        for (t, &tok) in text_token_ids.iter().enumerate() {
            tokens[t][TEXT_COL] = tok;
            mask[t][TEXT_COL]   = true;
        }

        Self { tokens, mask }
    }

    /// Create a frame segment from a `(n_codebooks, n_audio_frames)` token grid.
    ///
    /// `audio_tokens[cb][t]` is the token for codebook `cb` at time-step `t`.
    ///
    /// Each time-step `t` has:
    /// - `tokens[t][cb] = audio_tokens[cb][t]` for `cb` in `0..n_codebooks`
    /// - `mask[t][cb] = true` for all codebooks with valid tokens
    /// - The text column (`FRAME_WIDTH - 1`) is always `0` and `false`
    ///
    /// `frame_width` is `n_codebooks + 1` (the last column is reserved for text).
    ///
    /// # Panics
    /// Panics if `audio_tokens.len() != n_codebooks`, or if any inner `Vec` has a
    /// different length than `audio_tokens[0]`.
    pub fn from_audio_tokens(audio_tokens: &[Vec<i64>], n_codebooks: usize) -> Self {
        assert_eq!(
            audio_tokens.len(), n_codebooks,
            "from_audio_tokens: got {} codebook rows, expected {n_codebooks}",
            audio_tokens.len()
        );

        let frame_width = n_codebooks + 1; // last col = text (empty)
        let n_frames = if n_codebooks > 0 { audio_tokens[0].len() } else { 0 };

        // Verify all rows have the same length
        for (cb, row) in audio_tokens.iter().enumerate() {
            assert_eq!(
                row.len(), n_frames,
                "from_audio_tokens: codebook {cb} has {} frames, expected {n_frames}",
                row.len()
            );
        }

        let mut tokens = vec![vec![0i64; frame_width]; n_frames];
        let mut mask   = vec![vec![false; frame_width]; n_frames];

        for t in 0..n_frames {
            for cb in 0..n_codebooks {
                tokens[t][cb] = audio_tokens[cb][t];
                mask[t][cb]   = true;
            }
            // Text column remains 0 / false
        }

        Self { tokens, mask }
    }

    /// Concatenate two frames along the time (seq_len) axis.
    ///
    /// The resulting `Frame` has `seq_len = a.tokens.len() + b.tokens.len()`.
    /// Frame widths must match; panics if they differ.
    pub fn concat(mut a: Frame, b: Frame) -> Self {
        if !a.tokens.is_empty() && !b.tokens.is_empty() {
            assert_eq!(
                a.tokens[0].len(), b.tokens[0].len(),
                "Frame::concat: frame widths differ ({} vs {})",
                a.tokens[0].len(), b.tokens[0].len()
            );
        }
        a.tokens.extend(b.tokens);
        a.mask.extend(b.mask);
        a
    }

    /// Return the number of time-steps in this frame sequence.
    pub fn seq_len(&self) -> usize {
        self.tokens.len()
    }

    /// Return the frame width (number of columns per time-step).
    ///
    /// Returns `0` if the frame sequence is empty.
    pub fn frame_width(&self) -> usize {
        self.tokens.first().map(|row| row.len()).unwrap_or(0)
    }

    /// Build an all-zeros EOS frame of the given shape.
    ///
    /// All tokens are `0` and all mask entries are `false`.
    /// Use this to signal end-of-sequence to the generation loop.
    pub fn eos(seq_len: usize, frame_width: usize) -> Self {
        Self {
            tokens: vec![vec![0i64; frame_width]; seq_len],
            mask:   vec![vec![false; frame_width]; seq_len],
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_frame_col32_set() {
        let f = Frame::from_text_tokens(&[10, 20]);
        assert_eq!(f.tokens[0][32], 10);
        assert_eq!(f.tokens[1][32], 20);
        assert!(f.mask[0][32], "text col mask must be true");
        assert!(f.mask[1][32]);
    }

    #[test]
    fn test_text_frame_audio_cols_masked_false() {
        let f = Frame::from_text_tokens(&[5]);
        for col in 0..32 {
            assert!(!f.mask[0][col], "audio col {col} should be masked off in text frame");
            assert_eq!(f.tokens[0][col], 0);
        }
    }

    #[test]
    fn test_audio_frame_shape() {
        let n_cb = 32;
        let n_frames = 5;
        let audio: Vec<Vec<i64>> = (0..n_cb).map(|cb| vec![cb as i64; n_frames]).collect();
        let f = Frame::from_audio_tokens(&audio, n_cb);
        assert_eq!(f.seq_len(), n_frames);
        assert_eq!(f.frame_width(), n_cb + 1);  // 33
    }

    #[test]
    fn test_audio_frame_mask_correct() {
        let n_cb = 4;
        let audio: Vec<Vec<i64>> = (0..n_cb).map(|_| vec![1i64; 2]).collect();
        let f = Frame::from_audio_tokens(&audio, n_cb);
        // Audio cols 0..n_cb should be true
        for col in 0..n_cb {
            assert!(f.mask[0][col], "audio col {col} should be valid");
        }
        // Text col should be false
        assert!(!f.mask[0][n_cb], "text col should be masked in audio frame");
    }

    #[test]
    fn test_concat_seq_len() {
        let a = Frame::from_text_tokens(&[1, 2, 3]);
        let b = Frame::from_text_tokens(&[4, 5]);
        let c = Frame::concat(a, b);
        assert_eq!(c.seq_len(), 5);
    }

    #[test]
    fn test_concat_tokens_preserved() {
        let a = Frame::from_text_tokens(&[10]);
        let b = Frame::from_text_tokens(&[20]);
        let c = Frame::concat(a, b);
        assert_eq!(c.tokens[0][32], 10);
        assert_eq!(c.tokens[1][32], 20);
    }

    #[test]
    fn test_eos_frame_all_false() {
        let f = Frame::eos(3, 33);
        for t in 0..3 {
            for col in 0..33 {
                assert!(!f.mask[t][col], "EOS frame mask must be all false");
                assert_eq!(f.tokens[t][col], 0, "EOS frame tokens must be 0");
            }
        }
    }

    #[test]
    fn test_frame_width_text() {
        let f = Frame::from_text_tokens(&[1, 2, 3]);
        assert_eq!(f.frame_width(), 33);
    }
}
