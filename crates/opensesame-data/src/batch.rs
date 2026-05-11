//! Collated batch of [`AudioSample`](crate::sample::AudioSample)s ready for
//! training.

use crate::sample::AudioSample;

/// A collated mini-batch of audio samples.
///
/// Audio lengths and token lengths are stored separately so that the
/// training loop can apply masking or dynamic padding without carrying the
/// full padded tensors in Rust (the padding happens in the Tensor layer).
#[derive(Debug, Clone)]
pub struct AudioBatch {
    /// Raw PCM for each sample, variable length.  Shape: `[batch][samples]`.
    pub pcm: Vec<Vec<f32>>,
    /// BPE token IDs for each sample, variable length.  Shape: `[batch][tokens]`.
    pub text_tokens: Vec<Vec<u32>>,
    /// Numeric speaker ID for each sample.
    pub speaker_ids: Vec<u32>,
    /// Actual PCM lengths before any padding.
    pub lengths_pcm: Vec<usize>,
    /// Actual token sequence lengths before any padding.
    pub lengths_tokens: Vec<usize>,
}

impl AudioBatch {
    /// Collate a `Vec<AudioSample>` into an `AudioBatch`.
    ///
    /// The fields are simply gathered; no zero-padding is applied here —
    /// that is the responsibility of the tensor collator in the training loop.
    pub fn new(samples: Vec<AudioSample>) -> Self {
        let mut pcm = Vec::with_capacity(samples.len());
        let mut text_tokens = Vec::with_capacity(samples.len());
        let mut speaker_ids = Vec::with_capacity(samples.len());
        let mut lengths_pcm = Vec::with_capacity(samples.len());
        let mut lengths_tokens = Vec::with_capacity(samples.len());

        for s in samples {
            lengths_pcm.push(s.pcm.len());
            lengths_tokens.push(s.text_tokens.len());
            pcm.push(s.pcm);
            text_tokens.push(s.text_tokens);
            speaker_ids.push(s.speaker_id);
        }

        Self {
            pcm,
            text_tokens,
            speaker_ids,
            lengths_pcm,
            lengths_tokens,
        }
    }

    /// Number of samples in this batch.
    pub fn batch_size(&self) -> usize {
        self.pcm.len()
    }

    /// Maximum PCM length across all samples in this batch.
    ///
    /// Returns `0` for an empty batch.
    pub fn max_audio_len(&self) -> usize {
        self.lengths_pcm.iter().copied().max().unwrap_or(0)
    }

    /// Maximum token sequence length across all samples in this batch.
    ///
    /// Returns `0` for an empty batch.
    pub fn max_token_len(&self) -> usize {
        self.lengths_tokens.iter().copied().max().unwrap_or(0)
    }
}
