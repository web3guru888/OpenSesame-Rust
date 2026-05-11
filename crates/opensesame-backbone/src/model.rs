//! CSM backbone and depth-decoder model definitions.
//!
//! This module provides the top-level struct definitions for the CSM-1B model,
//! including the `ModelConfig` type that matches `ModelArgs` in
//! `SesameAILabs/csm/models.py` (Apache 2.0).
//!
//! # Architecture (confirmed from official source)
//!
//! ```text
//! Backbone  — Llama-3.2-1B (16L, d=2048, 32H/8KV, ffn=8192)
//! Decoder   — Llama-3.2-100M (4L, d=1024, 8H/2KV, ffn=8192)
//! Embeddings:
//!   text_embeddings:   Embedding(128_256, 2048)
//!   audio_embeddings:  Embedding(65_536, 2048)  ← 32 × 2048 codebook entries
//! Projection:          Linear(2048 → 1024, bias=False)
//! codebook0_head:      Linear(2048 → 2048, bias=False)  ← predicts CB0
//! audio_head:          Parameter(31, 1024, 2048)         ← depth decoder heads CB1..31
//! ```
//!
//! # Frame representation
//! Each sequence position = one Mimi frame.  Shape `(seq_len, 33)`:
//! - Columns 0..31: audio codebook tokens (CB0..CB31)
//! - Column 32: text token
//! Masked embeddings summed along dim=2 → `(batch, seq_len, backbone_dim)`.
//!
//! # References
//! - `SesameAILabs/csm models.py` (Apache 2.0)
//! - `SesameAILabs/csm generator.py` (Apache 2.0)

use crate::config::{
    AUDIO_NUM_CODEBOOKS, AUDIO_VOCAB_SIZE, BACKBONE_DIM, DECODER_DIM, FRAME_WIDTH,
    TEXT_VOCAB_SIZE,
};

// ── BackboneFlavor ────────────────────────────────────────────────────────────

/// Which Llama 3.2 variant to use for backbone or depth decoder.
///
/// Matches the `FLAVORS` dictionary in `SesameAILabs/csm models.py`:
/// ```text
/// FLAVORS = {"llama-1B": llama3_2_1B, "llama-100M": llama3_2_100M}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneFlavor {
    /// Llama-3.2-1B: 16 layers, d=2048, 32 heads, 8 KV heads, ffn=8192.
    ///
    /// Used as the main backbone (processes full multimodal context).
    Llama1B,
    /// Llama-3.2-100M: 4 layers, d=1024, 8 heads, 2 KV heads, ffn=8192.
    ///
    /// Used as the depth decoder (generates codebooks 1..31 per frame).
    Llama100M,
}

impl BackboneFlavor {
    /// Hidden dimension for this flavor.
    pub fn embed_dim(self) -> usize {
        match self {
            Self::Llama1B   => BACKBONE_DIM,   // 2048
            Self::Llama100M => DECODER_DIM,    // 1024
        }
    }

    /// Number of transformer layers for this flavor.
    pub fn num_layers(self) -> usize {
        match self {
            Self::Llama1B   => 16,
            Self::Llama100M => 4,
        }
    }

    /// Number of query attention heads for this flavor.
    pub fn num_heads(self) -> usize {
        match self {
            Self::Llama1B   => 32,
            Self::Llama100M => 8,
        }
    }

    /// Number of key-value heads (GQA) for this flavor.
    pub fn num_kv_heads(self) -> usize {
        match self {
            Self::Llama1B   => 8,
            Self::Llama100M => 2,
        }
    }

    /// FFN intermediate dimension for this flavor.
    pub fn intermediate_dim(self) -> usize {
        // Both 1B and 100M share the same intermediate dimension.
        8192
    }

    /// RoPE base frequency for this flavor.
    pub fn rope_base(self) -> f32 {
        // Both variants use the Llama 3.2 extended context RoPE base.
        500_000.0
    }

    /// RoPE YaRN scale factor for this flavor.
    pub fn scale_factor(self) -> f32 {
        32.0
    }

    /// RMSNorm epsilon for this flavor.
    pub fn norm_eps(self) -> f32 {
        1e-5
    }

    /// Maximum sequence length for this flavor.
    pub fn max_seq_len(self) -> usize {
        2048
    }
}

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// Full CSM-1B model configuration.
///
/// Matches `ModelArgs` in `SesameAILabs/csm models.py` (Apache 2.0):
/// ```text
/// @dataclass
/// class ModelArgs:
///     backbone_flavor: str
///     decoder_flavor: str
///     text_vocab_size: int
///     audio_vocab_size: int
///     audio_num_codebooks: int
/// ```
///
/// # CSM-1B defaults
/// ```ignore
/// let cfg = ModelConfig::csm_1b();
/// assert_eq!(cfg.text_vocab_size, 128_256);
/// assert_eq!(cfg.audio_vocab_size, 2048);
/// assert_eq!(cfg.audio_num_codebooks, 32);
/// ```
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Which Llama variant to use for the backbone transformer.
    pub backbone_flavor: BackboneFlavor,
    /// Which Llama variant to use for the depth decoder transformer.
    pub decoder_flavor: BackboneFlavor,
    /// Text (BPE) vocabulary size — 128_256 for Llama 3.2 tokenizer.
    pub text_vocab_size: usize,
    /// Audio codebook vocabulary size — 2048 for Mimi RVQ.
    pub audio_vocab_size: usize,
    /// Number of audio codebooks — 32 for CSM-1B.
    ///
    /// The backbone predicts CB0; the depth decoder predicts CB1..`audio_num_codebooks-1`.
    pub audio_num_codebooks: usize,
}

impl ModelConfig {
    /// Standard CSM-1B configuration (confirmed from `SesameAILabs/csm models.py`).
    ///
    /// ```text
    /// # From generator.py:
    /// model = Model.from_pretrained("sesame/csm-1b")
    /// # From models.py:
    /// ModelArgs(
    ///     backbone_flavor="llama-1B",
    ///     decoder_flavor="llama-100M",
    ///     text_vocab_size=128_256,
    ///     audio_vocab_size=2048,
    ///     audio_num_codebooks=32,
    /// )
    /// ```
    pub fn csm_1b() -> Self {
        Self {
            backbone_flavor:      BackboneFlavor::Llama1B,
            decoder_flavor:       BackboneFlavor::Llama100M,
            text_vocab_size:      TEXT_VOCAB_SIZE,    // 128_256
            audio_vocab_size:     AUDIO_VOCAB_SIZE,   // 2_048
            audio_num_codebooks:  AUDIO_NUM_CODEBOOKS, // 32
        }
    }

    /// Total number of audio embedding table entries.
    ///
    /// `audio_vocab_size × audio_num_codebooks = 2048 × 32 = 65_536`.
    /// Each codebook `k` occupies entries `[k × audio_vocab_size, (k+1) × audio_vocab_size)`.
    pub fn audio_embedding_count(&self) -> usize {
        self.audio_vocab_size * self.audio_num_codebooks
    }

    /// Frame width = number of token columns per sequence position.
    ///
    /// `audio_num_codebooks + 1` = 33 for CSM-1B.
    /// Column 32 is the text token; columns 0..31 are audio codebook tokens.
    pub fn frame_width(&self) -> usize {
        self.audio_num_codebooks + 1
    }

    /// Number of depth-decoder steps per frame (CB1..CB31 = 31 steps).
    ///
    /// The backbone predicts CB0 directly; the depth decoder handles the rest.
    pub fn depth_steps(&self) -> usize {
        self.audio_num_codebooks - 1
    }

    /// Shape of the `audio_head` parameter: `(depth_steps, decoder_dim, audio_vocab_size)`.
    ///
    /// = `(31, 1024, 2048)` for CSM-1B.
    pub fn audio_head_shape(&self) -> (usize, usize, usize) {
        (
            self.depth_steps(),
            self.decoder_flavor.embed_dim(),
            self.audio_vocab_size,
        )
    }
}

// ── CSM weight names ──────────────────────────────────────────────────────────

/// Top-level parameter names for a CSM-1B checkpoint, matching
/// `SesameAILabs/csm models.py` / HuggingFace safetensors layout.
pub struct CsmWeightKeys;

impl CsmWeightKeys {
    /// Text embedding table: `Embedding(128_256, 2048)`.
    pub const TEXT_EMBEDDINGS:   &'static str = "text_embeddings.weight";
    /// Audio embedding table: `Embedding(65_536, 2048)`.
    pub const AUDIO_EMBEDDINGS:  &'static str = "audio_embeddings.weight";
    /// Backbone → decoder projection: `Linear(2048, 1024, bias=False)`.
    pub const PROJECTION:        &'static str = "projection.weight";
    /// CB0 prediction head: `Linear(2048, 2048, bias=False)`.
    pub const CODEBOOK0_HEAD:    &'static str = "codebook0_head.weight";
    /// Depth decoder audio heads: `Parameter(31, 1024, 2048)`.
    pub const AUDIO_HEAD:        &'static str = "audio_head";
    /// Backbone transformer layers prefix.
    pub const BACKBONE_PREFIX:   &'static str = "backbone.";
    /// Depth decoder transformer layers prefix.
    pub const DECODER_PREFIX:    &'static str = "decoder.";
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_csm1b_vocab() {
        let cfg = ModelConfig::csm_1b();
        assert_eq!(cfg.text_vocab_size, 128_256, "text vocab = 128_256");
        assert_eq!(cfg.audio_vocab_size, 2_048, "audio vocab = 2048");
        assert_eq!(cfg.audio_num_codebooks, 32, "32 codebooks");
    }

    #[test]
    fn test_model_config_frame_width() {
        let cfg = ModelConfig::csm_1b();
        assert_eq!(cfg.frame_width(), 33, "frame_width = 33 (32 audio + 1 text)");
    }

    #[test]
    fn test_model_config_audio_embedding_count() {
        let cfg = ModelConfig::csm_1b();
        assert_eq!(cfg.audio_embedding_count(), 65_536, "65_536 audio embedding entries");
    }

    #[test]
    fn test_model_config_depth_steps() {
        let cfg = ModelConfig::csm_1b();
        assert_eq!(cfg.depth_steps(), 31, "31 depth steps (CB1..31)");
    }

    #[test]
    fn test_audio_head_shape() {
        let cfg = ModelConfig::csm_1b();
        let shape = cfg.audio_head_shape();
        assert_eq!(shape, (31, 1024, 2048), "audio_head shape (31, 1024, 2048)");
    }

    #[test]
    fn test_backbone_flavor_dims() {
        assert_eq!(BackboneFlavor::Llama1B.embed_dim(), 2048);
        assert_eq!(BackboneFlavor::Llama100M.embed_dim(), 1024);
        assert_eq!(BackboneFlavor::Llama1B.num_layers(), 16);
        assert_eq!(BackboneFlavor::Llama100M.num_layers(), 4);
        assert_eq!(BackboneFlavor::Llama1B.num_heads(), 32);
        assert_eq!(BackboneFlavor::Llama100M.num_heads(), 8);
        assert_eq!(BackboneFlavor::Llama1B.num_kv_heads(), 8);
        assert_eq!(BackboneFlavor::Llama100M.num_kv_heads(), 2);
    }

    #[test]
    fn test_backbone_flavor_ffn_rope() {
        assert_eq!(BackboneFlavor::Llama1B.intermediate_dim(), 8192);
        assert_eq!(BackboneFlavor::Llama100M.intermediate_dim(), 8192);
        assert!((BackboneFlavor::Llama1B.rope_base() - 500_000.0).abs() < 1.0);
        assert!((BackboneFlavor::Llama1B.scale_factor() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_width_constant() {
        assert_eq!(FRAME_WIDTH, 33, "FRAME_WIDTH = AUDIO_NUM_CODEBOOKS + 1 = 33");
    }

    #[test]
    fn test_constants_verified() {
        assert_eq!(AUDIO_NUM_CODEBOOKS, 32, "32 codebooks (CSM-1B)");
        assert_eq!(AUDIO_VOCAB_SIZE, 2_048, "2048 audio vocab");
        assert_eq!(TEXT_VOCAB_SIZE, 128_256, "128_256 text vocab");
        assert_eq!(BACKBONE_DIM, 2048, "backbone dim = 2048");
        assert_eq!(DECODER_DIM, 1024, "decoder dim = 1024");
    }

    #[test]
    fn test_weight_key_constants() {
        assert!(CsmWeightKeys::TEXT_EMBEDDINGS.contains("text_embeddings"));
        assert!(CsmWeightKeys::AUDIO_EMBEDDINGS.contains("audio_embeddings"));
        assert!(CsmWeightKeys::PROJECTION.contains("projection"));
        assert!(CsmWeightKeys::CODEBOOK0_HEAD.contains("codebook0_head"));
        assert!(CsmWeightKeys::AUDIO_HEAD.contains("audio_head"));
    }
}
