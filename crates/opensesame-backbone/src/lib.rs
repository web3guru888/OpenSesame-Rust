//! `opensesame-backbone` — CSM backbone and depth decoder.
//!
//! Implements the Sesame CSM-1B model components, exactly matching
//! `SesameAILabs/csm models.py` (Apache 2.0):
//!
//! - [`ModelConfig`] / [`BackboneFlavor`] — full model type definitions
//! - Backbone: Llama-3.2-1B (16L, d=2048, 32H/8KV, ffn=8192)
//! - Depth decoder: Llama-3.2-100M (4L, d=1024, 8H/2KV, ffn=8192)
//! - Multimodal embeddings: text (128_256 vocab) + audio (65_536 = 32×2048 entries)
//! - Projection: Linear(2048 → 1024)
//! - CB0 head: Linear(2048 → 2048)
//! - Audio head: Parameter(31, 1024, 2048)
//!
//! # Key constants (confirmed from official source)
//! - [`AUDIO_NUM_CODEBOOKS`] = 32
//! - [`AUDIO_VOCAB_SIZE`] = 2048
//! - [`TEXT_VOCAB_SIZE`] = 128_256
//! - [`FRAME_WIDTH`] = 33  (32 audio + 1 text column)
//! - [`BACKBONE_DIM`] = 2048
//! - [`DECODER_DIM`] = 1024

pub mod config;
pub mod embedding;
pub mod loader;
pub mod model;

// ── Public re-exports ─────────────────────────────────────────────────────────

/// Architecture constants — use these rather than hard-coded numbers.
pub use config::{
    AUDIO_NUM_CODEBOOKS,
    AUDIO_PAD_TOKEN,
    AUDIO_VOCAB_SIZE,
    BACKBONE_DIM,
    DECODER_DIM,
    FRAME_WIDTH,
    N_CODEBOOKS,
    TEXT_PAD_TOKEN,
    TEXT_VOCAB_SIZE,
};

/// Backbone and depth-decoder configuration structs.
pub use config::{BackboneConfig, CsmConfig};

/// Top-level model configuration matching Python `ModelArgs`.
pub use model::{BackboneFlavor, CsmWeightKeys, ModelConfig};
