//! `opensesame-backbone` — CSM backbone transformer.
//!
//! Implements the OpenSesame / Sesame CSM backbone:
//! - Multimodal embedding (text + 8 audio codebooks, summed)
//! - 16-layer Llama-3.2-1B transformer (via `atlas_model::OlmoModel`)
//! - CB0 audio prediction head (`Linear(2048 → 2048)`)
//! - Inner Monologue text prediction head (`Linear(2048 → 128000)`)
//!
//! # Quick start
//!
//! ```rust
//! use opensesame_backbone::{BackboneModel, BackboneConfig};
//!
//! let cfg = BackboneConfig::tiny();          // 2-layer model for tests
//! let mut model = BackboneModel::new(cfg.clone());
//!
//! let T = 2usize;
//! let B = 1usize;
//! let n_cb = cfg.n_audio_codebooks;
//! let tokens_text  = vec![42u32; B * T];    // text token IDs
//! let tokens_audio = vec![7u32; B * T * n_cb]; // audio code IDs
//!
//! let out = model.forward(&tokens_text, &tokens_audio, B, T);
//! assert_eq!(out.hidden_states.len(), B * T * cfg.d_model);
//! ```

pub mod config;
pub mod embedding;
pub mod heads;
pub mod loader;
pub mod model;
pub mod sampling;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use config::{
    BackboneConfig, CsmConfig,
    AUDIO_PAD_TOKEN, AUDIO_VOCAB_SIZE, N_CODEBOOKS, TEXT_PAD_TOKEN, TEXT_VOCAB_SIZE,
};
pub use embedding::{CsmEmbedding, EmbeddingTable, compute_position_embedding};
pub use heads::{CsmAudioHead, CsmCB0Head, CsmLinear, linear_forward};
pub use loader::{LlamaWeightMapper, TorchtuneWeightMapper, load_weights_from_safetensors};
pub use model::{BackboneModel, BackboneOutput};
pub use sampling::{argmax, sample_topk, sample_topk_with_u};
