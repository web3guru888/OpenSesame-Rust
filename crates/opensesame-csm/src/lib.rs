//! `opensesame-csm` — Full CSM model assembly and streaming session.
//!
//! Implements the complete Conversational Speech Model (CSM) by wiring together:
//! - **Mimi** audio codec ([`opensesame-mimi`]) — encode PCM ↔ discrete codes
//! - **Backbone** transformer ([`opensesame-backbone`]) — Llama-3.2-1B multimodal
//! - **Depformer** depth transformer ([`opensesame-depformer`]) — CB1..CB31 per frame
//!
//! # Quick start
//! ```ignore
//! use opensesame_csm::{CsmModel, CsmSession};
//!
//! // Build a tiny test model
//! let mut model = CsmModel::new_tiny();
//!
//! // Generate 4 frames (no context)
//! let out = model.generate(&[], &[], 4, 1.0, 50);
//! println!("Generated {} PCM samples", out.pcm.len());
//!
//! // Streaming session
//! let mut session = CsmSession::new(model);
//! session.push_audio(&[]);
//! let frame_pcm = session.generate_next_frame(Some(42));
//! ```
//!
//! # Architecture
//! ```text
//! CsmModel
//!  ├── mimi:      Mimi               — encode/decode PCM ↔ RVQ codes
//!  ├── backbone:  atlas_model::OlmoModel  — Llama-3.2-1B backbone transformer
//!  ├── embedding: CsmEmbedding        — multimodal text+audio embedding tables
//!  ├── depformer: Depformer           — Llama-3.2-100M depth decoder
//!  ├── cb0_head:  Projection          — backbone_dim → audio_vocab (CB0 head)
//!  └── proj:      Projection          — backbone_dim → decoder_dim (projection)
//! ```

pub mod config;
pub mod frame;
pub mod model;
pub mod projection;
pub mod session;
pub mod weight_loader;
pub mod loss;

// ── Public re-exports ─────────────────────────────────────────────────────────

/// Full model configuration (Mimi + Backbone + Depformer), composite style.
pub use config::CsmModelConfig;

/// Flat model configuration with all fields explicit (for weight loading).
pub use config::CsmConfig;

/// The CSM model struct and generation output.
pub use model::{CsmModel, GenerateOutput};

/// No-bias linear projection layer.
pub use projection::Projection;

/// Streaming generation session.
pub use session::CsmSession;

/// Frame tokenization helpers.
pub use frame::Frame;

/// Safetensors weight loader.
pub use weight_loader::{
    load_csm_from_safetensors,
    CsmError,
    KEY_TEXT_EMBEDDINGS,
    KEY_AUDIO_EMBEDDINGS,
    KEY_CODEBOOK0_HEAD,
    KEY_AUDIO_HEAD,
    KEY_EMBEDS_PROJECTOR,
    BACKBONE_LAYER_PREFIX,
    DECODER_LAYER_PREFIX,
};
