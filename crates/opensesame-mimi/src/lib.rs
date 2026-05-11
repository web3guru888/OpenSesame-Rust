//! `opensesame-mimi` — Full Mimi audio codec (Phase E).
//!
//! Implements the Kyutai Mimi neural audio codec (arXiv:2410.00037) in pure Rust,
//! as used by CSM-1B (`SesameAILabs/csm`).
//!
//! Mimi encodes 24 kHz mono audio to discrete tokens at **12.5 fps**
//! (1920 samples = 80 ms per frame) using:
//! - [`SEANetEncoder`] / [`SEANetDecoder`] — 960× strided causal convolutions
//! - [`MimiTransformer`] — 8-layer causal transformer (LayerNorm, RoPE, conv layout)
//! - [`ConvDownsample1d`] / [`ConvTrUpsample1d`] — 2× stride resampling
//! - [`SplitRVQ`] — 1 semantic + K-1 acoustic codebooks (K configurable 1..32)
//!
//! # CSM-1B usage (32 codebooks)
//! ```ignore
//! let mut codec = Mimi::new(MimiConfig::default()); // default: 32 codebooks
//! // or equivalently:
//! let mut codec = Mimi::new(MimiConfig::csm_32());
//! // set_num_codebooks mirrors Python mimi.set_num_codebooks(32)
//! codec.set_num_codebooks(32);
//! let codes = codec.encode(&audio, audio.len()); // shape: [32, T_frames]
//! ```
//!
//! # References
//! - Kyutai moshi-core (Apache 2.0): https://github.com/kyutai-labs/moshi
//! - SesameAILabs/csm (Apache 2.0): https://github.com/SesameAILabs/csm

#![warn(missing_docs)]

pub mod codec;
pub mod config;
pub mod conv;
pub mod loader;
pub mod streaming;
pub mod transformer;

// ── Public re-exports ──────────────────────────────────────────────────────

/// The full batch Mimi codec.
pub use codec::Mimi;

/// Mimi codec configuration (sample rate, codebook count, transformer dims, …).
pub use config::MimiConfig;

/// Temporal down/up-sampling convolutions.
pub use conv::{ConvDownsample1d, ConvTrUpsample1d};

/// Weight loading helpers for `kyutai/mimi` safetensors checkpoints.
pub use loader::{
    load_mimi, materialise_weight_norm, validate_mimi_checkpoint, LoadError, LoadResult,
    MIMI_KEY_PREFIXES,
};

/// Real-time streaming codec wrapper.
pub use streaming::StreamingMimi;

/// In-codec transformer and its sub-components.
pub use transformer::{
    CausalDepthwiseConv1d, FeedForward, LayerNorm, MimiTransformer, MultiHeadAttention, RoPE,
    TransformerLayer,
};

// ── Type aliases for API compatibility ────────────────────────────────────

/// Alias: [`Mimi`] — kept for consistency with the broader OpenSesame naming convention.
pub type MimiCodec = Mimi;
