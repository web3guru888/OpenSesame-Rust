//! `opensesame-mimi` — Full Mimi audio codec.
//!
//! Implements the complete Mimi tokenizer from Kyutai's Moshi system:
//!
//! * [`MimiConfig`] — exact production hyperparameters (sample_rate=24000, fps=12.5, …)
//! * [`MimiCodec`] — encode/decode pipeline (SEANet + transformer + SplitRVQ)
//! * [`MimiTransformer`] — in-codec causal transformer (8L, 8H, d=512, LayerNorm+RoPE)
//! * [`LayerNorm`] — standard layer normalisation used by the transformer
//! * [`SafetensorsFile`] — safetensors weight loader (F32 + BF16, no external deps)
//! * [`MimiStreamingSession`] — chunk-by-chunk streaming encode/decode session
//!
//! # Phase E
//! The transformer is an identity stub (pass-through); the ConvDownsample (25→12.5 fps)
//! is not yet wired. Full attention + weight loading will be added in Phase F.
//!
//! # References
//! * Moshi paper: arXiv:2410.00037
//! * Kyutai moshi-core Rust source: Apache 2.0

pub mod codec;
pub mod config;
pub mod loader;
pub mod transformer;
pub mod streaming;

pub use codec::MimiCodec;
pub use config::MimiConfig;
pub use loader::SafetensorsFile;
pub use streaming::MimiStreamingSession;
pub use transformer::{LayerNorm, MimiTransformer, MimiTransformerLayer};
