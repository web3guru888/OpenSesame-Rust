//! `opensesame-depformer` — Depth Transformer for per-frame audio codebook generation.
//!
//! Implements the **Depformer** component of the Sesame CSM architecture: a
//! lightweight 4-layer Llama-3.2-100M-scale transformer that autoregressively
//! generates audio codebook tokens CB1..CB7, conditioned on the backbone's
//! hidden state at each audio frame.
//!
//! # Architecture
//! - 4 transformer layers, `d_model=1024`, 8 query heads / 2 KV heads (GQA), `ffn=8192`
//! - Input at depth 0: projected backbone hidden state `Linear(2048 → 1024)`
//! - Input at depths 1–7: projected audio embedding of previous codebook token
//! - KV cache is **reset every frame** (8 positions maximum per frame)
//! - Per-depth output heads: `audio_head[i]` maps `[1024] → [2048]` logits
//!
//! # Key constants (from CSM-1B)
//! - `n_codebooks = 8` (CB0..CB7)
//! - `n_dep_codebooks = 7` (CB1..CB7; CB0 from backbone)
//! - `vocab_size = 2048` (audio tokens per codebook)
//! - `d_model = 1024` (depformer hidden dim)
//! - `d_backbone = 2048` (backbone hidden dim, for projection)
//!
//! # Quick example
//! ```ignore
//! let cfg = DepformerConfig::opensesame_1b();
//! let mut dep = Depformer::new(cfg.clone());
//! let proj_h = vec![0.0f32; cfg.d_model];   // projected backbone hidden
//! let codes = dep.generate_depth_sequence(
//!     proj_h,
//!     |_, _| vec![0.0f32; cfg.d_model],     // embed_fn (dummy)
//!     1.0,   // temperature
//!     50,    // topk
//!     0,     // cb0
//! );
//! assert_eq!(codes.len(), cfg.n_dep_codebooks);  // 7
//! ```

pub mod config;
pub mod depformer;
pub mod head;
pub mod linear;
pub mod sampling;
pub mod weight_loader;

// ── Public re-exports ─────────────────────────────────────────────────────────

/// Depformer configuration.
pub use config::DepformerConfig;

/// Depth Transformer struct and its generation logic.
pub use depformer::Depformer;

/// Per-depth audio output heads.
pub use head::CsmAudioHead;

/// Lightweight no-bias linear layer (for the backbone→depformer projection).
pub use linear::CsmLinear;

/// Top-k temperature sampling with greedy fallback.
pub use sampling::sample_topk;

/// Safetensors weight loader.
pub use weight_loader::{
    load_depformer_from_safetensors,
    DEPFORMER_AUDIO_HEAD_KEY,
    DEPFORMER_LAYER_PREFIX,
    DEPFORMER_NORM_KEY,
};
