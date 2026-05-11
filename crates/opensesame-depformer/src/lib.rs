//! `opensesame-depformer` — Depth Transformer for per-frame audio codebook generation.
//!
//! Implements the **Depformer** component of the Sesame CSM architecture: a
//! lightweight 4-layer Llama-3.2-100M-scale transformer that autoregressively
//! generates audio codebook tokens CB1..CB31, conditioned on the backbone's
//! hidden state at each audio frame.
//!
//! # Architecture
//! - 4 transformer layers, `d_model=1024`, 8 query heads / 2 KV heads (GQA), `ffn=8192`
//! - Input at depth 0: projected backbone hidden state (`Linear(2048 → 1024)`)
//! - Input at depths 1–31: projected audio embedding of previous codebook token
//! - KV cache is **reset every frame** (32 positions maximum per frame)
//! - Per-depth output heads: `audio_head[i]` maps `[1024] → [2051]` logits
//!
//! # Key constants (from CSM-1B HF checkpoint)
//! - `n_codebooks = 32` (CB0..CB31)
//! - `n_dep_codebooks = 31` (CB1..CB31; CB0 from backbone)
//! - `vocab_size = 2051` (EOS=0, normal=1..2048, pad=2050)
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
//! assert_eq!(codes.len(), cfg.n_dep_codebooks);  // 31
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
