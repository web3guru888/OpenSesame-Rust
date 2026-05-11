//! `opensesame-rvq` — Residual Vector Quantization for the Mimi audio codec.
//!
//! Implements three layers of vector quantisation:
//!
//! * [`VectorQuantizer`] — single VQ with EMA codebook updates (SoundStream §3-5)
//! * [`ResidualVQ`]      — chains N quantizers on successive residuals
//! * [`SplitRVQ`]        — Mimi variant: CB0 semantic (frozen) + CB1..7 acoustic
//!
//! All computation is pure Rust CPU (`Vec<f32>`). CUDA kernels (`vq_search.cu`,
//! `ema_update.cu`) will be wired in Phase B.
//!
//! # References
//! * SoundStream (arXiv:2107.03312) — EMA equations 3–5
//! * Encodec   (arXiv:2210.13438)  — STE aggregate fix
//! * Moshi     (arXiv:2410.00037)  — Split-RVQ, `core_vq.py`

pub mod config;
pub mod vq;
pub mod residual_vq;
pub mod split_rvq;

pub use config::RVQConfig;
pub use vq::{VectorQuantizer, VQOutput};
pub use residual_vq::{ResidualVQ, RVQOutput};
pub use split_rvq::{SplitRVQ, SplitRVQOutput};
