//! opensesame-eval — Audio quality evaluation metrics.
//!
//! Implements 5 metrics from scratch in pure Rust (no external audio crates):
//!
//! | Metric | Type | Higher = Better |
//! |--------|------|-----------------|
//! | SI-SNR | dB   | ✓              |
//! | WER    | rate | ✗ (lower)      |
//! | STOI   | 0–1  | ✓              |
//! | MCD    | dB   | ✗ (lower)      |
//! | ViSQOL | MOS 1–5 | ✓          |

pub mod dct;
pub mod fft;
pub mod filterbank;
pub mod mcd;
pub mod pesq;
pub mod resample;
pub mod sisnr;
pub mod stoi;
pub mod visqol;
pub mod wer;
pub mod window;
pub mod benchmark;

pub use benchmark::{BenchmarkResult, EvalSuite, EvalSuiteConfig, SingleResult};
pub use mcd::{Mcd, McdConfig};
pub use sisnr::SiSnr;
pub use stoi::{Stoi, StoiConfig};
pub use visqol::{ViSQOL, ViSQOLConfig};
pub use wer::{Wer, WerDetail};

/// Common error type for all evaluation metrics.
#[derive(Debug, Clone)]
pub enum EvalError {
    /// Signal lengths do not match.
    LengthMismatch { expected: usize, got: usize },
    /// Signal has zero length.
    EmptySignal,
    /// Unsupported or invalid sample rate.
    InvalidSampleRate(u32),
    /// FFT size is not a power of two.
    FftSizeNotPowerOfTwo(usize),
    /// Numerical problem during computation.
    NumericalError(&'static str),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {}, got {}", expected, got)
            }
            Self::EmptySignal => write!(f, "signal is empty"),
            Self::InvalidSampleRate(sr) => write!(f, "invalid sample rate: {}", sr),
            Self::FftSizeNotPowerOfTwo(n) => write!(f, "FFT size {} is not a power of two", n),
            Self::NumericalError(msg) => write!(f, "numerical error: {}", msg),
        }
    }
}

/// Convenience result alias.
pub type EvalResult<T> = Result<T, EvalError>;
