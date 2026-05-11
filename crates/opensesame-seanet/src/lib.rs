//! `opensesame-seanet` — SEANet causal conv encoder/decoder (1920× compression).
//!
//! Implements the Mimi variant of SEANet (used in Moshi/OpenSesame):
//! - Causal convolutions (left-pad only, streaming-capable)
//! - Weight normalisation on all convolutions
//! - Channel progression: 1 → 64 → 128 → 256 → 512 → 1024 → 512 (encoder)
//! - Decoder mirrors encoder with CausalConvTranspose1d + Tanh output
//! - 24 kHz input, 25 fps output (hop = 960 = 4×5×6×8)
//!
//! Phase D implementation — 50+ tests passing.

pub mod weight_norm;
pub mod causal_conv;
pub mod residual_unit;
pub mod encoder_block;
pub mod decoder_block;
pub mod encoder;
pub mod decoder;
pub mod streaming;

pub use encoder::SEANetEncoder;
pub use decoder::SEANetDecoder;
pub use streaming::StreamingEncoder;
pub use causal_conv::{CausalConv1d, CausalConvTranspose1d};
pub use residual_unit::ResidualUnit;
pub use encoder_block::EncoderBlock;
pub use decoder_block::DecoderBlock;
pub use weight_norm::WeightNorm;
