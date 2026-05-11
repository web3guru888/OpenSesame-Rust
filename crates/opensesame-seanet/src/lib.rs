//! opensesame-seanet — SEANet causal conv encoder/decoder (1920× compression).
//! Phase D implementation target: 50 tests.
pub mod causal_conv;
pub mod residual_unit;
pub mod encoder_block;
pub mod decoder_block;
pub mod encoder;
pub mod decoder;
pub mod streaming;
pub mod weight_norm;
