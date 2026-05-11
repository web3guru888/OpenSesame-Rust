//! opensesame-audio — Audio I/O, DSP, resampling, and streaming primitives.
//!
//! Phase A implementation target: 30 tests.
//! Zero external dependencies. Pure Rust + CUDA via atlas-tensor.

pub mod audio_buffer;
pub mod vad;
pub mod wav;
pub mod ring_buffer;
pub mod resample;
pub mod dsp;
