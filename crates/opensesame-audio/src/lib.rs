//! `opensesame-audio` — Audio I/O, DSP, resampling, and streaming primitives.
//!
//! This crate provides the foundational audio layer for OpenSesame-Rust.
//! Every other OpenSesame crate that touches audio depends on this one.
//!
//! # Modules
//! * [`audio_buffer`] — [`AudioBuffer`] struct with convenience methods
//! * [`wav`]          — Zero-dependency WAV file parser / writer
//! * [`resample`]     — Kaiser-windowed sinc resampler (CPU)
//! * [`ring_buffer`]  — Lock-free ring buffer for streaming audio
//! * [`vad`]          — Voice Activity Detection (energy + hangover)
//! * [`dsp`]          — Utility DSP functions (RMS, gain, clip, window, …)
//!
//! # Zero external dependencies
//! Only `atlas-core` (for `AtlasError` / `Result`) is used as a dependency.
//! All implementations are pure Rust.

#![warn(missing_docs)]

pub mod audio_buffer;
pub mod dsp;
pub mod resample;
pub mod ring_buffer;
pub mod vad;
pub mod wav;

// Re-export the most commonly used types at the crate root.
pub use audio_buffer::AudioBuffer;
pub use resample::Resampler;
pub use ring_buffer::RingBuffer;
pub use vad::VadState;
pub use wav::{WavReader, WavWriter};
