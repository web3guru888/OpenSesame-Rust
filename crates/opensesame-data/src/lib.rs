//! `opensesame-data` — Audio data pipeline for OpenSesame training.
//!
//! Provides a filesystem-based data loader for pre-extracted WAV+transcript
//! datasets (LibriTTS, GigaSpeech), a codec code cache, and batch collation.
//!
//! # Zero external crates
//! Only `atlas-*` and sibling `opensesame-*` crates are used.
//!
//! # Quick start
//! ```no_run
//! use opensesame_data::DataLoader;
//!
//! let loader = DataLoader::new("/data/libriTTS/train-clean-100", 8).unwrap();
//! for batch in loader.iter_batches() {
//!     let b = batch.unwrap();
//!     println!("batch size={}", b.batch_size());
//! }
//! ```

pub mod batch;
pub mod codec_cache;
pub mod compress;
pub mod error;
pub mod loader;
pub mod sample;

// Legacy stubs kept for workspace compatibility
pub mod cache;
pub mod deflate;
pub mod gigaspeech;
pub mod librispeech;

pub use batch::AudioBatch;
pub use codec_cache::CodeCache;
pub use error::DataError;
pub use loader::{BatchIterator, DataLoader};
pub use sample::AudioSample;
