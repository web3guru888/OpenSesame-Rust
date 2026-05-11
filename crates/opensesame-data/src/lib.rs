//! opensesame-data — Dataset pipeline (LibriSpeech, GigaSpeech, DEFLATE/tar).
//! Phase I implementation target: 25 tests.
pub mod librispeech;
pub mod gigaspeech;
pub mod batch;
pub mod loader;
pub mod cache;
pub mod deflate;
