//! TODO: DEFLATE / GZIP / TAR decompression for streaming dataset archives.
//!
//! Phase I focuses on pre-extracted filesystem datasets so compression is not
//! yet needed.  This module is a forward-compatibility stub.
//!
//! # Planned implementation
//! - `inflate(data: &[u8]) -> Result<Vec<u8>, DataError>` — RFC 1951 DEFLATE
//!   - BTYPE=00 stored blocks
//!   - BTYPE=01 static Huffman
//!   - BTYPE=10 dynamic Huffman
//! - `gzip_decompress(data: &[u8]) -> Result<Vec<u8>, DataError>` — RFC 1952
//! - `TarReader` — iterate 512-byte POSIX tar headers
//!
//! All implemented without external crates.

use crate::error::DataError;

/// Placeholder — will decompress a DEFLATE-compressed byte slice (RFC 1951).
///
/// # Errors
/// Always returns `Err(DataError::InvalidWav("DEFLATE not yet implemented"))`.
#[allow(dead_code)]
pub fn inflate(_data: &[u8]) -> Result<Vec<u8>, DataError> {
    Err(DataError::InvalidWav(
        "DEFLATE decompression not yet implemented (Phase I stub)".to_string(),
    ))
}

/// Placeholder — will decompress a GZIP-wrapped byte slice (RFC 1952).
///
/// # Errors
/// Always returns `Err(DataError::InvalidWav("GZIP not yet implemented"))`.
#[allow(dead_code)]
pub fn gzip_decompress(_data: &[u8]) -> Result<Vec<u8>, DataError> {
    Err(DataError::InvalidWav(
        "GZIP decompression not yet implemented (Phase I stub)".to_string(),
    ))
}
