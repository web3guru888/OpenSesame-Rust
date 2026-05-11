//! Error types for the opensesame-data pipeline.

use std::fmt;

/// Errors produced by the data loading pipeline.
#[derive(Debug)]
pub enum DataError {
    /// Underlying I/O error (file not found, permission denied, etc.)
    Io(std::io::Error),
    /// WAV file could not be parsed or has an unsupported format.
    InvalidWav(String),
    /// Transcript file could not be read or decoded as UTF-8.
    InvalidTranscript(String),
    /// Sample has zero or negligible content (below minimum duration).
    EmptySample,
    /// Requested batch index is out of range.
    BatchOutOfRange {
        /// The requested batch index.
        batch_idx: usize,
        /// Total number of available batches.
        num_batches: usize,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Io(e) => write!(f, "I/O error: {}", e),
            DataError::InvalidWav(msg) => write!(f, "invalid WAV: {}", msg),
            DataError::InvalidTranscript(msg) => write!(f, "invalid transcript: {}", msg),
            DataError::EmptySample => write!(f, "sample is empty or too short"),
            DataError::BatchOutOfRange { batch_idx, num_batches } => {
                write!(
                    f,
                    "batch index {} out of range (total batches: {})",
                    batch_idx, num_batches
                )
            }
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        DataError::Io(e)
    }
}
