//! On-disk cache for pre-computed Mimi codec codes.
//!
//! During training the codec encoding step is the bottleneck — pre-encoding
//! all audio offline and caching the integer codes avoids running the Mimi
//! encoder at every training iteration.
//!
//! # File format
//! A simple binary format with no external dependencies:
//!
//! ```text
//! [n_entries : u32 LE]
//! for each entry:
//!   [sample_id : u32 LE]
//!   [n_codebooks : u32 LE]
//!   [n_frames : u32 LE]
//!   [codes : u32 LE × (n_codebooks × n_frames)]
//! ```
//!
//! Entries are appended in write order; lookup is linear scan O(n).
//! For large caches (>10 k entries) an external index should be built.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::error::DataError;

/// Cached pre-tokenised Mimi codes for fast training iteration.
///
/// On first access, use [`write_entry`](CodeCache::write_entry) to populate
/// the cache; subsequent runs call [`read_entry`](CodeCache::read_entry).
pub struct CodeCache {
    path: PathBuf,
    /// In-memory index: sample_id → (n_codebooks, n_frames, byte_offset_of_codes).
    /// Populated lazily on first read or after a write.
    index: HashMap<u32, CacheEntry>,
}

/// Metadata for one cached entry held in the in-memory index.
#[derive(Clone)]
struct CacheEntry {
    /// Number of RVQ codebooks.
    n_codebooks: usize,
    /// Number of time frames.
    n_frames: usize,
    /// Flat code data `codes[cb * n_frames + t]`.
    codes: Vec<u32>,
}

impl CodeCache {
    /// Create (or open) a cache file at `path`.
    ///
    /// If the file already exists its entries are loaded into the in-memory
    /// index.  If it does not exist, an empty cache is created in memory and
    /// the file is written on the first [`write_entry`](Self::write_entry).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mut cache = Self {
            path,
            index: HashMap::new(),
        };
        // Best-effort: load existing data.  Errors are silently ignored —
        // the cache is merely a performance optimisation; correctness is
        // maintained by re-encoding on cache miss.
        let _ = cache.load_from_disk();
        cache
    }

    /// Write a codebook matrix for `sample_id` to the cache.
    ///
    /// `codes` is `codes[codebook][frame]` with shape `[n_codebooks][n_frames]`.
    /// Overwrites any previous entry for `sample_id`.
    pub fn write_entry(&mut self, sample_id: u32, codes: &[Vec<u32>]) -> Result<(), DataError> {
        if codes.is_empty() {
            return Err(DataError::EmptySample);
        }
        let n_codebooks = codes.len();
        let n_frames = codes[0].len();

        // Flatten row-major
        let mut flat: Vec<u32> = Vec::with_capacity(n_codebooks * n_frames);
        for cb in codes {
            flat.extend_from_slice(cb);
        }

        self.index.insert(
            sample_id,
            CacheEntry {
                n_codebooks,
                n_frames,
                codes: flat,
            },
        );

        self.flush_to_disk()
    }

    /// Read the codebook matrix for `sample_id` from the cache.
    ///
    /// Returns `codes[codebook][frame]`.
    pub fn read_entry(&self, sample_id: u32) -> Result<Vec<Vec<u32>>, DataError> {
        let entry = self.index.get(&sample_id).ok_or_else(|| {
            DataError::InvalidWav(format!("cache miss for sample_id={}", sample_id))
        })?;

        let mut codes = Vec::with_capacity(entry.n_codebooks);
        for cb in 0..entry.n_codebooks {
            let start = cb * entry.n_frames;
            let end = start + entry.n_frames;
            codes.push(entry.codes[start..end].to_vec());
        }
        Ok(codes)
    }

    /// Returns `true` if a cache entry exists for `sample_id`.
    pub fn contains(&self, sample_id: u32) -> bool {
        self.index.contains_key(&sample_id)
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Serialize the entire index to disk.
    ///
    /// Format: `[n_entries: u32][sample_id: u32][n_cb: u32][n_frames: u32][codes: u32×…]…`
    fn flush_to_disk(&self) -> Result<(), DataError> {
        let mut buf: Vec<u8> = Vec::new();

        let n = self.index.len() as u32;
        buf.extend_from_slice(&n.to_le_bytes());

        for (&sid, entry) in &self.index {
            buf.extend_from_slice(&sid.to_le_bytes());
            buf.extend_from_slice(&(entry.n_codebooks as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.n_frames as u32).to_le_bytes());
            for &c in &entry.codes {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }

        // Create parent directory if needed
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut f = std::fs::File::create(&self.path)?;
        f.write_all(&buf)?;
        Ok(())
    }

    /// Deserialize the index from disk.
    fn load_from_disk(&mut self) -> Result<(), DataError> {
        if !self.path.exists() {
            return Ok(());
        }
        let mut f = std::fs::File::open(&self.path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;

        if buf.len() < 4 {
            return Ok(());
        }

        let mut pos = 0usize;
        let n_entries = read_u32_le(&buf, pos);
        pos += 4;

        for _ in 0..n_entries {
            if pos + 12 > buf.len() {
                break;
            }
            let sample_id = read_u32_le(&buf, pos);
            pos += 4;
            let n_codebooks = read_u32_le(&buf, pos) as usize;
            pos += 4;
            let n_frames = read_u32_le(&buf, pos) as usize;
            pos += 4;

            let n_codes = n_codebooks * n_frames;
            if pos + n_codes * 4 > buf.len() {
                break;
            }
            let mut codes = Vec::with_capacity(n_codes);
            for _ in 0..n_codes {
                codes.push(read_u32_le(&buf, pos));
                pos += 4;
            }
            self.index.insert(
                sample_id,
                CacheEntry {
                    n_codebooks,
                    n_frames,
                    codes,
                },
            );
        }

        Ok(())
    }
}

/// Read a little-endian `u32` from `buf` at byte offset `pos`.
fn read_u32_le(buf: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
}
