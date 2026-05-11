//! Safetensors weight loader for Mimi pretrained models.
//!
//! The safetensors format (HuggingFace):
//! - Bytes 0–7: header size N as u64 little-endian.
//! - Bytes 8..8+N: UTF-8 JSON header describing every tensor.
//! - Bytes 8+N..: raw tensor data (tensor byte ranges are relative to this region).
//!
//! JSON header format per tensor:
//! ```json
//! {
//!   "tensor_name": {
//!     "dtype": "F32",              // or "BF16", "I32", etc.
//!     "shape": [2, 512],
//!     "data_offsets": [0, 4096]    // [begin, end) within data region
//!   },
//!   "__metadata__": { ... }        // ignored
//! }
//! ```
//!
//! This module supports `F32` and `BF16` dtypes. BF16→F32 conversion reinterprets
//! the u16 bits as the upper 16 bits of a u32 float (the lower 16 mantissa bits
//! are set to zero).
//!
//! Uses `atlas-json` for header parsing — no additional dependencies.

use std::collections::HashMap;
use atlas_json::Json;

// ─── TensorMeta ──────────────────────────────────────────────────────────────

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Data type string from the header (e.g. `"F32"`, `"BF16"`).
    pub dtype: String,
    /// Tensor shape, e.g. `[512, 256]`.
    pub shape: Vec<usize>,
    /// Byte offset where this tensor's data begins (relative to the data region).
    pub data_begin: usize,
    /// Byte offset where this tensor's data ends (exclusive; relative to data region).
    pub data_end: usize,
}

// ─── SafetensorsFile ─────────────────────────────────────────────────────────

/// A loaded safetensors file: parsed header + raw data bytes.
pub struct SafetensorsFile {
    /// Map from tensor name to its metadata.
    pub tensors: HashMap<String, TensorMeta>,
    /// Raw data region (all tensors' bytes concatenated).
    pub data: Vec<u8>,
}

impl SafetensorsFile {
    /// Open and parse a `.safetensors` file.
    ///
    /// Reads the 8-byte header size, parses the JSON header with `atlas-json`,
    /// and stores the data region for lazy tensor reads.
    ///
    /// # Errors
    /// Returns `Err(String)` on I/O failure, malformed header, or unsupported structure.
    pub fn open(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot open '{}': {}", path, e))?;

        if bytes.len() < 8 {
            return Err(format!(
                "'{}': file too small ({} bytes) for safetensors header size field",
                path,
                bytes.len()
            ));
        }

        // Read the 8-byte header size.
        let header_size = u64::from_le_bytes(
            bytes[0..8].try_into().expect("slice length 8")
        ) as usize;

        if bytes.len() < 8 + header_size {
            return Err(format!(
                "'{}': declared header size {} but file only has {} bytes after offset 8",
                path,
                header_size,
                bytes.len() - 8
            ));
        }

        // Parse the JSON header.
        let header_str = std::str::from_utf8(&bytes[8..8 + header_size])
            .map_err(|e| format!("'{}': header is not valid UTF-8: {}", path, e))?;

        let parsed = Json::parse(header_str)
            .map_err(|e| format!("'{}': JSON parse error: {}", path, e))?;

        let pairs = parsed
            .as_object()
            .ok_or_else(|| format!("'{}': JSON root must be an object", path))?;

        let mut tensors = HashMap::new();

        for (name, meta_json) in pairs {
            // Skip the metadata block.
            if name == "__metadata__" {
                continue;
            }

            let dtype = meta_json
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("'{}': tensor '{}' missing 'dtype'", path, name))?
                .to_string();

            let shape_arr = meta_json
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("'{}': tensor '{}' missing 'shape'", path, name))?;

            let shape: Vec<usize> = shape_arr
                .iter()
                .map(|v| {
                    v.as_usize()
                        .ok_or_else(|| format!("'{}': tensor '{}' has non-integer shape", path, name))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let offsets = meta_json
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    format!("'{}': tensor '{}' missing 'data_offsets'", path, name)
                })?;

            if offsets.len() != 2 {
                return Err(format!(
                    "'{}': tensor '{}' data_offsets must have exactly 2 elements",
                    path, name
                ));
            }

            let data_begin = offsets[0]
                .as_usize()
                .ok_or_else(|| format!("'{}': tensor '{}' data_offsets[0] not integer", path, name))?;
            let data_end = offsets[1]
                .as_usize()
                .ok_or_else(|| format!("'{}': tensor '{}' data_offsets[1] not integer", path, name))?;

            tensors.insert(
                name.clone(),
                TensorMeta { dtype, shape, data_begin, data_end },
            );
        }

        // Data region starts right after the header.
        let data = bytes[8 + header_size..].to_vec();

        Ok(SafetensorsFile { tensors, data })
    }

    /// Read a tensor as `Vec<f32>`.
    ///
    /// Handles `F32` (direct byte copy) and `BF16` (converted to F32).
    ///
    /// # Errors
    /// Returns `Err` if the tensor name is not found or the dtype is unsupported.
    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor '{}' not found in safetensors file", name))?;

        if meta.data_end > self.data.len() {
            return Err(format!(
                "tensor '{}': data_end {} exceeds data region size {}",
                name,
                meta.data_end,
                self.data.len()
            ));
        }

        let raw = &self.data[meta.data_begin..meta.data_end];

        match meta.dtype.as_str() {
            "F32" => {
                if raw.len() % 4 != 0 {
                    return Err(format!(
                        "tensor '{}': F32 data length {} not divisible by 4",
                        name,
                        raw.len()
                    ));
                }
                let mut out = vec![0.0_f32; raw.len() / 4];
                for (i, chunk) in raw.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes(chunk.try_into().expect("4 bytes"));
                }
                Ok(out)
            }
            "BF16" => {
                if raw.len() % 2 != 0 {
                    return Err(format!(
                        "tensor '{}': BF16 data length {} not divisible by 2",
                        name,
                        raw.len()
                    ));
                }
                let mut out = vec![0.0_f32; raw.len() / 2];
                for (i, chunk) in raw.chunks_exact(2).enumerate() {
                    let bf16 = u16::from_le_bytes(chunk.try_into().expect("2 bytes"));
                    out[i] = bf16_to_f32(bf16);
                }
                Ok(out)
            }
            other => Err(format!("tensor '{}': unsupported dtype '{}'", name, other)),
        }
    }

    /// List all tensor names in the file (excluding `__metadata__`).
    pub fn keys(&self) -> Vec<&str> {
        let mut ks: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        ks.sort();
        ks
    }

    /// Check whether a tensor with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

// ─── BF16 conversion ─────────────────────────────────────────────────────────

/// Convert a bfloat16 value to `f32`.
///
/// BF16 occupies bits [31:16] of a 32-bit float; the lower 16 mantissa bits
/// are zero-filled. This is a lossless layout conversion (no arithmetic).
///
/// ```text
/// BF16 0x3F80 → F32 0x3F80_0000 = 1.0
/// ```
pub(crate) fn bf16_to_f32(bf16: u16) -> f32 {
    let bits = (bf16 as u32) << 16;
    f32::from_bits(bits)
}

// ─── Synthetic safetensors builder (for tests) ───────────────────────────────

/// Build a minimal valid safetensors binary in memory.
///
/// Useful for unit tests that need a real file without depending on model weights.
///
/// # Arguments
/// * `tensors` – list of `(name, dtype_str, shape, flat_f32_data)`.
pub(crate) fn build_synthetic_safetensors(
    tensors: &[(&str, &str, Vec<usize>, Vec<f32>)],
) -> Vec<u8> {
    // First, lay out the data region and compute byte offsets.
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<(usize, usize)> = Vec::new();

    for (_, dtype, _, values) in tensors {
        let start = data_bytes.len();
        match *dtype {
            "F32" => {
                for &v in values {
                    data_bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            "BF16" => {
                for &v in values {
                    // Convert F32 → BF16 by dropping lower 16 bits.
                    let bits = v.to_bits();
                    let bf16 = (bits >> 16) as u16;
                    data_bytes.extend_from_slice(&bf16.to_le_bytes());
                }
            }
            _ => panic!("build_synthetic_safetensors: unknown dtype {}", dtype),
        }
        offsets.push((start, data_bytes.len()));
    }

    // Build the JSON header.
    let mut header = String::from("{");
    for (i, (name, dtype, shape, _)) in tensors.iter().enumerate() {
        if i > 0 { header.push(','); }
        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let (begin, end) = offsets[i];
        header.push_str(&format!(
            r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
            name,
            dtype,
            shape_str.join(","),
            begin,
            end
        ));
    }
    header.push('}');

    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_bytes);
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write bytes to a temporary file and return the path.
    fn write_temp(bytes: &[u8]) -> String {
        let path = format!("/tmp/opensesame_mimi_test_{}.safetensors", std::process::id());
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(bytes).expect("write temp file");
        path
    }

    #[test]
    fn test_safetensors_nonexistent() {
        let result = SafetensorsFile::open("/tmp/this_file_does_not_exist_opensesame.safetensors");
        assert!(result.is_err(), "open non-existent file must return Err");
    }

    #[test]
    fn test_safetensors_synthetic() {
        // Build a tiny synthetic safetensors with one F32 tensor [2, 3].
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = build_synthetic_safetensors(&[("my_tensor", "F32", vec![2, 3], data.clone())]);
        let path = write_temp(&bytes);

        let sf = SafetensorsFile::open(&path).expect("load synthetic safetensors");
        assert!(sf.contains("my_tensor"), "tensor must be present");

        let loaded = sf.get_f32("my_tensor").expect("get_f32");
        assert_eq!(loaded, data, "loaded tensor must match written data");
    }

    #[test]
    fn test_safetensors_bf16_convert() {
        // BF16 0x3F80 → F32 0x3F800000 = 1.0
        let result = bf16_to_f32(0x3F80);
        assert_eq!(result, 1.0_f32, "BF16 0x3F80 must convert to 1.0f32");
    }

    #[test]
    fn test_safetensors_bf16_convert_zero() {
        // BF16 0x0000 = 0.0
        assert_eq!(bf16_to_f32(0x0000), 0.0_f32);
    }

    #[test]
    fn test_safetensors_bf16_convert_neg_one() {
        // BF16 0xBF80 = -1.0
        assert_eq!(bf16_to_f32(0xBF80), -1.0_f32);
    }

    #[test]
    fn test_safetensors_f32_passthrough() {
        // F32 tensors must round-trip without modification.
        let data: Vec<f32> = vec![std::f32::consts::PI, -1.5, 42.0];
        let bytes = build_synthetic_safetensors(&[("pi_tensor", "F32", vec![3], data.clone())]);
        let path = write_temp(&bytes);

        let sf = SafetensorsFile::open(&path).expect("load");
        let loaded = sf.get_f32("pi_tensor").expect("get_f32");
        for (a, b) in data.iter().zip(loaded.iter()) {
            assert_eq!(*a, *b, "F32 must round-trip exactly");
        }
    }

    #[test]
    fn test_safetensors_keys_listed() {
        let bytes = build_synthetic_safetensors(&[
            ("weight", "F32", vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            ("bias", "F32", vec![4], vec![0.1, 0.2, 0.3, 0.4]),
        ]);
        let path = write_temp(&bytes);

        let sf = SafetensorsFile::open(&path).expect("load");
        let keys = sf.keys();
        assert!(keys.contains(&"weight"), "must list 'weight'");
        assert!(keys.contains(&"bias"), "must list 'bias'");
        assert_eq!(keys.len(), 2, "must list exactly 2 tensors");
    }

    #[test]
    fn test_safetensors_contains() {
        let bytes = build_synthetic_safetensors(&[
            ("alpha", "F32", vec![2], vec![0.0, 1.0]),
        ]);
        let path = write_temp(&bytes);
        let sf = SafetensorsFile::open(&path).expect("load");
        assert!(sf.contains("alpha"), "must contain 'alpha'");
        assert!(!sf.contains("beta"), "must not contain 'beta'");
    }

    #[test]
    fn test_safetensors_bf16_tensor() {
        // Write a BF16 tensor with value 2.0.
        // BF16 of 2.0 = 0x4000
        let data_bf16: Vec<u8> = vec![0x00, 0x40]; // 2.0 in BF16 LE
        // Use the builder with F32 values that convert to BF16 correctly.
        let data_f32 = vec![2.0_f32];
        let bytes = build_synthetic_safetensors(&[("x", "BF16", vec![1], data_f32)]);
        let path = write_temp(&bytes);
        let sf = SafetensorsFile::open(&path).expect("load");
        let vals = sf.get_f32("x").expect("get bf16 tensor");
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0], 2.0_f32, "BF16 2.0 must convert to f32 2.0");
        let _ = data_bf16; // suppress unused warning
    }

    #[test]
    fn test_safetensors_metadata_skipped() {
        // __metadata__ should not appear in tensor list.
        // Build manually since the helper doesn't emit __metadata__.
        let header = r#"{"__metadata__":{"version":"1.0"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let header_bytes = header.as_bytes();
        let hlen = header_bytes.len() as u64;
        let mut bytes: Vec<u8> = hlen.to_le_bytes().to_vec();
        bytes.extend_from_slice(header_bytes);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());

        let path = write_temp(&bytes);
        let sf = SafetensorsFile::open(&path).expect("load");
        assert!(
            !sf.contains("__metadata__"),
            "__metadata__ must not appear in tensor map"
        );
        assert!(sf.contains("w"), "real tensor 'w' must be present");
        assert_eq!(sf.keys().len(), 1, "only 1 real tensor");
    }
}
