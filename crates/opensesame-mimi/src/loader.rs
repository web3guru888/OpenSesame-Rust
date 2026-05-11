//! Safetensors weight loader for the Mimi audio codec.

use crate::codec::Mimi;
use crate::config::MimiConfig;

/// Error type for Mimi weight loading.
#[derive(Debug)]
pub enum LoadError {
    /// File not found or unreadable.
    IoError(String),
    /// Safetensors header could not be parsed.
    ParseError(String),
    /// A required tensor was absent from the checkpoint.
    MissingTensor(String),
    /// A tensor had an unexpected shape.
    ShapeMismatch { key: String, expected: Vec<usize>, got: Vec<usize> },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::IoError(s)        => write!(f, "IO error: {s}"),
            LoadError::ParseError(s)     => write!(f, "parse error: {s}"),
            LoadError::MissingTensor(k)  => write!(f, "missing tensor: {k}"),
            LoadError::ShapeMismatch { key, expected, got } =>
                write!(f, "shape mismatch for '{key}': expected {expected:?}, got {got:?}"),
        }
    }
}

/// Result alias for loading operations.
pub type LoadResult<T> = Result<T, LoadError>;

/// Expected top-level tensor key prefixes in a Mimi checkpoint.
pub const MIMI_KEY_PREFIXES: &[&str] = &[
    "encoder.model.",
    "decoder.model.",
    "encoder_transformer.layers.",
    "decoder_transformer.layers.",
    "downsample.",
    "upsample.",
    "quantizer.rvq_first.",
    "quantizer.rvq_rest.",
];

/// Materialise a weight-normed parameter: `W = weight_v * weight_g / ‖weight_v‖`.
///
/// At load time this collapses `weight_g + weight_v` into a single effective
/// weight, removing all runtime overhead (Kyutai's approach).
pub fn materialise_weight_norm(
    weight_v: &[f32],
    weight_g: &[f32],
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let stride = in_channels * kernel;
    let mut w = weight_v.to_vec();
    for o in 0..out_channels {
        let slice = &mut w[o * stride..(o + 1) * stride];
        let norm: f32 = slice.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = if norm > 1e-12 { weight_g[o] / norm } else { 0.0 };
        for v in slice.iter_mut() { *v *= scale; }
    }
    w
}

/// Load a Mimi codec from a safetensors checkpoint.
///
/// Returns a zero-weight model stub if the file exists (full parsing in Phase F).
pub fn load_mimi(path: &str, num_codebooks: usize) -> LoadResult<Mimi> {
    if !std::path::Path::new(path).exists() {
        return Err(LoadError::IoError(format!("file not found: {path}")));
    }
    let mut cfg = MimiConfig::default();
    cfg.set_num_codebooks(num_codebooks);
    Ok(Mimi::new(cfg))
}

/// Validate that a list of tensor names looks like a Mimi checkpoint.
pub fn validate_mimi_checkpoint(tensor_names: &[&str]) -> LoadResult<()> {
    for prefix in MIMI_KEY_PREFIXES {
        if !tensor_names.iter().any(|k| k.starts_with(prefix)) {
            return Err(LoadError::MissingTensor(format!("no tensor with prefix '{prefix}'")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_norm_identity() {
        let v = vec![1.0_f32, 0.0, 0.0, 1.0];
        let g = vec![1.0_f32, 1.0];
        let w = materialise_weight_norm(&v, &g, 2, 1, 2);
        assert!((w[0] - 1.0).abs() < 1e-5);
        assert!((w[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_weight_norm_scale() {
        let v = vec![3.0_f32, 4.0];
        let g = vec![2.0_f32];
        let w = materialise_weight_norm(&v, &g, 1, 1, 2);
        assert!((w[0] - 6.0 / 5.0).abs() < 1e-5);
        assert!((w[1] - 8.0 / 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_load_missing_file() {
        let r = load_mimi("/tmp/no_such_mimi_file.safetensors", 32);
        assert!(matches!(r, Err(LoadError::IoError(_))));
    }

    #[test]
    fn test_validate_ok() {
        let names: Vec<String> = MIMI_KEY_PREFIXES.iter().map(|p| format!("{p}x")).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        assert!(validate_mimi_checkpoint(&refs).is_ok());
    }

    #[test]
    fn test_validate_missing() {
        assert!(matches!(
            validate_mimi_checkpoint(&["encoder.model.0.weight"]),
            Err(LoadError::MissingTensor(_))
        ));
    }

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = LoadError::ShapeMismatch {
            key: "foo".to_string(), expected: vec![64, 1, 7], got: vec![64, 1, 3],
        };
        assert!(format!("{e}").contains("shape mismatch"));
    }
}
