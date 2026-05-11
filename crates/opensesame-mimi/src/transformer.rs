//! In-codec Mimi transformer (8 layers, 8 heads, d=512, LayerNorm, RoPE).
//!
//! For Phase E this is a **pass-through stub**: the transformer returns its
//! input unchanged. This lets the full pipeline compile and pass shape tests
//! while the detailed attention implementation is deferred to Phase F weight
//! loading verification.
//!
//! The [`LayerNorm`] helper IS fully implemented here because it is exercised
//! directly in the test suite.
//!
//! # References
//! * Kyutai moshi-core `transformer.rs` (d_model=512, n_heads=8, n_layers=8,
//!   causal=true, norm_first=true, layer_scale=0.01, context=250, RoPE).

use crate::config::MimiConfig;

// ─── LayerNorm ───────────────────────────────────────────────────────────────

/// Standard layer normalisation `(x − μ) / (σ + ε) · γ + β`.
///
/// Applies normalisation independently over the last axis of size `dim`.
/// The input tensor is interpreted as `[N, dim]` where N is any number of
/// leading dimensions flattened together.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter γ, shape `[dim]`. Initialised to 1.
    pub weight: Vec<f32>,
    /// Bias parameter β, shape `[dim]`. Initialised to 0.
    pub bias: Vec<f32>,
    /// Stabilisation epsilon added to the standard deviation (default 1e-5).
    pub eps: f32,
    /// Normalisation axis size.
    pub dim: usize,
}

impl LayerNorm {
    /// Construct a [`LayerNorm`] with identity initialisation (γ=1, β=0).
    pub fn new(dim: usize) -> Self {
        Self {
            weight: vec![1.0_f32; dim],
            bias: vec![0.0_f32; dim],
            eps: 1e-5,
            dim,
        }
    }

    /// Forward pass: normalise each `[dim]`-sized slice of `x`.
    ///
    /// `x` is interpreted as `[N, dim]`. Returns a vector of the same length.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(
            x.len() % self.dim,
            0,
            "LayerNorm: input length {} is not a multiple of dim {}",
            x.len(),
            self.dim
        );
        let n = x.len() / self.dim;
        let d = self.dim;
        let mut out = vec![0.0_f32; x.len()];

        for i in 0..n {
            let slice = &x[i * d..(i + 1) * d];

            // Compute mean.
            let mean = slice.iter().sum::<f32>() / d as f32;

            // Compute variance (population, not sample).
            let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;

            let inv_std = 1.0 / (var + self.eps).sqrt();

            for j in 0..d {
                out[i * d + j] = (slice[j] - mean) * inv_std * self.weight[j] + self.bias[j];
            }
        }

        out
    }
}

// ─── MimiTransformerLayer ────────────────────────────────────────────────────

/// A single transformer layer inside the Mimi codec.
///
/// In Phase E this is a **stub**: all projection matrices are zero-initialised
/// and the [`MimiTransformer::forward`] method bypasses this struct entirely,
/// returning the input unchanged. Weights are populated in Phase F from the
/// `.safetensors` file.
#[derive(Debug, Clone)]
pub struct MimiTransformerLayer {
    // Self-attention projections: [dim × dim].
    /// Query projection weight, shape `[dim, dim]`.
    pub q_proj: Vec<f32>,
    /// Key projection weight, shape `[dim, dim]`.
    pub k_proj: Vec<f32>,
    /// Value projection weight, shape `[dim, dim]`.
    pub v_proj: Vec<f32>,
    /// Output projection weight, shape `[dim, dim]`.
    pub o_proj: Vec<f32>,

    // Pre-attention layer norm.
    /// First layer normalisation (applied before attention).
    pub ln1: LayerNorm,
    // Pre-FFN layer norm.
    /// Second layer normalisation (applied before FFN).
    pub ln2: LayerNorm,

    // FFN: d_model → 4*d_model (GELU) → d_model.
    /// FFN first linear weight, shape `[dim, 4*dim]`.
    pub ffn1: Vec<f32>,
    /// FFN first linear bias, shape `[4*dim]`.
    pub ffn1_bias: Vec<f32>,
    /// FFN second linear weight, shape `[4*dim, dim]`.
    pub ffn2: Vec<f32>,
    /// FFN second linear bias, shape `[dim]`.
    pub ffn2_bias: Vec<f32>,
}

impl MimiTransformerLayer {
    /// Construct a stub layer for `dim`-dimensional model.
    pub fn new(dim: usize) -> Self {
        let ffn_dim = 4 * dim;
        Self {
            q_proj: vec![0.0_f32; dim * dim],
            k_proj: vec![0.0_f32; dim * dim],
            v_proj: vec![0.0_f32; dim * dim],
            o_proj: vec![0.0_f32; dim * dim],
            ln1: LayerNorm::new(dim),
            ln2: LayerNorm::new(dim),
            ffn1: vec![0.0_f32; dim * ffn_dim],
            ffn1_bias: vec![0.0_f32; ffn_dim],
            ffn2: vec![0.0_f32; ffn_dim * dim],
            ffn2_bias: vec![0.0_f32; dim],
        }
    }
}

// ─── MimiTransformer ─────────────────────────────────────────────────────────

/// 8-layer causal transformer embedded inside the Mimi codec.
///
/// Config (from Kyutai moshi-core):
/// - `d_model = 512`, `n_heads = 8`, `n_layers = 8`
/// - `causal = true`, `norm_first = true`, `layer_scale = 0.01`
/// - `positional_embedding = RoPE`, `norm = LayerNorm`
/// - `dim_feedforward = 2048` (4 × d_model), `context = 250`
///
/// **Phase E**: `forward()` is an identity pass-through. The full causal
/// self-attention + RoPE implementation is added when loading pretrained weights.
pub struct MimiTransformer {
    /// Transformer layers.
    pub layers: Vec<MimiTransformerLayer>,
    /// Final layer normalisation applied after all layers.
    pub norm: LayerNorm,
    /// Codec hyperparameters.
    pub config: MimiConfig,
}

impl MimiTransformer {
    /// Construct a [`MimiTransformer`] with stub (zero-weight) layers.
    pub fn new(config: &MimiConfig) -> Self {
        let dim = config.transformer_dim;
        let layers = (0..config.transformer_layers)
            .map(|_| MimiTransformerLayer::new(dim))
            .collect();
        Self {
            layers,
            norm: LayerNorm::new(dim),
            config: config.clone(),
        }
    }

    /// Forward pass.
    ///
    /// `x` is laid out as `[batch × seq_len × dim]` (i.e. `[B, T, D]` flattened).
    ///
    /// **Phase E**: returns a copy of the input unchanged (identity stub).
    /// The shape invariant `[B, T, D] → [B, T, D]` is upheld.
    pub fn forward(&self, x: &[f32], _batch: usize, _seq_len: usize) -> Vec<f32> {
        // Identity pass-through for Phase E.
        // Full causal self-attention + RoPE + layer-scale added in Phase F.
        x.to_vec()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MimiConfig;

    #[test]
    fn test_transformer_shape() {
        // [B=1, T=3, D=512] → [B=1, T=3, D=512] (identity stub keeps shape)
        let cfg = MimiConfig::v0_1();
        let t = MimiTransformer::new(&cfg);
        let x = vec![0.5_f32; 1 * 3 * 512];
        let y = t.forward(&x, 1, 3);
        assert_eq!(y.len(), x.len(), "transformer must preserve tensor length");
    }

    #[test]
    fn test_transformer_identity_stub() {
        // Phase E: output must equal input exactly.
        let cfg = MimiConfig::v0_1();
        let t = MimiTransformer::new(&cfg);
        let x: Vec<f32> = (0..512).map(|i| i as f32 * 0.001).collect();
        let y = t.forward(&x, 1, 1);
        for (a, b) in x.iter().zip(y.iter()) {
            assert_eq!(*a, *b, "transformer stub must be identity");
        }
    }

    #[test]
    fn test_layer_norm_values() {
        // LayerNorm([1.0, 2.0, 3.0]) ≈ [-1.2247, 0.0, 1.2247]
        let ln = LayerNorm::new(3);
        let x = [1.0_f32, 2.0, 3.0];
        let y = ln.forward(&x);
        assert_eq!(y.len(), 3);
        // mean=2, var=2/3, std=sqrt(2/3)≈0.8165 → normalized: -1.2247, 0, 1.2247
        assert!((y[0] - (-1.2247)).abs() < 1e-3, "y[0]={} ≠ -1.2247", y[0]);
        assert!(y[1].abs() < 1e-5, "y[1]={} ≠ 0.0", y[1]);
        assert!((y[2] - 1.2247).abs() < 1e-3, "y[2]={} ≠ 1.2247", y[2]);
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let ln = LayerNorm::new(16);
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.3 + 1.7).collect();
        let y = ln.forward(&x);
        let mean = y.iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-5, "LayerNorm output must have zero mean, got {}", mean);
    }

    #[test]
    fn test_layer_norm_unit_var() {
        let ln = LayerNorm::new(16);
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7 - 5.5)).collect();
        let y = ln.forward(&x);
        let mean = y.iter().sum::<f32>() / y.len() as f32;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / y.len() as f32;
        assert!((var - 1.0).abs() < 1e-4, "LayerNorm output must have unit variance, got {}", var);
    }
}
