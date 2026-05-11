//! Projection layer for the CSM model.
//!
//! A lightweight, no-bias linear layer `y = W x` used for:
//! - Backbone → Depformer projection: `Linear(backbone_dim → decoder_dim)`
//! - CB0 audio prediction head: `Linear(backbone_dim → audio_vocab_size)`
//! - Audio embedding projection: `Linear(backbone_dim → decoder_dim)` (inside embed_fn)
//!
//! Weight layout is row-major `[out_d × in_d]`.

/// Lightweight no-bias linear layer: `y = W x`.
///
/// # Shape
/// `weight`: `[out_d × in_d]` row-major — `weight[i * in_d + j]` is the weight
/// from input dimension `j` to output dimension `i`.
///
/// # Example
/// ```
/// # use opensesame_csm::Projection;
/// let proj = Projection::new_zeroed(4, 2);
/// let y = proj.forward(&[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(y.len(), 2);
/// ```
pub struct Projection {
    /// Row-major weight matrix `[out_d × in_d]`.
    pub weight: Vec<f32>,
    /// Input dimension.
    pub in_d: usize,
    /// Output dimension.
    pub out_d: usize,
}

impl Projection {
    /// Construct with all-zero weights.
    ///
    /// Use this before loading weights from a checkpoint.
    pub fn new_zeroed(in_d: usize, out_d: usize) -> Self {
        Self {
            weight: vec![0.0f32; out_d * in_d],
            in_d,
            out_d,
        }
    }

    /// Construct with deterministic pseudo-random weights.
    ///
    /// Uses an XORSHIFT-64 RNG seeded with `seed`, scaled by `1/√in_d`.
    /// Produces the same weights for the same `(in_d, out_d, seed)` triple,
    /// enabling reproducible unit tests.
    pub fn new_random(in_d: usize, out_d: usize, seed: u64) -> Self {
        let n = out_d * in_d;
        let scale = 1.0 / (in_d as f32).sqrt();
        let mut weight = Vec::with_capacity(n);
        let mut s = seed | 1; // ensure non-zero seed
        for _ in 0..n {
            // XORSHIFT-64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let f = (s >> 11) as f32 / (1u64 << 53) as f32; // ∈ [0, 1)
            weight.push((f * 2.0 - 1.0) * scale);             // ∈ [-scale, scale)
        }
        Self { weight, in_d, out_d }
    }

    /// Construct from an existing weight vector.
    ///
    /// Panics if `weight.len() != out_d * in_d`.
    pub fn from_data(weight: Vec<f32>, in_d: usize, out_d: usize) -> Self {
        assert_eq!(
            weight.len(),
            out_d * in_d,
            "Projection::from_data: weight len {} != out_d({}) * in_d({})",
            weight.len(), out_d, in_d
        );
        Self { weight, in_d, out_d }
    }

    /// Compute `y = W x`.
    ///
    /// `x`: input slice of length `in_d`.
    /// Returns output `Vec<f32>` of length `out_d`.
    ///
    /// Panics if `x.len() != in_d`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(), self.in_d,
            "Projection::forward: input len {} != in_d {}",
            x.len(), self.in_d
        );
        let mut out = vec![0.0f32; self.out_d];
        for i in 0..self.out_d {
            let row_start = i * self.in_d;
            let row = &self.weight[row_start..row_start + self.in_d];
            out[i] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_forward_shape() {
        let proj = Projection::new_random(8, 4, 42);
        let x = vec![1.0f32; 8];
        let y = proj.forward(&x);
        assert_eq!(y.len(), 4, "output shape = out_d");
    }

    #[test]
    fn test_projection_zero_input() {
        // Zero input → zero output (for any weight matrix)
        let proj = Projection::new_random(6, 3, 99);
        let x = vec![0.0f32; 6];
        let y = proj.forward(&x);
        for v in &y {
            assert!(*v == 0.0, "zero input → zero output");
        }
    }

    #[test]
    fn test_projection_zeroed() {
        // Zero weights + non-zero input → zero output
        let proj = Projection::new_zeroed(4, 2);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = proj.forward(&x);
        assert_eq!(y.len(), 2);
        for v in &y {
            assert!(*v == 0.0, "zero weights → zero output");
        }
    }

    #[test]
    fn test_projection_identity_approx() {
        // Build a 3×3 identity-like matrix and verify forward pass
        // Identity weight: W[i,i] = 1, W[i,j≠i] = 0
        let in_d = 3;
        let out_d = 3;
        let mut weight = vec![0.0f32; out_d * in_d];
        for i in 0..out_d {
            weight[i * in_d + i] = 1.0;
        }
        let proj = Projection::from_data(weight, in_d, out_d);
        let x = vec![1.0, 2.0, 3.0];
        let y = proj.forward(&x);
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 2.0).abs() < 1e-6);
        assert!((y[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_projection_from_pretrained() {
        // Load from a weight vector and verify the math
        // W = [[2, 0], [0, 3]]  →  y = [2x0, 3x1]
        let weight = vec![2.0f32, 0.0, 0.0, 3.0];
        let proj = Projection::from_data(weight, 2, 2);
        let x = vec![5.0f32, 7.0];
        let y = proj.forward(&x);
        assert!((y[0] - 10.0).abs() < 1e-6, "2*5 = 10");
        assert!((y[1] - 21.0).abs() < 1e-6, "3*7 = 21");
    }

    #[test]
    fn test_projection_non_square() {
        // 4-in, 2-out projection
        let proj = Projection::new_random(4, 2, 7777);
        let x = vec![0.5f32, 0.5, 0.5, 0.5];
        let y = proj.forward(&x);
        assert_eq!(y.len(), 2, "non-square output shape");
    }
}
