//! Lightweight no-bias linear layer for the Depformer.
//!
//! Used for the backbone→depformer projection (`Linear(2048 → 1024)`) in the
//! CSM generation loop, and for output heads where a full GPU dispatch wrapper
//! is unnecessary.

/// Lightweight no-bias linear layer.
///
/// Weight layout: row-major `[out_dim × in_dim]`.
/// `weight[i * in_dim + j]` is the weight from input dimension `j`
/// to output dimension `i`.
pub struct CsmLinear {
    /// Weight matrix: `[out_dim × in_dim]` row-major.
    pub weight: Vec<f32>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl CsmLinear {
    /// Construct from a weight vector.
    ///
    /// Panics if `weight.len() != out_dim * in_dim`.
    pub fn from_data(weight: Vec<f32>, in_dim: usize, out_dim: usize) -> Self {
        assert_eq!(
            weight.len(),
            out_dim * in_dim,
            "CsmLinear: weight len {} != out_dim({}) * in_dim({})",
            weight.len(), out_dim, in_dim
        );
        Self { weight, in_dim, out_dim }
    }

    /// Construct with zero-initialized weights.
    pub fn zeros(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight: vec![0.0f32; out_dim * in_dim],
            in_dim,
            out_dim,
        }
    }

    /// Compute `y = W · x` in place.
    ///
    /// `x`: `[in_dim]`, `out`: `[out_dim]`.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim, "CsmLinear::forward x len mismatch");
        assert_eq!(out.len(), self.out_dim, "CsmLinear::forward out len mismatch");
        for i in 0..self.out_dim {
            let row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
            out[i] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
    }

    /// Compute `y = W · x` and return as a `Vec<f32>`.
    pub fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim];
        self.forward(x, &mut out);
        out
    }
}
