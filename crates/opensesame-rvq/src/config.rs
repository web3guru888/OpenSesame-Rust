//! Configuration for Residual Vector Quantization.
//!
//! [`RVQConfig`] holds all hyperparameters for the VQ / RVQ stack.
//! The `Default` implementation matches the Mimi codec's production settings.

/// Configuration for a Residual Vector Quantizer stack.
///
/// Default values match the Mimi codec used in Moshi (arXiv:2410.00037):
/// K = 2048, D = 256, N = 8, EMA decay = 0.99, β = 0.25.
#[derive(Debug, Clone)]
pub struct RVQConfig {
    /// Number of codebooks in the RVQ chain (default 8 for Mimi).
    pub num_codebooks: usize,

    /// Number of entries per codebook, i.e. vocabulary size K (default 2048).
    pub codebook_size: usize,

    /// Dimensionality of each codebook vector D (default 256 for Mimi).
    pub quant_dim: usize,

    /// Commitment loss weight β — penalises encoder drift from nearest centroid.
    /// β = 0.25 is the SoundStream default.
    pub commitment_cost: f32,

    /// EMA decay factor γ for codebook updates.  0.99 in Moshi.
    pub ema_decay: f32,

    /// Laplace smoothing ε added to the denominator to prevent division by zero
    /// when a centroid has never been assigned. 1e-5 in Moshi.
    pub ema_epsilon: f32,

    /// Centroids with `cluster_size < dead_code_threshold × (mean_usage)` are
    /// reinitialised from a random input vector (1.0 means 100 % of mean usage).
    pub dead_code_threshold: f32,

    /// If true, initialise each codebook with k-means on the first training batch
    /// instead of using Xavier-like random vectors.
    pub kmeans_init: bool,
}

impl Default for RVQConfig {
    /// Returns Mimi-compatible defaults: K = 2048, D = 256, N = 8, γ = 0.99, β = 0.25.
    fn default() -> Self {
        Self {
            num_codebooks:     8,
            codebook_size:     2048,
            quant_dim:         256,
            commitment_cost:   0.25,
            ema_decay:         0.99,
            ema_epsilon:       1e-5,
            dead_code_threshold: 1.0,
            kmeans_init:       true,
        }
    }
}
