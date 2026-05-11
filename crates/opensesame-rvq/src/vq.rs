//! Single-level Vector Quantizer with EMA codebook updates.
//!
//! # Algorithm
//! 1. **Nearest-neighbour search** — for each input vector z_n, find the codebook
//!    entry e_k* that minimises L2 distance using the expansion
//!    `‖z - e‖² = ‖z‖² - 2·z·eᵀ + ‖e‖²`.
//! 2. **Commitment loss** — `β · MSE(z, sg[z_q])` back-propagated into the encoder.
//! 3. **EMA update** (training only) — updates cluster counts and embedding sums
//!    in-place; the actual codebook entries are derived by dividing sums by counts.
//! 4. **Dead-code reset** — reinitialise centroids that fall below a usage threshold.
//!
//! The **straight-through estimator** (STE) is expressed as
//! `z_q_ste = z + (z_q - z)` so gradients flow through to the encoder unchanged.
//! In this pure-CPU implementation no autograd graph is built; STE is applied by
//! the caller using `atlas-grad` when needed.

use crate::config::RVQConfig;

// ─── Output types ───────────────────────────────────────────────────────────

/// Output of a single VQ forward pass.
#[derive(Debug, Clone)]
pub struct VQOutput {
    /// Quantized embeddings — the looked-up codebook entries, shape [N × D].
    ///
    /// For gradient purposes the caller should apply the straight-through
    /// estimator: `z_q_ste = z + (z_q - z).detach()`.
    pub quantized: Vec<f32>,

    /// Code index for each input vector, shape [N].
    pub codes: Vec<u32>,

    /// Commitment loss `β · (1/N) · Σ_n ‖z_n - sg[z_q_n]‖²`.
    pub commit_loss: f32,

    /// Perplexity `exp(H)` where H is the entropy over code usage in this batch.
    /// Range: [1, K].  A value near K means uniform use of all codes; near 1
    /// indicates collapse to a single code.
    pub perplexity: f32,
}

// ─── VectorQuantizer ────────────────────────────────────────────────────────

/// Single-level vector quantizer with EMA codebook learning.
///
/// Storage layout: codebook is a flat `[K * D]` row-major array where
/// row k holds centroid e_k.
pub struct VectorQuantizer {
    /// Codebook entries, shape [K * D] row-major.  e_k = codebook[k*D .. k*D+D].
    pub codebook: Vec<f32>,

    /// EMA of per-centroid assignment counts N_k (SoundStream eq. 3).
    /// Accessible for checkpointing and testing; not a gradient parameter.
    pub cluster_size: Vec<f32>,

    /// EMA of per-centroid embedding sums m_k (SoundStream eq. 4).
    /// Accessible for checkpointing and testing; not a gradient parameter.
    pub embed_avg: Vec<f32>,

    /// Configuration used to construct this quantizer.
    pub config: RVQConfig,

    /// Dimension of each codebook vector D.
    pub dim: usize,

    /// Number of codebook entries K.
    pub num_codes: usize,

    /// Whether k-means initialisation has been applied on the first batch.
    initialized: bool,
}

impl VectorQuantizer {
    /// Construct a new [`VectorQuantizer`].
    ///
    /// Codebook entries are initialised with small uniform random values in
    /// `[-0.02, 0.02]` unless `config.kmeans_init` is true, in which case
    /// they will be replaced on the first training forward pass.
    pub fn new(config: RVQConfig) -> Self {
        let k = config.codebook_size;
        let d = config.quant_dim;

        // Deterministic small-random init (no external RNG crate):
        // Use a simple LCG so tests are reproducible.
        let codebook = small_random_init(k, d, 0x1234_5678_u64);
        let cluster_size = vec![1.0_f32; k]; // start at 1 to avoid dead-code triggers
        let embed_avg = codebook.clone();     // embed_avg = codebook × 1.0

        Self {
            codebook,
            cluster_size,
            embed_avg,
            dim: d,
            num_codes: k,
            initialized: !config.kmeans_init, // false if kmeans wanted
            config,
        }
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    /// Run a VQ forward pass.
    ///
    /// # Arguments
    /// * `z`        — input vectors, shape [N × D] row-major
    /// * `n`        — number of input vectors N
    /// * `d`        — vector dimension D (must equal `self.dim`)
    /// * `training` — if true, run EMA updates on codebook in-place
    ///
    /// # Returns
    /// [`VQOutput`] with quantized embeddings, codes, commitment loss, perplexity.
    pub fn forward(&mut self, z: &[f32], n: usize, d: usize, training: bool) -> VQOutput {
        assert_eq!(d, self.dim, "VQ forward: input dim {} ≠ codebook dim {}", d, self.dim);
        assert_eq!(z.len(), n * d, "VQ forward: z.len() {} ≠ n*d {}", z.len(), n * d);

        // K-means init on first training batch
        if training && !self.initialized {
            self.kmeans_init_codebook(z, n);
            self.initialized = true;
        }

        // Nearest-neighbour search
        let codes = self.encode(z, n, d);

        // Build quantized output
        let quantized = self.decode(&codes);

        // Commitment loss: β * (1/N) * Σ ||z_n - z_q_n||²
        let commit_loss = compute_commitment_loss(z, &quantized, n, d, self.config.commitment_cost);

        // Perplexity
        let perplexity = compute_perplexity(&codes, self.num_codes);

        // EMA update (training only, skip when semantic quantizer is frozen)
        if training {
            ema_update(
                &mut self.cluster_size,
                &mut self.embed_avg,
                &mut self.codebook,
                &codes,
                z,
                n,
                d,
                self.config.ema_decay,
                self.config.ema_epsilon,
            );
        }

        VQOutput { quantized, codes, commit_loss, perplexity }
    }

    // ── Encode ───────────────────────────────────────────────────────────────

    /// Encode input vectors to code indices (nearest-neighbour search only).
    ///
    /// # Arguments
    /// * `z` — input vectors [N × D] row-major
    /// * `n` — number of vectors
    /// * `d` — vector dimension (must equal `self.dim`)
    ///
    /// # Returns
    /// Code indices `[N]` (each in `0 .. K`).
    pub fn encode(&self, z: &[f32], n: usize, d: usize) -> Vec<u32> {
        assert_eq!(d, self.dim, "encode: dim mismatch");
        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let zn = &z[i * d..(i + 1) * d];
            let (code, _dist) = l2_nearest(zn, &self.codebook, self.num_codes, d);
            codes.push(code);
        }
        codes
    }

    // ── Decode ───────────────────────────────────────────────────────────────

    /// Decode code indices to quantized embeddings.
    ///
    /// # Arguments
    /// * `codes` — code indices `[N]`
    ///
    /// # Returns
    /// Quantized embeddings `[N × D]` row-major.
    pub fn decode(&self, codes: &[u32]) -> Vec<f32> {
        let d = self.dim;
        let mut out = Vec::with_capacity(codes.len() * d);
        for &c in codes {
            let c = c as usize;
            debug_assert!(c < self.num_codes, "code {} out of range [0, {})", c, self.num_codes);
            out.extend_from_slice(&self.codebook[c * d..(c + 1) * d]);
        }
        out
    }

    // ── Dead-code reset ───────────────────────────────────────────────────────

    /// Reinitialise centroids whose cluster size falls below the dead-code threshold.
    ///
    /// Threshold per centroid: `threshold = dead_code_threshold × mean_usage`.
    /// Dead centroids are replaced by a random input vector from `z`.
    ///
    /// # Arguments
    /// * `z` — current batch inputs [N × D] (source of replacement vectors)
    /// * `n` — number of vectors in `z`
    pub fn reset_dead_codes(&mut self, z: &[f32], n: usize) {
        let d = self.dim;
        let total: f32 = self.cluster_size.iter().sum();
        let threshold = self.config.dead_code_threshold * total / (self.num_codes as f32);

        // Simple pseudo-random selection (no external crate)
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_u64;

        for k in 0..self.num_codes {
            if self.cluster_size[k] < threshold {
                // Pick a random input vector
                rng_state = lcg_next(rng_state);
                let idx = (rng_state >> 33) as usize % n;
                let replacement = &z[idx * d..(idx + 1) * d];

                // Reset cluster stats to average usage so the centroid isn't
                // immediately marked dead again.
                let avg_usage = total / self.num_codes as f32;
                self.cluster_size[k] = avg_usage;
                for dim in 0..d {
                    self.embed_avg[k * d + dim] = replacement[dim] * avg_usage;
                    self.codebook[k * d + dim] = replacement[dim];
                }
            }
        }
    }

    // ── K-means initialisation ────────────────────────────────────────────────

    /// Initialise the codebook with k-means++ on the first training batch.
    ///
    /// Runs `num_iters = 10` Lloyd iterations. This replaces the small random
    /// init and substantially reduces the time to meaningful codebook utilisation.
    fn kmeans_init_codebook(&mut self, z: &[f32], n: usize) {
        let d = self.dim;
        let k = self.num_codes;
        // Cap iterations / sample size for speed on CPU
        let num_iters = 10usize;
        // Use at most 4096 samples to keep init fast on large batches
        let sample_n = n.min(4096);

        // --- Step 1: initialise centroids from random distinct input vectors ---
        let mut rng_state: u64 = 0xCAFE_BABE_1234_u64;
        let mut centroids: Vec<f32> = Vec::with_capacity(k * d);
        for idx in 0..k {
            rng_state = lcg_next(rng_state);
            let sample_idx = (rng_state >> 33) as usize % sample_n;
            let base = sample_idx * d;
            centroids.extend_from_slice(&z[base..base + d]);
            let _ = idx; // suppress unused warning
        }

        // --- Step 2: Lloyd iterations ---
        let mut assignments = vec![0u32; sample_n];
        for _iter in 0..num_iters {
            // Assignment step
            for i in 0..sample_n {
                let zn = &z[i * d..(i + 1) * d];
                let (code, _) = l2_nearest(zn, &centroids, k, d);
                assignments[i] = code;
            }
            // Update step: recompute centroids as mean of assigned vectors
            let mut sums = vec![0.0_f32; k * d];
            let mut counts = vec![0u32; k];
            for i in 0..sample_n {
                let c = assignments[i] as usize;
                counts[c] += 1;
                for dim in 0..d {
                    sums[c * d + dim] += z[i * d + dim];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for dim in 0..d {
                        centroids[c * d + dim] = sums[c * d + dim] * inv;
                    }
                }
                // else: leave centroid unchanged (will be reset by dead-code logic)
            }
        }

        // Write back to codebook and EMA state
        self.codebook.copy_from_slice(&centroids);
        // Initialise embed_avg = centroid and cluster_size = 1
        self.embed_avg.copy_from_slice(&centroids);
        for k_idx in 0..k {
            self.cluster_size[k_idx] = 1.0;
        }
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Find the nearest codebook entry to `query` using L2 distance.
///
/// Uses the expansion `‖z - e‖² = ‖z‖² - 2·z·eᵀ + ‖e‖²` to avoid an explicit
/// subtraction per dimension per centroid in the hot path.
///
/// Returns `(code_index, squared_distance)`.
fn l2_nearest(query: &[f32], codebook: &[f32], k: usize, d: usize) -> (u32, f32) {
    // Precompute ‖z‖²
    let z_norm_sq: f32 = query.iter().map(|&x| x * x).sum();

    let mut best_code = 0u32;
    let mut best_dist = f32::MAX;

    for c in 0..k {
        let e = &codebook[c * d..(c + 1) * d];
        // ‖e‖²
        let e_norm_sq: f32 = e.iter().map(|&x| x * x).sum();
        // -2 · z · eᵀ
        let dot: f32 = query.iter().zip(e.iter()).map(|(&a, &b)| a * b).sum();
        let dist = z_norm_sq - 2.0 * dot + e_norm_sq;
        if dist < best_dist {
            best_dist = dist;
            best_code = c as u32;
        }
    }

    (best_code, best_dist)
}

/// Apply EMA updates to `cluster_size`, `embed_avg`, and `codebook` in-place.
///
/// Implements SoundStream equations 3–5:
/// ```text
/// N_k^(t) = γ · N_k^(t-1) + (1-γ) · n_k^(t)
/// m_k^(t) = γ · m_k^(t-1) + (1-γ) · Σ_{z: code(z)=k} z
/// e_k^(t) = m_k^(t) / max(N_k^(t), ε)
/// ```
fn ema_update(
    cluster_size: &mut [f32],
    embed_avg: &mut [f32],
    codebook: &mut [f32],
    assignments: &[u32],
    z: &[f32],
    n: usize,
    d: usize,
    gamma: f32,
    eps: f32,
) {
    let k = cluster_size.len();

    // Accumulate per-centroid counts and embedding sums from the current batch
    let mut batch_counts = vec![0.0_f32; k];
    let mut batch_sums   = vec![0.0_f32; k * d];

    for i in 0..n {
        let c = assignments[i] as usize;
        batch_counts[c] += 1.0;
        for dim in 0..d {
            batch_sums[c * d + dim] += z[i * d + dim];
        }
    }

    // EMA update for cluster sizes
    for c in 0..k {
        cluster_size[c] = gamma * cluster_size[c] + (1.0 - gamma) * batch_counts[c];
    }

    // EMA update for embedding sums
    for c in 0..k {
        for dim in 0..d {
            embed_avg[c * d + dim] =
                gamma * embed_avg[c * d + dim] + (1.0 - gamma) * batch_sums[c * d + dim];
        }
    }

    // Recompute codebook entries: e_k = m_k / max(N_k, ε)
    for c in 0..k {
        let denom = cluster_size[c].max(eps);
        for dim in 0..d {
            codebook[c * d + dim] = embed_avg[c * d + dim] / denom;
        }
    }
}

/// Compute commitment loss: `β * (1/N) * Σ_n ||z_n - z_q_n||²`.
fn compute_commitment_loss(z: &[f32], z_q: &[f32], n: usize, d: usize, beta: f32) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..n * d {
        let diff = z[i] - z_q[i];
        sum += diff * diff;
    }
    beta * sum / (n as f32)
}

/// Compute perplexity `exp(H)` from code assignments.
///
/// `H = -Σ_k p_k · log(p_k)` where `p_k` is the fraction of vectors assigned to
/// centroid k.  Returns a value in `[1, K]`.
fn compute_perplexity(codes: &[u32], num_codes: usize) -> f32 {
    let n = codes.len();
    if n == 0 {
        return 1.0;
    }
    let mut counts = vec![0u32; num_codes];
    for &c in codes {
        counts[c as usize] += 1;
    }
    let inv_n = 1.0 / n as f32;
    let mut entropy = 0.0_f32;
    for &cnt in &counts {
        if cnt > 0 {
            let p = cnt as f32 * inv_n;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

// ─── Utilities ───────────────────────────────────────────────────────────────

/// Small uniform-random initialisation in [-0.02, 0.02] using a plain LCG.
/// Produces reproducible results without any external PRNG crate.
fn small_random_init(k: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(k * d);
    for _ in 0..k * d {
        state = lcg_next(state);
        // Map [0, 2^32) to [-0.02, 0.02]
        let u = (state >> 32) as u32 as f64 / u32::MAX as f64; // [0, 1]
        out.push((u as f32 * 0.04) - 0.02);
    }
    out
}

/// One step of a 64-bit LCG (Knuth's multiplicative congruential).
#[inline(always)]
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vq(k: usize, d: usize) -> VectorQuantizer {
        let mut cfg = RVQConfig::default();
        cfg.codebook_size = k;
        cfg.quant_dim = d;
        cfg.kmeans_init = false; // disable for unit tests
        VectorQuantizer::new(cfg)
    }

    #[test]
    fn test_lcg_not_zero() {
        let v = lcg_next(1);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_l2_nearest_exact() {
        // codebook has 4 entries of dim 2
        let cb: Vec<f32> = vec![
            1.0, 0.0,
            0.0, 1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ];
        let query = [0.9_f32, 0.1_f32];
        let (code, _dist) = l2_nearest(&query, &cb, 4, 2);
        assert_eq!(code, 0, "nearest to (0.9,0.1) should be entry 0 = (1,0)");
    }

    #[test]
    fn test_compute_perplexity_uniform() {
        // All 4 codes used equally → perplexity = 4
        let codes: Vec<u32> = (0..40).map(|i| (i % 4) as u32).collect();
        let p = compute_perplexity(&codes, 4);
        assert!((p - 4.0).abs() < 1e-4, "uniform perplexity should be 4, got {}", p);
    }

    #[test]
    fn test_compute_perplexity_collapsed() {
        // All codes are 0 → perplexity = 1
        let codes: Vec<u32> = vec![0u32; 20];
        let p = compute_perplexity(&codes, 4);
        assert!((p - 1.0).abs() < 1e-4, "collapsed perplexity should be 1, got {}", p);
    }
}
