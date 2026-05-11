//! Residual Vector Quantization (RVQ).
//!
//! Chains N [`VectorQuantizer`] layers where each layer quantizes the
//! **residual** left by the previous layers (SoundStream Algorithm 1).
//!
//! # Straight-through estimator (STE) note
//!
//! In a full autograd setting the STE is applied twice:
//! 1. Per quantizer: `z_q_k = residual + (codebook[code] - residual).detach()`
//! 2. At the aggregate level (Encodec bug fix):
//!    `total_quantized = z + (total_quantized - z).detach()`
//!
//! In this pure-CPU `Vec<f32>` implementation no gradient graph is built.
//! The commitment loss returned is the quantity the caller should back-propagate
//! into the encoder via the gradient framework.

use crate::config::RVQConfig;
use crate::vq::{VectorQuantizer, VQOutput};

// ─── Output ──────────────────────────────────────────────────────────────────

/// Output of an RVQ forward pass.
#[derive(Debug, Clone)]
pub struct RVQOutput {
    /// Sum of all quantized residuals across all codebook levels, shape [N × D].
    pub quantized: Vec<f32>,

    /// Per-codebook code assignments: `codes[k]` has shape [N].
    pub codes: Vec<Vec<u32>>,

    /// Sum of per-level commitment losses (each weighted by `β`).
    pub commit_loss: f32,

    /// Per-level perplexity values, length `num_codebooks`.
    pub perplexities: Vec<f32>,
}

// ─── ResidualVQ ───────────────────────────────────────────────────────────────

/// Residual Vector Quantizer — chains `num_codebooks` VQ layers.
///
/// Each layer receives the residual signal left by all previous layers.
/// Stacking layers progressively refines the reconstruction.
pub struct ResidualVQ {
    /// Constituent VQ layers, one per codebook level.
    quantizers: Vec<VectorQuantizer>,

    /// Configuration shared by all quantizers.
    pub config: RVQConfig,
}

impl ResidualVQ {
    /// Create a new [`ResidualVQ`] with `config.num_codebooks` VQ layers.
    ///
    /// All quantizers share the same `codebook_size`, `quant_dim`, and training
    /// hyper-parameters from `config`.
    pub fn new(config: RVQConfig) -> Self {
        let n = config.num_codebooks;
        let quantizers = (0..n).map(|_| VectorQuantizer::new(config.clone())).collect();
        Self { quantizers, config }
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    /// Run a residual VQ forward pass.
    ///
    /// # Algorithm (SoundStream Algorithm 1)
    /// ```text
    /// residual = z
    /// total_quantized = 0
    /// for k in 0..num_codebooks:
    ///     out_k = VQ_k.forward(residual)
    ///     residual -= out_k.quantized
    ///     total_quantized += out_k.quantized
    /// ```
    ///
    /// # Arguments
    /// * `z`        — input vectors [N × D] row-major
    /// * `n`        — number of input vectors N
    /// * `d`        — vector dimension D
    /// * `training` — passed through to each VQ layer for EMA updates
    pub fn forward(&mut self, z: &[f32], n: usize, d: usize, training: bool) -> RVQOutput {
        let mut residual = z.to_vec();
        let mut total_quantized = vec![0.0_f32; n * d];
        let mut all_codes: Vec<Vec<u32>> = Vec::with_capacity(self.quantizers.len());
        let mut perplexities = Vec::with_capacity(self.quantizers.len());
        let mut total_commit_loss = 0.0_f32;

        for vq in &mut self.quantizers {
            let out: VQOutput = vq.forward(&residual, n, d, training);

            // Subtract quantized from residual (in-place)
            for i in 0..n * d {
                residual[i] -= out.quantized[i];
            }

            // Accumulate quantized output
            for i in 0..n * d {
                total_quantized[i] += out.quantized[i];
            }

            total_commit_loss += out.commit_loss;
            all_codes.push(out.codes);
            perplexities.push(out.perplexity);
        }

        // Mean commit loss across all codebook levels
        let mean_commit_loss = total_commit_loss / self.quantizers.len() as f32;

        RVQOutput {
            quantized: total_quantized,
            codes: all_codes,
            commit_loss: mean_commit_loss,
            perplexities,
        }
    }

    // ── Encode ───────────────────────────────────────────────────────────────

    /// Encode input vectors to code indices for each codebook level.
    ///
    /// Runs the residual algorithm in inference mode (no EMA updates).
    ///
    /// # Returns
    /// `Vec<Vec<u32>>` with length `num_codebooks`; each inner vec has length N.
    pub fn encode(&self, z: &[f32], n: usize, d: usize) -> Vec<Vec<u32>> {
        let mut residual = z.to_vec();
        let mut all_codes = Vec::with_capacity(self.quantizers.len());

        for vq in &self.quantizers {
            let codes = vq.encode(&residual, n, d);
            // Subtract quantized residual
            let quantized = vq.decode(&codes);
            for i in 0..n * d {
                residual[i] -= quantized[i];
            }
            all_codes.push(codes);
        }

        all_codes
    }

    // ── Decode ───────────────────────────────────────────────────────────────

    /// Decode a set of code indices (one per codebook level) to vectors.
    ///
    /// # Arguments
    /// * `codes` — slice of `num_codebooks` code arrays, each of length N
    ///
    /// # Returns
    /// Sum of decoded embeddings across all levels, shape [N × D].
    pub fn decode(&self, codes: &[Vec<u32>]) -> Vec<f32> {
        assert_eq!(
            codes.len(),
            self.quantizers.len(),
            "decode: expected {} code vectors, got {}",
            self.quantizers.len(),
            codes.len()
        );
        let n = codes[0].len();
        let d = self.config.quant_dim;
        let mut out = vec![0.0_f32; n * d];

        for (vq, level_codes) in self.quantizers.iter().zip(codes.iter()) {
            let decoded = vq.decode(level_codes);
            for i in 0..n * d {
                out[i] += decoded[i];
            }
        }
        out
    }

    /// Access individual quantizer layers (read-only).
    pub fn quantizers(&self) -> &[VectorQuantizer] {
        &self.quantizers
    }
}
