//! Split-RVQ — Mimi's two-group audio tokenizer.
//!
//! Mimi uses a **split** architecture (arXiv:2410.00037, §3.2):
//!
//! * **CB0 (semantic)** — one [`VectorQuantizer`] trained with WavLM distillation.
//!   During acoustic fine-tuning it is frozen (no EMA updates).
//! * **CB1..CB7 (acoustic)** — a [`ResidualVQ`] of 7 layers trained with EMA.
//!
//! Crucially, both sub-quantizers receive the **original z** as input
//! (not z - semantic_output). The acoustic group learns complementary
//! residuals from the full z, not from the semantic residual.

use crate::config::RVQConfig;
use crate::residual_vq::ResidualVQ;
use crate::vq::VectorQuantizer;

// ─── Output ──────────────────────────────────────────────────────────────────

/// Output of a [`SplitRVQ`] forward pass.
#[derive(Debug, Clone)]
pub struct SplitRVQOutput {
    /// All code assignments: `codes[0]` is CB0 (semantic), `codes[1..8]` are
    /// CB1..CB7 (acoustic). Total length = `1 + num_acoustic_codebooks`.
    pub codes: Vec<Vec<u32>>,

    /// Full reconstruction: sum of semantic and acoustic quantized outputs,
    /// shape [N × D].
    pub quantized: Vec<f32>,

    /// Total commitment loss (semantic + acoustic).
    pub commit_loss: f32,

    /// Per-codebook perplexity values, length = `1 + num_acoustic_codebooks`.
    pub perplexities: Vec<f32>,
}

// ─── SplitRVQ ─────────────────────────────────────────────────────────────────

/// Split Residual VQ — Mimi's tokenizer (CB0 semantic + CB1..CB7 acoustic).
///
/// Both the semantic and acoustic sub-quantizers receive the **same** input z.
pub struct SplitRVQ {
    /// CB0: single semantic VQ.
    pub semantic_vq: VectorQuantizer,

    /// CB1..CB7: residual acoustic quantizers.
    pub acoustic_rvq: ResidualVQ,

    /// If true, CB0's EMA is not updated during training (used in stage 2).
    pub semantic_frozen: bool,
}

impl SplitRVQ {
    /// Construct a [`SplitRVQ`].
    ///
    /// # Arguments
    /// * `config`     — shared RVQ hyperparameters (K, D, EMA params, etc.)
    /// * `n_semantic` — number of semantic codebooks (1 for Mimi; CB0 only)
    ///
    /// The acoustic sub-quantizer uses `config.num_codebooks - n_semantic` levels.
    pub fn new(config: RVQConfig, n_semantic: usize) -> Self {
        assert!(
            config.num_codebooks > n_semantic,
            "num_codebooks {} must be > n_semantic {}",
            config.num_codebooks,
            n_semantic
        );
        let n_acoustic = config.num_codebooks - n_semantic;

        let semantic_vq = VectorQuantizer::new(config.clone());

        let mut acoustic_config = config.clone();
        acoustic_config.num_codebooks = n_acoustic;
        let acoustic_rvq = ResidualVQ::new(acoustic_config);

        Self {
            semantic_vq,
            acoustic_rvq,
            semantic_frozen: false,
        }
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    /// Run a SplitRVQ forward pass.
    ///
    /// Both semantic (CB0) and acoustic (CB1..7) sub-quantizers receive the
    /// full input `z`. Their outputs are summed for the final reconstruction.
    ///
    /// If `semantic_frozen` is true, CB0 runs in inference mode (no EMA).
    ///
    /// # Arguments
    /// * `z`        — input vectors [N × D] row-major
    /// * `n`        — number of input vectors N
    /// * `d`        — vector dimension D
    /// * `training` — enable EMA updates for non-frozen quantizers
    pub fn forward(&mut self, z: &[f32], n: usize, d: usize, training: bool) -> SplitRVQOutput {
        // Semantic VQ: frozen flag overrides training mode
        let sem_training = training && !self.semantic_frozen;
        let sem_out = self.semantic_vq.forward(z, n, d, sem_training);

        // Acoustic RVQ: always respects training flag
        let aco_out = self.acoustic_rvq.forward(z, n, d, training);

        // Combine: quantized = semantic + acoustic
        let mut quantized = vec![0.0_f32; n * d];
        for i in 0..n * d {
            quantized[i] = sem_out.quantized[i] + aco_out.quantized[i];
        }

        // Merge codes: CB0 first, then CB1..7
        let mut codes = Vec::with_capacity(1 + aco_out.codes.len());
        codes.push(sem_out.codes);
        codes.extend(aco_out.codes);

        // Merge perplexities
        let mut perplexities = Vec::with_capacity(1 + aco_out.perplexities.len());
        perplexities.push(sem_out.perplexity);
        perplexities.extend(aco_out.perplexities);

        let commit_loss = sem_out.commit_loss + aco_out.commit_loss;

        SplitRVQOutput { codes, quantized, commit_loss, perplexities }
    }

    // ── Encode ───────────────────────────────────────────────────────────────

    /// Encode input vectors to code indices for all codebook levels.
    ///
    /// Runs both semantic and acoustic sub-quantizers in inference mode.
    ///
    /// # Returns
    /// `Vec<Vec<u32>>` with length `1 + n_acoustic`; index 0 = CB0.
    pub fn encode(&self, z: &[f32], n: usize, d: usize) -> Vec<Vec<u32>> {
        let sem_codes = self.semantic_vq.encode(z, n, d);
        let aco_codes = self.acoustic_rvq.encode(z, n, d);

        let mut codes = Vec::with_capacity(1 + aco_codes.len());
        codes.push(sem_codes);
        codes.extend(aco_codes);
        codes
    }

    // ── Decode ───────────────────────────────────────────────────────────────

    /// Decode code indices to quantized embeddings.
    ///
    /// # Arguments
    /// * `codes` — slice with `1 + n_acoustic` code vectors (CB0 first)
    ///
    /// # Returns
    /// Sum of semantic + acoustic decoded embeddings, shape [N × D].
    pub fn decode(&self, codes: &[Vec<u32>]) -> Vec<f32> {
        assert!(
            !codes.is_empty(),
            "decode: expected at least 1 code vector"
        );
        let n = codes[0].len();
        let d = self.semantic_vq.dim;

        // CB0: semantic
        let sem_decoded = self.semantic_vq.decode(&codes[0]);

        // CB1..CB7: acoustic
        let aco_decoded = if codes.len() > 1 {
            self.acoustic_rvq.decode(&codes[1..].to_vec())
        } else {
            vec![0.0_f32; n * d]
        };

        // Sum
        let mut out = vec![0.0_f32; n * d];
        for i in 0..n * d {
            out[i] = sem_decoded[i] + aco_decoded[i];
        }
        out
    }

    // ── Freeze ────────────────────────────────────────────────────────────────

    /// Freeze the semantic quantizer (CB0).
    ///
    /// After calling this, `forward()` will not update CB0's EMA buffers even
    /// when `training = true`. This is used during Mimi's stage-2 acoustic
    /// fine-tuning.
    pub fn freeze_semantic(&mut self) {
        self.semantic_frozen = true;
    }

    /// Return the currently active number of codebooks (1 semantic + N acoustic).
    pub fn num_codebooks(&self) -> usize {
        1 + self.acoustic_rvq.active_quantizers
    }

    /// Return the total number of trained codebooks (active + inactive).
    pub fn max_codebooks(&self) -> usize {
        1 + self.acoustic_rvq.quantizers.len()
    }

    /// Set the number of active codebooks at runtime.
    ///
    /// Mirrors the Python `mimi.set_num_codebooks(n)` API.
    /// CB0 (semantic) is always active; `n - 1` acoustic codebooks are enabled.
    ///
    /// # Panics
    /// Panics if `n < 1` or if `n - 1` exceeds the number of trained acoustic
    /// quantizers.
    pub fn set_num_codebooks(&mut self, n: usize) {
        assert!(n >= 1, "set_num_codebooks: need at least 1 (semantic CB0)");
        let n_acoustic = n - 1;
        assert!(
            n_acoustic <= self.acoustic_rvq.quantizers.len(),
            "set_num_codebooks: requested {n} total but only {} trained",
            1 + self.acoustic_rvq.quantizers.len()
        );
        self.acoustic_rvq.active_quantizers = n_acoustic;
    }
}
