//! Per-codebook audio output heads for the Depformer (CB1..CB7).
//!
//! Each depth step in the Depformer produces a hidden state vector that is
//! projected to logits over the 2048-token audio vocabulary via a dedicated
//! linear head.  The heads are stored as a flat weight tensor and indexed by
//! depth (1-indexed: depth 1 → CB1 head, …, depth 7 → CB7 head).

/// Per-depth output projection: one `Linear(d_model → vocab_size)` per
/// generated codebook (CB1..CB7).
///
/// Weight layout: `weights[i]` is `[vocab_size × d_model]` row-major for
/// codebook `i+1` (0-indexed storage, 1-indexed access via `forward`).
///
/// Corresponds to the `audio_head` parameter tensor in the Sesame CSM-1B
/// safetensors checkpoint (shape `[31, 1024, 2048]` for the 32-CB variant).
pub struct CsmAudioHead {
    /// Weight matrices, one per generated codebook (0-indexed, CB1 = index 0).
    /// `weights[i]` layout: `[vocab_size × d_model]` row-major.
    pub weights: Vec<Vec<f32>>,
    /// Number of generated codebooks (= `n_dep_codebooks` = 7 for CSM-1B).
    pub n_codebooks: usize,
    /// Depformer hidden dimension (1024 for CSM-1B).
    pub d_model: usize,
    /// Audio token vocabulary size (2048 for CSM-1B).
    pub vocab_size: usize,
}

impl CsmAudioHead {
    /// Construct from a flat `[n_dep_codebooks × vocab_size × d_model]` tensor.
    ///
    /// Matches the safetensors layout of the CSM-1B `audio_head` parameter.
    /// Panics if `flat.len() != n_dep_codebooks × vocab_size × d_model`.
    pub fn from_flat(flat: Vec<f32>, n_dep_codebooks: usize, d_model: usize, vocab_size: usize) -> Self {
        let stride = vocab_size * d_model;
        assert_eq!(
            flat.len(),
            n_dep_codebooks * stride,
            "from_flat: expected {} floats ({}×{}×{}), got {}",
            n_dep_codebooks * stride,
            n_dep_codebooks, vocab_size, d_model,
            flat.len()
        );
        let weights = (0..n_dep_codebooks)
            .map(|i| flat[i * stride..(i + 1) * stride].to_vec())
            .collect();
        Self { weights, n_codebooks: n_dep_codebooks, d_model, vocab_size }
    }

    /// Construct with zero-initialized weights.
    ///
    /// Use before loading weights from a checkpoint.
    pub fn zeros(n_dep_codebooks: usize, d_model: usize, vocab_size: usize) -> Self {
        let weights = (0..n_dep_codebooks)
            .map(|_| vec![0.0f32; vocab_size * d_model])
            .collect();
        Self { weights, n_codebooks: n_dep_codebooks, d_model, vocab_size }
    }

    /// Construct with deterministic random weights (for testing).
    ///
    /// Uses seeded XORSHIFT-64 RNG scaled by `1/sqrt(d_model)` for each
    /// codebook head independently.  Two calls with the same arguments produce
    /// identical weights (no entropy), enabling reproducible test comparisons.
    pub fn random(n_dep_codebooks: usize, d_model: usize, vocab_size: usize, seed: u64) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        let weights = (0..n_dep_codebooks)
            .map(|i| {
                let n = vocab_size * d_model;
                let mut w = Vec::with_capacity(n);
                // Per-codebook seed to ensure independent distributions
                let mut s = seed
                    .wrapping_add(i as u64 * 6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    | 1;
                for _ in 0..n {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let f = (s >> 11) as f32 / ((1u64 << 53) as f32);
                    w.push((f * 2.0 - 1.0) * scale);
                }
                w
            })
            .collect();
        Self { weights, n_codebooks: n_dep_codebooks, d_model, vocab_size }
    }

    /// Compute logits for codebook `depth` (1-indexed).
    ///
    /// `depth`: 1 = CB1, …, n_codebooks = CB7.
    /// `hidden`: `[d_model]` depformer hidden state at this depth step.
    /// `logits`: `[vocab_size]` output buffer (overwritten).
    ///
    /// # Panics
    /// Panics if `depth` is not in `1..=n_codebooks`.
    pub fn forward(&self, depth: usize, hidden: &[f32], logits: &mut [f32]) {
        assert!(
            depth >= 1 && depth <= self.n_codebooks,
            "depth {depth} out of [1, {}]",
            self.n_codebooks
        );
        assert_eq!(hidden.len(), self.d_model, "CsmAudioHead hidden len mismatch");
        assert_eq!(logits.len(), self.vocab_size, "CsmAudioHead logits len mismatch");
        let w = &self.weights[depth - 1];
        for j in 0..self.vocab_size {
            let row = &w[j * self.d_model..(j + 1) * self.d_model];
            logits[j] = row.iter().zip(hidden.iter()).map(|(&wi, &hi)| wi * hi).sum();
        }
    }

    /// Compute logits for codebook `depth` and return as `Vec<f32>`.
    pub fn forward_vec(&self, depth: usize, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.vocab_size];
        self.forward(depth, hidden, &mut logits);
        logits
    }
}
