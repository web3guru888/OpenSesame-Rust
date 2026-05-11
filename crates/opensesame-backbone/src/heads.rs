//! Output heads for the CSM backbone.
//!
//! Three head types:
//! - `CsmLinear`   — lightweight no-bias linear layer (for projection + heads)
//! - `CsmCB0Head`  — CB0 audio prediction: `Linear(d_backbone → audio_vocab_size)`
//! - `CsmAudioHead`— per-codebook CB1..CB7 heads: `7 × Linear(dep_dim → audio_vocab_size)`

use crate::config::AUDIO_VOCAB_SIZE;

// ── CsmLinear ─────────────────────────────────────────────────────────────────

/// Lightweight no-bias linear layer.
///
/// Used for the backbone→depformer projection (`Linear(2048→1024)`) and for
/// output heads where a GPU dispatch wrapper is unnecessary.
///
/// Weight layout: row-major `[out_dim × in_dim]` — `weight[i * in_dim + j]` is
/// the weight connecting input `j` to output `i`.
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
            "CsmLinear weight len {} != out_dim*in_dim {}",
            weight.len(),
            out_dim * in_dim
        );
        Self { weight, in_dim, out_dim }
    }

    /// Construct with zeros (useful for bias-less heads that are later loaded).
    pub fn zeros(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight: vec![0.0f32; out_dim * in_dim],
            in_dim,
            out_dim,
        }
    }

    /// Construct with small random weights (deterministic, no external crate).
    pub fn random(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let n = out_dim * in_dim;
        let scale = 1.0 / (in_dim as f32).sqrt();
        let mut weight = Vec::with_capacity(n);
        let mut s = seed | 1;
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let f = (s >> 11) as f32 / (1u64 << 53) as f32;
            weight.push((f * 2.0 - 1.0) * scale);
        }
        Self { weight, in_dim, out_dim }
    }

    /// Compute `y = W x` in place.
    ///
    /// `x`: `[in_dim]`, `out`: `[out_dim]`.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim, "CsmLinear::forward input len mismatch");
        assert_eq!(out.len(), self.out_dim, "CsmLinear::forward output len mismatch");
        for i in 0..self.out_dim {
            let row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
            out[i] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
    }

    /// Compute `y = W x` and return as a new `Vec<f32>`.
    pub fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim];
        self.forward(x, &mut out);
        out
    }
}

// ── CsmCB0Head ───────────────────────────────────────────────────────────────

/// CB0 audio prediction head: `Linear(d_backbone → AUDIO_VOCAB_SIZE)`, no bias.
///
/// Receives the backbone final hidden state `[d_backbone]` at each position
/// and predicts logits over the `AUDIO_VOCAB_SIZE = 2048` audio tokens for
/// codebook 0.
pub struct CsmCB0Head {
    /// Weight matrix: `[AUDIO_VOCAB_SIZE × d_backbone]` row-major.
    pub weight: Vec<f32>,
    /// Backbone hidden dimension.
    pub d_backbone: usize,
}

impl CsmCB0Head {
    /// Construct with zeros.
    pub fn zeros(d_backbone: usize) -> Self {
        Self {
            weight: vec![0.0f32; AUDIO_VOCAB_SIZE * d_backbone],
            d_backbone,
        }
    }

    /// Construct from weight data.
    pub fn from_data(weight: Vec<f32>, d_backbone: usize) -> Self {
        assert_eq!(weight.len(), AUDIO_VOCAB_SIZE * d_backbone);
        Self { weight, d_backbone }
    }

    /// Construct with random weights.
    pub fn random(d_backbone: usize, seed: u64) -> Self {
        let n = AUDIO_VOCAB_SIZE * d_backbone;
        let scale = 1.0 / (d_backbone as f32).sqrt();
        let mut weight = Vec::with_capacity(n);
        let mut s = seed | 1;
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let f = (s >> 11) as f32 / (1u64 << 53) as f32;
            weight.push((f * 2.0 - 1.0) * scale);
        }
        Self { weight, d_backbone }
    }

    /// Compute CB0 logits.
    ///
    /// `hidden`: `[d_backbone]` — backbone final hidden state.
    /// `logits`: `[AUDIO_VOCAB_SIZE]` — output buffer.
    pub fn forward(&self, hidden: &[f32], logits: &mut [f32]) {
        assert_eq!(hidden.len(), self.d_backbone);
        assert_eq!(logits.len(), AUDIO_VOCAB_SIZE);
        for out_i in 0..AUDIO_VOCAB_SIZE {
            let row = &self.weight[out_i * self.d_backbone..(out_i + 1) * self.d_backbone];
            logits[out_i] = row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum();
        }
    }

    /// Compute CB0 logits and return as a Vec.
    pub fn forward_vec(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; AUDIO_VOCAB_SIZE];
        self.forward(hidden, &mut logits);
        logits
    }
}

// ── CsmAudioHead ─────────────────────────────────────────────────────────────

/// Per-codebook audio output heads for CB1..CB7.
///
/// Contains `n_codebooks - 1 = 7` separate `Linear(dep_dim → AUDIO_VOCAB_SIZE)`
/// heads (one per non-zero codebook).  Each head receives the depformer hidden
/// state and predicts logits over the 2048 audio tokens for its codebook.
pub struct CsmAudioHead {
    /// Weights per codebook: `weights[i]` is the head for codebook `i+1`.
    /// Each entry has shape `[AUDIO_VOCAB_SIZE × dep_dim]` row-major.
    pub weights: Vec<Vec<f32>>,
    /// Number of non-zero codebooks (= N_CODEBOOKS - 1 = 7).
    pub n_codebooks: usize,
    /// Depformer hidden dimension.
    pub dep_dim: usize,
}

impl CsmAudioHead {
    /// Construct with zeros for all heads.
    pub fn zeros(n_dep_codebooks: usize, dep_dim: usize) -> Self {
        let weights = (0..n_dep_codebooks)
            .map(|_| vec![0.0f32; AUDIO_VOCAB_SIZE * dep_dim])
            .collect();
        Self { weights, n_codebooks: n_dep_codebooks, dep_dim }
    }

    /// Construct with random weights.
    pub fn random(n_dep_codebooks: usize, dep_dim: usize) -> Self {
        let scale = 1.0 / (dep_dim as f32).sqrt();
        let weights = (0..n_dep_codebooks)
            .map(|i| {
                let n = AUDIO_VOCAB_SIZE * dep_dim;
                let mut w = Vec::with_capacity(n);
                let mut s = (7777u64 + i as u64 * 12345) | 1;
                for _ in 0..n {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    let f = (s >> 11) as f32 / (1u64 << 53) as f32;
                    w.push((f * 2.0 - 1.0) * scale);
                }
                w
            })
            .collect();
        Self { weights, n_codebooks: n_dep_codebooks, dep_dim }
    }

    /// Compute logits for one codebook.
    ///
    /// `depth`:  codebook index, **1-indexed** (1 = CB1, …, 7 = CB7).
    /// `hidden`: `[dep_dim]` depformer hidden state.
    /// `logits`: `[AUDIO_VOCAB_SIZE]` output buffer.
    pub fn forward(&self, depth: usize, hidden: &[f32], logits: &mut [f32]) {
        assert!(
            depth >= 1 && depth <= self.n_codebooks,
            "depth {depth} out of range 1..={}",
            self.n_codebooks
        );
        assert_eq!(hidden.len(), self.dep_dim);
        assert_eq!(logits.len(), AUDIO_VOCAB_SIZE);
        let w = &self.weights[depth - 1];
        for out_i in 0..AUDIO_VOCAB_SIZE {
            let row = &w[out_i * self.dep_dim..(out_i + 1) * self.dep_dim];
            logits[out_i] = row.iter().zip(hidden.iter()).map(|(&wi, &hi)| wi * hi).sum();
        }
    }

    /// Compute logits for one codebook and return as a Vec.
    pub fn forward_vec(&self, depth: usize, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; AUDIO_VOCAB_SIZE];
        self.forward(depth, hidden, &mut logits);
        logits
    }
}

// ── Linear helper (for use in BackboneModel) ─────────────────────────────────

/// Apply a linear transformation with optional bias.
///
/// `input`:   `[N × in_dim]` flat
/// `weight`:  `[out_dim × in_dim]` row-major
/// `bias`:    `[out_dim]` (may be empty / all zeros)
/// `n`:       batch size (number of row vectors in `input`)
/// Returns:   `[N × out_dim]` flat
pub fn linear_forward(
    input:   &[f32],
    weight:  &[f32],
    bias:    &[f32],
    n:       usize,
    in_dim:  usize,
    out_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(),  n * in_dim);
    assert_eq!(weight.len(), out_dim * in_dim);
    assert!(bias.is_empty() || bias.len() == out_dim);

    let mut out = vec![0.0f32; n * out_dim];
    for row in 0..n {
        let x = &input[row * in_dim..(row + 1) * in_dim];
        let y = &mut out[row * out_dim..(row + 1) * out_dim];
        for i in 0..out_dim {
            let w = &weight[i * in_dim..(i + 1) * in_dim];
            y[i] = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            if !bias.is_empty() {
                y[i] += bias[i];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csm_linear_forward() {
        // 2×2 identity matrix
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        let lin = CsmLinear::from_data(w, 2, 2);
        let x   = vec![3.0f32, 5.0];
        let y   = lin.forward_vec(&x);
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn test_audio_head_shape() {
        // Use tiny dep_dim instead of full 1024 for speed
        let dep_dim = 16;
        let head = CsmCB0Head::random(dep_dim, 42);
        let hidden = vec![1.0f32; dep_dim];
        let logits = head.forward_vec(&hidden);
        assert_eq!(logits.len(), AUDIO_VOCAB_SIZE);
    }

    #[test]
    fn test_audio_head_finite() {
        let dep_dim = 16;
        let head = CsmCB0Head::random(dep_dim, 99);
        let hidden = vec![0.5f32; dep_dim];
        let logits = head.forward_vec(&hidden);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_text_head_shape() {
        // Mimic Linear(d_model → text_vocab) but with tiny dims
        let in_dim  = 16;
        let out_dim = 50;  // tiny text vocab for test
        let head = CsmLinear::random(in_dim, out_dim, 7);
        let hidden = vec![0.1f32; in_dim];
        let logits = head.forward_vec(&hidden);
        assert_eq!(logits.len(), out_dim);
    }

    #[test]
    fn test_audio_logits_softmax() {
        // Verify that softmax over tiny CB0 logits sums to 1
        let dep_dim = 8;
        let mut head = CsmCB0Head::zeros(dep_dim);
        // Give it non-trivial weights
        for (i, w) in head.weight.iter_mut().enumerate() {
            *w = (i as f32 * 0.01) - 0.5;
        }
        let hidden = vec![1.0f32; dep_dim];
        let logits = head.forward_vec(&hidden);
        // softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "softmax sum = {}", total);
    }

    #[test]
    fn test_text_logits_bounded() {
        // Verify text logits are finite with large output dimension
        let in_dim  = 16;
        let out_dim = 128_000;  // full text vocab
        let head = CsmLinear::zeros(in_dim, out_dim);
        let hidden = vec![1.0f32; in_dim];
        let logits = head.forward_vec(&hidden);
        assert_eq!(logits.len(), out_dim);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_csm_audio_head_n_codebooks() {
        let n_dep = 7;
        let dep_dim = 8;
        let head = CsmAudioHead::random(n_dep, dep_dim);
        assert_eq!(head.n_codebooks, 7);
        let hidden = vec![0.5f32; dep_dim];
        for depth in 1..=7 {
            let logits = head.forward_vec(depth, &hidden);
            assert_eq!(logits.len(), AUDIO_VOCAB_SIZE);
            assert!(logits.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_linear_forward_helper() {
        // 1 row, 2→3 linear (all ones weight, no bias)
        let w = vec![1.0f32; 3 * 2];
        let x = vec![1.0f32, 2.0f32];
        let y = linear_forward(&x, &w, &[], 1, 2, 3);
        assert_eq!(y.len(), 3);
        // each output = sum(x) = 3.0
        for &v in &y { assert!((v - 3.0).abs() < 1e-6); }
    }
}
