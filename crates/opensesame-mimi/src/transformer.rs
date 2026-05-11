//! In-codec causal transformer for Mimi (LayerNorm + RoPE, 8L/8H/512d).

use crate::config::MimiConfig;

// ── GELU ─────────────────────────────────────────────────────────────────────
#[inline]
fn gelu_f(x: f32) -> f32 {
    let c = 0.797_884_56_f32;
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

/// Standard LayerNorm: `y = γ * (x - μ) / √(σ²+ε) + β`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale γ, shape `[d]`.
    pub gamma: Vec<f32>,
    /// Bias β, shape `[d]`.
    pub beta:  Vec<f32>,
    /// Epsilon for stability.
    pub eps:   f32,
    /// Feature dimension.
    pub dim:   usize,
}

impl LayerNorm {
    /// Create identity LayerNorm (γ=1, β=0).
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { gamma: vec![1.0_f32; dim], beta: vec![0.0_f32; dim], eps, dim }
    }

    /// Normalise a single vector `x` of length `dim`.
    pub fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var  = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv  = 1.0 / (var + self.eps).sqrt();
        x.iter().enumerate().map(|(i, v)| self.gamma[i] * (v - mean) * inv + self.beta[i]).collect()
    }

    /// Normalise a batch `x` of shape `[T, D]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0_f32; t * d];
        for ti in 0..t {
            let row = self.forward_vec(&x[ti * d..(ti + 1) * d]);
            out[ti * d..(ti + 1) * d].copy_from_slice(&row);
        }
        out
    }
}

// ── RoPE ──────────────────────────────────────────────────────────────────────

/// Rotary Position Embeddings (RoPE).
#[derive(Debug, Clone)]
pub struct RoPE {
    /// half of head_dim
    pub half_dim: usize,
    /// θ_i = 1 / base^(2i/d)
    pub freqs: Vec<f32>,
}

impl RoPE {
    /// Construct for a given head_dim and base period.
    pub fn new(head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        let freqs = (0..half).map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32)).collect();
        Self { half_dim: half, freqs }
    }

    /// Rotate a single Q or K vector in-place at position `pos`.
    pub fn rotate_vec(&self, v: &mut [f32], pos: usize) {
        let h = self.half_dim;
        for i in 0..h {
            let (sin, cos) = (self.freqs[i] * pos as f32).sin_cos();
            let x = v[i]; let y = v[i + h];
            v[i]     = x * cos - y * sin;
            v[i + h] = x * sin + y * cos;
        }
    }
}

// ── Linear ────────────────────────────────────────────────────────────────────

/// Dense linear: `y = x W^T + b`. Weights `[out, in]`.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `[out_dim × in_dim]`
    pub weight:  Vec<f32>,
    /// `[out_dim]` or empty.
    pub bias:    Vec<f32>,
    /// Input dim.
    pub in_dim:  usize,
    /// Output dim.
    pub out_dim: usize,
}

impl Linear {
    /// Zero-initialised layer.
    pub fn new(in_dim: usize, out_dim: usize, has_bias: bool) -> Self {
        Self {
            weight:  vec![0.0_f32; out_dim * in_dim],
            bias:    if has_bias { vec![0.0_f32; out_dim] } else { vec![] },
            in_dim, out_dim,
        }
    }

    /// Apply to a single vector.
    pub fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut y = if self.bias.is_empty() { vec![0.0_f32; self.out_dim] } else { self.bias.clone() };
        for o in 0..self.out_dim {
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            y[o] += row.iter().zip(x.iter()).map(|(w, x)| w * x).sum::<f32>();
        }
        y
    }

    /// Apply to batch `[T, in_dim]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; t * self.out_dim];
        for ti in 0..t {
            let row = self.forward_vec(&x[ti * self.in_dim..(ti + 1) * self.in_dim]);
            out[ti * self.out_dim..(ti + 1) * self.out_dim].copy_from_slice(&row);
        }
        out
    }
}

// ── CausalDepthwiseConv1d ─────────────────────────────────────────────────────

/// Causal depthwise Conv1d, kernel=5, preserves sequence length.
#[derive(Debug, Clone)]
pub struct CausalDepthwiseConv1d {
    /// `[C, k]`
    pub weight:      Vec<f32>,
    /// `[C]` or empty.
    pub bias:        Vec<f32>,
    /// Channels.
    pub channels:    usize,
    /// Kernel size.
    pub kernel_size: usize,
}

impl CausalDepthwiseConv1d {
    /// Zero-init.
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        Self { weight: vec![0.0_f32; channels * kernel_size], bias: vec![0.0_f32; channels], channels, kernel_size }
    }

    /// Forward on `[T, C]`, returns `[T, C]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = self.channels; let k = self.kernel_size; let pad = k - 1;
        let mut y = vec![0.0_f32; t * c];
        for ot in 0..t {
            for ch in 0..c {
                let mut acc = if self.bias.is_empty() { 0.0 } else { self.bias[ch] };
                for ki in 0..k {
                    let tp = ot as isize + ki as isize - pad as isize;
                    if tp >= 0 && (tp as usize) < t {
                        acc += self.weight[ch * k + ki] * x[tp as usize * c + ch];
                    }
                }
                y[ot * c + ch] = acc;
            }
        }
        y
    }
}

// ── MultiHeadAttention ────────────────────────────────────────────────────────

/// Causal MHA with RoPE and optional local window.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub q_proj:   Linear,
    pub k_proj:   Linear,
    pub v_proj:   Linear,
    pub out_proj: Linear,
    /// Number of heads.
    pub n_heads:  usize,
    /// Dim per head.
    pub head_dim: usize,
    /// Model dim.
    pub dim:      usize,
    /// Local attention window (0 = global).
    pub context:  usize,
    /// RoPE.
    pub rope:     RoPE,
}

impl MultiHeadAttention {
    /// Zero-init MHA.
    pub fn new(dim: usize, n_heads: usize, context: usize, rope_base: f32) -> Self {
        let head_dim = dim / n_heads;
        Self {
            q_proj:   Linear::new(dim, dim, false),
            k_proj:   Linear::new(dim, dim, false),
            v_proj:   Linear::new(dim, dim, false),
            out_proj: Linear::new(dim, dim, false),
            n_heads, head_dim, dim, context,
            rope: RoPE::new(head_dim, rope_base),
        }
    }

    /// Forward on `[T, D]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let d = self.dim; let nh = self.n_heads; let hd = self.head_dim;
        let scale = (hd as f32).sqrt().recip();
        let mut q = self.q_proj.forward(x, t);
        let mut k = self.k_proj.forward(x, t);
        let v     = self.v_proj.forward(x, t);
        for ti in 0..t {
            for h in 0..nh {
                let qs = ti * d + h * hd;
                self.rope.rotate_vec(&mut q[qs..qs + hd], ti);
                let ks = ti * d + h * hd;
                self.rope.rotate_vec(&mut k[ks..ks + hd], ti);
            }
        }
        let mut out = vec![0.0_f32; t * d];
        let mut scores = vec![0.0_f32; t];
        for h in 0..nh {
            for ti in 0..t {
                let lo = if self.context > 0 && ti + 1 > self.context { ti + 1 - self.context } else { 0 };
                let win = ti - lo + 1;
                let q_v = &q[ti * d + h * hd..ti * d + (h + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for tj in lo..=ti {
                    let k_v = &k[tj * d + h * hd..tj * d + (h + 1) * hd];
                    let sc = q_v.iter().zip(k_v.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                    scores[tj - lo] = sc;
                    if sc > mx { mx = sc; }
                }
                let mut sum_exp = 0.0_f32;
                for j in 0..win { scores[j] = (scores[j] - mx).exp(); sum_exp += scores[j]; }
                for j in 0..win { scores[j] /= sum_exp; }
                for tj in lo..=ti {
                    let w = scores[tj - lo];
                    let v_v = &v[tj * d + h * hd..tj * d + (h + 1) * hd];
                    for di in 0..hd { out[ti * d + h * hd + di] += w * v_v[di]; }
                }
            }
        }
        self.out_proj.forward(&out, t)
    }
}

// ── FeedForward ───────────────────────────────────────────────────────────────

/// FFN: Linear(d, ffn) → GELU → Linear(ffn, d).
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, ffn_dim: usize) -> Self {
        Self { fc1: Linear::new(dim, ffn_dim, false), fc2: Linear::new(ffn_dim, dim, false) }
    }
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let h: Vec<f32> = self.fc1.forward(x, t).into_iter().map(gelu_f).collect();
        self.fc2.forward(&h, t)
    }
}

// ── TransformerLayer ──────────────────────────────────────────────────────────

/// Single Mimi transformer layer: LN→Attn→DWConv→LN→FFN (pre-norm, layer-scale).
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub ln1:     LayerNorm,
    pub attn:    MultiHeadAttention,
    pub ln2:     LayerNorm,
    pub ffn:     FeedForward,
    pub dw_conv: CausalDepthwiseConv1d,
    pub ls1:     Vec<f32>,
    pub ls2:     Vec<f32>,
}

impl TransformerLayer {
    pub fn new(dim: usize, n_heads: usize, ffn_dim: usize, context: usize,
               conv_k: usize, rope_base: f32, norm_eps: f32, ls_init: f32) -> Self {
        Self {
            ln1:     LayerNorm::new(dim, norm_eps),
            attn:    MultiHeadAttention::new(dim, n_heads, context, rope_base),
            ln2:     LayerNorm::new(dim, norm_eps),
            ffn:     FeedForward::new(dim, ffn_dim),
            dw_conv: CausalDepthwiseConv1d::new(dim, conv_k),
            ls1:     vec![ls_init; dim],
            ls2:     vec![ls_init; dim],
        }
    }

    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let d = self.attn.dim;
        let attn_out = self.attn.forward(&self.ln1.forward(x, t), t);
        let mut h: Vec<f32> = (0..t * d).map(|i| x[i] + self.ls1[i % d] * attn_out[i]).collect();
        let conv_out = self.dw_conv.forward(&h, t);
        for i in 0..t * d { h[i] += conv_out[i]; }
        let ffn_out = self.ffn.forward(&self.ln2.forward(&h, t), t);
        (0..t * d).map(|i| h[i] + self.ls2[i % d] * ffn_out[i]).collect()
    }
}

// ── MimiTransformer ───────────────────────────────────────────────────────────

/// Causal in-codec transformer used by Mimi encoder and decoder.
pub struct MimiTransformer {
    pub layers: Vec<TransformerLayer>,
    pub norm:   LayerNorm,
    pub dim:    usize,
}

impl MimiTransformer {
    /// Create from [`MimiConfig`].
    pub fn new(cfg: &MimiConfig) -> Self {
        let layers = (0..cfg.transformer_layers).map(|_| TransformerLayer::new(
            cfg.transformer_dim, cfg.transformer_heads, cfg.ffn_dim,
            cfg.transformer_context, cfg.conv_kernel_size, cfg.rope_base,
            cfg.norm_eps, cfg.layer_scale_init,
        )).collect();
        Self { layers, norm: LayerNorm::new(cfg.transformer_dim, cfg.norm_eps), dim: cfg.transformer_dim }
    }

    /// Forward on `[T, D]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        for layer in &self.layers { h = layer.forward(&h, t); }
        self.norm.forward(&h, t)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MimiConfig;

    fn tiny_cfg() -> MimiConfig {
        MimiConfig {
            transformer_dim: 8, transformer_heads: 2, transformer_layers: 1,
            transformer_context: 4, ffn_dim: 32, conv_kernel_size: 5,
            layer_scale_init: 0.01, rope_base: 10_000.0, norm_eps: 1e-5,
            ..MimiConfig::default()
        }
    }

    #[test]
    fn test_layernorm_mean_zero() {
        let ln = LayerNorm::new(4, 1e-5);
        let y = ln.forward_vec(&[1.0, 2.0, 3.0, 4.0]);
        let mean = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_layernorm_batch_shape() {
        let ln = LayerNorm::new(8, 1e-5);
        let y = ln.forward(&vec![0.5_f32; 16 * 8], 16);
        assert_eq!(y.len(), 16 * 8);
    }

    #[test]
    fn test_rope_pos0_identity() {
        let rope = RoPE::new(4, 10_000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let mut v = orig.clone();
        rope.rotate_vec(&mut v, 0);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rope_no_crash() {
        let rope = RoPE::new(64, 10_000.0);
        let mut v = vec![0.1_f32; 64];
        rope.rotate_vec(&mut v, 7);
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn test_linear_matmul() {
        let mut lin = Linear::new(2, 3, false);
        lin.weight = vec![1.0, 0.0,  0.0, 1.0,  1.0, 1.0];
        let y = lin.forward_vec(&[2.0, 3.0]);
        assert!((y[0] - 2.0).abs() < 1e-5);
        assert!((y[1] - 3.0).abs() < 1e-5);
        assert!((y[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_dw_conv_shape() {
        let conv = CausalDepthwiseConv1d::new(8, 5);
        let y = conv.forward(&vec![1.0_f32; 20 * 8], 20);
        assert_eq!(y.len(), 20 * 8);
    }

    #[test]
    fn test_mha_zero_weights_zero_output() {
        let mha = MultiHeadAttention::new(8, 2, 4, 10_000.0);
        let y = mha.forward(&vec![1.0_f32; 3 * 8], 3);
        for v in &y { assert!(v.abs() < 1e-5); }
    }

    #[test]
    fn test_mha_output_shape() {
        let mha = MultiHeadAttention::new(16, 4, 10, 10_000.0);
        let y = mha.forward(&vec![0.1_f32; 6 * 16], 6);
        assert_eq!(y.len(), 6 * 16);
    }

    #[test]
    fn test_ffn_shape() {
        let ffn = FeedForward::new(8, 32);
        assert_eq!(ffn.forward(&vec![0.2_f32; 5 * 8], 5).len(), 5 * 8);
    }

    #[test]
    fn test_transformer_layer_shape() {
        let layer = TransformerLayer::new(8, 2, 32, 4, 5, 10_000.0, 1e-5, 0.01);
        assert_eq!(layer.forward(&vec![0.1_f32; 6 * 8], 6).len(), 6 * 8);
    }

    #[test]
    fn test_mimi_transformer_shape() {
        let trm = MimiTransformer::new(&tiny_cfg());
        let y = trm.forward(&vec![0.1_f32; 10 * 8], 10);
        assert_eq!(y.len(), 10 * 8);
    }

    #[test]
    fn test_transformer_output_finite() {
        let trm = MimiTransformer::new(&tiny_cfg());
        let x: Vec<f32> = (0..5 * 8).map(|i| i as f32 * 0.01).collect();
        let y = trm.forward(&x, 5);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rope_rotates_nontrivially() {
        // pos > 0 should rotate the vector away from its original direction.
        let rope = RoPE::new(8, 10_000.0);
        let original = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut rotated = original.clone();
        rope.rotate_vec(&mut rotated, 1);
        // At pos=1 with base=10000, at least some components should differ.
        let changed = original.iter().zip(rotated.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "RoPE at pos=1 should change the vector");
    }

    #[test]
    fn test_rope_pos_differences() {
        // Two different positions should produce different rotations.
        let rope = RoPE::new(8, 10_000.0);
        let v = vec![1.0f32, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];
        let mut v1 = v.clone();
        let mut v2 = v.clone();
        rope.rotate_vec(&mut v1, 1);
        rope.rotate_vec(&mut v2, 5);
        assert_ne!(v1, v2, "different positions should produce different rotations");
    }

    #[test]
    fn test_causal_attention_no_future_leakage() {
        // With non-trivial weights, token 0's output should not change when we
        // append a second token after it (causal masking guarantee).
        let d = 8usize; let nh = 2usize;
        let mut mha = MultiHeadAttention::new(d, nh, 0, 10_000.0);
        // Give q/k/v/out identity-ish weights (diagonal matrices) so non-zero output.
        for o in 0..d {
            mha.q_proj.weight[o * d + o]   = 1.0;
            mha.k_proj.weight[o * d + o]   = 1.0;
            mha.v_proj.weight[o * d + o]   = 1.0;
            mha.out_proj.weight[o * d + o] = 1.0;
        }
        // Token 0: same in both sequences.
        let token0: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        // Token 1: deliberately very different (would contaminate if causality breaks).
        let token1: Vec<f32> = (0..d).map(|i| 100.0 + i as f32).collect();

        // Single-token sequence (only token 0).
        let y1 = mha.forward(&token0, 1);

        // Two-token sequence: [token0, token1].
        let mut x2 = token0.clone();
        x2.extend_from_slice(&token1);
        let y2 = mha.forward(&x2, 2);

        // Token 0's output must be identical in both cases (causal masking).
        let first_token_same = y1.iter().zip(y2[..d].iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        assert!(first_token_same, "causal: first token output must not depend on future tokens");
    }

    #[test]
    fn test_layernorm_gamma_beta_applied() {
        // With gamma=2 and beta=1, output should be 2*(normalized)+1.
        let mut ln = LayerNorm::new(4, 1e-5);
        ln.gamma = vec![2.0; 4];
        ln.beta  = vec![1.0; 4];
        let y = ln.forward_vec(&[1.0, 2.0, 3.0, 4.0]);
        // mean=2.5, variance=1.25, so normalized values are spread around 0.
        // All outputs should be 1.0 + 2.0 * normalized_i.
        let mean_out = y.iter().sum::<f32>() / 4.0;
        assert!((mean_out - 1.0).abs() < 1e-4, "with beta=1, mean should be ~1.0");
    }

    #[test]
    fn test_linear_bias_applied() {
        let mut lin = Linear::new(2, 2, true);
        lin.weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        lin.bias   = vec![10.0, 20.0];
        let y = lin.forward_vec(&[1.0, 2.0]);
        assert!((y[0] - 11.0).abs() < 1e-5);
        assert!((y[1] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn test_dw_conv_causality() {
        // With a kernel that is all-zero except the last (past) position,
        // the output at t=0 should depend only on the (zero-padded) past.
        let c = 2usize; let k = 3usize;
        let mut conv = CausalDepthwiseConv1d::new(c, k);
        // Only the earliest kernel position (oldest past) has weight 1.
        // kernel layout: [c, k] — for ch=0, position 0 (oldest) = 1.
        conv.weight[0 * k + 0] = 1.0; // ch=0, oldest position
        conv.weight[1 * k + 0] = 1.0; // ch=1, oldest position
        conv.bias = vec![];
        // Input: only t=1 is non-zero.
        let mut x = vec![0.0f32; 4 * c];
        x[1 * c + 0] = 5.0;
        x[1 * c + 1] = 7.0;
        let y = conv.forward(&x, 4);
        // At t=2, the oldest kernel position reaches t=0 (zero). 
        // At t=3, it reaches t=1 (non-zero) → output should be 5, 7.
        assert!((y[3 * c + 0] - 5.0).abs() < 1e-5, "causal conv output mismatch ch0");
        assert!((y[3 * c + 1] - 7.0).abs() < 1e-5, "causal conv output mismatch ch1");
    }

    #[test]
    fn test_local_window_respected() {
        // With context=1 (each token only sees itself), the output at each
        // position should depend only on local info — test that using context=1
        // is different from context=0 (full attention) with non-trivial input.
        let d = 8usize; let nh = 2usize;
        let mut mha_full   = MultiHeadAttention::new(d, nh, 0, 10_000.0);
        let mut mha_window = MultiHeadAttention::new(d, nh, 1, 10_000.0);
        // Same non-trivial weights
        for o in 0..d {
            mha_full.q_proj.weight[o * d + o]   = 0.5;
            mha_window.q_proj.weight[o * d + o] = 0.5;
            mha_full.k_proj.weight[o * d + o]   = 0.5;
            mha_window.k_proj.weight[o * d + o] = 0.5;
            mha_full.v_proj.weight[o * d + o]   = 0.5;
            mha_window.v_proj.weight[o * d + o] = 0.5;
            mha_full.out_proj.weight[o * d + o]   = 0.5;
            mha_window.out_proj.weight[o * d + o] = 0.5;
        }
        let x: Vec<f32> = (0..4 * d).map(|i| (i % 7) as f32 * 0.1).collect();
        let y_full   = mha_full.forward(&x, 4);
        let y_window = mha_window.forward(&x, 4);
        // With window=1, token 3 sees only itself (not tokens 0-2).
        // They should differ at token 3 if earlier tokens carry different values.
        assert_ne!(y_full[3*d..4*d], y_window[3*d..4*d],
            "window=1 vs full attention should differ at token 3");
    }
}
