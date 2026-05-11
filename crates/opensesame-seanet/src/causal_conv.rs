//! Causal 1D convolution and transposed convolution for streaming audio.
//!
//! CausalConv1d applies left-padding only — no future context leaks into any output sample.
//! CausalConvTranspose1d upsamples and trims from the right (causal trim).
//!
//! The conv is implemented inline (no separate padded buffer) to avoid O(B×C×T) allocation.

use crate::weight_norm::WeightNorm;

// ─────────────────────────────────────────────────────────────────────────────
// ELU activation (shared within this crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential Linear Unit: x if x≥0, else exp(x)-1.  α=1.0.
#[inline]
pub(crate) fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

// ─────────────────────────────────────────────────────────────────────────────
// CausalConv1d
// ─────────────────────────────────────────────────────────────────────────────

/// 1D causal convolution — left-pad only, zero future context.
///
/// Weight shape: [out_ch, in_ch, kernel]. Bias shape: [out_ch].
/// Left padding = (kernel - 1) * dilation ensures T_out = ceil(T_in / stride).
pub struct CausalConv1d {
    /// Weight normalisation (direction + magnitude).
    pub weight_norm: WeightNorm,
    /// Bias term, shape [out_ch].
    pub bias: Vec<f32>,
    /// Convolution stride.
    pub stride: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Left-pad size = (kernel - 1) * dilation.
    pub padding: usize,
}

impl CausalConv1d {
    /// Construct with Kaiming-uniform weight initialisation.
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, dilation: usize) -> Self {
        let weight_norm = WeightNorm::new(out_ch, in_ch, kernel);
        let bias = vec![0.0f32; out_ch];
        let padding = (kernel.saturating_sub(1)) * dilation;
        CausalConv1d { weight_norm, bias, stride, dilation, padding }
    }

    /// Compute the output time length given an input time length.
    ///
    /// T_out = floor((T_in - 1) / stride) + 1 = ceil(T_in / stride).
    pub fn output_len(&self, t_in: usize) -> usize {
        (t_in.saturating_sub(1)) / self.stride + 1
    }

    /// Forward pass: [B, in_ch, T_in] → [B, out_ch, T_out].
    ///
    /// Input flat: index = (b * in_ch + c) * t_in + t.
    /// Returns (output_flat, t_out). Padding is handled inline (no allocation).
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        in_ch: usize,
        t_in: usize,
    ) -> (Vec<f32>, usize) {
        let out_ch = self.weight_norm.out_ch;
        let kernel = self.weight_norm.kernel;
        let t_out = self.output_len(t_in);
        let weight = self.weight_norm.weight();
        let padding = self.padding;

        let mut output = vec![0.0f32; batch * out_ch * t_out];

        for b in 0..batch {
            for oc in 0..out_ch {
                let bias_oc = self.bias[oc];
                let w_base = oc * in_ch * kernel;
                for t in 0..t_out {
                    let mut acc = bias_oc;
                    let t_base = t * self.stride; // position in padded space
                    for ic in 0..in_ch {
                        let x_base = (b * in_ch + ic) * t_in;
                        let wic_base = w_base + ic * kernel;
                        for k in 0..kernel {
                            let t_pad = t_base + k * self.dilation;
                            // Translate from padded space to original space.
                            if t_pad >= padding {
                                let t_orig = t_pad - padding;
                                if t_orig < t_in {
                                    acc += input[x_base + t_orig] * weight[wic_base + k];
                                }
                            }
                            // else: in the zero-padded region, contributes 0
                        }
                    }
                    output[(b * out_ch + oc) * t_out + t] = acc;
                }
            }
        }
        (output, t_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CausalConvTranspose1d
// ─────────────────────────────────────────────────────────────────────────────

/// 1D transposed causal convolution — upsampling with causal right-trim.
///
/// Weight stored as WeightNorm with shape [in_ch, out_ch, kernel].
/// T_out = T_in * stride (after trimming kernel - stride from raw output).
pub struct CausalConvTranspose1d {
    /// Weight normalisation (first dim = in_ch of transposed conv).
    pub weight_norm: WeightNorm,
    /// Bias, shape [out_ch].
    pub bias: Vec<f32>,
    /// Upsample stride.
    pub stride: usize,
    /// Samples trimmed from the right = kernel - stride.
    pub trim: usize,
    /// Output channels (second dim of weight).
    pub out_ch: usize,
    /// Input channels (first dim of weight).
    pub in_ch: usize,
}

impl CausalConvTranspose1d {
    /// Construct a CausalConvTranspose1d layer.
    ///
    /// `kernel` = 2 * stride following Mimi convention.
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Self {
        // WeightNorm first dim = in_ch (input channels of the transposed conv)
        let weight_norm = WeightNorm::new(in_ch, out_ch, kernel);
        let bias = vec![0.0f32; out_ch];
        let trim = kernel.saturating_sub(stride);
        CausalConvTranspose1d { weight_norm, bias, stride, trim, out_ch, in_ch }
    }

    /// Forward pass: [B, in_ch, T_in] → [B, out_ch, T_in * stride].
    ///
    /// Input flat: index = (b * in_ch + c) * t_in + t.
    /// Returns (output_flat, t_out) where t_out = T_in * stride.
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        in_ch: usize,
        t_in: usize,
    ) -> (Vec<f32>, usize) {
        let kernel = self.weight_norm.kernel;
        // Raw output before trim.
        let raw_t_out = if t_in == 0 { 0 } else { (t_in - 1) * self.stride + kernel };
        let t_out = t_in * self.stride;

        let weight = self.weight_norm.weight(); // [in_ch, out_ch, kernel]

        // Accumulate into raw output, then add bias and trim.
        let mut raw = vec![0.0f32; batch * self.out_ch * raw_t_out];

        for b in 0..batch {
            for ic in 0..in_ch {
                let x_base = (b * in_ch + ic) * t_in;
                let w_base = ic * self.out_ch * kernel;
                for t in 0..t_in {
                    let xval = input[x_base + t];
                    let t_out_start = t * self.stride;
                    for oc in 0..self.out_ch {
                        let y_base = (b * self.out_ch + oc) * raw_t_out;
                        let woc_base = w_base + oc * kernel;
                        for k in 0..kernel {
                            raw[y_base + t_out_start + k] += weight[woc_base + k] * xval;
                        }
                    }
                }
            }
        }

        // Copy t_out samples per channel (trim) and add bias.
        let mut output = vec![0.0f32; batch * self.out_ch * t_out];
        for b in 0..batch {
            for oc in 0..self.out_ch {
                let boc = self.bias[oc];
                let src = (b * self.out_ch + oc) * raw_t_out;
                let dst = (b * self.out_ch + oc) * t_out;
                for t in 0..t_out {
                    output[dst + t] = raw[src + t] + boc;
                }
            }
        }
        (output, t_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // All tests use small T to stay fast (T=960 or less).

    #[test]
    fn test_causal_conv_shape() {
        // [B,1,960] with stride=1 should produce [B,32,960].
        let conv = CausalConv1d::new(1, 32, 7, 1, 1);
        let input = vec![0.0f32; 1 * 1 * 960];
        let (out, t_out) = conv.forward(&input, 1, 1, 960);
        assert_eq!(t_out, 960);
        assert_eq!(out.len(), 1 * 32 * 960);
    }

    #[test]
    fn test_causal_conv_stride8() {
        // [B,1,960] with stride=8 → [B,out,120].
        let conv = CausalConv1d::new(1, 4, 16, 8, 1);
        let input = vec![0.0f32; 1 * 1 * 960];
        let (out, t_out) = conv.forward(&input, 1, 1, 960);
        assert_eq!(t_out, 120);
        assert_eq!(out.len(), 1 * 4 * 120);
    }

    #[test]
    fn test_causal_no_future_leakage() {
        // Perturbing input[t+1] must not change output[t].
        let conv = CausalConv1d::new(1, 1, 7, 1, 1);
        let t_in = 30usize;
        let input_a = vec![0.5f32; t_in];
        let mut input_b = input_a.clone();
        // Perturb position 20.
        input_b[20] = 999.0;
        let (out_a, _) = conv.forward(&input_a, 1, 1, t_in);
        let (out_b, _) = conv.forward(&input_b, 1, 1, t_in);
        // output[t] for t < 20 must be identical (future hasn't leaked).
        for t in 0..20 {
            assert_eq!(out_a[t], out_b[t], "future leaked at t={}", t);
        }
        // output[20] sees input[14..21], so it will differ.
        assert_ne!(out_a[20], out_b[20], "expected difference at t=20");
    }

    #[test]
    fn test_causal_conv_dilation_3() {
        // Dilation=3, kernel=3: padding = (3-1)*3 = 6. Shape preserved for stride=1.
        let conv = CausalConv1d::new(1, 2, 3, 1, 3);
        assert_eq!(conv.padding, 6);
        let input = vec![0.0f32; 1 * 1 * 50];
        let (_, t_out) = conv.forward(&input, 1, 1, 50);
        assert_eq!(t_out, 50);
    }

    #[test]
    fn test_causal_conv_bias() {
        // Default bias is zero.
        let conv = CausalConv1d::new(2, 4, 3, 1, 1);
        assert!(conv.bias.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_causal_conv_output_len_formula() {
        // Verify output_len for all Mimi strides with T=960.
        // 960 is divisible by 8, 6, 5 (no), 4 — so use T=960 for strides 8, 4 and T=120 for 6, 5.
        let cases = [(960usize, 8usize, 120usize), (120, 6, 20), (20, 5, 4), (4, 4, 1)];
        for (t_in, s, expected) in cases {
            let conv = CausalConv1d::new(1, 1, 2 * s, s, 1);
            assert_eq!(conv.output_len(t_in), expected, "stride={}", s);
        }
    }

    #[test]
    fn test_causal_conv_stride1_preserves_length() {
        let conv = CausalConv1d::new(4, 8, 3, 1, 1);
        let input = vec![0.0f32; 2 * 4 * 50];
        let (_, t_out) = conv.forward(&input, 2, 4, 50);
        assert_eq!(t_out, 50);
    }

    #[test]
    fn test_causal_conv_multi_channel() {
        let conv = CausalConv1d::new(8, 16, 3, 1, 1);
        let input = vec![1.0f32; 1 * 8 * 10];
        let (out, t_out) = conv.forward(&input, 1, 8, 10);
        assert_eq!(t_out, 10);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_causal_conv_k1() {
        // kernel_size=1: no padding.
        let conv = CausalConv1d::new(4, 8, 1, 1, 1);
        assert_eq!(conv.padding, 0);
        let input = vec![1.0f32; 1 * 4 * 20];
        let (_, t_out) = conv.forward(&input, 1, 4, 20);
        assert_eq!(t_out, 20);
    }

    #[test]
    fn test_causal_conv_large_dilation() {
        // dilation=16, k=3: padding = 16*(3-1) = 32.
        let conv = CausalConv1d::new(1, 1, 3, 1, 16);
        assert_eq!(conv.padding, 32);
        let input = vec![0.0f32; 1 * 1 * 50];
        let (_, t_out) = conv.forward(&input, 1, 1, 50);
        assert_eq!(t_out, 50);
    }

    #[test]
    fn test_causal_conv_output_len_formula_large() {
        // 120 → stride=6 should give exactly 20.
        let conv = CausalConv1d::new(64, 128, 12, 6, 1);
        assert_eq!(conv.output_len(120), 20);
    }

    #[test]
    fn test_causal_conv_mimi_strides_chain() {
        // Chain: 960 → stride=8 → 120 → stride=6 → 20 → stride=5 → 4 → stride=4 → 1
        let t = 960usize;
        let conv8 = CausalConv1d::new(1, 1, 16, 8, 1);
        let conv6 = CausalConv1d::new(1, 1, 12, 6, 1);
        let conv5 = CausalConv1d::new(1, 1, 10, 5, 1);
        let conv4 = CausalConv1d::new(1, 1, 8, 4, 1);
        assert_eq!(conv8.output_len(t), 120);
        assert_eq!(conv6.output_len(120), 20);
        assert_eq!(conv5.output_len(20), 4);
        assert_eq!(conv4.output_len(4), 1);
    }

    // ── CausalConvTranspose1d ─────────────────────────────────────────────────

    #[test]
    fn test_transposed_upsample_stride4() {
        // [B,4,25] → [B,2,100].
        let conv = CausalConvTranspose1d::new(4, 2, 8, 4);
        let input = vec![0.0f32; 1 * 4 * 25];
        let (out, t_out) = conv.forward(&input, 1, 4, 25);
        assert_eq!(t_out, 100);
        assert_eq!(out.len(), 1 * 2 * 100);
    }

    #[test]
    fn test_transposed_upsample_stride8() {
        // [B,4,1] → [B,2,8].
        let conv = CausalConvTranspose1d::new(4, 2, 16, 8);
        let input = vec![0.0f32; 1 * 4 * 1];
        let (out, t_out) = conv.forward(&input, 1, 4, 1);
        assert_eq!(t_out, 8);
        assert_eq!(out.len(), 1 * 2 * 8);
    }

    #[test]
    fn test_transposed_shape_formula() {
        // T_out = T_in * stride after trim.
        for &(t_in, stride) in &[(1usize, 8usize), (4, 4), (20, 5), (20, 6)] {
            let kernel = 2 * stride;
            let conv = CausalConvTranspose1d::new(2, 4, kernel, stride);
            assert_eq!(conv.trim, kernel - stride);
            let input = vec![0.0f32; 1 * 2 * t_in];
            let (_, t_out) = conv.forward(&input, 1, 2, t_in);
            assert_eq!(t_out, t_in * stride, "stride={}", stride);
        }
    }

    #[test]
    fn test_transposed_trim_correct() {
        let conv = CausalConvTranspose1d::new(4, 2, 16, 8);
        assert_eq!(conv.trim, 8);
        let conv2 = CausalConvTranspose1d::new(4, 2, 12, 6);
        assert_eq!(conv2.trim, 6);
    }

    #[test]
    fn test_transposed_output_finite() {
        let conv = CausalConvTranspose1d::new(4, 2, 8, 4);
        let input: Vec<f32> = (0..1 * 4 * 5).map(|i| i as f32 * 0.01).collect();
        let (out, _) = conv.forward(&input, 1, 4, 5);
        assert!(out.iter().all(|x| x.is_finite()), "non-finite output");
    }
}
