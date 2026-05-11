//! Residual unit used in both SEANetEncoder and SEANetDecoder.
//!
//! Mimi config: compress=2 (hidden = dim/2), true_skip=True (identity shortcut),
//! dilation=1 (dilation_base^0 = 1 with n_residual_layers=1).
//!
//! Forward: x + ELU(conv2(ELU(conv1(ELU(x)))))  — applies ELU before conv1.

use crate::causal_conv::{elu, CausalConv1d};

/// One residual block with a bottleneck (compress=2) and identity shortcut.
///
/// Architecture:
/// ```text
/// x ─────────────────────────────────────────────── +
///   ↓ ELU                                           │
///   ↓ CausalConv1d(dim → dim/2, k=3, dilation=1)   │
///   ↓ ELU                                           │
///   ↓ CausalConv1d(dim/2 → dim, k=1)               │
/// ──────────────────────────────────────────────────┘
/// ```
pub struct ResidualUnit {
    /// Bottleneck compression conv: dim → dim/2, kernel=3.
    pub conv1: CausalConv1d,
    /// Point-wise restore conv: dim/2 → dim, kernel=1.
    pub conv2: CausalConv1d,
    /// Full channel count (input and output).
    pub channels: usize,
}

impl ResidualUnit {
    /// Construct a ResidualUnit for the given channel count.
    ///
    /// `channels` must be even (compress=2 halves the hidden dim).
    pub fn new(channels: usize) -> Self {
        let hidden = channels / 2;
        // dilation = dilation_base^0 = 2^0 = 1
        let conv1 = CausalConv1d::new(channels, hidden, 3, 1, 1);
        let conv2 = CausalConv1d::new(hidden, channels, 1, 1, 1);
        ResidualUnit { conv1, conv2, channels }
    }

    /// Forward pass: [B, channels, T] → [B, channels, T].
    ///
    /// Applies ELU before conv1 and between conv1/conv2, then adds identity.
    pub fn forward(&self, x: &[f32], batch: usize, channels: usize, t: usize) -> Vec<f32> {
        let hidden = channels / 2;

        // Apply ELU to input before first conv.
        let h: Vec<f32> = x.iter().map(|&v| elu(v)).collect();

        // conv1: [B, channels, T] → [B, hidden, T]
        let (h, _) = self.conv1.forward(&h, batch, channels, t);

        // ELU between convs.
        let h: Vec<f32> = h.iter().map(|&v| elu(v)).collect();

        // conv2: [B, hidden, T] → [B, channels, T]
        let (h, _) = self.conv2.forward(&h, batch, hidden, t);

        // Identity residual (true_skip): output = x + h
        x.iter().zip(h.iter()).map(|(&a, &b)| a + b).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_unit_shape() {
        // Input shape must be preserved: [B, C, T] → [B, C, T].
        let ru = ResidualUnit::new(64);
        let batch = 2;
        let t = 50;
        let input = vec![0.5f32; batch * 64 * t];
        let out = ru.forward(&input, batch, 64, t);
        assert_eq!(out.len(), batch * 64 * t);
    }

    #[test]
    fn test_residual_unit_elu_applied() {
        // With negative input, ELU produces negative outputs — confirm output can be < 0.
        let ru = ResidualUnit::new(4);
        // Use large negative input; ELU(-inf) → -1, and identity adds negative.
        let input = vec![-10.0f32; 1 * 4 * 10];
        let out = ru.forward(&input, 1, 4, 10);
        // At least some outputs should be negative (ELU is not ReLU).
        assert!(out.iter().any(|&v| v < 0.0), "expected some negative outputs from ELU");
    }

    #[test]
    fn test_residual_unit_identity_adds() {
        // Zero weights in conv2 → conv branch ≈ 0 → output ≈ input.
        let mut ru = ResidualUnit::new(4);
        // Zero out all weights and biases in both convs.
        ru.conv1.weight_norm.v.iter_mut().for_each(|w| *w = 0.0);
        ru.conv1.bias.iter_mut().for_each(|b| *b = 0.0);
        ru.conv2.weight_norm.v.iter_mut().for_each(|w| *w = 0.0);
        ru.conv2.bias.iter_mut().for_each(|b| *b = 0.0);
        ru.conv2.weight_norm.g.iter_mut().for_each(|g| *g = 1e-12);
        ru.conv1.weight_norm.g.iter_mut().for_each(|g| *g = 1e-12);

        let input: Vec<f32> = (0..1 * 4 * 10).map(|i| i as f32 * 0.1).collect();
        let out = ru.forward(&input, 1, 4, 10);
        // With effectively zero conv weights, output ≈ input.
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "identity failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_residual_unit_output_finite() {
        let ru = ResidualUnit::new(32);
        let input: Vec<f32> = (0..1 * 32 * 20).map(|i| (i as f32) * 0.001).collect();
        let out = ru.forward(&input, 1, 32, 20);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output in ResidualUnit");
    }
}
