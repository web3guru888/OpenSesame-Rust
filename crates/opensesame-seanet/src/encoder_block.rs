//! Encoder block: one residual unit followed by strided downsampling.
//!
//! Mimi strides: [8, 6, 5, 4] (block 0 has stride=8, block 3 has stride=4).
//! Block: ResidualUnit(in_ch) → ELU → CausalConv1d(in_ch→out_ch, k=2*stride, s=stride).

use crate::causal_conv::{elu, CausalConv1d};
use crate::residual_unit::ResidualUnit;

/// One SEANet encoder stage: residual unit(s) + ELU + strided downsampling conv.
///
/// Mimi config: n_residual_layers=1 (one ResidualUnit, dilation=1).
pub struct EncoderBlock {
    /// Residual processing units before downsampling.
    pub residual_units: Vec<ResidualUnit>,
    /// Strided causal conv: in_ch → out_ch, kernel=2*stride.
    pub downsample: CausalConv1d,
    /// Input channel count.
    pub in_ch: usize,
    /// Output channel count.
    pub out_ch: usize,
    /// Downsample stride.
    pub stride: usize,
}

impl EncoderBlock {
    /// Construct an EncoderBlock.
    ///
    /// `stride` is the downsampling ratio; kernel = 2 * stride.
    pub fn new(in_ch: usize, out_ch: usize, stride: usize) -> Self {
        let residual_units = vec![ResidualUnit::new(in_ch)];
        let downsample = CausalConv1d::new(in_ch, out_ch, 2 * stride, stride, 1);
        EncoderBlock { residual_units, downsample, in_ch, out_ch, stride }
    }

    /// Forward pass: [B, in_ch, T_in] → [B, out_ch, T_in / stride].
    pub fn forward(&self, x: &[f32], batch: usize, t_in: usize) -> (Vec<f32>, usize) {
        let mut h = x.to_vec();
        for unit in &self.residual_units {
            h = unit.forward(&h, batch, self.in_ch, t_in);
        }
        h.iter_mut().for_each(|v| *v = elu(*v));
        self.downsample.forward(&h, batch, self.in_ch, t_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use T compatible with each stride: 960 for stride=8, 120 for stride=6, etc.

    #[test]
    fn test_encoder_block_stride8() {
        // [B,32,960] → [B,64,120]   (960/8 = 120)
        let block = EncoderBlock::new(32, 64, 8);
        let input = vec![0.0f32; 1 * 32 * 960];
        let (out, t_out) = block.forward(&input, 1, 960);
        assert_eq!(t_out, 120);
        assert_eq!(out.len(), 1 * 64 * 120);
    }

    #[test]
    fn test_encoder_block_stride6() {
        // [B,64,120] → [B,128,20]   (120/6 = 20)
        let block = EncoderBlock::new(64, 128, 6);
        let input = vec![0.0f32; 1 * 64 * 120];
        let (out, t_out) = block.forward(&input, 1, 120);
        assert_eq!(t_out, 20);
        assert_eq!(out.len(), 1 * 128 * 20);
    }

    #[test]
    fn test_encoder_block_stride5() {
        // [B,128,20] → [B,256,4]   (20/5 = 4)
        let block = EncoderBlock::new(128, 256, 5);
        let input = vec![0.0f32; 1 * 128 * 20];
        let (out, t_out) = block.forward(&input, 1, 20);
        assert_eq!(t_out, 4);
        assert_eq!(out.len(), 1 * 256 * 4);
    }

    #[test]
    fn test_encoder_block_stride4() {
        // [B,512,4] → [B,1024,1]   (4/4 = 1) — Mimi block 3: 512→1024, stride=4
        let block = EncoderBlock::new(512, 1024, 4);
        let input = vec![0.0f32; 1 * 512 * 4];
        let (out, t_out) = block.forward(&input, 1, 4);
        assert_eq!(t_out, 1);
        assert_eq!(out.len(), 1 * 1024 * 1);
    }

    #[test]
    fn test_encoder_block_channels_correct() {
        let block = EncoderBlock::new(64, 128, 8);
        assert_eq!(block.in_ch, 64);
        assert_eq!(block.out_ch, 128);
        assert_eq!(block.stride, 8);
        assert_eq!(block.downsample.weight_norm.out_ch, 128);
        assert_eq!(block.downsample.weight_norm.in_ch, 64);
    }

    #[test]
    fn test_encoder_block_output_finite() {
        let block = EncoderBlock::new(4, 8, 4);
        let input: Vec<f32> = (0..1 * 4 * 8).map(|i| (i as f32) * 0.001).collect();
        let (out, _) = block.forward(&input, 1, 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
