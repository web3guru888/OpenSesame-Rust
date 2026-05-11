//! Decoder block: transposed upsampling conv followed by residual unit(s).
//!
//! Mimi decoder strides: [4, 5, 6, 8] (reverse of encoder [8, 6, 5, 4]).
//! Block: CausalConvTranspose1d (upsample) → ResidualUnit.
//! ELU after each block is applied by SEANetDecoder's forward method.

use crate::causal_conv::CausalConvTranspose1d;
use crate::residual_unit::ResidualUnit;

/// One SEANet decoder stage: transposed upsampling followed by residual unit(s).
///
/// Mimi config: n_residual_layers=1 (one ResidualUnit per block).
pub struct DecoderBlock {
    /// Transposed conv for upsampling: in_ch → out_ch.
    pub upsample: CausalConvTranspose1d,
    /// Residual processing after upsampling.
    pub residual_units: Vec<ResidualUnit>,
    /// Input channel count (before upsampling).
    pub in_ch: usize,
    /// Output channel count (after upsampling).
    pub out_ch: usize,
    /// Upsampling stride.
    pub stride: usize,
}

impl DecoderBlock {
    /// Construct a DecoderBlock.
    ///
    /// `in_ch` → `out_ch` via transposed conv with kernel = 2 * stride.
    pub fn new(in_ch: usize, out_ch: usize, stride: usize) -> Self {
        let kernel = 2 * stride;
        let upsample = CausalConvTranspose1d::new(in_ch, out_ch, kernel, stride);
        let residual_units = vec![ResidualUnit::new(out_ch)];
        DecoderBlock { upsample, residual_units, in_ch, out_ch, stride }
    }

    /// Forward pass: [B, in_ch, T_in] → [B, out_ch, T_in * stride].
    ///
    /// Caller applies ELU after this block's output.
    pub fn forward(&self, x: &[f32], batch: usize, t_in: usize) -> (Vec<f32>, usize) {
        let (h, t_up) = self.upsample.forward(x, batch, self.in_ch, t_in);
        let mut h = h;
        for unit in &self.residual_units {
            h = unit.forward(&h, batch, self.out_ch, t_up);
        }
        (h, t_up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_block_stride4() {
        // [B,8,1] → [B,4,4]   (1*4 = 4)
        let block = DecoderBlock::new(8, 4, 4);
        let input = vec![0.0f32; 1 * 8 * 1];
        let (out, t_out) = block.forward(&input, 1, 1);
        assert_eq!(t_out, 4);
        assert_eq!(out.len(), 1 * 4 * 4);
    }

    #[test]
    fn test_decoder_block_stride8() {
        // [B,8,1] → [B,4,8]   (1*8 = 8)
        let block = DecoderBlock::new(8, 4, 8);
        let input = vec![0.0f32; 1 * 8 * 1];
        let (out, t_out) = block.forward(&input, 1, 1);
        assert_eq!(t_out, 8);
        assert_eq!(out.len(), 1 * 4 * 8);
    }

    #[test]
    fn test_decoder_block_all_strides() {
        // Mimi decoder chain: strides [4, 5, 6, 8] from T_lat=1.
        // 1 → 4 → 20 → 120 → 960
        let configs = [(1024usize, 512usize, 4usize), (512, 256, 5), (256, 128, 6), (128, 64, 8)];
        let mut t = 1usize;
        let mut h = vec![0.0f32; 1024 * t];

        for (in_ch, out_ch, stride) in configs {
            let block = DecoderBlock::new(in_ch, out_ch, stride);
            let (out, t_out) = block.forward(&h, 1, t);
            assert_eq!(t_out, t * stride, "stride={}", stride);
            assert_eq!(out.len(), out_ch * t_out);
            h = out;
            t = t_out;
        }
        assert_eq!(t, 960);
    }

    #[test]
    fn test_decoder_block_output_finite() {
        let block = DecoderBlock::new(4, 2, 4);
        let input: Vec<f32> = (0..4).map(|i| (i as f32) * 0.01).collect();
        let (out, _) = block.forward(&input, 1, 1);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
