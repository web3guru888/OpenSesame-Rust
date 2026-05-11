//! SEANet encoder: raw audio → latent representation at 25 fps.
//!
//! Mimi exact config (verified from moshi-core Rust source):
//! - Channel progression: 1 → 64 → 128 → 256 → 512 → 1024 → 512
//! - Strides: [8, 6, 5, 4] (first block has stride=8, last has stride=4)
//! - Initial conv: 1→64, k=7
//! - Final conv: 1024→512, k=3 (last_kernel_size=3)
//! - ELU before final conv
//! - Hop length: 8×6×5×4 = 960 → 25 fps at 24 kHz

use crate::causal_conv::{elu, CausalConv1d};
use crate::encoder_block::EncoderBlock;

/// Full SEANet encoder mapping raw waveform to latent frames.
///
/// Input:  [B, 1, T]          (mono 24 kHz audio)
/// Output: [B, 512, T/960]    (≈25 frames/sec for T=24000)
pub struct SEANetEncoder {
    /// Initial projection: 1 → 64 channels, kernel=7.
    pub input_conv: CausalConv1d,
    /// Four downsampling blocks with strides [8, 6, 5, 4].
    pub blocks: Vec<EncoderBlock>,
    /// Final projection: 1024 → 512, kernel=3 (last_kernel_size).
    pub output_conv: CausalConv1d,
}

impl SEANetEncoder {
    /// Construct the encoder with Mimi default configuration.
    ///
    /// Strides [8, 6, 5, 4] verified from moshi-core Rust source.
    /// Channel progression: 1 → 64 → 128 → 256 → 512 → 1024 → 512.
    pub fn new() -> Self {
        let n_filters = 64usize;
        // Strides verified from Kyutai's moshi-core seanet.rs
        let strides = [8usize, 6, 5, 4];

        // initial conv: 1 → 64, kernel=7
        let input_conv = CausalConv1d::new(1, n_filters, 7, 1, 1);

        // Build blocks: channels double at each stage (no cap at 512)
        let mut blocks = Vec::with_capacity(strides.len());
        let mut in_ch = n_filters;
        for &stride in &strides {
            let out_ch = in_ch * 2;
            blocks.push(EncoderBlock::new(in_ch, out_ch, stride));
            in_ch = out_ch;
        }
        // After 4 blocks: in_ch = 64 * 2^4 = 1024

        // Final conv: 1024 → 512, kernel=3 (last_kernel_size=3)
        let output_conv = CausalConv1d::new(in_ch, 512, 3, 1, 1);

        SEANetEncoder { input_conv, blocks, output_conv }
    }

    /// Forward pass.
    ///
    /// `audio` is flat [B, 1, T]. Returns (latent, t_out) where latent is [B, 512, T/960].
    pub fn forward(&self, audio: &[f32], batch: usize, t: usize) -> (Vec<f32>, usize) {
        // Initial conv: [B, 1, T] → [B, 64, T]
        let (mut h, mut t_cur) = self.input_conv.forward(audio, batch, 1, t);
        let mut ch = 64usize;

        // Downsampling blocks.
        for block in &self.blocks {
            let (out, t_out) = block.forward(&h, batch, t_cur);
            h = out;
            t_cur = t_out;
            ch = block.out_ch;
        }
        // ch = 1024 after all blocks

        // ELU before final conv.
        h.iter_mut().for_each(|v| *v = elu(*v));

        // Final conv: [B, 1024, T/960] → [B, 512, T/960]
        let (out, t_out) = self.output_conv.forward(&h, batch, ch, t_cur);
        (out, t_out)
    }
}

impl Default for SEANetEncoder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use T=960 (1 frame) or T=1920 (2 frames) for speed.
    // The full architecture is covered — only the time dimension is small.

    #[test]
    fn test_encoder_full_1s() {
        // T=960 → 1 frame: [B=1, 1, 960] → [B=1, 512, 1]
        let enc = SEANetEncoder::new();
        let input = vec![0.0f32; 960];
        let (out, t_out) = enc.forward(&input, 1, 960);
        assert_eq!(t_out, 1, "960 samples = 1 frame");
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_encoder_full_2s() {
        // T=1920 → 2 frames.
        let enc = SEANetEncoder::new();
        let input = vec![0.0f32; 1920];
        let (out, t_out) = enc.forward(&input, 1, 1920);
        assert_eq!(t_out, 2);
        assert_eq!(out.len(), 512 * 2);
    }

    #[test]
    fn test_encoder_batch2() {
        // B=2, T=960: two clips, 1 frame each.
        let enc = SEANetEncoder::new();
        let input = vec![0.1f32; 2 * 1 * 960];
        let (out, t_out) = enc.forward(&input, 2, 960);
        assert_eq!(t_out, 1);
        assert_eq!(out.len(), 2 * 512);
    }

    #[test]
    fn test_encoder_output_range() {
        // All outputs must be finite.
        let enc = SEANetEncoder::new();
        let input: Vec<f32> = (0..960).map(|i| (i as f32 * 0.001).sin()).collect();
        let (out, _) = enc.forward(&input, 1, 960);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite in encoder output");
    }

    #[test]
    fn test_encoder_deterministic() {
        // Same input → same output.
        let enc = SEANetEncoder::new();
        let input: Vec<f32> = (0..960).map(|i| (i as f32).sin() * 0.1).collect();
        let (out1, _) = enc.forward(&input, 1, 960);
        let (out2, _) = enc.forward(&input, 1, 960);
        assert_eq!(out1, out2, "encoder not deterministic");
    }

    #[test]
    fn test_encoder_short_input() {
        // T=960 (min 1 frame): should not crash.
        let enc = SEANetEncoder::new();
        let input = vec![0.0f32; 960];
        let (out, t_out) = enc.forward(&input, 1, 960);
        assert_eq!(t_out, 1);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_encoder_mimi_channel_progression() {
        // Verify the 1024-channel bottleneck.
        let enc = SEANetEncoder::new();
        // blocks: 64→128→256→512→1024
        assert_eq!(enc.blocks[0].out_ch, 128); // stride=8
        assert_eq!(enc.blocks[1].out_ch, 256); // stride=6
        assert_eq!(enc.blocks[2].out_ch, 512); // stride=5
        assert_eq!(enc.blocks[3].out_ch, 1024); // stride=4 → bottleneck!
        // Final conv: 1024→512, k=3
        assert_eq!(enc.output_conv.weight_norm.in_ch, 1024);
        assert_eq!(enc.output_conv.weight_norm.out_ch, 512);
        assert_eq!(enc.output_conv.weight_norm.kernel, 3);
    }

    #[test]
    fn test_encoder_mimi_strides() {
        // Strides are [8, 6, 5, 4] — not reversed.
        let enc = SEANetEncoder::new();
        let expected = [8usize, 6, 5, 4];
        for (i, block) in enc.blocks.iter().enumerate() {
            assert_eq!(block.stride, expected[i], "block {} stride", i);
        }
    }

    #[test]
    fn test_encoder_bottleneck_1024() {
        // Verify block[3] (stride=4) outputs 1024 channels.
        let enc = SEANetEncoder::new();
        assert_eq!(enc.blocks[3].in_ch, 512);
        assert_eq!(enc.blocks[3].out_ch, 1024);
        assert_eq!(enc.blocks[3].stride, 4);
    }
}
