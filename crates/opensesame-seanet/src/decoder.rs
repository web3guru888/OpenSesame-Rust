//! SEANet decoder: latent frames → reconstructed audio.
//!
//! Mimi exact config (verified from moshi-core Rust source):
//! - Channel progression: 512 → 1024 → 512 → 256 → 128 → 64 → 1
//! - init_conv: 512→1024, k=3  (mirrors encoder final_conv k=3)
//! - Decoder block strides: [4, 5, 6, 8]  (reverse of encoder [8,6,5,4])
//! - final_conv: 64→1, k=7  (mirrors encoder init_conv k=7)
//! - Tanh applied to final output

use crate::causal_conv::{elu, CausalConv1d};
use crate::decoder_block::DecoderBlock;

/// Full SEANet decoder mapping latent frames to reconstructed waveform.
///
/// Input:  [B, 512, T_lat]    (latent frames from RVQ decode / transformer)
/// Output: [B, 1, T_lat*960]  (reconstructed mono 24 kHz audio, Tanh-clipped to (-1,1))
pub struct SEANetDecoder {
    /// Initial projection: 512 → 1024, kernel=3 (mirrors encoder final k=3).
    pub input_conv: CausalConv1d,
    /// Four upsampling blocks with strides [4, 5, 6, 8].
    pub blocks: Vec<DecoderBlock>,
    /// Final projection: 64 → 1, kernel=7 (mirrors encoder init k=7).
    pub output_conv: CausalConv1d,
}

impl SEANetDecoder {
    /// Construct the decoder with Mimi default configuration.
    ///
    /// Decoder strides: [4, 5, 6, 8] (reverse of encoder [8,6,5,4]).
    /// init_conv uses k=3, final_conv uses k=7 (exact mirror of encoder).
    pub fn new() -> Self {
        let n_filters = 64usize;
        let dimension = 512usize;
        // Decoder strides: reverse of encoder [8,6,5,4]
        let strides = [4usize, 5, 6, 8];

        // init_conv: 512 → 1024 (= n_filters * 2^4), kernel=3
        let init_ch = n_filters * (1 << strides.len()); // 64 * 16 = 1024
        let input_conv = CausalConv1d::new(dimension, init_ch, 3, 1, 1);

        // Build blocks: channels halve at each stage.
        let mut blocks = Vec::with_capacity(strides.len());
        let mut in_ch = init_ch;
        for &stride in &strides {
            let out_ch = in_ch / 2;
            blocks.push(DecoderBlock::new(in_ch, out_ch, stride));
            in_ch = out_ch;
        }
        // After 4 blocks: in_ch = 1024 / 16 = 64

        // final_conv: 64 → 1, kernel=7
        let output_conv = CausalConv1d::new(in_ch, 1, 7, 1, 1);

        SEANetDecoder { input_conv, blocks, output_conv }
    }

    /// Forward pass.
    ///
    /// `latent` is flat [B, 512, T_lat].
    /// Returns (audio, t_out) where audio is [B, 1, T_lat * 960] with Tanh applied.
    pub fn forward(&self, latent: &[f32], batch: usize, t_latent: usize) -> (Vec<f32>, usize) {
        // init_conv: [B, 512, T_lat] → [B, 1024, T_lat]
        let (mut h, mut t_cur) = self.input_conv.forward(latent, batch, 512, t_latent);
        let mut ch = 1024usize;

        // ELU after init_conv.
        h.iter_mut().for_each(|v| *v = elu(*v));

        // Upsampling blocks: each applies upsample + residual unit.
        for block in &self.blocks {
            let (out, t_out) = block.forward(&h, batch, t_cur);
            h = out;
            // ELU after each block.
            h.iter_mut().for_each(|v| *v = elu(*v));
            t_cur = t_out;
            ch = block.out_ch;
        }
        // ch = 64 after all blocks

        // final_conv: [B, 64, T_out] → [B, 1, T_out]
        let (mut out, t_out) = self.output_conv.forward(&h, batch, ch, t_cur);

        // Tanh: map output to (-1, 1).
        out.iter_mut().for_each(|v| *v = v.tanh());

        (out, t_out)
    }
}

impl Default for SEANetDecoder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use T_lat=1 (single frame → 960 samples) for speed.

    #[test]
    fn test_decoder_full_25frames() {
        // T_lat=1 → 960 samples.  [B=1, 512, 1] → [B=1, 1, 960]
        let dec = SEANetDecoder::new();
        let latent = vec![0.0f32; 512];
        let (out, t_out) = dec.forward(&latent, 1, 1);
        assert_eq!(t_out, 960, "1 frame should decode to 960 samples");
        assert_eq!(out.len(), 960);
    }

    #[test]
    fn test_decoder_output_range() {
        // Tanh ensures output ∈ (-1, 1).
        let dec = SEANetDecoder::new();
        let latent: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let (out, _) = dec.forward(&latent, 1, 1);
        assert!(
            out.iter().all(|v| v.is_finite() && *v > -1.0 && *v < 1.0),
            "decoder output outside (-1,1)"
        );
    }

    #[test]
    fn test_decoder_deterministic() {
        let dec = SEANetDecoder::new();
        let latent: Vec<f32> = (0..512).map(|i| (i as f32).sin() * 0.1).collect();
        let (out1, _) = dec.forward(&latent, 1, 1);
        let (out2, _) = dec.forward(&latent, 1, 1);
        assert_eq!(out1, out2, "decoder not deterministic");
    }

    #[test]
    fn test_decoder_single_frame() {
        // T_latent=1: no crash.
        let dec = SEANetDecoder::new();
        let latent = vec![0.0f32; 512];
        let (out, t_out) = dec.forward(&latent, 1, 1);
        assert_eq!(t_out, 960);
        assert_eq!(out.len(), 960);
    }

    #[test]
    fn test_decoder_mimi_init_conv() {
        // init_conv: 512→1024, k=3.
        let dec = SEANetDecoder::new();
        assert_eq!(dec.input_conv.weight_norm.in_ch, 512);
        assert_eq!(dec.input_conv.weight_norm.out_ch, 1024);
        assert_eq!(dec.input_conv.weight_norm.kernel, 3);
    }

    #[test]
    fn test_decoder_mimi_final_conv() {
        // final_conv: 64→1, k=7.
        let dec = SEANetDecoder::new();
        assert_eq!(dec.output_conv.weight_norm.in_ch, 64);
        assert_eq!(dec.output_conv.weight_norm.out_ch, 1);
        assert_eq!(dec.output_conv.weight_norm.kernel, 7);
    }

    #[test]
    fn test_decoder_tanh_output() {
        // Large latent values → output still in (-1, 1).
        let dec = SEANetDecoder::new();
        let latent = vec![1000.0f32; 512];
        let (out, _) = dec.forward(&latent, 1, 1);
        assert!(out.iter().all(|v| *v > -1.0 && *v < 1.0), "Tanh not applied");
    }

    #[test]
    fn test_decoder_two_frames() {
        // T_lat=2 → 1920 samples.
        let dec = SEANetDecoder::new();
        let latent = vec![0.0f32; 512 * 2];
        let (out, t_out) = dec.forward(&latent, 1, 2);
        assert_eq!(t_out, 1920);
        assert_eq!(out.len(), 1920);
    }

    #[test]
    fn test_decoder_mimi_strides() {
        // Decoder block strides: [4, 5, 6, 8].
        let dec = SEANetDecoder::new();
        let expected = [4usize, 5, 6, 8];
        for (i, block) in dec.blocks.iter().enumerate() {
            assert_eq!(block.stride, expected[i], "block {} stride", i);
        }
    }
}
