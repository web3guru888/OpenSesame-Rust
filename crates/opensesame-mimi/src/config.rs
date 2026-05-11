//! Mimi codec configuration.
//!
//! [`MimiConfig`] holds all hyperparameters for the full Mimi audio codec
//! (SEANet + in-codec transformer + ConvDownsample + SplitRVQ + ConvTrUpsample
//! + in-codec decoder transformer + SEANetDecoder).
//!
//! The default values match the Kyutai Mimi codec used in CSM-1B
//! (`mimi.set_num_codebooks(32)`). The historic 8-codebook mode is obtained
//! by calling `config.set_num_codebooks(8)`.

// ─── MimiConfig ──────────────────────────────────────────────────────────────

/// Full configuration for the Mimi audio codec.
///
/// Default values: 24 kHz input, 32 codebooks (CSM-1B), 2048 vocab,
/// 512-dim SEANet latent, 8-layer / 8-head in-codec transformer.
#[derive(Debug, Clone)]
pub struct MimiConfig {
    /// Audio sample rate in Hz. Default 24 000.
    pub sample_rate: u32,

    /// Number of audio channels. Default 1 (mono).
    pub channels: usize,

    /// SEANet encoder/decoder output dimension. Default 512.
    pub encoder_dim: usize,

    /// Inner VQ projection dimension (codebook vectors). Default 256.
    pub quant_dim: usize,

    /// Number of RVQ codebooks active during encode/decode.
    ///
    /// Mirrors Python `mimi.set_num_codebooks(n)`.
    /// CSM-1B uses 32; historic Moshi default is 8.
    /// Must be ≤ `max_codebooks`.
    pub num_codebooks: usize,

    /// Total number of trained codebook slots. Default 32.
    pub max_codebooks: usize,

    /// Vocabulary size per codebook (number of centroids K). Default 2048.
    pub codebook_size: usize,

    /// In-codec transformer hidden dimension. Default 512.
    pub transformer_dim: usize,

    /// Number of attention heads in the in-codec transformer. Default 8.
    pub transformer_heads: usize,

    /// Number of transformer layers. Default 8.
    pub transformer_layers: usize,

    /// Causal local-attention window size in frames. Default 250.
    ///
    /// Tokens beyond `transformer_context` positions back are masked out.
    pub transformer_context: usize,

    /// Feed-forward expansion dimension (4 × transformer_dim). Default 2048.
    pub ffn_dim: usize,

    /// CaiT-style initial layer-scale value. Default 0.01.
    pub layer_scale_init: f32,

    /// Depthwise-conv kernel size inside each transformer layer. Default 5.
    pub conv_kernel_size: usize,

    /// ConvDownsample / ConvTrUpsample stride (25 fps → 12.5 fps). Default 2.
    pub downsample_stride: usize,

    /// Downsample / upsample kernel size (= 2 × stride). Default 4.
    pub downsample_kernel: usize,

    /// Apply RMS normalisation to input PCM before encoding. Default true.
    pub renormalize: bool,

    /// RoPE base period for the in-codec transformer. Default 10 000.
    pub rope_base: f32,

    /// Small epsilon for LayerNorm in the in-codec transformer. Default 1e-5.
    pub norm_eps: f32,
}

impl Default for MimiConfig {
    /// Returns the default Mimi config matching CSM-1B usage:
    /// 32 codebooks, 24 kHz, 12.5 fps output, 512-dim latent.
    fn default() -> Self {
        Self {
            sample_rate:          24_000,
            channels:             1,
            encoder_dim:          512,
            quant_dim:            256,
            num_codebooks:        32,   // CSM-1B: mimi.set_num_codebooks(32)
            max_codebooks:        32,
            codebook_size:        2048,
            transformer_dim:      512,
            transformer_heads:    8,
            transformer_layers:   8,
            transformer_context:  250,
            ffn_dim:              2048,
            layer_scale_init:     0.01,
            conv_kernel_size:     5,
            downsample_stride:    2,
            downsample_kernel:    4,
            renormalize:          true,
            rope_base:            10_000.0,
            norm_eps:             1e-5,
        }
    }
}

impl MimiConfig {
    /// Set the number of active codebooks (1 ≤ n ≤ max_codebooks).
    ///
    /// Mirrors Python `mimi.set_num_codebooks(n)`. Call before encoding/decoding
    /// to select how many RVQ levels are used.
    ///
    /// # Panics
    /// Panics if `n == 0` or `n > max_codebooks`.
    pub fn set_num_codebooks(&mut self, n: usize) {
        assert!(n >= 1, "set_num_codebooks: n must be ≥ 1");
        assert!(
            n <= self.max_codebooks,
            "set_num_codebooks: n={n} exceeds max_codebooks={}",
            self.max_codebooks
        );
        self.num_codebooks = n;
    }

    /// SEANet hop length in PCM samples per encoder frame.
    ///
    /// Product of strides [8, 6, 5, 4] = 960 → 25 fps at 24 kHz.
    pub fn seanet_hop(&self) -> usize {
        960
    }

    /// Total codec hop length: SEANet hop × downsample stride.
    ///
    /// Default 960 × 2 = 1920 samples/frame → 12.5 fps at 24 kHz.
    pub fn hop_length(&self) -> usize {
        self.seanet_hop() * self.downsample_stride
    }

    /// Output frame rate in frames per second.
    ///
    /// Default 24000 / 1920 = 12.5 fps.
    pub fn frame_rate(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length() as f32
    }

    /// Return a legacy 8-codebook config (original Moshi/Mimi defaults).
    pub fn moshi_default() -> Self {
        let mut cfg = Self::default();
        cfg.num_codebooks  = 8;
        cfg.max_codebooks  = 8;
        cfg
    }
}
