//! Mimi codec configuration.

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
    /// CSM-1B uses 32; historic Moshi default is 8.
    pub num_codebooks: usize,
    /// Total number of trained codebook slots. Default 32.
    pub max_codebooks: usize,
    /// Vocabulary size per codebook. Default 2048.
    pub codebook_size: usize,
    /// In-codec transformer hidden dimension. Default 512.
    pub transformer_dim: usize,
    /// Number of attention heads in the in-codec transformer. Default 8.
    pub transformer_heads: usize,
    /// Number of transformer layers. Default 8.
    pub transformer_layers: usize,
    /// Causal local-attention window size in frames. Default 250.
    pub transformer_context: usize,
    /// Feed-forward expansion dimension. Default 2048.
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
    /// Small epsilon for LayerNorm. Default 1e-5.
    pub norm_eps: f32,
}

impl Default for MimiConfig {
    fn default() -> Self {
        Self {
            sample_rate:          24_000,
            channels:             1,
            encoder_dim:          512,
            quant_dim:            256,
            num_codebooks:        32,
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
    pub fn set_num_codebooks(&mut self, n: usize) {
        assert!(n >= 1, "set_num_codebooks: n must be ≥ 1");
        assert!(n <= self.max_codebooks,
            "set_num_codebooks: n={n} exceeds max_codebooks={}", self.max_codebooks);
        self.num_codebooks = n;
    }

    /// SEANet hop length: strides [8,6,5,4] → 960 samples/frame at 25 fps.
    pub fn seanet_hop(&self) -> usize { 960 }

    /// Total codec hop length: seanet_hop × downsample_stride = 1920 samples.
    pub fn hop_length(&self) -> usize { self.seanet_hop() * self.downsample_stride }

    /// Output frame rate: 24000 / 1920 = 12.5 fps.
    pub fn frame_rate(&self) -> f32 { self.sample_rate as f32 / self.hop_length() as f32 }

    /// Legacy 8-codebook config (original Moshi defaults).
    pub fn moshi_default() -> Self {
        let mut cfg = Self::default();
        cfg.num_codebooks = 8;
        cfg.max_codebooks = 8;
        cfg
    }

    /// Alias for [`moshi_default`]: 8-codebook Kyutai Moshi v0.1 config.
    ///
    /// Convenient alias used throughout the codebase and tests.
    pub fn v0_1() -> Self {
        Self::moshi_default()
    }

    /// 32-codebook config for CSM-1B (`SesameAILabs/csm`).
    ///
    /// Matches `mimi.set_num_codebooks(32)` in generator.py.
    pub fn csm_32() -> Self {
        Self::default() // default already has num_codebooks=32
    }

    /// Number of audio samples per Mimi code frame (1920 at 24 kHz / 12.5 fps).
    ///
    /// Alias for [`hop_length`].
    pub fn samples_per_frame(&self) -> usize {
        self.hop_length()
    }
}
