//! Full Mimi audio codec: SEANet + transformer + SplitRVQ.
//!
//! Data format conventions:
//! - SEANet encoder output: `[B, C, T]` channel-major (batch × channels × time)
//! - Transformer / Projection: `[T, C]` time-major
//! - SEANet decoder input:  `[B, C, T]` channel-major
//!
//! Transpositions are applied at the SEANet boundary.

use crate::config::MimiConfig;
use crate::conv::{ConvDownsample1d, ConvTrUpsample1d};
use crate::transformer::MimiTransformer;
use opensesame_rvq::{RVQConfig, SplitRVQ};
use opensesame_seanet::{SEANetEncoder, SEANetDecoder};

// ── Format helpers ────────────────────────────────────────────────────────────

/// Transpose `[B, C, T]` → `[T, C]` (batch=1 assumed).
///
/// SEANet returns channel-major; transformer / projection use time-major.
fn bct_to_tc(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * c];
    for ti in 0..t {
        for ci in 0..c {
            out[ti * c + ci] = x[ci * t + ti];
        }
    }
    out
}

/// Transpose `[T, C]` → `[B, C, T]` (batch=1 assumed).
///
/// Transformer output is time-major; SEANet decoder expects channel-major.
fn tc_to_bct(x: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; c * t];
    for ti in 0..t {
        for ci in 0..c {
            out[ci * t + ti] = x[ti * c + ci];
        }
    }
    out
}

// ── Linear projection ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Projection {
    weight: Vec<f32>,
    in_d:   usize,
    out_d:  usize,
}

impl Projection {
    fn new(in_d: usize, out_d: usize) -> Self {
        Self { weight: vec![0.0_f32; out_d * in_d], in_d, out_d }
    }
    /// Apply to `x` of shape `[T, in_d]`, returns `[T, out_d]`.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0.0_f32; t * self.out_d];
        for ti in 0..t {
            let xi = &x[ti * self.in_d..(ti + 1) * self.in_d];
            for o in 0..self.out_d {
                let w = &self.weight[o * self.in_d..(o + 1) * self.in_d];
                y[ti * self.out_d + o] = w.iter().zip(xi.iter()).map(|(a, b)| a * b).sum();
            }
        }
        y
    }
}

// ── RMS normalisation ─────────────────────────────────────────────────────────

fn rms_normalize(x: &[f32], t: usize) -> Vec<f32> {
    if t == 0 { return x.to_vec(); }
    let rms = (x[..t].iter().map(|v| v * v).sum::<f32>() / t as f32).sqrt();
    if rms < 1e-7 { return x.to_vec(); }
    x[..t].iter().map(|v| v / rms).collect()
}

// ── Mimi ─────────────────────────────────────────────────────────────────────

/// Full Mimi audio codec.
///
/// Combines SEANet, in-codec transformers, ConvDownsample/Upsample, and
/// SplitRVQ. The number of active codebooks can be changed at runtime via
/// [`Mimi::set_num_codebooks`], mirroring Python `mimi.set_num_codebooks(32)`.
///
/// # Important
/// `config.transformer_dim` must equal `config.encoder_dim` (both 512 in the
/// production Mimi model). The in-codec transformer operates on the same
/// feature space as the SEANet latent.
pub struct Mimi {
    /// Full codec configuration.
    pub config:              MimiConfig,
    /// SEANet encoder (1→512 at 25 fps, returns `[B, C, T]`).
    pub encoder:             SEANetEncoder,
    /// SEANet decoder (512→1 at 25 fps, expects `[B, C, T]`).
    pub decoder:             SEANetDecoder,
    /// In-codec transformer applied after SEANet encoding (operates on `[T, C]`).
    pub encoder_transformer: MimiTransformer,
    /// In-codec transformer applied before SEANet decoding (operates on `[T, C]`).
    pub decoder_transformer: MimiTransformer,
    /// Temporal downsampler: 25 fps → 12.5 fps (operates on `[T, C]`).
    pub downsample:          ConvDownsample1d,
    /// Temporal upsampler: 12.5 fps → 25 fps (operates on `[T, C]`).
    pub upsample:            ConvTrUpsample1d,
    /// Split RVQ quantizer (CB0 semantic + CB1..N-1 acoustic).
    pub quantizer:           SplitRVQ,
    /// Linear 512→256 (encoder_dim → quant_dim).
    proj_down: Projection,
    /// Linear 256→512 (quant_dim → encoder_dim).
    proj_up:   Projection,
}

impl Mimi {
    /// Construct a new [`Mimi`] codec with zero-initialised weights.
    ///
    /// For production use, load weights from a safetensors checkpoint
    /// via [`crate::loader::load_mimi`].
    pub fn new(cfg: MimiConfig) -> Self {
        let rvq_cfg = RVQConfig {
            num_codebooks:       cfg.max_codebooks,
            codebook_size:       cfg.codebook_size,
            quant_dim:           cfg.quant_dim,
            commitment_cost:     0.25,
            ema_decay:           0.99,
            ema_epsilon:         1e-5,
            dead_code_threshold: 1.0,
            kmeans_init:         true,
        };
        let mut quantizer = SplitRVQ::new(rvq_cfg, 1);
        quantizer.set_num_codebooks(cfg.num_codebooks);

        Self {
            encoder:             SEANetEncoder::new(),
            decoder:             SEANetDecoder::new(),
            encoder_transformer: MimiTransformer::new(&cfg),
            decoder_transformer: MimiTransformer::new(&cfg),
            downsample:          ConvDownsample1d::new(cfg.encoder_dim, cfg.downsample_stride),
            upsample:            ConvTrUpsample1d::new(cfg.encoder_dim, cfg.downsample_stride),
            proj_down:           Projection::new(cfg.encoder_dim, cfg.quant_dim),
            proj_up:             Projection::new(cfg.quant_dim, cfg.encoder_dim),
            quantizer,
            config: cfg,
        }
    }

    /// Set the number of active RVQ codebooks (mirrors Python `set_num_codebooks`).
    pub fn set_num_codebooks(&mut self, n: usize) {
        self.config.set_num_codebooks(n);
        self.quantizer.set_num_codebooks(n);
    }

    /// Return the currently active number of codebooks.
    pub fn num_codebooks(&self) -> usize { self.quantizer.num_codebooks() }

    /// Encode PCM audio to RVQ code indices.
    ///
    /// # Data flow
    /// ```text
    /// pcm [T]
    ///   → (RMS norm)
    ///   → SEANetEncoder → [1, 512, T_s] channel-major  (T_s = T / 960)
    ///   → bct_to_tc     → [T_s, 512] time-major
    ///   → encoder_transformer → [T_s, 512]
    ///   → proj_down 512→256   → [T_s, 256]
    ///   → ConvDownsample       → [T_q, 256]  (T_q = T_s / 2)
    ///   → SplitRVQ.encode      → [K, T_q]
    /// ```
    pub fn encode(&self, pcm: &[f32], t: usize) -> Vec<Vec<u32>> {
        let pcm_norm = if self.config.renormalize { rms_normalize(pcm, t) } else { pcm.to_vec() };

        // SEANet encoder: returns [1, 512, T_s] channel-major
        let (enc_bct, t_s) = self.encoder.forward(&pcm_norm, 1, t);
        let enc_dim = self.config.encoder_dim; // 512

        // Transpose to [T_s, 512] time-major
        let enc_tc = bct_to_tc(&enc_bct, enc_dim, t_s);

        // In-codec transformer: [T_s, 512] → [T_s, 512]
        let trm_out = self.encoder_transformer.forward(&enc_tc, t_s);

        // ConvDownsample FIRST (on 512-dim data): [T_s, 512] → [T_q, 512]
        let (ds_out, t_q) = self.downsample.forward(&trm_out, t_s);

        // THEN project 512 → 256: [T_q, 256]
        let proj_out = self.proj_down.forward(&ds_out, t_q);

        // SplitRVQ encode
        self.quantizer.encode(&proj_out, t_q, self.config.quant_dim)
    }

    /// Decode RVQ code indices back to PCM.
    ///
    /// # Data flow
    /// ```text
    /// codes [K][T_q]
    ///   → SplitRVQ.decode   → [T_q, 256] time-major
    ///   → proj_up 256→512   → [T_q, 512]
    ///   → ConvTrUpsample    → [T_s, 512]  (T_s = T_q * 2)
    ///   → decoder_transformer → [T_s, 512]
    ///   → tc_to_bct          → [1, 512, T_s] channel-major
    ///   → SEANetDecoder      → pcm [T]
    /// ```
    pub fn decode(&self, codes: &[Vec<u32>]) -> (Vec<f32>, usize) {
        if codes.is_empty() || codes[0].is_empty() { return (vec![], 0); }
        let t_q = codes[0].len();
        let enc_dim = self.config.encoder_dim; // 512

        // SplitRVQ decode → [T_q, 256]
        let quant_out = self.quantizer.decode(codes);

        // Project 256 → 512 (after RVQ, before upsample)
        let proj_out = self.proj_up.forward(&quant_out, t_q);

        // ConvTrUpsample: [T_q, 512] → [T_s, 512]  (upsample at full dim)
        let (us_out, t_s) = self.upsample.forward(&proj_out, t_q);

        // In-codec transformer → [T_s, 512]
        let trm_out = self.decoder_transformer.forward(&us_out, t_s);

        // Transpose to [1, 512, T_s] channel-major for SEANet decoder
        let dec_bct = tc_to_bct(&trm_out, t_s, enc_dim);

        // SEANet decoder
        self.decoder.forward(&dec_bct, 1, t_s)
    }

    /// Encode then decode (codec round-trip).
    pub fn round_trip(&self, pcm: &[f32], t: usize) -> (Vec<f32>, usize) {
        self.decode(&self.encode(pcm, t))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Small config: 1 transformer layer, real 512-dim (matches SEANet output).
    fn small_cfg() -> MimiConfig {
        MimiConfig {
            // transformer_dim MUST equal encoder_dim (512) — no separate projection
            transformer_dim:     512,
            transformer_heads:   8,
            transformer_layers:  1,
            transformer_context: 4,
            ffn_dim:             2048,
            conv_kernel_size:    5,
            layer_scale_init:    0.01,
            rope_base:           10_000.0,
            norm_eps:            1e-5,
            num_codebooks:       8,
            max_codebooks:       8,
            ..MimiConfig::default()
        }
    }

    #[test]
    fn test_bct_to_tc_basic() {
        // [1, 2, 3] → [3, 2]: x[c*T+t] → y[t*C+c]
        let x = vec![1.0, 2.0, 3.0,   4.0, 5.0, 6.0]; // C=2, T=3
        let y = bct_to_tc(&x, 2, 3);
        // t=0: [x[0*3+0], x[1*3+0]] = [1, 4]
        // t=1: [x[0*3+1], x[1*3+1]] = [2, 5]
        // t=2: [x[0*3+2], x[1*3+2]] = [3, 6]
        assert!((y[0] - 1.0).abs() < 1e-5);
        assert!((y[1] - 4.0).abs() < 1e-5);
        assert!((y[2] - 2.0).abs() < 1e-5);
        assert!((y[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_tc_to_bct_roundtrip() {
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect(); // [3, 2] time-major
        let bct = tc_to_bct(&x, 3, 2);
        let back = bct_to_tc(&bct, 2, 3);
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mimi_construction() {
        let mimi = Mimi::new(small_cfg());
        assert_eq!(mimi.num_codebooks(), 8);
    }

    #[test]
    fn test_set_num_codebooks_32() {
        let mut mimi = Mimi::new(MimiConfig::default());
        assert_eq!(mimi.num_codebooks(), 32);
        mimi.set_num_codebooks(8);
        assert_eq!(mimi.num_codebooks(), 8);
        mimi.set_num_codebooks(32);
        assert_eq!(mimi.num_codebooks(), 32);
    }

    #[test]
    fn test_encode_shape_1frame() {
        // 1920 samples = 1920/960=2 SEANet frames → 1 codec frame (T_q=1)
        let mimi = Mimi::new(small_cfg());
        let codes = mimi.encode(&vec![0.1_f32; 1920], 1920);
        assert_eq!(codes.len(), 8, "8 codebooks");
        assert_eq!(codes[0].len(), 1, "1 codec frame from 1920 samples");
    }

    #[test]
    fn test_encode_shape_2frames() {
        let mimi = Mimi::new(small_cfg());
        let codes = mimi.encode(&vec![0.05_f32; 3840], 3840);
        assert_eq!(codes.len(), 8);
        assert_eq!(codes[0].len(), 2);
    }

    #[test]
    fn test_decode_nonempty() {
        let mimi = Mimi::new(small_cfg());
        let codes: Vec<Vec<u32>> = (0..8).map(|_| vec![0u32; 1]).collect();
        let (pcm, _) = mimi.decode(&codes);
        assert!(!pcm.is_empty());
    }

    #[test]
    fn test_round_trip_shape() {
        let mimi = Mimi::new(small_cfg());
        let (out, _) = mimi.round_trip(&vec![0.1_f32; 1920], 1920);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_codes_in_range() {
        let mimi = Mimi::new(small_cfg());
        let pcm: Vec<f32> = (0..1920).map(|i| (i as f32 * 0.001).sin()).collect();
        let codes = mimi.encode(&pcm, 1920);
        for (cb, row) in codes.iter().enumerate() {
            for &code in row {
                assert!((code as usize) < mimi.config.codebook_size,
                    "cb{cb}: code {code} ≥ vocab {}", mimi.config.codebook_size);
            }
        }
    }

    #[test]
    fn test_rms_normalize_unit_rms() {
        let x = vec![1.0_f32; 100];
        let y = rms_normalize(&x, 100);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 100.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_normalize_silent() {
        let x = vec![0.0_f32; 50];
        let y = rms_normalize(&x, 50);
        assert!(y.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_projection_shape() {
        let p = Projection::new(512, 256);
        assert_eq!(p.forward(&vec![0.5_f32; 4 * 512], 4).len(), 4 * 256);
    
    // ── CSM-1B integration tests (32 codebooks) ──────────────────────────────

    #[test]
    fn test_csm_32cb_encode_1frame() {
        // CSM-1B uses 32 codebooks: mimi.set_num_codebooks(32)
        let mut mimi = Mimi::new(MimiConfig {
            transformer_layers: 1,
            transformer_context: 16,
            num_codebooks: 32,
            max_codebooks: 32,
            ..MimiConfig::default()
        });
        mimi.set_num_codebooks(32);
        let codes = mimi.encode(&vec![0.1_f32; 1920], 1920);
        assert_eq!(codes.len(), 32, "CSM-1B: 32 codebooks");
        assert_eq!(codes[0].len(), 1, "1 code frame");
    }

    #[test]
    fn test_csm_32cb_encode_decode_roundtrip() {
        // Encode 2 frames with 32 codebooks, decode, check shape
        let mimi = Mimi::new(MimiConfig {
            transformer_layers: 1,
            transformer_context: 16,
            num_codebooks: 32,
            max_codebooks: 32,
            ..MimiConfig::default()
        });
        let audio = vec![0.05_f32; 3840]; // 2 frames
        let codes = mimi.encode(&audio, 3840);
        assert_eq!(codes.len(), 32);
        assert_eq!(codes[0].len(), 2);
        let (decoded, _) = mimi.decode(&codes);
        assert!(!decoded.is_empty(), "decoded output must be non-empty");
        // Decoded length: 2 frames × 1920 samples/frame = 3840 samples
        assert_eq!(decoded.len(), 3840, "decoded length = 2 frames = 3840 samples");
    }

    #[test]
    fn test_csm_32cb_codes_in_range() {
        // All 32-codebook tokens must be < vocab_size (2048)
        let mimi = Mimi::new(MimiConfig {
            transformer_layers: 1,
            transformer_context: 16,
            num_codebooks: 32,
            max_codebooks: 32,
            ..MimiConfig::default()
        });
        let codes = mimi.encode(&vec![0.0_f32; 1920], 1920);
        for (cb, row) in codes.iter().enumerate() {
            for &token in row {
                assert!(
                    (token as usize) < 2048,
                    "codebook {cb} token {token} >= vocab_size 2048"
                );
            }
        }
    }

    #[test]
    fn test_set_num_codebooks_then_encode_32() {
        // Start with 8-codebook config, switch to 32 via set_num_codebooks
        let mut mimi = Mimi::new(MimiConfig {
            transformer_layers: 1,
            transformer_context: 16,
            num_codebooks: 8,
            max_codebooks: 32,  // allow up to 32
            ..MimiConfig::default()
        });
        assert_eq!(mimi.num_codebooks(), 8);
        // Matches: mimi.set_num_codebooks(32) in generator.py
        mimi.set_num_codebooks(32);
        assert_eq!(mimi.num_codebooks(), 32);
        let codes = mimi.encode(&vec![0.0_f32; 1920], 1920);
        assert_eq!(codes.len(), 32, "32 codebooks after set_num_codebooks");
    }

    #[test]
    fn test_decode_output_finite() {
        let mimi = Mimi::new(small_cfg());
        let codes: Vec<Vec<u32>> = (0..8).map(|_| vec![42u32, 100u32]).collect();
        let (pcm, _) = mimi.decode(&codes);
        for (i, &v) in pcm.iter().enumerate() {
            assert!(v.is_finite(), "decoded[{i}] = {v} is not finite");
        }
    }

}
}
