//! Mimi codec: full encode/decode pipeline.
//!
//! [`MimiCodec`] wraps [`SEANetEncoder`], [`SEANetDecoder`], a pair of
//! [`MimiTransformer`]s (encoder-side + decoder-side), and a [`SplitRVQ`].
//!
//! # Data flow
//!
//! ## Encode
//! ```text
//! audio [B, 1, T]
//!   ──► SEANetEncoder          → latent_cm [B, 512, T/960]  (channel-major)
//!   ──► transpose to row-major → latent_rm [B*T/960, 512]
//!   ──► enc_transformer        → latent_rm [B*T/960, 512]   (identity in Phase E)
//!   ──► enc_proj (512→256)     → proj [B*T/960, 256]
//!   ──► SplitRVQ.encode        → codes [8][B*T/960]
//! ```
//!
//! ## Decode
//! ```text
//! codes [8][N]  (N = B * T_frames)
//!   ──► SplitRVQ.decode       → quant [N, 256]
//!   ──► dec_proj (256→512)    → dec_rm [N, 512]
//!   ──► dec_transformer       → dec_rm [N, 512]  (identity in Phase E)
//!   ──► transpose to ch-major → latent_cm [B, 512, T_frames]
//!   ──► SEANetDecoder          → audio [B, 1, T_frames*960]
//! ```
//!
//! **Phase E note**: Both transformers are identity pass-throughs. The ConvDownsample
//! (25 fps → 12.5 fps) is not wired yet — the codec hop is 960 samples in Phase E.

use crate::config::MimiConfig;
use crate::loader::SafetensorsFile;
use crate::transformer::MimiTransformer;
use opensesame_rvq::{RVQConfig, SplitRVQ};
use opensesame_seanet::{SEANetDecoder, SEANetEncoder};

// ─── MimiCodec ───────────────────────────────────────────────────────────────

/// Full Mimi audio codec.
///
/// Encodes mono 24 kHz audio to discrete tokens and decodes tokens back to audio.
/// Wraps SEANet, two in-codec transformers, linear projections, and Split-RVQ.
pub struct MimiCodec {
    /// SEANet encoder: audio → latent (hop = 960).
    pub encoder: SEANetEncoder,
    /// In-codec encoder-side transformer (causal, 8L, 8H, d=512, LayerNorm+RoPE).
    pub enc_transformer: MimiTransformer,
    /// Split-RVQ: 1 semantic codebook + 7 acoustic codebooks, each with 2048 entries in 256d.
    pub quantizer: SplitRVQ,
    /// In-codec decoder-side transformer.
    pub dec_transformer: MimiTransformer,
    /// SEANet decoder: latent → audio.
    pub decoder: SEANetDecoder,
    /// Encoder projection weight: [encoder_dim=512, quant_dim=256] (row-major: [in, out]).
    pub enc_proj: Vec<f32>,
    /// Encoder projection bias: [quant_dim=256].
    pub enc_proj_bias: Vec<f32>,
    /// Decoder projection weight: [quant_dim=256, encoder_dim=512].
    pub dec_proj: Vec<f32>,
    /// Decoder projection bias: [encoder_dim=512].
    pub dec_proj_bias: Vec<f32>,
    /// Codec hyperparameters.
    pub config: MimiConfig,
}

impl MimiCodec {
    /// Construct a [`MimiCodec`] with zero-initialised weights.
    ///
    /// The resulting codec is valid for shape-testing: encoder/decoder use
    /// deterministic random weights (from their own `::new()` constructors), and
    /// the projections are zero-biased so the output is deterministic.
    pub fn new(config: MimiConfig) -> Self {
        let encoder = SEANetEncoder::new();
        let decoder = SEANetDecoder::new();
        let enc_transformer = MimiTransformer::new(&config);
        let dec_transformer = MimiTransformer::new(&config);

        let rvq_cfg = RVQConfig {
            num_codebooks: config.num_codebooks, // 8
            codebook_size: config.codebook_size, // 2048
            quant_dim: config.quant_dim,         // 256
            ..RVQConfig::default()
        };
        // 1 semantic + 7 acoustic = 8 total codebooks.
        let quantizer = SplitRVQ::new(rvq_cfg, 1);

        let enc_dim = config.encoder_dim; // 512
        let q_dim = config.quant_dim;     // 256

        Self {
            encoder,
            enc_transformer,
            quantizer,
            dec_transformer,
            decoder,
            enc_proj: vec![0.0_f32; enc_dim * q_dim],   // [512, 256]
            enc_proj_bias: vec![0.0_f32; q_dim],          // [256]
            dec_proj: vec![0.0_f32; q_dim * enc_dim],    // [256, 512]
            dec_proj_bias: vec![0.0_f32; enc_dim],        // [512]
            config,
        }
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /// General matrix-vector multiply: `[N, in_dim] × [in_dim, out_dim] + bias → [N, out_dim]`.
    ///
    /// `weight` is stored row-major as `[in_dim, out_dim]`:
    /// `weight[k * out_dim + j]` is the weight connecting input index k to output index j.
    fn linear(
        x: &[f32],
        n: usize,
        in_dim: usize,
        weight: &[f32],
        bias: &[f32],
        out_dim: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(weight.len(), in_dim * out_dim);
        debug_assert_eq!(bias.len(), out_dim);
        debug_assert_eq!(x.len(), n * in_dim);

        let mut out = vec![0.0_f32; n * out_dim];
        for i in 0..n {
            for j in 0..out_dim {
                let mut acc = bias[j];
                for k in 0..in_dim {
                    acc += x[i * in_dim + k] * weight[k * out_dim + j];
                }
                out[i * out_dim + j] = acc;
            }
        }
        out
    }

    /// Transpose channel-major `[B, C, T]` → row-major `[B*T, C]`.
    ///
    /// SEANet outputs channel-major: `data[b * C * T + c * T + t]`.
    /// SplitRVQ wants row-major: `data[(b*T + t) * C + c]`.
    fn chan_to_row(cm: &[f32], batch: usize, ch: usize, t: usize) -> Vec<f32> {
        let n = batch * t;
        let mut rm = vec![0.0_f32; n * ch];
        for b in 0..batch {
            for t_idx in 0..t {
                for c in 0..ch {
                    rm[(b * t + t_idx) * ch + c] = cm[b * ch * t + c * t + t_idx];
                }
            }
        }
        rm
    }

    /// Transpose row-major `[B*T, C]` → channel-major `[B, C, T]`.
    fn row_to_chan(rm: &[f32], batch: usize, ch: usize, t: usize) -> Vec<f32> {
        let mut cm = vec![0.0_f32; batch * ch * t];
        for b in 0..batch {
            for t_idx in 0..t {
                for c in 0..ch {
                    cm[b * ch * t + c * t + t_idx] = rm[(b * t + t_idx) * ch + c];
                }
            }
        }
        cm
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /// Encode audio to discrete codes.
    ///
    /// # Arguments
    /// * `audio` — flat `[B, 1, T]` mono PCM (24 kHz)
    /// * `batch` — batch size B
    /// * `t`     — number of audio samples T (must be a multiple of 960)
    ///
    /// # Returns
    /// `Vec<Vec<u32>>` of length `num_codebooks` (8); each inner Vec has length `B * (T/960)`.
    pub fn encode(&self, audio: &[f32], batch: usize, t: usize) -> Vec<Vec<u32>> {
        let enc_dim = self.config.encoder_dim; // 512
        let q_dim = self.config.quant_dim;     // 256

        // 1. SEANet encoder: [B, 1, T] → channel-major [B, 512, T/960]
        let (latent_cm, t_out) = self.encoder.forward(audio, batch, t);
        let n = batch * t_out; // total frame count

        // 2. Transpose to row-major [N, 512] for transformer + projection.
        let latent_rm = Self::chan_to_row(&latent_cm, batch, enc_dim, t_out);

        // 3. In-codec transformer (identity in Phase E): [N, 512] → [N, 512].
        let latent_rm = self.enc_transformer.forward(&latent_rm, batch, t_out);

        // 4. Project 512 → 256.
        let proj = Self::linear(&latent_rm, n, enc_dim, &self.enc_proj, &self.enc_proj_bias, q_dim);

        // 5. Quantize.
        self.quantizer.encode(&proj, n, q_dim)
    }

    /// Decode discrete codes to audio.
    ///
    /// # Arguments
    /// * `codes` — slice of `num_codebooks` code vectors; each has length `B * T_frames`
    /// * `batch` — batch size B (used to reshape codes into per-batch frames)
    ///
    /// # Returns
    /// Flat `[B, 1, T_frames * 960]` audio (24 kHz, Tanh-bounded to (−1, 1)).
    /// Returns an empty Vec if `codes` is empty or each code vector is empty.
    pub fn decode(&self, codes: &[Vec<u32>], batch: usize) -> Vec<f32> {
        if codes.is_empty() || codes[0].is_empty() {
            return Vec::new();
        }

        let enc_dim = self.config.encoder_dim; // 512
        let q_dim = self.config.quant_dim;     // 256

        let n = codes[0].len();           // B * T_frames
        let t_frames = n / batch;         // frames per batch item

        // 1. Dequantize: codes → [N, 256].
        let quant = self.quantizer.decode(codes);

        // 2. Project 256 → 512.
        let dec_rm =
            Self::linear(&quant, n, q_dim, &self.dec_proj, &self.dec_proj_bias, enc_dim);

        // 3. In-codec transformer (identity in Phase E).
        let dec_rm = self.dec_transformer.forward(&dec_rm, batch, t_frames);

        // 4. Transpose row-major [N, 512] → channel-major [B, 512, T_frames].
        let latent_cm = Self::row_to_chan(&dec_rm, batch, enc_dim, t_frames);

        // 5. SEANet decoder: [B, 512, T_frames] → [B, 1, T_frames * 960].
        let (audio, _t_audio) = self.decoder.forward(&latent_cm, batch, t_frames);
        audio
    }

    /// Encode then immediately decode (measure reconstruction quality).
    ///
    /// # Returns
    /// Audio of the same length as the input (if T is a multiple of 960).
    pub fn reconstruct(&self, audio: &[f32], batch: usize, t: usize) -> Vec<f32> {
        let codes = self.encode(audio, batch, t);
        self.decode(&codes, batch)
    }

    /// Load pretrained weights from a `.safetensors` file.
    ///
    /// Opens and parses the file; maps Kyutai weight names to our struct fields.
    /// Returns `Err` if the file cannot be read or parsed.
    ///
    /// **Phase E**: the loader is wired up but weight-population is deferred.
    /// After `from_pretrained`, the returned codec uses default (zero-projection)
    /// weights except for SEANet which retains its own random init.
    pub fn from_pretrained(path: &str) -> Result<Self, String> {
        // Open and validate the file; fail fast if path is bad.
        let _file = SafetensorsFile::open(path)?;
        // TODO (Phase F): iterate _file.tensors and populate encoder/decoder/quantizer weights
        // following the Kyutai weight name mapping documented in
        // opensesame-rustymimi-analysis.md §5-7.
        let config = MimiConfig::v0_1();
        Ok(Self::new(config))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build codec with default Mimi config.
    fn make_codec() -> MimiCodec {
        MimiCodec::new(MimiConfig::v0_1())
    }

    // ── Projection shape tests ────────────────────────────────────────────────

    #[test]
    fn test_enc_proj_shape() {
        // enc_proj maps [N, 512] → [N, 256].
        let codec = make_codec();
        let n = 4;
        let x = vec![0.0_f32; n * 512];
        let y = MimiCodec::linear(
            &x,
            n,
            512,
            &codec.enc_proj,
            &codec.enc_proj_bias,
            256,
        );
        assert_eq!(y.len(), n * 256, "enc_proj output must be [N, 256]");
    }

    #[test]
    fn test_dec_proj_shape() {
        // dec_proj maps [N, 256] → [N, 512].
        let codec = make_codec();
        let n = 3;
        let x = vec![0.0_f32; n * 256];
        let y = MimiCodec::linear(
            &x,
            n,
            256,
            &codec.dec_proj,
            &codec.dec_proj_bias,
            512,
        );
        assert_eq!(y.len(), n * 512, "dec_proj output must be [N, 512]");
    }

    // ── Encode shape tests ────────────────────────────────────────────────────

    #[test]
    fn test_codec_encode_shape() {
        // encode([B=1, 1, T=960]) → 8 code vectors of length 1
        let codec = make_codec();
        let audio = vec![0.0_f32; 960];
        let codes = codec.encode(&audio, 1, 960);
        assert_eq!(codes.len(), 8, "must return 8 code vectors (one per codebook)");
        for (i, cv) in codes.iter().enumerate() {
            assert_eq!(cv.len(), 1, "codebook {} must have 1 frame", i);
        }
    }

    #[test]
    fn test_codec_encode_2frames() {
        // T=1920 → 8 code vectors of length 2
        let codec = make_codec();
        let audio = vec![0.0_f32; 1920];
        let codes = codec.encode(&audio, 1, 1920);
        assert_eq!(codes.len(), 8);
        for cv in &codes {
            assert_eq!(cv.len(), 2, "each codebook must have 2 frames for T=1920");
        }
    }

    #[test]
    fn test_codec_batch2() {
        // B=2, T=960 → 8 code vectors of length 2 (1 frame × 2 batch items)
        let codec = make_codec();
        let audio = vec![0.1_f32; 2 * 960];
        let codes = codec.encode(&audio, 2, 960);
        assert_eq!(codes.len(), 8, "8 codebooks");
        for cv in &codes {
            assert_eq!(cv.len(), 2, "2 entries (B=2 × T/960=1)");
        }
    }

    // ── Decode shape tests ────────────────────────────────────────────────────

    #[test]
    fn test_codec_decode_shape() {
        // decode(8 code vecs of len 1, batch=1) → [1, 1, 960] = 960 samples
        let codec = make_codec();
        let codes: Vec<Vec<u32>> = (0..8).map(|_| vec![0u32]).collect();
        let audio = codec.decode(&codes, 1);
        assert_eq!(audio.len(), 960, "1 frame × 960 samples/frame = 960 samples");
    }

    #[test]
    fn test_codec_batch2_decode_shape() {
        // decode(8 code vecs of len 2, batch=2) → [2, 1, 960] = 1920 samples
        let codec = make_codec();
        let codes: Vec<Vec<u32>> = (0..8).map(|_| vec![0u32, 1u32]).collect();
        let audio = codec.decode(&codes, 2);
        assert_eq!(audio.len(), 1920, "B=2 × 1 frame × 960 = 1920 samples");
    }

    // ── Reconstruct ───────────────────────────────────────────────────────────

    #[test]
    fn test_codec_reconstruct_shape() {
        // reconstruct: output length == input length for T=960
        let codec = make_codec();
        let audio = vec![0.0_f32; 960];
        let out = codec.reconstruct(&audio, 1, 960);
        assert_eq!(out.len(), audio.len(), "reconstruct must return same length");
    }

    // ── Code range ────────────────────────────────────────────────────────────

    #[test]
    fn test_codec_codes_in_range() {
        // All code values must be in [0, 2047] (codebook_size = 2048)
        let codec = make_codec();
        let audio: Vec<f32> = (0..1920).map(|i| (i as f32 * 0.001).sin()).collect();
        let codes = codec.encode(&audio, 1, 1920);
        for (cb, cv) in codes.iter().enumerate() {
            for &c in cv {
                assert!(
                    (c as usize) < 2048,
                    "codebook {} code {} out of range [0, 2048)",
                    cb,
                    c
                );
            }
        }
    }

    // ── Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn test_codec_deterministic() {
        // Same input → same codes on every call
        let codec = make_codec();
        let audio: Vec<f32> = (0..960).map(|i| (i as f32).sin() * 0.1).collect();
        let codes1 = codec.encode(&audio, 1, 960);
        let codes2 = codec.encode(&audio, 1, 960);
        assert_eq!(codes1, codes2, "encode must be deterministic");
    }

    // ── Pipeline no crash ─────────────────────────────────────────────────────

    #[test]
    fn test_codec_pipeline_no_crash() {
        // Full encode → decode on T=960 must not panic.
        let codec = make_codec();
        let audio: Vec<f32> = (0..960).map(|i| (i as f32 * 0.0001).sin()).collect();
        let codes = codec.encode(&audio, 1, 960);
        let decoded = codec.decode(&codes, 1);
        assert_eq!(decoded.len(), 960, "decoded length must be 960");
        assert!(decoded.iter().all(|v| v.is_finite()), "decoded audio must be finite");
    }

    // ── Output bounded ────────────────────────────────────────────────────────

    #[test]
    fn test_codec_output_bounded() {
        // SEANetDecoder applies Tanh → output ∈ (−1, 1) ⊂ [−2, 2]
        let codec = make_codec();
        let audio = vec![0.5_f32; 960];
        let codes = codec.encode(&audio, 1, 960);
        let decoded = codec.decode(&codes, 1);
        for &v in &decoded {
            assert!(
                v >= -2.0 && v <= 2.0,
                "decoded sample {} outside [-2, 2]",
                v
            );
        }
    }

    // ── Empty codes ───────────────────────────────────────────────────────────

    #[test]
    fn test_codec_empty_codes() {
        // decode with empty codes → empty audio (no panic)
        let codec = make_codec();
        let empty: Vec<Vec<u32>> = Vec::new();
        let out = codec.decode(&empty, 1);
        assert!(out.is_empty(), "decode(empty) must return empty Vec");
    }
}
