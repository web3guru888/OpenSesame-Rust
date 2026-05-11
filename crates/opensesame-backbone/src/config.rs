//! Configuration types for the OpenSesame/CSM backbone transformer.
//!
//! Provides `BackboneConfig` (backbone-only, Phase F) and `CsmConfig` (full
//! CSM including depformer) with their standard constructors.

use atlas_model::{ModelConfig, RopeScaling};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Text vocabulary size (Llama 3 BPE tokenizer).
pub const TEXT_VOCAB_SIZE: usize = 128_000;

/// Audio vocabulary size per codebook (Mimi quantizer).
pub const AUDIO_VOCAB_SIZE: usize = 2_048;

/// Number of audio codebooks used by OpenSesame (CB0..CB7).
pub const N_CODEBOOKS: usize = 8;

/// Padding token index in the text vocabulary.
/// Any token id ≥ TEXT_VOCAB_SIZE is treated as padding.
pub const TEXT_PAD_TOKEN: u32 = TEXT_VOCAB_SIZE as u32;

/// Padding token index within a single codebook's slice.
/// Per codebook `k`, the actual flat index is `k * (AUDIO_VOCAB_SIZE + 1) + AUDIO_PAD_TOKEN`.
pub const AUDIO_PAD_TOKEN: u32 = AUDIO_VOCAB_SIZE as u32;

// ── BackboneConfig ────────────────────────────────────────────────────────────

/// Configuration for the CSM backbone transformer (Llama-3.2-1B architecture).
///
/// The backbone is a 16-layer transformer with grouped query attention (GQA,
/// 32 query heads / 8 KV heads) and SwiGLU FFN.  It processes sequences of
/// multimodal (text + audio) embeddings and produces hidden states used by:
/// - The CB0 audio prediction head (immediate output)
/// - The Depformer (CB1..CB7 generation)
/// - The Inner Monologue text head (training only)
#[derive(Debug, Clone)]
pub struct BackboneConfig {
    /// Number of transformer layers (16 for Llama-3.2-1B).
    pub n_layers: usize,
    /// Hidden / embedding dimension (2048).
    pub d_model: usize,
    /// Number of query attention heads (32).
    pub n_heads: usize,
    /// Number of KV attention heads for GQA (8).
    pub n_kv_heads: usize,
    /// FFN intermediate dimension (8192 = 4 × d_model).
    pub ffn_dim: usize,
    /// Text vocabulary size (128000 for Llama 3 tokenizer).
    pub text_vocab_size: usize,
    /// Audio vocabulary size per codebook (2048 for Mimi).
    pub audio_vocab_size: usize,
    /// Number of audio codebooks (8 for OpenSesame).
    pub n_audio_codebooks: usize,
    /// Maximum sequence length (2048 for CSM training context).
    pub max_seq_len: usize,
    /// RoPE base theta (500_000.0 for Llama 3.2 extended context).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
}

impl BackboneConfig {
    /// Standard OpenSesame 1B backbone configuration.
    ///
    /// Matches the Sesame CSM-1B backbone: Llama-3.2-1B with 16 layers,
    /// d=2048, 32 query heads, 8 KV heads, ffn=8192, rope_theta=500_000.
    pub fn csm_1b() -> Self {
        Self {
            n_layers:          16,
            d_model:           2048,
            n_heads:           32,
            n_kv_heads:        8,
            ffn_dim:           8192,
            text_vocab_size:   TEXT_VOCAB_SIZE,
            audio_vocab_size:  AUDIO_VOCAB_SIZE,
            n_audio_codebooks: N_CODEBOOKS,
            max_seq_len:       2048,
            rope_theta:        500_000.0,
            norm_eps:          1e-5,
        }
    }

    /// Tiny configuration for fast unit tests (2 layers, d=64).
    pub fn tiny() -> Self {
        Self {
            n_layers:          2,
            d_model:           64,
            n_heads:           4,
            n_kv_heads:        2,
            ffn_dim:           128,
            text_vocab_size:   200,
            audio_vocab_size:  16,
            n_audio_codebooks: 4,
            max_seq_len:       64,
            rope_theta:        10_000.0,
            norm_eps:          1e-5,
        }
    }

    /// Convert to atlas-model `ModelConfig` for use with `OlmoModel`.
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            vocab_size:   self.text_vocab_size,
            d_model:      self.d_model,
            n_layers:     self.n_layers,
            n_heads:      self.n_heads,
            n_kv_heads:   self.n_kv_heads,
            ffn_hidden:   self.ffn_dim,
            max_seq_len:  self.max_seq_len,
            rope_theta:   self.rope_theta,
            rms_norm_eps: self.norm_eps,
            layer_types:  Vec::new(),
            sliding_window: None,
            rope_scaling: RopeScaling::None,
            eos_token_id: None,
        }
    }

    /// Approximate total parameter count for this backbone configuration.
    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let n = self.n_layers;
        let kv_dim = (d / self.n_heads) * self.n_kv_heads;
        let ffn = self.ffn_dim;

        // Attention: Q(d×d) + K(kv_dim×d) + V(kv_dim×d) + O(d×d) per layer
        let attn_per_layer = d * d + kv_dim * d * 2 + d * d;
        // FFN: gate(ffn×d) + up(ffn×d) + down(d×ffn) per layer
        let ffn_per_layer = ffn * d * 2 + d * ffn;
        // Norms: attn_norm(d) + ffn_norm(d) + final_norm(d) per layer
        let norm_params = d * (n * 2 + 1);

        n * (attn_per_layer + ffn_per_layer) + norm_params
    }
}

// ── CsmConfig (full model including depformer) ────────────────────────────────

/// Complete configuration for the OpenSesame / Sesame CSM model.
///
/// Includes both the backbone (16-layer Llama-3.2-1B) and the depformer
/// (4-layer smaller transformer for CB1..CB7 generation).
#[derive(Debug, Clone)]
pub struct CsmConfig {
    // ── Backbone ────────────────────────────────────────────────────────────
    /// Hidden dimension of the backbone transformer (2048).
    pub backbone_d_model: usize,
    /// Number of backbone transformer layers (16).
    pub backbone_n_layers: usize,
    /// Number of backbone query heads (32).
    pub backbone_n_heads: usize,
    /// Number of backbone KV heads (8, GQA).
    pub backbone_n_kv_heads: usize,
    /// Backbone FFN intermediate dimension (8192).
    pub backbone_ffn_dim: usize,
    /// Backbone maximum sequence length (2048).
    pub backbone_max_seq: usize,
    /// Backbone RoPE theta (500_000.0).
    pub backbone_rope_base: f32,
    /// Backbone RMSNorm epsilon (1e-5).
    pub backbone_norm_eps: f32,

    // ── Multimodal ───────────────────────────────────────────────────────────
    /// Text vocabulary size (128_000 for Llama 3).
    pub text_vocab_size: usize,
    /// Audio vocabulary size per codebook (2_048 for Mimi).
    pub audio_vocab_size: usize,
    /// Number of audio codebooks (8 for OpenSesame).
    pub n_codebooks: usize,

    // ── Depformer ────────────────────────────────────────────────────────────
    /// Depformer hidden dimension (1024).
    pub dep_d_model: usize,
    /// Number of depformer layers (4).
    pub dep_n_layers: usize,
    /// Depformer query heads (8).
    pub dep_n_heads: usize,
    /// Depformer KV heads (2, GQA).
    pub dep_n_kv_heads: usize,
    /// Depformer FFN dimension (8192).
    pub dep_ffn_dim: usize,
    /// Depformer max sequence length (= n_codebooks = 8).
    pub dep_max_seq: usize,
    /// Depformer RoPE theta (500_000.0).
    pub dep_rope_base: f32,
}

impl CsmConfig {
    /// Standard OpenSesame-1B full model configuration (8 codebooks, Mimi codec).
    pub fn opensesame_1b() -> Self {
        Self {
            backbone_d_model:    2048,
            backbone_n_layers:   16,
            backbone_n_heads:    32,
            backbone_n_kv_heads: 8,
            backbone_ffn_dim:    8192,
            backbone_max_seq:    2048,
            backbone_rope_base:  500_000.0,
            backbone_norm_eps:   1e-5,
            text_vocab_size:     TEXT_VOCAB_SIZE,
            audio_vocab_size:    AUDIO_VOCAB_SIZE,
            n_codebooks:         N_CODEBOOKS,
            dep_d_model:         1024,
            dep_n_layers:        4,
            dep_n_heads:         8,
            dep_n_kv_heads:      2,
            dep_ffn_dim:         8192,
            dep_max_seq:         N_CODEBOOKS,
            dep_rope_base:       500_000.0,
        }
    }

    /// Extract the backbone-only `BackboneConfig`.
    pub fn backbone_config(&self) -> BackboneConfig {
        BackboneConfig {
            n_layers:          self.backbone_n_layers,
            d_model:           self.backbone_d_model,
            n_heads:           self.backbone_n_heads,
            n_kv_heads:        self.backbone_n_kv_heads,
            ffn_dim:           self.backbone_ffn_dim,
            text_vocab_size:   self.text_vocab_size,
            audio_vocab_size:  self.audio_vocab_size,
            n_audio_codebooks: self.n_codebooks,
            max_seq_len:       self.backbone_max_seq,
            rope_theta:        self.backbone_rope_base,
            norm_eps:          self.backbone_norm_eps,
        }
    }

    /// atlas-model `ModelConfig` for the backbone transformer.
    pub fn backbone_model_config(&self) -> ModelConfig {
        ModelConfig {
            vocab_size:   self.text_vocab_size,
            d_model:      self.backbone_d_model,
            n_layers:     self.backbone_n_layers,
            n_heads:      self.backbone_n_heads,
            n_kv_heads:   self.backbone_n_kv_heads,
            ffn_hidden:   self.backbone_ffn_dim,
            max_seq_len:  self.backbone_max_seq,
            rope_theta:   self.backbone_rope_base,
            rms_norm_eps: self.backbone_norm_eps,
            layer_types:  Vec::new(),
            sliding_window: None,
            rope_scaling: RopeScaling::None,
            eos_token_id: None,
        }
    }

    /// atlas-model `ModelConfig` for the depformer transformer.
    pub fn depformer_model_config(&self) -> ModelConfig {
        ModelConfig {
            vocab_size:   self.audio_vocab_size,
            d_model:      self.dep_d_model,
            n_layers:     self.dep_n_layers,
            n_heads:      self.dep_n_heads,
            n_kv_heads:   self.dep_n_kv_heads,
            ffn_hidden:   self.dep_ffn_dim,
            max_seq_len:  self.dep_max_seq,
            rope_theta:   self.dep_rope_base,
            rms_norm_eps: 1e-5,
            layer_types:  Vec::new(),
            sliding_window: None,
            rope_scaling: RopeScaling::None,
            eos_token_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_config_csm1b() {
        let cfg = BackboneConfig::csm_1b();
        assert_eq!(cfg.n_layers, 16);
        assert_eq!(cfg.d_model, 2048);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.ffn_dim, 8192);
        assert_eq!(cfg.text_vocab_size, 128_000);
        assert_eq!(cfg.audio_vocab_size, 2_048);
        assert_eq!(cfg.n_audio_codebooks, 8);
        assert_eq!(cfg.max_seq_len, 2048);
        assert!((cfg.rope_theta - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn test_backbone_config_total_params() {
        let cfg = BackboneConfig::csm_1b();
        let p = cfg.param_count();
        // Should be roughly 1B for backbone weights only (without embeddings)
        // 16 layers × (attn + ffn): ~1B params
        assert!(p > 500_000_000, "param_count {} too low", p);
        assert!(p < 2_000_000_000, "param_count {} too high", p);
    }

    #[test]
    fn test_csm_config_opensesame_1b() {
        let cfg = CsmConfig::opensesame_1b();
        assert_eq!(cfg.backbone_n_layers, 16);
        assert_eq!(cfg.dep_n_layers, 4);
        assert_eq!(cfg.n_codebooks, 8);
    }

    #[test]
    fn test_backbone_config_tiny() {
        let cfg = BackboneConfig::tiny();
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.d_model, 64);
        assert!(cfg.ffn_dim > cfg.d_model);
    }

    #[test]
    fn test_to_model_config_fields() {
        let cfg = BackboneConfig::csm_1b();
        let mc = cfg.to_model_config();
        assert_eq!(mc.d_model, 2048);
        assert_eq!(mc.n_layers, 16);
    }
}
