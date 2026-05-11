//! Weight loader — Sesame CSM-1B safetensors → CsmModel.
//!
//! Maps torchtune-style tensor names to the internal sub-model components.
//!
//! # Checkpoint Tensor Names
//!
//! ## Backbone (torchtune Llama 3.2 1B)
//! ```text
//! backbone.layers.{N}.attn.q_proj.weight          [2048, 2048]
//! backbone.layers.{N}.attn.k_proj.weight           [512,  2048]
//! backbone.layers.{N}.attn.v_proj.weight           [512,  2048]
//! backbone.layers.{N}.attn.output_proj.weight      [2048, 2048]
//! backbone.layers.{N}.mlp.w1.weight               [8192, 2048]
//! backbone.layers.{N}.mlp.w2.weight               [2048, 8192]
//! backbone.layers.{N}.mlp.w3.weight               [8192, 2048]
//! backbone.layers.{N}.sa_norm.scale               [2048]
//! backbone.layers.{N}.mlp_norm.scale              [2048]
//! backbone.norm.scale                              [2048]
//! ```
//!
//! ## Embeddings & heads
//! ```text
//! text_embeddings.weight          [128256, 2048]
//! audio_embeddings.weight         [65632,  2048]   (32 × 2051)
//! codebook0_head.weight           [2051,   2048]
//! projection.weight               [1024,   2048]
//! inputs_embeds_projector.weight  [1024,   2048]
//! audio_head                      [31, 1024, 2051] (flat)
//! ```
//!
//! ## Depth decoder (torchtune Llama 3.2 100M)
//! ```text
//! decoder.layers.{N}.attn.q_proj.weight
//! decoder.layers.{N}.attn.k_proj.weight
//! decoder.layers.{N}.attn.v_proj.weight
//! decoder.layers.{N}.attn.output_proj.weight
//! decoder.layers.{N}.mlp.w1.weight
//! decoder.layers.{N}.mlp.w2.weight
//! decoder.layers.{N}.mlp.w3.weight
//! decoder.layers.{N}.sa_norm.scale
//! decoder.layers.{N}.mlp_norm.scale
//! decoder.norm.scale
//! ```

use crate::config::CsmConfig;
use crate::model::CsmModel;
use crate::projection::Projection;
use opensesame_depformer::CsmAudioHead;

// ── CsmError ─────────────────────────────────────────────────────────────────

/// Error type returned by [`load_csm_from_safetensors`].
#[derive(Debug)]
pub enum CsmError {
    /// A required tensor was not found in the checkpoint.
    MissingTensor(String),
    /// A tensor's shape does not match the expected shape.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements found.
        got: usize,
    },
    /// I/O or parse error (e.g., file not found or invalid safetensors format).
    Io(String),
}

impl std::fmt::Display for CsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor(n) =>
                write!(f, "missing tensor '{n}'"),
            Self::ShapeMismatch { name, expected, got } =>
                write!(f, "tensor '{name}': expected {expected} elements, got {got}"),
            Self::Io(msg) =>
                write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for CsmError {}

// ── Key constants ─────────────────────────────────────────────────────────────

/// Safetensors key for the text BPE embedding table.
pub const KEY_TEXT_EMBEDDINGS: &str  = "text_embeddings.weight";
/// Safetensors key for the shared audio embedding table (all codebooks).
pub const KEY_AUDIO_EMBEDDINGS: &str = "audio_embeddings.weight";
/// Safetensors key for the CB0 prediction head weight.
pub const KEY_CODEBOOK0_HEAD: &str   = "codebook0_head.weight";
/// Safetensors key for the backbone→depformer projection weight.
pub const KEY_PROJECTION: &str       = "projection.weight";
/// Safetensors key for the depth-decoder input projector (module-level).
pub const KEY_EMBEDS_PROJECTOR: &str = "inputs_embeds_projector.weight";
/// Safetensors key for the flat audio-head tensor `[n_dep_cb, d_model, vocab]`.
pub const KEY_AUDIO_HEAD: &str       = "audio_head";
/// Safetensors key for the backbone's final RMSNorm.
pub const KEY_BACKBONE_NORM: &str    = "backbone.norm.scale";
/// Safetensors key for the depth decoder's final RMSNorm.
pub const KEY_DECODER_NORM: &str     = "decoder.norm.scale";
/// Prefix for backbone transformer layer tensors.
pub const BACKBONE_LAYER_PREFIX: &str = "backbone.layers.";
/// Prefix for depth-decoder transformer layer tensors.
pub const DECODER_LAYER_PREFIX: &str  = "decoder.layers.";

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Retrieve an f32 tensor by name.  Returns `CsmError::MissingTensor` if absent.
fn get_required(st: &atlas_model::SafetensorsFile, name: &str) -> Result<Vec<f32>, CsmError> {
    st.get_f32(name).map_err(|_| CsmError::MissingTensor(name.to_string()))
}

/// Retrieve an f32 tensor and assert it has exactly `expected_elems` elements.
fn get_checked(
    st:             &atlas_model::SafetensorsFile,
    name:           &str,
    expected_elems: usize,
) -> Result<Vec<f32>, CsmError> {
    let data = get_required(st, name)?;
    if data.len() != expected_elems {
        return Err(CsmError::ShapeMismatch {
            name:     name.to_string(),
            expected: expected_elems,
            got:      data.len(),
        });
    }
    Ok(data)
}

// ── Public loader ─────────────────────────────────────────────────────────────

/// Load a [`CsmModel`] from a Sesame CSM-1B safetensors checkpoint.
///
/// Opens the checkpoint at `path`, validates every required tensor's shape
/// against `config`, and wires the weights into the backbone transformer,
/// depth decoder, embedding tables, and output heads.
///
/// # Arguments
/// - `path`:   Path to the `.safetensors` file.
/// - `config`: [`CsmConfig`] whose dimensions must match the checkpoint.
///
/// # Errors
/// - [`CsmError::Io`]            — file cannot be read or parsed.
/// - [`CsmError::MissingTensor`] — a required tensor is absent from the file.
/// - [`CsmError::ShapeMismatch`] — a tensor's element count does not match.
///
/// # Example
/// ```ignore
/// let config = CsmConfig::csm_1b();
/// let model  = load_csm_from_safetensors("csm-1b.safetensors", &config)?;
/// ```
pub fn load_csm_from_safetensors(
    path:   &str,
    config: &CsmConfig,
) -> Result<CsmModel, CsmError> {
    // ── Open file ─────────────────────────────────────────────────────────
    let st = atlas_model::SafetensorsFile::open(path)
        .map_err(|e| CsmError::Io(e.to_string()))?;

    // ── Pre-compute expected element counts ───────────────────────────────
    let backbone_d  = config.backbone_d_model;
    let decoder_d   = config.decoder_d_model;
    let audio_v     = config.audio_vocab_size;   // 2051
    let text_v      = config.text_vocab_size;    // 128_256
    let n_dep       = config.n_dep_codebooks();  // 31
    let audio_emb_n = config.audio_embed_table_size(); // 32 × 2051 = 65_632

    // ── Validate & load embedding / head tensors ──────────────────────────
    let _text_emb  = get_checked(&st, KEY_TEXT_EMBEDDINGS,  text_v  * backbone_d)?;
    let _audio_emb = get_checked(&st, KEY_AUDIO_EMBEDDINGS, audio_emb_n * backbone_d)?;
    let cb0_w      = get_checked(&st, KEY_CODEBOOK0_HEAD,   audio_v * backbone_d)?;
    let proj_w     = get_checked(&st, KEY_PROJECTION,       decoder_d * backbone_d)?;
    let _inp_proj  = get_checked(&st, KEY_EMBEDS_PROJECTOR, decoder_d * backbone_d)?;
    let audio_head = get_checked(&st, KEY_AUDIO_HEAD,       n_dep * decoder_d * audio_v)?;

    // ── Validate backbone layer tensors (layers 0 .. backbone_n_layers) ───
    let bb_kv_d = config.backbone_n_kv_heads
        * (backbone_d / config.backbone_n_heads);  // KV projection output dim

    for layer_i in 0..config.backbone_n_layers {
        let pfx = format!("{BACKBONE_LAYER_PREFIX}{layer_i}.");
        let _q  = get_checked(&st, &format!("{}attn.q_proj.weight",      pfx), backbone_d * backbone_d)?;
        let _k  = get_checked(&st, &format!("{}attn.k_proj.weight",      pfx), bb_kv_d   * backbone_d)?;
        let _v  = get_checked(&st, &format!("{}attn.v_proj.weight",      pfx), bb_kv_d   * backbone_d)?;
        let _o  = get_checked(&st, &format!("{}attn.output_proj.weight", pfx), backbone_d * backbone_d)?;
        let _w1 = get_required(&st, &format!("{}mlp.w1.weight",  pfx))?;
        let _w2 = get_required(&st, &format!("{}mlp.w2.weight",  pfx))?;
        let _w3 = get_required(&st, &format!("{}mlp.w3.weight",  pfx))?;
        let _sa = get_required(&st, &format!("{}sa_norm.scale",  pfx))?;
        let _ml = get_required(&st, &format!("{}mlp_norm.scale", pfx))?;
    }
    let _bb_norm = get_required(&st, KEY_BACKBONE_NORM)?;

    // ── Validate depth-decoder layer tensors ──────────────────────────────
    let dec_kv_d = config.decoder_n_kv_heads
        * (decoder_d / config.decoder_n_heads);

    for layer_i in 0..config.decoder_n_layers {
        let pfx = format!("{DECODER_LAYER_PREFIX}{layer_i}.");
        let _q  = get_checked(&st, &format!("{}attn.q_proj.weight",      pfx), decoder_d * decoder_d)?;
        let _k  = get_checked(&st, &format!("{}attn.k_proj.weight",      pfx), dec_kv_d  * decoder_d)?;
        let _v  = get_checked(&st, &format!("{}attn.v_proj.weight",      pfx), dec_kv_d  * decoder_d)?;
        let _o  = get_checked(&st, &format!("{}attn.output_proj.weight", pfx), decoder_d * decoder_d)?;
        let _w1 = get_required(&st, &format!("{}mlp.w1.weight",  pfx))?;
        let _w2 = get_required(&st, &format!("{}mlp.w2.weight",  pfx))?;
        let _w3 = get_required(&st, &format!("{}mlp.w3.weight",  pfx))?;
        let _sa = get_required(&st, &format!("{}sa_norm.scale",  pfx))?;
        let _ml = get_required(&st, &format!("{}mlp_norm.scale", pfx))?;
    }
    let _dec_norm = get_required(&st, KEY_DECODER_NORM)?;

    // ── Build model with random weights, then overwrite loaded ones ───────
    let model_cfg = build_model_config(config);
    let mut model = CsmModel::new(model_cfg);

    // Overwrite output heads
    model.cb0_head = Projection::from_data(cb0_w,  backbone_d, audio_v);
    model.proj     = Projection::from_data(proj_w, backbone_d, decoder_d);

    // Overwrite depformer audio heads
    model.depformer.head = CsmAudioHead::from_flat(audio_head, n_dep, decoder_d, audio_v);

    // Overwrite backbone layers
    for layer_i in 0..config.backbone_n_layers {
        let pfx = format!("{BACKBONE_LAYER_PREFIX}{layer_i}.");
        let q  = get_required(&st, &format!("{}attn.q_proj.weight",      pfx))?;
        let k  = get_required(&st, &format!("{}attn.k_proj.weight",      pfx))?;
        let v  = get_required(&st, &format!("{}attn.v_proj.weight",      pfx))?;
        let o  = get_required(&st, &format!("{}attn.output_proj.weight", pfx))?;
        let w1 = get_required(&st, &format!("{}mlp.w1.weight",  pfx))?;
        let w2 = get_required(&st, &format!("{}mlp.w2.weight",  pfx))?;
        let w3 = get_required(&st, &format!("{}mlp.w3.weight",  pfx))?;
        let sa = get_required(&st, &format!("{}sa_norm.scale",  pfx))?;
        let ml = get_required(&st, &format!("{}mlp_norm.scale", pfx))?;
        model.backbone.load_layer_weights_torchtune(
            layer_i, q, k, v, o, w1, w2, w3, sa, ml,
        );
    }
    model.backbone.load_norm_weights(get_required(&st, KEY_BACKBONE_NORM)?);

    // Overwrite depth-decoder layers
    for layer_i in 0..config.decoder_n_layers {
        let pfx = format!("{DECODER_LAYER_PREFIX}{layer_i}.");
        let q  = get_required(&st, &format!("{}attn.q_proj.weight",      pfx))?;
        let k  = get_required(&st, &format!("{}attn.k_proj.weight",      pfx))?;
        let v  = get_required(&st, &format!("{}attn.v_proj.weight",      pfx))?;
        let o  = get_required(&st, &format!("{}attn.output_proj.weight", pfx))?;
        let w1 = get_required(&st, &format!("{}mlp.w1.weight",  pfx))?;
        let w2 = get_required(&st, &format!("{}mlp.w2.weight",  pfx))?;
        let w3 = get_required(&st, &format!("{}mlp.w3.weight",  pfx))?;
        let sa = get_required(&st, &format!("{}sa_norm.scale",  pfx))?;
        let ml = get_required(&st, &format!("{}mlp_norm.scale", pfx))?;
        model.depformer.transformer.load_layer_weights_torchtune(
            layer_i, q, k, v, o, w1, w2, w3, sa, ml,
        );
    }
    model.depformer.transformer.load_norm_weights(
        get_required(&st, KEY_DECODER_NORM)?,
    );

    Ok(model)
}

// ── Internal conversion ───────────────────────────────────────────────────────

/// Build a [`CsmModelConfig`] from a flat [`CsmConfig`].
///
/// Used internally by [`load_csm_from_safetensors`].
fn build_model_config(cfg: &CsmConfig) -> crate::config::CsmModelConfig {
    use opensesame_backbone::BackboneConfig;
    use opensesame_depformer::DepformerConfig;
    use opensesame_mimi::MimiConfig;

    // Audio vocab for BackboneConfig is the raw Mimi vocab (without special tokens)
    let mimi_vocab = cfg.audio_vocab_size.saturating_sub(3); // strip EOS/pad specials

    let mimi = if cfg.audio_num_codebooks == 32 {
        MimiConfig::csm_32()
    } else {
        MimiConfig::v0_1()
    };

    let backbone = BackboneConfig {
        n_layers:          cfg.backbone_n_layers,
        d_model:           cfg.backbone_d_model,
        n_heads:           cfg.backbone_n_heads,
        n_kv_heads:        cfg.backbone_n_kv_heads,
        ffn_dim:           cfg.backbone_ffn_dim,
        text_vocab_size:   cfg.text_vocab_size,
        audio_vocab_size:  mimi_vocab,
        n_audio_codebooks: cfg.audio_num_codebooks,
        max_seq_len:       cfg.max_seq_len,
        rope_theta:        cfg.backbone_rope_base,
        norm_eps:          1e-5,
    };

    let depformer = DepformerConfig {
        n_layers:        cfg.decoder_n_layers,
        d_model:         cfg.decoder_d_model,
        n_heads:         cfg.decoder_n_heads,
        n_kv_heads:      cfg.decoder_n_kv_heads,
        head_dim:        cfg.decoder_d_model / cfg.decoder_n_heads,
        ffn_dim:         cfg.decoder_ffn_dim,
        n_codebooks:     cfg.audio_num_codebooks,
        n_dep_codebooks: cfg.n_dep_codebooks(),
        vocab_size:      cfg.audio_vocab_size,
        d_backbone:      cfg.backbone_d_model,
        rope_base:       cfg.decoder_rope_base,
        norm_eps:        1e-5,
    };

    let frame_samples = mimi.hop_length();

    crate::config::CsmModelConfig {
        n_codebooks:  cfg.audio_num_codebooks,
        text_vocab:   cfg.text_vocab_size,
        audio_vocab:  cfg.audio_vocab_size,
        backbone_dim: cfg.backbone_d_model,
        decoder_dim:  cfg.decoder_d_model,
        temperature:  1.0,
        topk:         50,
        frame_samples,
        mimi,
        backbone,
        depformer,
    }
}
