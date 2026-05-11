//! Weight loader — Sesame CSM-1B safetensors → Depformer.
//!
//! Maps torchtune-style tensor names (used in the `sesame/csm-1b` checkpoint)
//! to the `atlas_model::OlmoModel` weight fields.
//!
//! # Tensor name conventions
//! The Sesame CSM checkpoint uses torchtune naming:
//! - `decoder.layers.N.attn.q_proj.weight`  → query projection
//! - `decoder.layers.N.attn.k_proj.weight`  → key projection
//! - `decoder.layers.N.attn.v_proj.weight`  → value projection
//! - `decoder.layers.N.attn.output_proj.weight` → output projection
//! - `decoder.layers.N.mlp.w1.weight`       → gate (SwiGLU)
//! - `decoder.layers.N.mlp.w2.weight`       → down (SwiGLU)
//! - `decoder.layers.N.mlp.w3.weight`       → up (SwiGLU)
//! - `decoder.layers.N.sa_norm.scale`        → attention pre-norm
//! - `decoder.layers.N.mlp_norm.scale`       → FFN pre-norm
//! - `decoder.norm.scale`                    → final RMSNorm
//! - `audio_head`                            → `[n_dep_cb, vocab_size, d_model]`
//!
//! # Torchtune SwiGLU naming
//! ```text
//! w1 (gate): in_dim=d_model, out_dim=ffn_dim  → atlas w_gate
//! w2 (down): in_dim=ffn_dim, out_dim=d_model  → atlas w_down
//! w3 (up):   in_dim=d_model, out_dim=ffn_dim  → atlas w_up
//! ```

use crate::config::DepformerConfig;
use crate::depformer::Depformer;
use crate::head::CsmAudioHead;

/// Safetensors key for the Depformer's final RMSNorm.
pub const DEPFORMER_NORM_KEY: &str = "decoder.norm.scale";

/// Safetensors key for the `audio_head` parameter tensor.
pub const DEPFORMER_AUDIO_HEAD_KEY: &str = "audio_head";

/// Safetensors key prefix for Depformer transformer layers.
///
/// Full layer key format: `"{DEPFORMER_LAYER_PREFIX}{layer_index}.{suffix}"`.
pub const DEPFORMER_LAYER_PREFIX: &str = "decoder.layers.";

/// Load a [`Depformer`] from a Sesame CSM-1B safetensors checkpoint.
///
/// Opens the checkpoint via `atlas_model::SafetensorsFile`, reads all
/// depformer-related tensors, and wires them into an `OlmoModel` and a
/// `CsmAudioHead`.
///
/// # Parameters
/// - `st`:     Opened `SafetensorsFile` for the checkpoint.
/// - `config`: `DepformerConfig` whose dimensions must match the checkpoint.
///
/// # Errors
/// Returns an error if any required tensor is missing or dimensionally inconsistent.
///
/// # Example
/// ```ignore
/// let st = atlas_model::SafetensorsFile::open("csm-1b.safetensors")?;
/// let config = DepformerConfig::opensesame_1b();
/// let depformer = load_depformer_from_safetensors(&st, config)?;
/// ```
pub fn load_depformer_from_safetensors(
    st: &atlas_model::SafetensorsFile,
    config: DepformerConfig,
) -> Result<Depformer, Box<dyn std::error::Error>> {
    let mut dep = Depformer::new_zeroed(config.clone());

    // ── Transformer layers ────────────────────────────────────────────────────
    for layer_i in 0..config.n_layers {
        let pfx = format!("{}{layer_i}.", DEPFORMER_LAYER_PREFIX);
        dep.transformer.load_layer_weights_torchtune(
            layer_i,
            st.get_f32(&format!("{}attn.q_proj.weight", pfx))?,
            st.get_f32(&format!("{}attn.k_proj.weight", pfx))?,
            st.get_f32(&format!("{}attn.v_proj.weight", pfx))?,
            st.get_f32(&format!("{}attn.output_proj.weight", pfx))?,
            st.get_f32(&format!("{}mlp.w1.weight", pfx))?,   // gate
            st.get_f32(&format!("{}mlp.w2.weight", pfx))?,   // down
            st.get_f32(&format!("{}mlp.w3.weight", pfx))?,   // up
            st.get_f32(&format!("{}sa_norm.scale", pfx))?,
            st.get_f32(&format!("{}mlp_norm.scale", pfx))?,
        );
    }

    // ── Final RMSNorm ─────────────────────────────────────────────────────────
    dep.transformer.load_norm_weights(st.get_f32(DEPFORMER_NORM_KEY)?);

    // ── Audio head ────────────────────────────────────────────────────────────
    // The CSM-1B checkpoint stores all 31 dep-codebook heads in `audio_head`
    // shape [31, vocab_size, d_model].  We slice only the first n_dep_codebooks.
    let n_dep = config.n_dep_codebooks;
    let head_stride = config.vocab_size * config.d_model;
    let audio_head_flat = st.get_f32(DEPFORMER_AUDIO_HEAD_KEY)?;
    if audio_head_flat.len() < n_dep * head_stride {
        return Err(format!(
            "audio_head too small: need >= {} floats ({n_dep}×{}×{}), got {}",
            n_dep * head_stride,
            config.vocab_size,
            config.d_model,
            audio_head_flat.len()
        )
        .into());
    }
    dep.head = CsmAudioHead::from_flat(
        audio_head_flat[..n_dep * head_stride].to_vec(),
        n_dep,
        config.d_model,
        config.vocab_size,
    );

    // ── GPU init ──────────────────────────────────────────────────────────────
    dep.transformer.init_gpu_resources();

    Ok(dep)
}
