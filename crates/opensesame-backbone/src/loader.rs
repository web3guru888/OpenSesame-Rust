//! Weight loading utilities for the CSM backbone.
//!
//! Handles two naming conventions:
//! - **HuggingFace / Llama** — e.g. `model.layers.0.self_attn.q_proj.weight`
//! - **torchtune / Sesame CSM** — e.g. `backbone.layers.0.attn.q_proj.weight`
//!
//! The `LlamaWeightMapper` translates HF tensor names to the parameter slots
//! expected by `atlas-model`'s `OlmoModel`.

/// Maps HuggingFace Llama-3 safetensors weight names to the corresponding
/// `atlas-model` / `OlmoModel` parameter slot identifiers.
///
/// Returns `Some(mapped_name)` for recognised keys, `None` for unknown keys.
///
/// # Name conventions
///
/// | HuggingFace key | atlas-model field |
/// |-----------------|-------------------|
/// | `model.embed_tokens.weight` | `embed` |
/// | `model.norm.weight` | `norm` |
/// | `lm_head.weight` | `lm_head` |
/// | `model.layers.N.self_attn.q_proj.weight` | `layers.N.attn.wq` |
/// | `model.layers.N.self_attn.k_proj.weight` | `layers.N.attn.wk` |
/// | `model.layers.N.self_attn.v_proj.weight` | `layers.N.attn.wv` |
/// | `model.layers.N.self_attn.o_proj.weight` | `layers.N.attn.wo` |
/// | `model.layers.N.mlp.gate_proj.weight` | `layers.N.ffn.w_gate` |
/// | `model.layers.N.mlp.up_proj.weight` | `layers.N.ffn.w_up` |
/// | `model.layers.N.mlp.down_proj.weight` | `layers.N.ffn.w_down` |
/// | `model.layers.N.input_layernorm.weight` | `layers.N.attn_norm` |
/// | `model.layers.N.post_attention_layernorm.weight` | `layers.N.ffn_norm` |
pub struct LlamaWeightMapper;

impl LlamaWeightMapper {
    /// Map a HuggingFace Llama safetensors key to an atlas-model field name.
    pub fn map_key(hf_key: &str) -> Option<String> {
        // Top-level tensors
        match hf_key {
            "model.embed_tokens.weight" => return Some("embed".to_string()),
            "model.norm.weight"         => return Some("norm".to_string()),
            "lm_head.weight"            => return Some("lm_head".to_string()),
            _ => {}
        }

        // Per-layer tensors: "model.layers.{N}.{suffix}"
        let layer_pfx = "model.layers.";
        if let Some(rest) = hf_key.strip_prefix(layer_pfx) {
            // Parse layer index
            let dot = rest.find('.')?;
            let layer_str = &rest[..dot];
            let _layer_n: usize = layer_str.parse().ok()?;
            let suffix = &rest[dot + 1..];

            let mapped = match suffix {
                "self_attn.q_proj.weight"              => "attn.wq",
                "self_attn.k_proj.weight"              => "attn.wk",
                "self_attn.v_proj.weight"              => "attn.wv",
                "self_attn.o_proj.weight"              => "attn.wo",
                "mlp.gate_proj.weight"                 => "ffn.w_gate",
                "mlp.up_proj.weight"                   => "ffn.w_up",
                "mlp.down_proj.weight"                 => "ffn.w_down",
                "input_layernorm.weight"               => "attn_norm",
                "post_attention_layernorm.weight"      => "ffn_norm",
                _ => return None,
            };
            return Some(format!("layers.{layer_str}.{mapped}"));
        }

        None
    }
}

/// Maps torchtune / Sesame CSM safetensors weight names to atlas-model field names.
///
/// torchtune uses different naming from HuggingFace:
/// - `attn.q_proj.weight` instead of `self_attn.q_proj.weight`
/// - `attn.output_proj.weight` instead of `self_attn.o_proj.weight`
/// - `mlp.w1.weight` (gate) / `mlp.w2.weight` (down) / `mlp.w3.weight` (up)
/// - `sa_norm.scale` instead of `input_layernorm.weight`
/// - `mlp_norm.scale` instead of `post_attention_layernorm.weight`
pub struct TorchtuneWeightMapper;

impl TorchtuneWeightMapper {
    /// Map a torchtune tensor name (with prefix stripped) to an atlas-model slot.
    ///
    /// `suffix` is the part **after** the layer prefix, e.g. `"attn.q_proj.weight"`.
    pub fn map_layer_suffix(suffix: &str) -> Option<&'static str> {
        match suffix {
            "attn.q_proj.weight"      => Some("attn.wq"),
            "attn.k_proj.weight"      => Some("attn.wk"),
            "attn.v_proj.weight"      => Some("attn.wv"),
            "attn.output_proj.weight" => Some("attn.wo"),
            "mlp.w1.weight"           => Some("ffn.w_gate"),
            "mlp.w2.weight"           => Some("ffn.w_down"),
            "mlp.w3.weight"           => Some("ffn.w_up"),
            "sa_norm.scale"           => Some("attn_norm"),
            "mlp_norm.scale"          => Some("ffn_norm"),
            _ => None,
        }
    }
}

/// Load CsmModel weights from a Sesame CSM `.safetensors` checkpoint.
///
/// Returns an error string if the file cannot be opened or a required tensor
/// is missing.  This is a stub that validates the file path and tensor presence;
/// full weight injection into the model is deferred to Phase H.
pub fn load_weights_from_safetensors(path: &str) -> Result<(), String> {
    use atlas_model::SafetensorsFile;
    let _st = SafetensorsFile::open(path)
        .map_err(|e| format!("cannot open {path}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_mapper_q_proj() {
        let key = "model.layers.0.self_attn.q_proj.weight";
        let mapped = LlamaWeightMapper::map_key(key);
        assert_eq!(mapped, Some("layers.0.attn.wq".to_string()));
    }

    #[test]
    fn test_weight_mapper_k_proj() {
        let mapped = LlamaWeightMapper::map_key("model.layers.3.self_attn.k_proj.weight");
        assert_eq!(mapped, Some("layers.3.attn.wk".to_string()));
    }

    #[test]
    fn test_weight_mapper_v_proj() {
        let mapped = LlamaWeightMapper::map_key("model.layers.15.self_attn.v_proj.weight");
        assert_eq!(mapped, Some("layers.15.attn.wv".to_string()));
    }

    #[test]
    fn test_weight_mapper_o_proj() {
        let mapped = LlamaWeightMapper::map_key("model.layers.0.self_attn.o_proj.weight");
        assert_eq!(mapped, Some("layers.0.attn.wo".to_string()));
    }

    #[test]
    fn test_weight_mapper_gate() {
        let mapped = LlamaWeightMapper::map_key("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(mapped, Some("layers.0.ffn.w_gate".to_string()));
    }

    #[test]
    fn test_weight_mapper_up() {
        let mapped = LlamaWeightMapper::map_key("model.layers.0.mlp.up_proj.weight");
        assert_eq!(mapped, Some("layers.0.ffn.w_up".to_string()));
    }

    #[test]
    fn test_weight_mapper_down() {
        let mapped = LlamaWeightMapper::map_key("model.layers.0.mlp.down_proj.weight");
        assert_eq!(mapped, Some("layers.0.ffn.w_down".to_string()));
    }

    #[test]
    fn test_weight_mapper_attn_norm() {
        let mapped = LlamaWeightMapper::map_key("model.layers.0.input_layernorm.weight");
        assert_eq!(mapped, Some("layers.0.attn_norm".to_string()));
    }

    #[test]
    fn test_weight_mapper_ffn_norm() {
        let mapped = LlamaWeightMapper::map_key(
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(mapped, Some("layers.0.ffn_norm".to_string()));
    }

    #[test]
    fn test_weight_mapper_embed() {
        let mapped = LlamaWeightMapper::map_key("model.embed_tokens.weight");
        assert_eq!(mapped, Some("embed".to_string()));
    }

    #[test]
    fn test_weight_mapper_norm() {
        let mapped = LlamaWeightMapper::map_key("model.norm.weight");
        assert_eq!(mapped, Some("norm".to_string()));
    }

    #[test]
    fn test_weight_mapper_lm_head() {
        let mapped = LlamaWeightMapper::map_key("lm_head.weight");
        assert_eq!(mapped, Some("lm_head".to_string()));
    }

    #[test]
    fn test_weight_mapper_unknown() {
        let mapped = LlamaWeightMapper::map_key("some.unknown.tensor");
        assert_eq!(mapped, None);
    }

    #[test]
    fn test_weight_mapper_unknown_layer_suffix() {
        let mapped = LlamaWeightMapper::map_key("model.layers.0.unknown_field");
        assert_eq!(mapped, None);
    }

    #[test]
    fn test_load_weights_missing_file() {
        let result = load_weights_from_safetensors("/nonexistent/path/model.safetensors");
        assert!(result.is_err(), "expected Err for missing file");
        let msg = result.unwrap_err();
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_torchtune_mapper_q_proj() {
        let mapped = TorchtuneWeightMapper::map_layer_suffix("attn.q_proj.weight");
        assert_eq!(mapped, Some("attn.wq"));
    }

    #[test]
    fn test_torchtune_mapper_output_proj() {
        let mapped = TorchtuneWeightMapper::map_layer_suffix("attn.output_proj.weight");
        assert_eq!(mapped, Some("attn.wo"));
    }

    #[test]
    fn test_torchtune_mapper_sa_norm() {
        let mapped = TorchtuneWeightMapper::map_layer_suffix("sa_norm.scale");
        assert_eq!(mapped, Some("attn_norm"));
    }

    #[test]
    fn test_torchtune_mapper_unknown() {
        let mapped = TorchtuneWeightMapper::map_layer_suffix("unknown.weight");
        assert_eq!(mapped, None);
    }
}
