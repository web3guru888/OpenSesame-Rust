//! External integration tests for load_csm_from_safetensors (Phase H).
//!
//! These tests use `atlas_model::SafetensorsFile::build_f32` to construct
//! tiny in-memory checkpoints and verify the loader's validation logic.

use opensesame_csm::{
    CsmConfig, CsmError,
    KEY_TEXT_EMBEDDINGS, KEY_AUDIO_EMBEDDINGS, KEY_CODEBOOK0_HEAD,
    KEY_AUDIO_HEAD, KEY_EMBEDS_PROJECTOR, BACKBONE_LAYER_PREFIX, DECODER_LAYER_PREFIX,
};
use opensesame_csm::weight_loader::{
    load_csm_from_safetensors, KEY_PROJECTION,
    KEY_BACKBONE_NORM, KEY_DECODER_NORM,
};

// ── Tiny config for tests ─────────────────────────────────────────────────────

/// A minimal CsmConfig for fast tests.
///
/// `backbone_n_layers = 1`, `decoder_n_layers = 1`, tiny dimensions.
fn tiny_cfg() -> CsmConfig {
    CsmConfig {
        backbone_n_layers:   1,
        backbone_d_model:    16,
        backbone_n_heads:    2,
        backbone_n_kv_heads: 2,
        backbone_ffn_dim:    32,
        backbone_rope_base:  10_000.0,
        backbone_rope_scale: 1.0,

        decoder_n_layers:   1,
        decoder_d_model:    8,
        decoder_n_heads:    2,
        decoder_n_kv_heads: 2,
        decoder_ffn_dim:    16,
        decoder_rope_base:  10_000.0,
        decoder_rope_scale: 1.0,

        text_vocab_size:    32,
        audio_vocab_size:   8,   // tiny: 5 + 3 special
        audio_num_codebooks: 4,
        frame_width:        5,   // 4 + 1

        max_seq_len: 64,
    }
}

// ── Build a minimal valid safetensors bytes for tiny_cfg ──────────────────────

/// Build a complete set of correctly-shaped tensors for `tiny_cfg()`.
fn build_valid_st_bytes(cfg: &CsmConfig) -> Vec<u8> {
    let d  = cfg.backbone_d_model;
    let dd = cfg.decoder_d_model;
    let v  = cfg.audio_vocab_size;
    let tv = cfg.text_vocab_size;
    let n_dep = cfg.n_dep_codebooks(); // 3
    let kv_d  = cfg.backbone_n_kv_heads * (d / cfg.backbone_n_heads);
    let dkv_d = cfg.decoder_n_kv_heads  * (dd / cfg.decoder_n_heads);
    let ffn_b = cfg.backbone_ffn_dim;
    let ffn_d = cfg.decoder_ffn_dim;
    let audio_emb_n = cfg.audio_embed_table_size(); // 4 × 8 = 32

    let zero_tv   = vec![0.0f32; tv   * d];
    let zero_av   = vec![0.0f32; audio_emb_n * d];
    let zero_cb0  = vec![0.0f32; v  * d];
    let zero_proj = vec![0.0f32; dd * d];
    let zero_head = vec![0.0f32; n_dep * dd * v];

    // Backbone layer 0
    let bb_q  = vec![0.0f32; d   * d];
    let bb_k  = vec![0.0f32; kv_d * d];
    let bb_v  = vec![0.0f32; kv_d * d];
    let bb_o  = vec![0.0f32; d   * d];
    let bb_w1 = vec![0.0f32; ffn_b * d];
    let bb_w2 = vec![0.0f32; d * ffn_b];
    let bb_w3 = vec![0.0f32; ffn_b * d];
    let bb_sa = vec![0.0f32; d];
    let bb_ml = vec![0.0f32; d];
    let bb_norm = vec![0.0f32; d];

    // Decoder layer 0
    let dc_q  = vec![0.0f32; dd    * dd];
    let dc_k  = vec![0.0f32; dkv_d * dd];
    let dc_v  = vec![0.0f32; dkv_d * dd];
    let dc_o  = vec![0.0f32; dd    * dd];
    let dc_w1 = vec![0.0f32; ffn_d * dd];
    let dc_w2 = vec![0.0f32; dd * ffn_d];
    let dc_w3 = vec![0.0f32; ffn_d * dd];
    let dc_sa = vec![0.0f32; dd];
    let dc_ml = vec![0.0f32; dd];
    let dc_norm = vec![0.0f32; dd];

    let b0 = format!("{BACKBONE_LAYER_PREFIX}0.");
    let d0 = format!("{DECODER_LAYER_PREFIX}0.");

    atlas_model::SafetensorsFile::build_f32(&[
        (KEY_TEXT_EMBEDDINGS,    &[tv, d],           &zero_tv),
        (KEY_AUDIO_EMBEDDINGS,   &[audio_emb_n, d],  &zero_av),
        (KEY_CODEBOOK0_HEAD,     &[v, d],             &zero_cb0),
        (KEY_PROJECTION,         &[dd, d],            &zero_proj),
        (KEY_EMBEDS_PROJECTOR,   &[dd, d],            &zero_proj),
        (KEY_AUDIO_HEAD,         &[n_dep, dd, v],    &zero_head),
        (&format!("{}attn.q_proj.weight",      b0), &[d,    d],    &bb_q),
        (&format!("{}attn.k_proj.weight",      b0), &[kv_d, d],    &bb_k),
        (&format!("{}attn.v_proj.weight",      b0), &[kv_d, d],    &bb_v),
        (&format!("{}attn.output_proj.weight", b0), &[d,    d],    &bb_o),
        (&format!("{}mlp.w1.weight",           b0), &[ffn_b, d],   &bb_w1),
        (&format!("{}mlp.w2.weight",           b0), &[d, ffn_b],   &bb_w2),
        (&format!("{}mlp.w3.weight",           b0), &[ffn_b, d],   &bb_w3),
        (&format!("{}sa_norm.scale",           b0), &[d],          &bb_sa),
        (&format!("{}mlp_norm.scale",          b0), &[d],          &bb_ml),
        (KEY_BACKBONE_NORM,                         &[d],          &bb_norm),
        (&format!("{}attn.q_proj.weight",      d0), &[dd,    dd],   &dc_q),
        (&format!("{}attn.k_proj.weight",      d0), &[dkv_d, dd],   &dc_k),
        (&format!("{}attn.v_proj.weight",      d0), &[dkv_d, dd],   &dc_v),
        (&format!("{}attn.output_proj.weight", d0), &[dd,    dd],   &dc_o),
        (&format!("{}mlp.w1.weight",           d0), &[ffn_d, dd],   &dc_w1),
        (&format!("{}mlp.w2.weight",           d0), &[dd, ffn_d],   &dc_w2),
        (&format!("{}mlp.w3.weight",           d0), &[ffn_d, dd],   &dc_w3),
        (&format!("{}sa_norm.scale",           d0), &[dd],          &dc_sa),
        (&format!("{}mlp_norm.scale",          d0), &[dd],          &dc_ml),
        (KEY_DECODER_NORM,                          &[dd],          &dc_norm),
    ])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// TEST W01: loading a correctly-shaped dummy safetensors succeeds.
#[test]
fn load_valid_safetensors_succeeds() {
    let cfg  = tiny_cfg();
    let bytes = build_valid_st_bytes(&cfg);

    // Write to a temp file
    let path = "/tmp/opensesame_csm_test_valid.safetensors";
    std::fs::write(path, &bytes).unwrap();

    let result = load_csm_from_safetensors(path, &cfg);
    assert!(result.is_ok(), "valid safetensors must load without error: {:?}", result.err());
}

/// TEST W02: non-existent file returns CsmError::Io.
#[test]
fn load_nonexistent_file_is_io_error() {
    let cfg = tiny_cfg();
    let result = load_csm_from_safetensors("/tmp/DOES_NOT_EXIST_opensesame.safetensors", &cfg);
    match result {
        Err(CsmError::Io(_)) => {} // expected
        other => panic!("expected Io error, got: {:?}", other.err()),
    }
}

/// TEST W03: missing text_embeddings tensor returns MissingTensor.
#[test]
fn load_missing_text_embeddings_fails() {
    let cfg  = tiny_cfg();
    let bytes = build_valid_st_bytes(&cfg);
    let mut st = atlas_model::SafetensorsFile::from_bytes(bytes).unwrap();
    // Remove text_embeddings
    st.tensors.retain(|t| t.name != KEY_TEXT_EMBEDDINGS);

    let path = "/tmp/opensesame_csm_test_missing_text_emb.safetensors";
    let rebuilt = atlas_model::SafetensorsFile::build_f32(&[]); // placeholder rebuild
    // Write original bytes but with header modified — instead, just test st directly
    // We can't easily remove from the file, so we test via a fresh build without that key
    let _ = (st, rebuilt, path); // suppress unused

    // Build a safetensors without text_embeddings
    let d  = cfg.backbone_d_model;
    let dd = cfg.decoder_d_model;
    let v  = cfg.audio_vocab_size;
    let n_dep = cfg.n_dep_codebooks();
    let audio_emb_n = cfg.audio_embed_table_size();

    let bytes2 = atlas_model::SafetensorsFile::build_f32(&[
        (KEY_AUDIO_EMBEDDINGS, &[audio_emb_n, d], &vec![0.0f32; audio_emb_n * d]),
        (KEY_CODEBOOK0_HEAD,   &[v, d],            &vec![0.0f32; v * d]),
        (KEY_PROJECTION,       &[dd, d],            &vec![0.0f32; dd * d]),
        (KEY_AUDIO_HEAD,       &[n_dep, dd, v],    &vec![0.0f32; n_dep * dd * v]),
    ]);

    let path2 = "/tmp/opensesame_csm_test_no_text_emb.safetensors";
    std::fs::write(path2, &bytes2).unwrap();

    let result = load_csm_from_safetensors(path2, &cfg);
    assert!(matches!(result, Err(CsmError::MissingTensor(_))),
        "missing text_embeddings should give MissingTensor");
}

/// TEST W04: wrong shape for audio_head returns ShapeMismatch.
#[test]
fn load_wrong_audio_head_shape_fails() {
    let cfg   = tiny_cfg();
    let mut bytes = build_valid_st_bytes(&cfg);

    // Build an alternative with wrong audio_head size (way too small)
    let d   = cfg.backbone_d_model;
    let dd  = cfg.decoder_d_model;
    let v   = cfg.audio_vocab_size;
    let tv  = cfg.text_vocab_size;
    let audio_emb_n = cfg.audio_embed_table_size();
    let n_dep = cfg.n_dep_codebooks();
    let kv_d  = cfg.backbone_n_kv_heads * (d / cfg.backbone_n_heads);
    let dkv_d = cfg.decoder_n_kv_heads  * (dd / cfg.decoder_n_heads);
    let ffn_b = cfg.backbone_ffn_dim;
    let ffn_d = cfg.decoder_ffn_dim;
    let b0 = format!("{BACKBONE_LAYER_PREFIX}0.");
    let d0 = format!("{DECODER_LAYER_PREFIX}0.");

    // Use wrong audio_head size
    let wrong_head = vec![0.0f32; 3]; // obviously wrong
    let bytes_wrong = atlas_model::SafetensorsFile::build_f32(&[
        (KEY_TEXT_EMBEDDINGS,    &[tv, d],            &vec![0.0f32; tv * d]),
        (KEY_AUDIO_EMBEDDINGS,   &[audio_emb_n, d],  &vec![0.0f32; audio_emb_n * d]),
        (KEY_CODEBOOK0_HEAD,     &[v, d],             &vec![0.0f32; v * d]),
        (KEY_PROJECTION,         &[dd, d],            &vec![0.0f32; dd * d]),
        (KEY_EMBEDS_PROJECTOR,   &[dd, d],            &vec![0.0f32; dd * d]),
        (KEY_AUDIO_HEAD,         &[1, 1, 1],          &wrong_head),
        (&format!("{}attn.q_proj.weight",      b0), &[d, d],       &vec![0.0f32; d * d]),
        (&format!("{}attn.k_proj.weight",      b0), &[kv_d, d],    &vec![0.0f32; kv_d * d]),
        (&format!("{}attn.v_proj.weight",      b0), &[kv_d, d],    &vec![0.0f32; kv_d * d]),
        (&format!("{}attn.output_proj.weight", b0), &[d, d],       &vec![0.0f32; d * d]),
        (&format!("{}mlp.w1.weight",           b0), &[ffn_b, d],   &vec![0.0f32; ffn_b * d]),
        (&format!("{}mlp.w2.weight",           b0), &[d, ffn_b],   &vec![0.0f32; d * ffn_b]),
        (&format!("{}mlp.w3.weight",           b0), &[ffn_b, d],   &vec![0.0f32; ffn_b * d]),
        (&format!("{}sa_norm.scale",           b0), &[d],          &vec![0.0f32; d]),
        (&format!("{}mlp_norm.scale",          b0), &[d],          &vec![0.0f32; d]),
        (KEY_BACKBONE_NORM,                         &[d],          &vec![0.0f32; d]),
        (&format!("{}attn.q_proj.weight",      d0), &[dd, dd],     &vec![0.0f32; dd * dd]),
        (&format!("{}attn.k_proj.weight",      d0), &[dkv_d, dd],  &vec![0.0f32; dkv_d * dd]),
        (&format!("{}attn.v_proj.weight",      d0), &[dkv_d, dd],  &vec![0.0f32; dkv_d * dd]),
        (&format!("{}attn.output_proj.weight", d0), &[dd, dd],     &vec![0.0f32; dd * dd]),
        (&format!("{}mlp.w1.weight",           d0), &[ffn_d, dd],  &vec![0.0f32; ffn_d * dd]),
        (&format!("{}mlp.w2.weight",           d0), &[dd, ffn_d],  &vec![0.0f32; dd * ffn_d]),
        (&format!("{}mlp.w3.weight",           d0), &[ffn_d, dd],  &vec![0.0f32; ffn_d * dd]),
        (&format!("{}sa_norm.scale",           d0), &[dd],         &vec![0.0f32; dd]),
        (&format!("{}mlp_norm.scale",          d0), &[dd],         &vec![0.0f32; dd]),
        (KEY_DECODER_NORM,                          &[dd],         &vec![0.0f32; dd]),
    ]);
    let _ = bytes; // suppress unused warning

    let path = "/tmp/opensesame_csm_test_wrong_head.safetensors";
    std::fs::write(path, &bytes_wrong).unwrap();

    let result = load_csm_from_safetensors(path, &cfg);
    assert!(matches!(result, Err(CsmError::ShapeMismatch { .. })),
        "wrong audio_head shape should give ShapeMismatch");
}

/// TEST W05: audio_head loaded correctly — shape [n_dep, d_model, vocab_size].
#[test]
fn audio_head_shape_correct() {
    let cfg  = tiny_cfg();
    let n_dep = cfg.n_dep_codebooks();
    let dd   = cfg.decoder_d_model;
    let v    = cfg.audio_vocab_size;

    let bytes = build_valid_st_bytes(&cfg);
    let path  = "/tmp/opensesame_csm_test_audio_head.safetensors";
    std::fs::write(path, &bytes).unwrap();

    let model = load_csm_from_safetensors(path, &cfg).unwrap();
    assert_eq!(model.depformer.head.n_codebooks, n_dep,
        "audio_head n_codebooks mismatch");
    assert_eq!(model.depformer.head.d_model, dd);
    assert_eq!(model.depformer.head.vocab_size, v);
}

/// TEST W06: inputs_embeds_projector weight shape matches config.
#[test]
fn inputs_embeds_projector_weight_size() {
    // The projector is [decoder_d_model, backbone_d_model] = [dd, d]
    // After loading, model.proj has in_d = backbone_d_model, out_d = decoder_d_model
    let cfg   = tiny_cfg();
    let bytes = build_valid_st_bytes(&cfg);
    let path  = "/tmp/opensesame_csm_test_projector.safetensors";
    std::fs::write(path, &bytes).unwrap();

    let model = load_csm_from_safetensors(path, &cfg).unwrap();
    assert_eq!(model.proj.in_d,  cfg.backbone_d_model,
        "projection in_d must be backbone_d_model");
    assert_eq!(model.proj.out_d, cfg.decoder_d_model,
        "projection out_d must be decoder_d_model");
    assert_eq!(model.proj.weight.len(),
        cfg.backbone_d_model * cfg.decoder_d_model,
        "projection weight count mismatch");
}

/// TEST W07: text_embeddings shape matches config.
#[test]
fn text_embeddings_shape_correct() {
    let cfg   = tiny_cfg();
    let bytes = build_valid_st_bytes(&cfg);
    let path  = "/tmp/opensesame_csm_test_text_emb_shape.safetensors";
    std::fs::write(path, &bytes).unwrap();

    // We validate that loading succeeds (shape was checked by loader)
    let result = load_csm_from_safetensors(path, &cfg);
    assert!(result.is_ok(), "text_embeddings shape must match config");
}

/// TEST W08: audio_embeddings table_size = audio_vocab_size × audio_num_codebooks.
#[test]
fn audio_embeddings_table_size_correct() {
    let cfg = tiny_cfg();
    let expected = cfg.audio_vocab_size * cfg.audio_num_codebooks;
    assert_eq!(cfg.audio_embed_table_size(), expected,
        "audio embed table = {expected}");
    // 4 codebooks × 8 vocab = 32 for tiny config
    assert_eq!(cfg.audio_embed_table_size(), 32);
}
