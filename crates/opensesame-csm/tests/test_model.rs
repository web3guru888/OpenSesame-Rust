//! External integration tests for CsmModel (Phase H).

use opensesame_csm::{CsmModel, CsmModelConfig};

// ── Construction ──────────────────────────────────────────────────────────────

/// TEST M01: CsmModel::new_tiny() constructs without panic.
#[test]
fn csm_model_new_tiny_no_panic() {
    let _m = CsmModel::new_tiny();
}

/// TEST M02: CsmModel::new(config) constructs without panic.
#[test]
fn csm_model_new_config_no_panic() {
    let cfg = CsmModelConfig::tiny();
    let _m = CsmModel::new(cfg);
}

// ── Embed frame shape ─────────────────────────────────────────────────────────

/// TEST M03: embed_frame output has shape [backbone_dim].
#[test]
fn embed_frame_shape_is_backbone_dim() {
    let m = CsmModel::new_tiny();
    let d = m.config.backbone_dim;
    let audio = vec![0u32; m.config.n_codebooks];
    let embed = m.embed_frame(None, &audio);
    assert_eq!(embed.len(), d, "embed shape must be backbone_dim = {d}");
}

/// TEST M04: masked text frame uses text embedding for col 32.
#[test]
fn embed_frame_text_only_nonzero() {
    let m = CsmModel::new_tiny();
    // Text token 1 → some non-zero embedding (with random weights)
    let audio = vec![u32::MAX; m.config.n_codebooks]; // all-padding audio
    let embed_t1 = m.embed_frame(Some(1), &audio);
    let embed_t2 = m.embed_frame(Some(2), &audio);
    // Different tokens should produce different embeddings (with very high probability)
    assert_ne!(embed_t1, embed_t2,
        "different text tokens should give different embeddings");
}

/// TEST M05: masked audio frame uses sum of audio embeddings.
#[test]
fn embed_frame_audio_contributes() {
    let m = CsmModel::new_tiny();
    let n_cb = m.config.n_codebooks;
    // Compare: no audio vs some audio
    let no_audio  = m.embed_frame(None, &vec![u32::MAX; n_cb]);
    let with_audio = m.embed_frame(None, &vec![0u32; n_cb]);
    // With random weights, these should almost certainly differ
    assert_ne!(no_audio, with_audio,
        "audio tokens should contribute to embedding (non-zero weights)");
}

// ── Backbone forward ──────────────────────────────────────────────────────────

/// TEST M06: backbone_forward output shapes are correct.
#[test]
fn backbone_forward_shapes() {
    let mut m = CsmModel::new_tiny();
    let d = m.config.backbone_dim;
    let embed = vec![0.01f32; d];
    let (cb0_logits, hidden) = m.backbone_forward(&embed, 1);
    assert_eq!(cb0_logits.len(), m.config.audio_vocab,
        "CB0 logits must have audio_vocab entries");
    assert_eq!(hidden.len(), d,
        "backbone hidden must be backbone_dim");
}

/// TEST M07: backbone handles multi-frame input.
#[test]
fn backbone_forward_multi_frame() {
    let mut m = CsmModel::new_tiny();
    let d = m.config.backbone_dim;
    let embeds = vec![0.01f32; d * 5];
    let (logits, hidden) = m.backbone_forward(&embeds, 5);
    assert_eq!(logits.len(), m.config.audio_vocab);
    assert_eq!(hidden.len(), d);
}

// ── Frame code generation ─────────────────────────────────────────────────────

/// TEST M08: generate_frame_codes returns n_codebooks tokens.
#[test]
fn generate_frame_codes_length() {
    let mut m = CsmModel::new_tiny();
    let d = m.config.backbone_dim;
    let hidden = vec![0.1f32; d];
    let logits = vec![1.0f32; m.config.audio_vocab];
    let codes = m.generate_frame_codes(&hidden, &logits);
    assert_eq!(codes.len(), m.config.n_codebooks,
        "must return exactly n_codebooks tokens");
}

/// TEST M09: all generated tokens are in [0, audio_vocab).
#[test]
fn generate_frame_codes_all_in_range() {
    let mut m = CsmModel::new_tiny();
    let d = m.config.backbone_dim;
    let hidden = vec![0.5f32; d];
    let logits: Vec<f32> = (0..m.config.audio_vocab).map(|i| i as f32 * 0.01).collect();
    let codes = m.generate_frame_codes(&hidden, &logits);
    for &c in &codes {
        assert!((c as usize) < m.config.audio_vocab,
            "token {c} must be < audio_vocab {}", m.config.audio_vocab);
    }
}

// ── Full generation ───────────────────────────────────────────────────────────

/// TEST M10: generate returns correct code shape for n frames.
#[test]
fn generate_output_codes_shape() {
    let mut m = CsmModel::new_tiny();
    let n = 3;
    let out = m.generate(&[], &[], n, 1.0, 4);
    assert_eq!(out.codes.len(), m.config.n_codebooks,
        "codes must have n_codebooks rows");
    for (cb, row) in out.codes.iter().enumerate() {
        assert_eq!(row.len(), n,
            "codes[{cb}] must have {n} frames");
    }
}

/// TEST M11: two greedy (temperature=0) calls give identical codes.
#[test]
fn generate_greedy_is_deterministic() {
    let mut m = CsmModel::new_tiny();
    let out1 = m.generate(&[1], &[], 2, 0.0, 0);
    let out2 = m.generate(&[1], &[], 2, 0.0, 0);
    assert_eq!(out1.codes, out2.codes,
        "greedy generation must be deterministic");
}

/// TEST M12: generate with context runs without panic.
#[test]
fn generate_with_context_no_panic() {
    let mut m = CsmModel::new_tiny();
    let ctx = vec![0.01f32; m.config.frame_samples];
    let out = m.generate(&[], &ctx, 1, 1.0, 1);
    assert_eq!(out.codes.len(), m.config.n_codebooks);
}
