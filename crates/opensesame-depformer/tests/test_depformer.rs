// Tests 11–18: Depformer forward pass and generation

use opensesame_depformer::{Depformer, DepformerConfig};

/// Tiny depformer config for fast testing (exercises all code paths).
fn tiny_cfg() -> DepformerConfig {
    DepformerConfig {
        n_layers:         1,
        d_model:          64,
        n_heads:          4,
        n_kv_heads:       2,
        head_dim:         16,    // 64 / 4
        ffn_dim:          128,
        n_codebooks:      8,
        n_dep_codebooks:  7,
        vocab_size:       32,
        d_backbone:       64,
        rope_base:        10_000.0,
        norm_eps:         1e-5,
    }
}

/// TEST 11: generate returns exactly n_dep_codebooks tokens.
#[test]
fn generate_correct_token_count() {
    let cfg = tiny_cfg();
    let n_dep = cfg.n_dep_codebooks;
    let d = cfg.d_model;
    let mut dep = Depformer::new(cfg);
    let codes = dep.generate_depth_sequence(
        vec![0.0f32; d],
        |_, _| vec![0.0f32; d],
        1.0, 0, 0,
    );
    assert_eq!(codes.len(), n_dep);
}

/// TEST 12: all generated tokens are within [0, vocab_size).
#[test]
fn generated_tokens_in_vocab_range() {
    let cfg = tiny_cfg();
    let vocab = cfg.vocab_size;
    let d = cfg.d_model;
    let mut dep = Depformer::new(cfg);
    let codes = dep.generate_depth_sequence(
        vec![0.0f32; d],
        |_, _| vec![0.0f32; d],
        0.0, 0, 0,   // temperature=0 (greedy)
    );
    for &code in &codes {
        assert!((code as usize) < vocab, "token {code} >= vocab_size {vocab}");
    }
}

/// TEST 13: reset_for_frame sets pos back to 0.
#[test]
fn depformer_reset_clears_pos() {
    let cfg = tiny_cfg();
    let d = cfg.d_model;
    let mut dep = Depformer::new(cfg);
    // Advance pos manually
    dep.transformer.forward_hidden_raw(vec![0.0f32; d]);
    dep.transformer.forward_hidden_raw(vec![0.0f32; d]);
    assert!(dep.transformer.pos() >= 2, "pos should be at least 2");
    // Reset
    dep.reset_for_frame();
    assert_eq!(dep.transformer.pos(), 0, "pos should be 0 after reset");
}

/// TEST 14: two consecutive frames each complete without error.
#[test]
fn two_frames_independent() {
    let cfg = tiny_cfg();
    let d = cfg.d_model;
    let n_dep = cfg.n_dep_codebooks;
    let mut dep = Depformer::new(cfg);
    let c1 = dep.generate_depth_sequence(vec![0.0f32; d], |_, _| vec![0.0f32; d], 0.0, 0, 0);
    let c2 = dep.generate_depth_sequence(vec![0.0f32; d], |_, _| vec![0.0f32; d], 0.0, 0, 0);
    assert_eq!(c1.len(), n_dep);
    assert_eq!(c2.len(), n_dep);
}

/// TEST 15: greedy decode is deterministic across two identically initialised models.
#[test]
fn depformer_greedy_deterministic() {
    let cfg = tiny_cfg();
    let d = cfg.d_model;
    let proj_h: Vec<f32> = (0..d).map(|i| (i % 7) as f32 * 0.01).collect();
    let embed_fn = |_: usize, code: u32| -> Vec<f32> {
        (0..d).map(|i| (i as f32 + code as f32) * 0.001).collect()
    };
    let mut dep1 = Depformer::new(cfg.clone());
    let mut dep2 = Depformer::new(cfg.clone());
    let c1 = dep1.generate_depth_sequence(proj_h.clone(), embed_fn, 0.0, 0, 10);
    let c2 = dep2.generate_depth_sequence(proj_h, embed_fn, 0.0, 0, 10);
    assert_eq!(c1, c2, "greedy decode must be deterministic given identical weights");
}

/// TEST 16: different backbone hiddens produce different codes (statistical test).
///
/// With random weights (seeded), different inputs produce different outputs with
/// overwhelming probability (~(1/vocab)^n_dep ≈ 0 chance of spurious equality).
#[test]
fn different_backbone_hidden_different_codes() {
    let cfg = tiny_cfg();
    let d = cfg.d_model;
    let mut dep = Depformer::new(cfg);
    let codes_zero = dep.generate_depth_sequence(
        vec![0.0f32; d],
        |_, _| vec![0.0f32; d],
        0.0, 0, 0,
    );
    let codes_one = dep.generate_depth_sequence(
        vec![1.0f32; d],
        |_, _| vec![0.0f32; d],
        0.0, 0, 0,
    );
    assert_ne!(codes_zero, codes_one,
        "different backbone hiddens should produce different codes with random weights");
}

/// TEST 17: KV cache capacity (max_seq_len) equals n_codebooks.
#[test]
fn depformer_kv_cache_capacity() {
    let cfg = DepformerConfig::opensesame_1b();
    let mc = cfg.to_model_config();
    assert_eq!(mc.max_seq_len, 8, "KV cache should hold exactly n_codebooks=8 positions");
}

/// TEST 18: after one full frame, the position counter equals n_codebooks.
///
/// One depth-0 step (backbone proj) + 7 depth steps = 8 positions written.
#[test]
fn depformer_kv_fills_n_codebooks_positions() {
    let cfg = tiny_cfg();
    let d = cfg.d_model;
    let n_cb = cfg.n_codebooks;
    let mut dep = Depformer::new(cfg);
    dep.generate_depth_sequence(
        vec![0.1f32; d],
        |_, code| (0..d).map(|i| (i as f32 + code as f32) * 0.001).collect(),
        0.0, 0, 5,
    );
    assert_eq!(
        dep.transformer.pos(), n_cb,
        "pos should be {n_cb} (= 1 backbone + 7 depth steps)"
    );
}
