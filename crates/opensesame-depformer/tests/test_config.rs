// Tests 1–4: DepformerConfig

use opensesame_depformer::DepformerConfig;

/// TEST 1: All CSM-1B constants are correct.
#[test]
fn depformer_config_constants() {
    let cfg = DepformerConfig::opensesame_1b();
    assert_eq!(cfg.n_layers,         4);
    assert_eq!(cfg.d_model,          1024);
    assert_eq!(cfg.n_heads,          8);
    assert_eq!(cfg.n_kv_heads,       2);
    assert_eq!(cfg.head_dim,         128);    // 1024 / 8
    assert_eq!(cfg.ffn_dim,          8192);
    assert_eq!(cfg.n_codebooks,      8);      // CB0..CB7
    assert_eq!(cfg.n_dep_codebooks,  7);      // CB1..CB7
    assert_eq!(cfg.vocab_size,       2048);
    assert_eq!(cfg.d_backbone,       2048);
}

/// TEST 2: to_model_config round-trips all fields.
#[test]
fn depformer_to_model_config() {
    let cfg = DepformerConfig::opensesame_1b();
    let mc = cfg.to_model_config();
    assert_eq!(mc.d_model,     1024);
    assert_eq!(mc.n_layers,    4);
    assert_eq!(mc.n_heads,     8);
    assert_eq!(mc.n_kv_heads,  2);
    assert_eq!(mc.ffn_hidden,  8192);
    assert_eq!(mc.max_seq_len, 8);    // n_codebooks
    assert!((mc.rope_theta - 500_000.0).abs() < 1.0);
}

/// TEST 3: GQA group size is 4.
#[test]
fn depformer_gqa_groups() {
    let cfg = DepformerConfig::opensesame_1b();
    let groups = cfg.n_heads / cfg.n_kv_heads;
    assert_eq!(groups, 4);
}

/// TEST 4: head_dim = d_model / n_heads.
#[test]
fn depformer_head_dim_consistent() {
    let cfg = DepformerConfig::opensesame_1b();
    assert_eq!(cfg.head_dim, cfg.d_model / cfg.n_heads);
    assert_eq!(cfg.head_dim, 128);
}
