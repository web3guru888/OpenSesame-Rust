//! External integration tests for CsmConfig (Phase H).

use opensesame_csm::CsmConfig;

/// TEST C01: csm_1b() returns correct backbone layer count.
#[test]
fn csm_1b_backbone_n_layers() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(cfg.backbone_n_layers, 16, "Llama-3.2-1B: 16 layers");
}

/// TEST C02: csm_1b() backbone dim = 2048.
#[test]
fn csm_1b_backbone_d_model() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(cfg.backbone_d_model, 2048);
}

/// TEST C03: frame_width = audio_num_codebooks + 1.
#[test]
fn csm_1b_frame_width_invariant() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(cfg.frame_width, cfg.audio_num_codebooks + 1,
        "frame_width must be audio_num_codebooks + 1");
    assert_eq!(cfg.frame_width, 33);
}

/// TEST C04: audio_embed_table_size = audio_vocab_size × audio_num_codebooks.
#[test]
fn csm_1b_audio_embed_table_size() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(
        cfg.audio_embed_table_size(),
        cfg.audio_vocab_size * cfg.audio_num_codebooks,
        "embed table = vocab × n_codebooks"
    );
    // 32 × 2051 = 65_632
    assert_eq!(cfg.audio_embed_table_size(), 65_632);
}

/// TEST C05: audio_vocab_size = 2051 (EOS=0, normal=1..2048, pad=2050).
#[test]
fn csm_1b_audio_vocab_size() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(cfg.audio_vocab_size, 2051);
}

/// TEST C06: n_dep_codebooks = audio_num_codebooks - 1 = 31.
#[test]
fn csm_1b_n_dep_codebooks() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(cfg.n_dep_codebooks(), 31,
        "depth decoder generates CB1..CB31 (backbone handles CB0)");
    assert_eq!(cfg.n_dep_codebooks(), cfg.audio_num_codebooks - 1);
}

/// TEST C07: csm_1b() decoder (Llama-100M) dimensions.
#[test]
fn csm_1b_decoder_dims() {
    let cfg = CsmConfig::csm_1b();
    assert_eq!(cfg.decoder_n_layers,   4);
    assert_eq!(cfg.decoder_d_model,    1024);
    assert_eq!(cfg.decoder_n_heads,    8);
    assert_eq!(cfg.decoder_n_kv_heads, 2);
    assert_eq!(cfg.decoder_ffn_dim,    8192);
}

/// TEST C08: csm_1b() max_seq_len sanity.
#[test]
fn csm_1b_max_seq_len() {
    let cfg = CsmConfig::csm_1b();
    assert!(cfg.max_seq_len >= 1024, "max_seq_len should be at least 1024");
    assert_eq!(cfg.max_seq_len, 2048);
}
