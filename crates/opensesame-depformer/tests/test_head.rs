// Tests 5–10: CsmAudioHead

use opensesame_depformer::{CsmAudioHead, DepformerConfig};

/// TEST 5: forward output shape matches vocab_size.
#[test]
fn audio_head_output_shape() {
    let (n_dep, d_model, vocab_size) = (7, 1024, 2048);
    let flat = vec![0.0f32; n_dep * vocab_size * d_model];
    let head = CsmAudioHead::from_flat(flat, n_dep, d_model, vocab_size);
    let hidden = vec![0.5f32; d_model];
    let mut logits = vec![0.0f32; vocab_size];
    head.forward(1, &hidden, &mut logits);
    assert_eq!(logits.len(), vocab_size);
}

/// TEST 6: weight tensor has correct count and per-head size.
#[test]
fn audio_head_weight_count() {
    let cfg = DepformerConfig::opensesame_1b();
    let flat = vec![0.0f32; cfg.n_dep_codebooks * cfg.vocab_size * cfg.d_model];
    let head = CsmAudioHead::from_flat(flat, cfg.n_dep_codebooks, cfg.d_model, cfg.vocab_size);
    assert_eq!(head.weights.len(), 31);  // n_dep_codebooks = 31 (CB1..CB31)
    for w in &head.weights {
        assert_eq!(w.len(), cfg.vocab_size * cfg.d_model);  // 2051 * 1024
    }
}

/// TEST 7: forward is deterministic (no hidden random state).
#[test]
fn audio_head_deterministic() {
    let (n_dep, d, v) = (7usize, 64usize, 128usize);  // tiny dims for speed
    let flat: Vec<f32> = (0..n_dep * v * d).map(|i| (i % 17) as f32 * 0.01).collect();
    let head = CsmAudioHead::from_flat(flat, n_dep, d, v);
    let hidden: Vec<f32> = (0..d).map(|i| (i % 5) as f32 * 0.1).collect();
    let mut l1 = vec![0.0f32; v];
    let mut l2 = vec![0.0f32; v];
    head.forward(3, &hidden, &mut l1);
    head.forward(3, &hidden, &mut l2);
    assert_eq!(l1, l2, "same input must produce same output");
}

/// TEST 8: depth=0 panics with "out of [1" message.
#[test]
#[should_panic(expected = "out of [1")]
fn audio_head_depth_zero_panics() {
    let head = CsmAudioHead::zeros(7, 16, 32);
    let mut logits = vec![0.0f32; 32];
    head.forward(0, &vec![0.0f32; 16], &mut logits);
}

/// TEST 9: depth > n_codebooks panics with "out of [1" message.
#[test]
#[should_panic(expected = "out of [1")]
fn audio_head_depth_overflow_panics() {
    let head = CsmAudioHead::zeros(7, 16, 32);
    let mut logits = vec![0.0f32; 32];
    head.forward(8, &vec![0.0f32; 16], &mut logits);  // valid range: 1..=7
}

/// TEST 10: zero weights → zero logits for any non-zero input.
#[test]
fn audio_head_zero_weights_zero_logits() {
    let head = CsmAudioHead::zeros(7, 1024, 2048);
    let hidden = vec![1.0f32; 1024];
    let mut logits = vec![0.0f32; 2048];
    head.forward(1, &hidden, &mut logits);
    assert!(logits.iter().all(|&v| v.abs() < 1e-9));
}
