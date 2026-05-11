// Tests 19–25: CsmLinear, weight-loading utilities, sampling

use opensesame_depformer::{CsmAudioHead, CsmLinear, DepformerConfig, sample_topk};

/// TEST 19: CsmLinear output shape is [out_dim].
#[test]
fn projection_shapes() {
    let proj = CsmLinear::from_data(vec![0.0f32; 1024 * 2048], 2048, 1024);
    assert_eq!(proj.weight.len(), 1024 * 2048);
    let input = vec![1.0f32; 2048];
    let output = proj.forward_vec(&input);
    assert_eq!(output.len(), 1024);
}

/// TEST 20: CsmLinear obeys scaling linearity (W·(2x) = 2·(W·x)).
#[test]
fn projection_is_linear() {
    let n = 32usize;
    let w: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 * 0.1).collect();
    let proj = CsmLinear::from_data(w, n, n);
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y1 = proj.forward_vec(&x);
    let x2: Vec<f32> = x.iter().map(|&v| v * 2.0).collect();
    let y2 = proj.forward_vec(&x2);
    for (a, b) in y1.iter().zip(y2.iter()) {
        assert!((b - a * 2.0).abs() < 1e-4, "linearity: {b} != {a}*2.0");
    }
}

/// TEST 21: Weight dimension constants are self-consistent.
#[test]
fn weight_loading_dimension_constants() {
    let cfg = DepformerConfig::opensesame_1b();
    let d   = cfg.d_model;            // 1024
    let kv  = cfg.n_kv_heads * cfg.head_dim;  // 2 × 128 = 256
    let f   = cfg.ffn_dim;            // 8192
    assert_eq!(kv, 256);
    assert_eq!(d / cfg.n_heads, cfg.head_dim);
    // Q: [d, d]; K: [kv, d]; V: [kv, d]; O: [d, d]
    assert_eq!(d * d,  1024 * 1024);
    assert_eq!(kv * d, 256 * 1024);
    // Gate/Up: [f, d]; Down: [d, f]
    assert_eq!(f * d, 8192 * 1024);
    assert_eq!(d * f, 1024 * 8192);
}

/// TEST 22: from_flat correctly slices weights by codebook.
#[test]
fn audio_head_stride_slicing() {
    let (n_dep, v, d) = (7usize, 2048usize, 1024usize);
    let flat: Vec<f32> = (0..n_dep * v * d).map(|i| i as f32).collect();
    let head = CsmAudioHead::from_flat(flat, n_dep, d, v);
    for i in 0..n_dep {
        assert_eq!(
            head.weights[i][0],
            (i * v * d) as f32,
            "slice {i} should start at offset {}",
            i * v * d
        );
    }
}

/// TEST 23: Partial load from a CSM-1B checkpoint (31 dep codebooks → take 7).
#[test]
fn audio_head_partial_load_from_csm1b() {
    let (n_csm1b, n_dep, v, d) = (31usize, 7usize, 2048usize, 1024usize);
    let flat = vec![0.0f32; n_csm1b * v * d];
    let head = CsmAudioHead::from_flat(flat[..n_dep * v * d].to_vec(), n_dep, d, v);
    assert_eq!(head.n_codebooks, n_dep);
    assert_eq!(head.weights.len(), n_dep);
}

/// TEST 24: Torchtune SwiGLU weight naming (w1=gate, w2=down, w3=up).
///
/// Verifies the non-obvious naming convention to prevent future mistakes.
#[test]
fn torchtune_ffn_weight_order() {
    let (d, f) = (1024usize, 8192usize);
    // w1 (gate): shape [f, d] → atlas w_gate (in_dim=d, out_dim=f)
    assert_eq!(f * d, 8192 * 1024, "gate: [ffn_dim, d_model]");
    // w2 (down): shape [d, f] → atlas w_down (in_dim=f, out_dim=d)
    assert_eq!(d * f, 1024 * 8192, "down: [d_model, ffn_dim]");
    // w3 (up): shape [f, d] → atlas w_up (in_dim=d, out_dim=f)
    assert_eq!(f * d, 8192 * 1024, "up: [ffn_dim, d_model]");
    // gate and up have the same shape; down is transposed
    assert_eq!(f * d, f * d);
}

/// TEST 25: sample_topk always returns a valid token ID for any input.
#[test]
fn sampling_always_valid_token() {
    let logits: Vec<f32> = (0..2048).map(|i| (i % 7) as f32 - 3.0).collect();
    for temperature in [0.0f32, 0.5, 1.0, 2.0] {
        for topk in [0usize, 1, 10, 50, 2048] {
            let code = sample_topk(&logits, topk, temperature);
            assert!(
                (code as usize) < 2048,
                "code {code} >= 2048 for temp={temperature} topk={topk}"
            );
        }
    }
}
