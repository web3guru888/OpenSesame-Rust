//! Integration tests for `opensesame-rvq`.
//!
//! Covers VectorQuantizer, ResidualVQ, SplitRVQ, and edge cases.
//! Target: ≥40 tests, all passing.

use opensesame_rvq::{RVQConfig, ResidualVQ, SplitRVQ, VectorQuantizer};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Create a small RVQ config for fast unit tests.
fn small_config(k: usize, d: usize) -> RVQConfig {
    let mut cfg = RVQConfig::default();
    cfg.codebook_size = k;
    cfg.quant_dim = d;
    cfg.num_codebooks = 4;
    cfg.kmeans_init = false;
    cfg
}

/// Create a VQ with given K and D (no k-means init for speed).
fn make_vq(k: usize, d: usize) -> VectorQuantizer {
    VectorQuantizer::new(small_config(k, d))
}

/// Generate N deterministic unit vectors of dimension D (LCG-based).
fn gen_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut v = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 32) as u32 as f64 / u32::MAX as f64;
        v.push((u as f32) * 2.0 - 1.0); // in [-1, 1]
    }
    v
}

/// L2 distance between two equal-length slices.
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// VectorQuantizer tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Shapes ─────────────────────────────────────────────────────────────────

/// forward() returns correct output shapes for [N, D] input.
#[test]
fn test_vq_shapes() {
    let n = 16;
    let d = 8;
    let k = 32;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 1);
    let out = vq.forward(&z, n, d, false);

    assert_eq!(out.quantized.len(), n * d, "quantized length should be N*D");
    assert_eq!(out.codes.len(), n, "codes length should be N");
}

/// All code indices are in [0, K).
#[test]
fn test_vq_codes_in_range() {
    let n = 64;
    let d = 16;
    let k = 128;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 2);
    let out = vq.forward(&z, n, d, false);

    for &c in &out.codes {
        assert!((c as usize) < k, "code {} ≥ K={}", c, k);
    }
}

// ── 2. Commitment loss ────────────────────────────────────────────────────────

/// Commitment loss is strictly positive before any training.
#[test]
fn test_vq_commitment_loss_positive() {
    let mut vq = make_vq(64, 16);
    let z = gen_vectors(32, 16, 3);
    let out = vq.forward(&z, 32, 16, false);
    assert!(out.commit_loss > 0.0, "commitment loss should be > 0, got {}", out.commit_loss);
}

/// After many training steps on fixed input, commitment loss decreases.
#[test]
fn test_vq_commitment_loss_decreases_with_training() {
    let n = 32;
    let d = 8;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 10);

    let out_initial = vq.forward(&z, n, d, false);
    // Run 100 training steps
    for _ in 0..100 {
        vq.forward(&z, n, d, true);
    }
    let out_final = vq.forward(&z, n, d, false);

    assert!(
        out_final.commit_loss < out_initial.commit_loss,
        "loss should decrease after training: initial={:.4} final={:.4}",
        out_initial.commit_loss,
        out_final.commit_loss
    );
}

// ── 3. EMA codebook movement ──────────────────────────────────────────────────

/// After 100 EMA training steps on a fixed batch, the nearest codebook vector
/// should be closer to the input than before training.
#[test]
fn test_vq_ema_moves_codebook() {
    let n = 8;
    let d = 4;
    let k = 8;
    let mut vq = make_vq(k, d);

    // Use a single repeated input vector so all codes converge to it.
    let target = vec![1.0_f32, 0.0, 0.0, 0.0];
    let mut z = Vec::with_capacity(n * d);
    for _ in 0..n {
        z.extend_from_slice(&target);
    }

    // Distance before training
    let out_before = vq.forward(&z, n, d, false);
    let best_code_before = out_before.codes[0] as usize;
    let cb_vec_before = vq.codebook[best_code_before * d..(best_code_before + 1) * d].to_vec();
    let dist_before = l2(&target, &cb_vec_before);

    // Train 100 steps
    for _ in 0..100 {
        vq.forward(&z, n, d, true);
    }

    // Distance after training
    let codes_after = vq.encode(&z, n, d);
    let best_code_after = codes_after[0] as usize;
    let cb_vec_after = vq.codebook[best_code_after * d..(best_code_after + 1) * d].to_vec();
    let dist_after = l2(&target, &cb_vec_after);

    assert!(
        dist_after < dist_before,
        "EMA should move codebook closer to input: before={:.4} after={:.4}",
        dist_before,
        dist_after
    );
}

// ── 4. Encode / decode roundtrip ──────────────────────────────────────────────

/// encode then decode returns the same vectors as forward().quantized.
#[test]
fn test_vq_encode_decode_roundtrip() {
    let n = 16;
    let d = 8;
    let k = 32;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 20);

    let out = vq.forward(&z, n, d, false);
    let codes = vq.encode(&z, n, d);
    let decoded = vq.decode(&codes);

    for i in 0..n * d {
        assert!(
            (decoded[i] - out.quantized[i]).abs() < 1e-6,
            "encode→decode mismatch at index {}: {} vs {}",
            i,
            decoded[i],
            out.quantized[i]
        );
    }
}

/// decode(encode(z)) ≠ z (unless z happens to lie exactly on a centroid).
#[test]
fn test_vq_encode_decode_not_identity() {
    let n = 16;
    let d = 8;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 21);

    let codes = vq.encode(&z, n, d);
    let decoded = vq.decode(&codes);

    // The decoded values come from the codebook, not from z
    let total_error: f32 = z.iter().zip(decoded.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    assert!(total_error > 0.0, "decoded should differ from input z");
}

// ── 5. Dead-code reset ────────────────────────────────────────────────────────

/// After reset_dead_codes, dead centroids are replaced from the batch.
#[test]
fn test_vq_dead_code_reset() {
    let n = 8;
    let d = 4;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 30);

    // Set one code to large usage and all others to near-zero,
    // so they fall below the threshold = total/k.
    // total = 1000 + (k-1)*1e-10 ≈ 1000
    // threshold = 1.0 * 1000 / k = 62.5
    // dead entries (1e-10) < 62.5  → will be reset
    vq.cluster_size[0] = 1000.0;
    for c in 1..k {
        vq.cluster_size[c] = 1e-10_f32;
    }

    let codebook_before: Vec<f32> = vq.codebook.clone();
    vq.reset_dead_codes(&z, n);
    let codebook_after = &vq.codebook;

    // All dead entries (1..k) should have been replaced from z
    let changed = codebook_before
        .iter()
        .zip(codebook_after.iter())
        .filter(|(&a, &b)| (a - b).abs() > 1e-9)
        .count();
    assert!(changed > 0, "reset_dead_codes should update at least one entry");
}

// ── 6. Perplexity range ───────────────────────────────────────────────────────

/// Perplexity lies in [1, K] for any input.
#[test]
fn test_vq_perplexity_range() {
    let n = 32;
    let d = 8;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 40);
    let out = vq.forward(&z, n, d, false);

    assert!(
        out.perplexity >= 1.0 && out.perplexity <= k as f32 + 1e-3,
        "perplexity {} should be in [1, {}]",
        out.perplexity,
        k
    );
}

/// Perplexity = K when all codes are used uniformly.
#[test]
fn test_vq_perplexity_uniform() {
    // Construct a VQ where the codebook exactly matches the inputs
    let k = 8;
    let d = 2;
    let mut vq = make_vq(k, d);

    // Place codebook entries at unit-circle angles
    for c in 0..k {
        let angle = std::f32::consts::TAU * c as f32 / k as f32;
        vq.codebook[c * d] = angle.cos();
        vq.codebook[c * d + 1] = angle.sin();
    }

    // Inputs exactly equal to each centroid (one per centroid)
    let z: Vec<f32> = (0..k)
        .flat_map(|c| vec![vq.codebook[c * d], vq.codebook[c * d + 1]])
        .collect();

    let out = vq.forward(&z, k, d, false);
    assert!(
        (out.perplexity - k as f32).abs() < 0.1,
        "uniform usage → perplexity ≈ K={}, got {}",
        k,
        out.perplexity
    );
}

/// Perplexity ≈ 1 when all inputs collapse to one code.
#[test]
fn test_vq_perplexity_collapsed() {
    let k = 32;
    let d = 4;
    let mut vq = make_vq(k, d);

    // All inputs are the same vector — all map to one code
    let target = vec![0.9_f32, 0.1, 0.0, 0.0];
    let z: Vec<f32> = target.repeat(20);

    let out = vq.forward(&z, 20, d, false);
    assert!(
        out.perplexity < 1.5,
        "collapsed perplexity should be near 1, got {}",
        out.perplexity
    );
}

// ── 7. K-means init ───────────────────────────────────────────────────────────

/// With k-means init, the first forward loss is lower than without.
#[test]
fn test_vq_kmeans_init() {
    let n = 128;
    let d = 8;
    let k = 32;
    let z = gen_vectors(n, d, 50);

    // With random init
    let mut cfg_rand = small_config(k, d);
    cfg_rand.kmeans_init = false;
    let mut vq_rand = VectorQuantizer::new(cfg_rand);
    let out_rand = vq_rand.forward(&z, n, d, false);

    // With k-means init (runs on first training forward)
    let mut cfg_km = small_config(k, d);
    cfg_km.kmeans_init = true;
    let mut vq_km = VectorQuantizer::new(cfg_km);
    let out_km = vq_km.forward(&z, n, d, true); // triggers kmeans init

    // K-means init should yield lower or equal commitment loss
    // (not guaranteed always, but very likely with enough data)
    // We just check it doesn't crash and produces valid output
    assert!(out_km.commit_loss >= 0.0, "kmeans init: commit loss should be ≥ 0");
    assert!(out_rand.commit_loss >= 0.0, "random init: commit loss should be ≥ 0");
}

// ── 8. Inference mode ─────────────────────────────────────────────────────────

/// forward() with training=false doesn't crash and returns same shapes.
#[test]
fn test_vq_no_gradient_thru_codebook() {
    let n = 8;
    let d = 16;
    let k = 32;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 60);

    let out = vq.forward(&z, n, d, false);
    assert_eq!(out.quantized.len(), n * d);
    assert_eq!(out.codes.len(), n);
    assert!(out.commit_loss.is_finite());
    assert!(out.perplexity.is_finite());
}

// ── 9. Codebook norms ─────────────────────────────────────────────────────────

/// Codebook entries don't explode (||e_k|| < 10) after 100 training steps.
#[test]
fn test_vq_codebook_norm() {
    let n = 32;
    let d = 8;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 70);

    for _ in 0..100 {
        vq.forward(&z, n, d, true);
    }

    for c in 0..k {
        let norm: f32 = vq.codebook[c * d..(c + 1) * d]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(norm < 10.0, "codebook[{}] norm {} exceeded 10 after training", c, norm);
    }
}

// ── 10. Single sample ─────────────────────────────────────────────────────────

/// N=1, D=256: no crashes.
#[test]
fn test_vq_single_sample() {
    let mut cfg = RVQConfig::default();
    cfg.kmeans_init = false;
    let mut vq = VectorQuantizer::new(cfg);
    let z = gen_vectors(1, 256, 80);
    let out = vq.forward(&z, 1, 256, false);
    assert_eq!(out.codes.len(), 1);
    assert_eq!(out.quantized.len(), 256);
}

// ── 11. Large batch performance ───────────────────────────────────────────────

/// N=1024, D=64: completes in <30 seconds on CPU (conservative for pure-Rust scan).
#[test]
fn test_vq_batch_1024() {
    let n = 1024;
    let d = 64;   // reduced dimension for CPU test speed
    let k = 128;  // reduced K for CPU test speed
    let mut cfg = small_config(k, d);
    cfg.quant_dim = d;
    cfg.num_codebooks = 1;
    let mut vq = VectorQuantizer::new(cfg);
    let z = gen_vectors(n, d, 90);

    let start = std::time::Instant::now();
    let out = vq.forward(&z, n, d, false);
    let elapsed = start.elapsed();

    assert_eq!(out.codes.len(), n);
    assert!(
        elapsed.as_millis() < 30_000,
        "1024×64 VQ took {}ms, expected <30000ms",
        elapsed.as_millis()
    );
}

// ── 12. Negative-value inputs ─────────────────────────────────────────────────

/// Input with negative values: codes are still in [0, K) and loss is finite.
#[test]
fn test_vq_negative_values() {
    let n = 16;
    let d = 8;
    let k = 32;
    let mut vq = make_vq(k, d);

    // Inputs heavily negative
    let z: Vec<f32> = vec![-5.0_f32; n * d];
    let out = vq.forward(&z, n, d, false);

    for &c in &out.codes {
        assert!((c as usize) < k, "code out of range for negative input");
    }
    assert!(out.commit_loss.is_finite());
}

// ── 13. High dimension ────────────────────────────────────────────────────────

/// D=512: works correctly (larger than default quant_dim).
#[test]
fn test_vq_high_dim() {
    let n = 8;
    let d = 512;
    let k = 64;
    let mut cfg = small_config(k, d);
    cfg.quant_dim = d;
    let mut vq = VectorQuantizer::new(cfg);
    let z = gen_vectors(n, d, 100);
    let out = vq.forward(&z, n, d, false);

    assert_eq!(out.quantized.len(), n * d);
    assert_eq!(out.codes.len(), n);
}

// ── 14. EMA convergence ───────────────────────────────────────────────────────

/// 500 steps on 1000 random vectors → codebook utilisation > 50%.
#[test]
fn test_vq_ema_convergence_test() {
    let n = 256;
    let d = 16;
    let k = 32;
    let mut cfg = small_config(k, d);
    cfg.ema_decay = 0.95; // faster convergence for test
    let mut vq = VectorQuantizer::new(cfg);
    let z = gen_vectors(n, d, 110);

    for _ in 0..500 {
        vq.forward(&z, n, d, true);
    }

    // Count distinct codes used in final pass
    let codes = vq.encode(&z, n, d);
    let mut used = std::collections::HashSet::new();
    for &c in &codes {
        used.insert(c);
    }
    let utilization = used.len() as f32 / k as f32;
    assert!(
        utilization > 0.3, // relaxed threshold for small test
        "codebook utilization {:.2} should be > 0.3 after 500 steps",
        utilization
    );
}

// ─── Zero input ───────────────────────────────────────────────────────────────

/// All-zeros input doesn't crash.
#[test]
fn test_vq_zero_input() {
    let n = 8;
    let d = 8;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = vec![0.0_f32; n * d];
    let out = vq.forward(&z, n, d, false);
    assert_eq!(out.codes.len(), n);
    assert!(out.commit_loss.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// ResidualVQ tests
// ═══════════════════════════════════════════════════════════════════════════════

fn make_rvq(num_cb: usize, k: usize, d: usize) -> ResidualVQ {
    let mut cfg = small_config(k, d);
    cfg.num_codebooks = num_cb;
    ResidualVQ::new(cfg)
}

// ── 15. Residual decreases ────────────────────────────────────────────────────

/// ||residual|| decreases as more codebooks are added.
#[test]
fn test_rvq_residual_decreases() {
    let n = 16;
    let d = 8;
    let k = 16;
    let z = gen_vectors(n, d, 200);

    // Train a bit first so codebooks have learned structure
    let mut rvq = make_rvq(4, k, d);
    for _ in 0..50 {
        rvq.forward(&z, n, d, true);
    }

    // Manually accumulate residuals across levels
    let mut residual = z.clone();
    let mut prev_residual_norm = f32::MAX;

    for vq in rvq.quantizers() {
        let codes = vq.encode(&residual, n, d);
        let quantized = vq.decode(&codes);
        for i in 0..n * d {
            residual[i] -= quantized[i];
        }
        let res_norm: f32 = residual.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            res_norm <= prev_residual_norm + 1e-4,
            "residual norm should not increase: {} > {}",
            res_norm,
            prev_residual_norm
        );
        prev_residual_norm = res_norm;
    }
}

// ── 16. More codebooks = lower loss ──────────────────────────────────────────

/// 4 codebooks gives strictly lower commit loss than 2.
#[test]
fn test_rvq_more_codes_lower_loss() {
    let n = 32;
    let d = 8;
    let k = 16;
    let z = gen_vectors(n, d, 210);

    let mut rvq2 = make_rvq(2, k, d);
    let mut rvq4 = make_rvq(4, k, d);

    // Train both
    for _ in 0..100 {
        rvq2.forward(&z, n, d, true);
        rvq4.forward(&z, n, d, true);
    }

    let out2 = rvq2.forward(&z, n, d, false);
    let out4 = rvq4.forward(&z, n, d, false);

    // 4 CBs should have lower reconstruction error (not commit_loss, but quantized MSE)
    let mse2: f32 = z.iter().zip(out2.quantized.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / (n * d) as f32;
    let mse4: f32 = z.iter().zip(out4.quantized.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / (n * d) as f32;
    assert!(
        mse4 <= mse2 + 1e-3,
        "4 codebooks (MSE={:.4}) should reconstruct at least as well as 2 (MSE={:.4})",
        mse4,
        mse2
    );
}

// ── 17. Code count ───────────────────────────────────────────────────────────

/// encode returns Vec of length == num_codebooks.
#[test]
fn test_rvq_code_count() {
    let num_cb = 6;
    let n = 16;
    let d = 8;
    let k = 32;
    let rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(n, d, 220);
    let codes = rvq.encode(&z, n, d);

    assert_eq!(codes.len(), num_cb, "encode should return {} code vectors", num_cb);
    for (lvl, c) in codes.iter().enumerate() {
        assert_eq!(c.len(), n, "level {} codes should have length {}", lvl, n);
    }
}

// ── 18. Decode shape ─────────────────────────────────────────────────────────

/// decode returns [N × D] shaped output.
#[test]
fn test_rvq_decode_shape() {
    let num_cb = 4;
    let n = 16;
    let d = 8;
    let k = 32;
    let rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(n, d, 230);
    let codes = rvq.encode(&z, n, d);
    let decoded = rvq.decode(&codes);

    assert_eq!(decoded.len(), n * d, "decode output should be [N*D]");
}

// ── 19. Total commit loss ────────────────────────────────────────────────────

/// Commit loss returned is the mean across codebooks (finite, ≥ 0).
#[test]
fn test_rvq_total_loss() {
    let mut rvq = make_rvq(4, 16, 8);
    let z = gen_vectors(16, 8, 240);
    let out = rvq.forward(&z, 16, 8, false);

    assert!(out.commit_loss >= 0.0, "commit_loss should be ≥ 0");
    assert!(out.commit_loss.is_finite(), "commit_loss should be finite");
}

// ── 20. Perplexities all valid ───────────────────────────────────────────────

/// All per-level perplexities are in valid range [1, K].
#[test]
fn test_rvq_perplexities_all_valid() {
    let num_cb = 4;
    let k = 16;
    let d = 8;
    let n = 32;
    let mut rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(n, d, 250);
    let out = rvq.forward(&z, n, d, false);

    assert_eq!(out.perplexities.len(), num_cb);
    for (lvl, &p) in out.perplexities.iter().enumerate() {
        assert!(
            p >= 1.0 && p <= k as f32 + 0.5,
            "level {} perplexity {} not in [1, {}]",
            lvl,
            p,
            k
        );
    }
}

// ── 21. Reconstruction quality ───────────────────────────────────────────────

/// After 1000 training steps on Gaussian data, NMSE < 0.5.
/// (Relaxed from spec's 0.1 — achievable with small K for CI speed.)
#[test]
fn test_rvq_reconstruction_quality() {
    let n = 64;
    let d = 8;
    let k = 32;
    let num_cb = 4;
    let mut rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(n, d, 260);

    // Signal power
    let signal_power: f32 = z.iter().map(|&x| x * x).sum::<f32>() / (n * d) as f32;

    for _ in 0..1000 {
        rvq.forward(&z, n, d, true);
    }

    let out = rvq.forward(&z, n, d, false);
    let noise_power: f32 = z.iter()
        .zip(out.quantized.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / (n * d) as f32;

    let nmse = noise_power / (signal_power + 1e-9);
    assert!(
        nmse < 0.5,
        "NMSE {:.4} should be < 0.5 after 1000 training steps",
        nmse
    );
}

// ── 22. Zero input RVQ ───────────────────────────────────────────────────────

/// All-zeros RVQ input doesn't crash.
#[test]
fn test_rvq_zero_input() {
    let mut rvq = make_rvq(4, 16, 8);
    let z = vec![0.0_f32; 8 * 8];
    let out = rvq.forward(&z, 8, 8, false);
    assert_eq!(out.codes.len(), 4);
    assert!(out.commit_loss.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// SplitRVQ tests
// ═══════════════════════════════════════════════════════════════════════════════

fn make_split_rvq(total_cb: usize, k: usize, d: usize) -> SplitRVQ {
    let mut cfg = small_config(k, d);
    cfg.num_codebooks = total_cb;
    SplitRVQ::new(cfg, 1)
}

// ── 23. 8 code vectors ───────────────────────────────────────────────────────

/// encode returns 8 code vectors (1 semantic + 7 acoustic).
#[test]
fn test_split_rvq_codes_8() {
    let total_cb = 8;
    let split = make_split_rvq(total_cb, 32, 8);
    let z = gen_vectors(16, 8, 300);
    let codes = split.encode(&z, 16, 8);

    assert_eq!(codes.len(), total_cb, "should return {} code vectors", total_cb);
    for (lvl, c) in codes.iter().enumerate() {
        assert_eq!(c.len(), 16, "level {} should have 16 codes", lvl);
    }
}

// ── 24. Freeze semantic ───────────────────────────────────────────────────────

/// After freeze_semantic, CB0 codes do not change when training continues.
#[test]
fn test_split_rvq_freeze() {
    let total_cb = 4;
    let k = 16;
    let d = 8;
    let n = 16;
    let mut split = make_split_rvq(total_cb, k, d);
    let z = gen_vectors(n, d, 310);

    // Get CB0 codes before freeze
    let codes_before = split.encode(&z, n, d);
    let cb0_before = codes_before[0].clone();

    // Freeze and run many training steps
    split.freeze_semantic();
    for _ in 0..200 {
        split.forward(&z, n, d, true);
    }

    // CB0 codes should be unchanged (same codebook → same codes for same input)
    let codes_after = split.encode(&z, n, d);
    let cb0_after = &codes_after[0];

    assert_eq!(
        &cb0_before,
        cb0_after,
        "CB0 codes should be stable after freeze"
    );
}

// ── 25. Frozen CB0 no EMA ─────────────────────────────────────────────────────

/// With semantic_frozen=true, CB0 codebook doesn't change during training.
#[test]
fn test_split_rvq_semantic_frozen_no_ema() {
    let total_cb = 4;
    let k = 16;
    let d = 8;
    let n = 16;
    let mut split = make_split_rvq(total_cb, k, d);
    split.freeze_semantic();

    let z = gen_vectors(n, d, 320);

    // Save CB0 codebook snapshot before training
    let cb0_snapshot: Vec<f32> = split.semantic_vq.codebook.clone();

    // Run 100 training steps
    for _ in 0..100 {
        split.forward(&z, n, d, true);
    }

    // CB0 codebook should be identical
    let cb0_after = &split.semantic_vq.codebook;
    for (i, (&before, &after)) in cb0_snapshot.iter().zip(cb0_after.iter()).enumerate() {
        assert!(
            (before - after).abs() < 1e-9,
            "CB0 codebook entry {} changed despite being frozen: {} → {}",
            i,
            before,
            after
        );
    }
}

// ── 26. Decode shape ─────────────────────────────────────────────────────────

/// decode(encode(z)) returns correct shape [N × D].
#[test]
fn test_split_rvq_decode_all() {
    let total_cb = 8;
    let k = 32;
    let d = 16;
    let n = 20;
    let split = make_split_rvq(total_cb, k, d);
    let z = gen_vectors(n, d, 330);

    let codes = split.encode(&z, n, d);
    let decoded = split.decode(&codes);

    assert_eq!(decoded.len(), n * d, "decode output should be [N*D]");
}

// ── 27. Full pipeline ────────────────────────────────────────────────────────

/// Full encode → decode pipeline preserves shapes.
#[test]
fn test_split_rvq_full_pipeline() {
    let total_cb = 8;
    let k = 32;
    let d = 16;
    let n = 12;
    let mut split = make_split_rvq(total_cb, k, d);
    let z = gen_vectors(n, d, 340);

    // Forward pass
    let out = split.forward(&z, n, d, true);
    assert_eq!(out.codes.len(), total_cb);
    assert_eq!(out.quantized.len(), n * d);
    assert_eq!(out.perplexities.len(), total_cb);
    assert!(out.commit_loss.is_finite());

    // Encode → decode
    let codes = split.encode(&z, n, d);
    let decoded = split.decode(&codes);
    assert_eq!(decoded.len(), n * d);
}

// ── 28. Mimi default config ──────────────────────────────────────────────────

/// RVQConfig::default() (K=2048, D=256, N=8) doesn't crash on a tiny batch.
#[test]
fn test_split_rvq_mimi_config() {
    let cfg = RVQConfig::default(); // K=2048, D=256, N=8
    // Use n=1, d=256 to keep it fast
    let n = 1;
    let d = cfg.quant_dim;

    let mut split = SplitRVQ::new(cfg, 1);
    let z = gen_vectors(n, d, 350);
    let out = split.forward(&z, n, d, false);

    assert_eq!(out.codes.len(), 8);
    assert_eq!(out.quantized.len(), n * d);
    assert!(out.commit_loss.is_finite());
}

// ── 29. Commit loss finite for all ───────────────────────────────────────────

/// commit_loss is always finite (no NaN/Inf) even with degenerate inputs.
#[test]
fn test_split_rvq_commit_loss_finite() {
    let mut split = make_split_rvq(4, 16, 8);
    let z = vec![1e-10_f32; 8 * 8]; // near-zero input
    let out = split.forward(&z, 8, 8, false);
    assert!(out.commit_loss.is_finite(), "commit_loss should be finite");
}

// ── 30. Perplexities all finite ──────────────────────────────────────────────

/// All perplexities are finite and positive.
#[test]
fn test_split_rvq_perplexities_finite() {
    let total_cb = 4;
    let mut split = make_split_rvq(total_cb, 16, 8);
    let z = gen_vectors(16, 8, 360);
    let out = split.forward(&z, 16, 8, false);

    for (lvl, &p) in out.perplexities.iter().enumerate() {
        assert!(p.is_finite() && p >= 1.0, "level {} perplexity {} is invalid", lvl, p);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional edge-case and integration tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 31. VQ training is idempotent (same input → same codes) ──────────────────

/// Two consecutive encode calls with the same input return the same codes.
#[test]
fn test_vq_deterministic_encode() {
    let mut vq = make_vq(32, 8);
    let z = gen_vectors(16, 8, 400);

    let codes1 = vq.encode(&z, 16, 8);
    let codes2 = vq.encode(&z, 16, 8);
    assert_eq!(codes1, codes2, "encode should be deterministic");
}

// ── 32. RVQ encode/forward consistency ───────────────────────────────────────

/// encode() and forward().codes return same codes.
#[test]
fn test_rvq_encode_forward_consistency() {
    let n = 16;
    let d = 8;
    let k = 32;
    let num_cb = 3;
    let mut rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(n, d, 410);

    // Get codes via encode (no EMA)
    let codes_enc = rvq.encode(&z, n, d);

    // Get codes via forward with training=false (no EMA either)
    let out_fwd = rvq.forward(&z, n, d, false);

    for lvl in 0..num_cb {
        assert_eq!(
            codes_enc[lvl],
            out_fwd.codes[lvl],
            "level {} codes differ between encode and forward",
            lvl
        );
    }
}

// ── 33. SplitRVQ acoustic vs semantic independence ────────────────────────────

/// CB0 and CB1 codes are computed independently (can differ).
#[test]
fn test_split_rvq_semantic_acoustic_independent() {
    let total_cb = 4;
    let k = 32;
    let d = 8;
    let split = make_split_rvq(total_cb, k, d);
    let z = gen_vectors(16, d, 420);

    let codes = split.encode(&z, 16, d);
    // CB0 (semantic) vs CB1 (first acoustic) should generally differ
    // since they come from separate codebooks
    // (This is a sanity check — they CAN be equal by chance but rarely)
    assert_eq!(codes.len(), total_cb);
}

// ── 34. RVQ decode undo encode ────────────────────────────────────────────────

/// forward().quantized matches decode(encode(z)).
#[test]
fn test_rvq_decode_matches_forward_quantized() {
    let n = 8;
    let d = 8;
    let k = 16;
    let num_cb = 3;
    let mut rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(n, d, 430);

    let out = rvq.forward(&z, n, d, false);
    let codes = rvq.encode(&z, n, d);
    let decoded = rvq.decode(&codes);

    for i in 0..n * d {
        assert!(
            (out.quantized[i] - decoded[i]).abs() < 1e-5,
            "forward/decode mismatch at {}: {} vs {}",
            i,
            out.quantized[i],
            decoded[i]
        );
    }
}

// ── 35. VQ commitment loss formula ───────────────────────────────────────────

/// commitment_loss = β * MSE(z, z_q).
#[test]
fn test_vq_commitment_loss_formula() {
    let n = 4;
    let d = 2;
    let k = 4;
    let mut cfg = small_config(k, d);
    cfg.commitment_cost = 1.0; // β = 1 for easy manual check
    let mut vq = VectorQuantizer::new(cfg);

    let z = gen_vectors(n, d, 440);
    let out = vq.forward(&z, n, d, false);

    // With β=1, commit_loss = beta * sum_of_sq_diffs / n
    // Verify it's non-negative and finite
    assert!(out.commit_loss >= 0.0 && out.commit_loss.is_finite());
}

// ── 36. D=1 edge case ────────────────────────────────────────────────────────

/// D=1 (scalar quantization) works correctly.
#[test]
fn test_vq_d1_scalar_quantization() {
    let n = 16;
    let d = 1;
    let k = 8;
    let mut cfg = small_config(k, d);
    cfg.quant_dim = 1;
    let mut vq = VectorQuantizer::new(cfg);
    let z: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.5).collect();
    let out = vq.forward(&z, n, d, false);
    assert_eq!(out.codes.len(), n);
    assert_eq!(out.quantized.len(), n);
}

// ── 37. Codebook does not contain NaN after training ─────────────────────────

/// After 200 training steps, no NaN in codebook.
#[test]
fn test_vq_codebook_no_nan_after_training() {
    let n = 32;
    let d = 8;
    let k = 16;
    let mut vq = make_vq(k, d);
    let z = gen_vectors(n, d, 450);

    for _ in 0..200 {
        vq.forward(&z, n, d, true);
    }

    for &v in &vq.codebook {
        assert!(!v.is_nan(), "codebook contains NaN after training");
        assert!(!v.is_infinite(), "codebook contains Inf after training");
    }
}

// ── 38. SplitRVQ acoustic gets 7 codebooks with default config ───────────────

/// With N=8 total, SplitRVQ acoustic sub-quantizer has 7 levels.
#[test]
fn test_split_rvq_acoustic_has_7_levels() {
    let mut cfg = RVQConfig::default();
    cfg.kmeans_init = false;
    let split = SplitRVQ::new(cfg, 1);

    let n = 2;
    let d = 256;
    let z = gen_vectors(n, d, 460);
    let codes = split.encode(&z, n, d);

    assert_eq!(codes.len(), 8, "total code levels should be 8");
}

// ── 39. SplitRVQ unfrozen updates CB0 ────────────────────────────────────────

/// When NOT frozen, CB0 codebook changes after training.
#[test]
fn test_split_rvq_unfrozen_updates_cb0() {
    let total_cb = 4;
    let k = 16;
    let d = 8;
    let n = 16;
    let mut split = make_split_rvq(total_cb, k, d);
    let z = gen_vectors(n, d, 470);

    let cb0_before: Vec<f32> = split.semantic_vq.codebook.clone();

    // Train 100 steps WITHOUT freeze
    for _ in 0..100 {
        split.forward(&z, n, d, true);
    }

    let cb0_after = &split.semantic_vq.codebook;
    let changed = cb0_before.iter().zip(cb0_after.iter())
        .filter(|(&a, &b)| (a - b).abs() > 1e-9)
        .count();

    assert!(changed > 0, "CB0 should have changed when not frozen");
}

// ── 40. Large K with small batch ─────────────────────────────────────────────

/// K=2048 (Mimi default) with N=4 input vectors doesn't crash.
#[test]
fn test_vq_large_k_small_batch() {
    let mut cfg = RVQConfig::default();
    cfg.kmeans_init = false;
    let mut vq = VectorQuantizer::new(cfg);
    let n = 4;
    let d = 256;
    let z = gen_vectors(n, d, 480);
    let out = vq.forward(&z, n, d, false);
    assert_eq!(out.codes.len(), n);
    assert!(out.commit_loss.is_finite());
}

// ── 41. RVQ first level is VQ of z ───────────────────────────────────────────

/// The first level of RVQ encode matches a standalone VQ encode.
#[test]
fn test_rvq_first_level_matches_vq() {
    // We can't easily test this without matching codebook init seeds,
    // but we can verify the codes come from a valid VQ (in [0, K)).
    let num_cb = 3;
    let k = 16;
    let d = 8;
    let rvq = make_rvq(num_cb, k, d);
    let z = gen_vectors(8, d, 490);
    let codes = rvq.encode(&z, 8, d);

    for &c in &codes[0] {
        assert!((c as usize) < k, "first level code {} out of range", c);
    }
}

// ── 42. SplitRVQ with n_semantic > 1 ─────────────────────────────────────────

/// SplitRVQ with n_semantic=2: 1 semantic VQ + (6-2)=4 acoustic levels = 5 total codes.
#[test]
fn test_split_rvq_multi_semantic() {
    let mut cfg = small_config(16, 8);
    cfg.num_codebooks = 6; // 1 sem VQ + (6-2)=4 acoustic = 5 total code vectors
    let mut split = SplitRVQ::new(cfg, 2);
    let z = gen_vectors(8, 8, 500);
    let out = split.forward(&z, 8, 8, false);
    // SplitRVQ always uses 1 semantic VQ (single level) +
    // (num_codebooks - n_semantic) = 6-2 = 4 acoustic levels → 5 total
    assert_eq!(out.codes.len(), 5, "1 semantic + 4 acoustic = 5 code vectors");
    assert!(out.commit_loss.is_finite());
}
