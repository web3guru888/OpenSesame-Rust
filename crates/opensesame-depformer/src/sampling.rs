//! Sampling utilities for the Depformer depth decode loop.
//!
//! Provides top-k temperature sampling used in `Depformer::generate_depth_sequence`.

/// Sample a token from `logits` using temperature scaling and optional top-k filtering.
///
/// # Modes
/// - `temperature ≈ 0.0` (`< 1e-7`): **greedy argmax** — returns the highest-logit token.
/// - `topk = 0`: no filtering (pure temperature sampling over all tokens).
/// - `topk > 0`: restrict sampling to the `topk` highest-logit tokens.
///
/// # Algorithm
/// 1. Handle greedy edge case.
/// 2. Scale: `logits /= temperature`.
/// 3. Keep only top-k (mask rest to `−∞`).
/// 4. Compute numerically-stable softmax.
/// 5. Sample via inverse CDF with an internal LCG uniform random.
///
/// Returns the sampled token index as `u32` in `[0, logits.len())`.
///
/// # Panics
/// Panics if `logits` is empty.
pub fn sample_topk(logits: &[f32], topk: usize, temperature: f32) -> u32 {
    let n = logits.len();
    assert!(n > 0, "sample_topk: empty logits");

    // Greedy path
    if temperature < 1e-7 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Scale by temperature
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Top-k masking
    if topk > 0 && topk < n {
        let mut sorted = scaled.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[topk - 1];
        for v in scaled.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Numerically-stable softmax
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = scaled
        .iter()
        .map(|&s| if s.is_finite() { (s - max).exp() } else { 0.0 })
        .sum();
    let probs: Vec<f32> = scaled
        .iter()
        .map(|&s| if s.is_finite() { (s - max).exp() / sum_exp } else { 0.0 })
        .collect();

    // Sample via inverse CDF using an internal LCG
    let u = lcg_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i as u32;
        }
    }
    (n - 1) as u32
}

/// Internal LCG pseudo-random uniform float in `[0, 1)`.
///
/// Uses a module-level atomic state — not thread-safe for reproducibility,
/// but sufficient for single-threaded sampling in tests and inference.
fn lcg_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853C_49E6_748F_EA9Bu64);
    let prev = STATE.fetch_add(0, Ordering::Relaxed);
    let next = prev
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    STATE.store(next, Ordering::Relaxed);
    ((next >> 11) as f32) / ((1u64 << 53) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let mut logits = vec![0.0f32; 64];
        logits[42] = 10.0;
        assert_eq!(sample_topk(&logits, 0, 0.0), 42);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits: Vec<f32> = (0..32).map(|i| i as f32).collect();
        // With temperature=0, should always return last element (highest)
        assert_eq!(sample_topk(&logits, 0, 0.0), 31);
    }

    #[test]
    fn topk_one_returns_argmax() {
        let mut logits = vec![0.0f32; 32];
        logits[7] = 5.0;
        // topk=1 keeps only the best; always samples it
        let code = sample_topk(&logits, 1, 1.0);
        assert_eq!(code, 7);
    }

    #[test]
    fn result_in_range() {
        let logits: Vec<f32> = (0..2048).map(|i| (i % 7) as f32 - 3.0).collect();
        for temperature in [0.0f32, 0.5, 1.0, 2.0] {
            for topk in [0usize, 1, 10, 50, 2048] {
                let code = sample_topk(&logits, topk, temperature);
                assert!((code as usize) < 2048);
            }
        }
    }
}
