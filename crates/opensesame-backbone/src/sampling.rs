//! Sampling utilities for autoregressive decoding.
//!
//! Implements top-k sampling with temperature scaling and Gumbel-noise-based
//! token selection — matching the Sesame CSM sampling strategy.

/// Sample from `logits` using temperature + top-k + Gumbel noise.
///
/// Algorithm:
/// 1. Scale: `logits /= temperature`
/// 2. Keep only the top-k values; set all others to `−∞`
/// 3. Compute log-softmax for numerical stability
/// 4. Convert log-probs to probabilities and sample via inverse CDF
///
/// `topk = 0` means "keep all logits" (pure temperature sampling).
/// `temperature` must be > 0.
///
/// Returns the sampled token index as `u32`.
pub fn sample_topk(logits: &[f32], topk: usize, temperature: f32) -> u32 {
    let n = logits.len();
    assert!(n > 0, "empty logits");
    assert!(temperature > 0.0, "temperature must be positive");

    // Step 1: scale by temperature
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Step 2: apply top-k masking
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

    // Step 3: numerically stable log-softmax
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = scaled
        .iter()
        .map(|&s| if s.is_finite() { (s - max).exp() } else { 0.0 })
        .sum::<f32>()
        .ln();
    let log_probs: Vec<f32> = scaled
        .iter()
        .map(|&s| if s.is_finite() { s - max - log_sum } else { f32::NEG_INFINITY })
        .collect();

    // Step 4: convert to probabilities
    let max_lp = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = log_probs.iter().map(|&v| (v - max_lp).exp()).sum();
    let probs: Vec<f32> = log_probs
        .iter()
        .map(|&v| (v - max_lp).exp() / sum_exp)
        .collect();

    // Step 5: sample via inverse CDF using internal LCG RNG
    // Use strict inequality (u < cumsum) so that a probability-0 token at index 0
    // with cumsum=0 is never selected when u=0.
    let u = lcg_next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i as u32;
        }
    }
    (n - 1) as u32
}

/// Deterministic top-k sample using an explicit uniform random value.
///
/// Use this in tests for reproducibility.
///
/// `u` should be in `[0, 1)`.  Passing `0.0` always returns the first
/// non-masked token (argmax-like with ties broken early).
pub fn sample_topk_with_u(logits: &[f32], topk: usize, temperature: f32, u: f32) -> u32 {
    let n = logits.len();
    assert!(n > 0, "empty logits");
    assert!(temperature > 0.0, "temperature must be positive");

    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

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

    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = scaled
        .iter()
        .map(|&s| if s.is_finite() { (s - max).exp() } else { 0.0 })
        .sum::<f32>()
        .ln();
    let log_probs: Vec<f32> = scaled
        .iter()
        .map(|&s| if s.is_finite() { s - max - log_sum } else { f32::NEG_INFINITY })
        .collect();

    let max_lp = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = log_probs.iter().map(|&v| (v - max_lp).exp()).sum();
    let probs: Vec<f32> = log_probs
        .iter()
        .map(|&v| (v - max_lp).exp() / sum_exp)
        .collect();

    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i as u32;
        }
    }
    (n - 1) as u32
}

/// Argmax over logits (greedy decoding, deterministic).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Very lightweight LCG-based random f32 in `[0, 1)`.
///
/// Not thread-safe and not cryptographically secure.  Sufficient for sampling.
fn lcg_next_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x517cc1b727220a95);
    let prev = STATE.load(Ordering::Relaxed);
    let next = prev
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    STATE.store(next, Ordering::Relaxed);
    (next >> 11) as f32 / (1u64 << 53) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_topk_basic() {
        // Logits with a clear winner at index 2
        let logits = vec![0.0f32, 0.5, 10.0, 0.1, 0.2];
        // With temperature=1.0, topk=1, should always return index 2
        let tok = sample_topk_with_u(&logits, 1, 1.0, 0.5);
        assert_eq!(tok, 2, "expected argmax=2");
    }

    #[test]
    fn test_sample_topk_uniform_at_zero() {
        // u=0.0 always picks first non-zero-probability token
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tok = sample_topk_with_u(&logits, 3, 1.0, 0.0);
        // With topk=3 and u=0, should be one of the top 3 (indices 2,3,4)
        assert!(tok >= 2 && tok <= 4, "got {}", tok);
    }

    #[test]
    fn test_sample_topk_temperature_effect() {
        // Very low temperature → near-greedy
        let logits = vec![0.0f32, 0.0, 10.0];
        let tok = sample_topk_with_u(&logits, 0, 0.01, 0.99);
        assert_eq!(tok, 2);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_sample_topk_stochastic_returns_valid() {
        let logits = vec![1.0f32; 2048];
        let tok = sample_topk(&logits, 50, 1.0);
        assert!((tok as usize) < 2048);
    }
}
