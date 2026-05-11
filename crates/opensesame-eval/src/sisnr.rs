//! Scale-Invariant Signal-to-Noise Ratio (SI-SNR / SI-SDR).
//!
//! Reference: Luo & Mesgarani, "TasNet: Time-domain Audio Separation Network …", 2019.
//!
//! SI-SNR is scale-invariant: multiplying the estimate by any scalar leaves the
//! score unchanged.  Polarity inversion also scores +∞.

use crate::{EvalError, EvalResult};

/// SI-SNR metric.
pub struct SiSnr;

impl SiSnr {
    /// Compute SI-SNR in dB between `reference` and `estimate`.
    ///
    /// Both slices must have equal length ≥ 1.
    ///
    /// Returns `+∞` when the estimate is a scaled replica of the reference
    /// (perfect reconstruction apart from amplitude).
    /// Returns `Err` if the reference is silent (all zeros) or lengths differ.
    pub fn compute(reference: &[f32], estimate: &[f32]) -> EvalResult<f32> {
        let n = reference.len();
        if n != estimate.len() {
            return Err(EvalError::LengthMismatch {
                expected: n,
                got: estimate.len(),
            });
        }
        if n == 0 {
            return Err(EvalError::EmptySignal);
        }

        // 1. Zero-mean both signals (use f64 to avoid accumulation error)
        let mean_ref: f64 = reference.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let mean_est: f64 = estimate.iter().map(|&x| x as f64).sum::<f64>() / n as f64;

        let s: Vec<f64> = reference.iter().map(|&x| x as f64 - mean_ref).collect();
        let s_hat: Vec<f64> = estimate.iter().map(|&x| x as f64 - mean_est).collect();

        // 2. Scale-invariant projection: s_target = (<s_hat, s> / <s, s>) * s
        let dot_ss: f64 = s.iter().map(|&x| x * x).sum();
        if dot_ss < 1e-10 {
            return Err(EvalError::NumericalError("reference is effectively silent"));
        }
        let dot_sh_s: f64 = s_hat.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum();
        let scale = dot_sh_s / dot_ss;

        // 3. Signal and noise power
        let s_target_sq: f64 = s.iter().map(|&x| (scale * x) * (scale * x)).sum();
        let e_noise_sq: f64 = s_hat
            .iter()
            .zip(s.iter())
            .map(|(&sh, &sr)| {
                let e = sh - scale * sr;
                e * e
            })
            .sum();

        if e_noise_sq < 1e-20 {
            return Ok(f32::INFINITY); // perfect reconstruction
        }

        Ok((10.0 * (s_target_sq / e_noise_sq).log10()) as f32)
    }

    /// Mean SI-SNR over a batch of `(reference, estimate)` pairs.
    ///
    /// Returns `Err` if the batch is empty or any pair errors.
    pub fn batch_mean(pairs: &[(&[f32], &[f32])]) -> EvalResult<f32> {
        if pairs.is_empty() {
            return Err(EvalError::EmptySignal);
        }
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (r, e) in pairs {
            match Self::compute(r, e) {
                Ok(v) if v.is_finite() => {
                    sum += v as f64;
                    count += 1;
                }
                Ok(_inf) => {
                    // +∞ contributes ∞ to mean — treat as very large
                    sum += 100.0;
                    count += 1;
                }
                Err(e) => return Err(e),
            }
        }
        if count == 0 {
            return Err(EvalError::EmptySignal);
        }
        Ok((sum / count as f64) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn test_sisnr_identical() {
        let s = sine(440.0, 16000, 16000);
        let r = SiSnr::compute(&s, &s).unwrap();
        assert!(r.is_infinite() && r > 0.0, "identical: SI-SNR = {}", r);
    }

    #[test]
    fn test_sisnr_scaled_estimate() {
        // SI-SNR is scale-invariant: 10×s should give +inf (exact f32 arithmetic)
        // or a very high value (≥100 dB) when f64 rounding is present.
        let s = sine(440.0, 16000, 16000);
        let s_scaled: Vec<f32> = s.iter().map(|&x| 10.0 * x).collect();
        let r = SiSnr::compute(&s, &s_scaled).unwrap();
        assert!(
            r.is_infinite() || r > 100.0,
            "scaled estimate: SI-SNR should be +inf or very high, got {}",
            r
        );
    }

    #[test]
    fn test_sisnr_polarity_inversion() {
        // -s has same SI-SNR as s (scale invariant)
        let s = sine(220.0, 16000, 8000);
        let s_inv: Vec<f32> = s.iter().map(|&x| -x).collect();
        let r = SiSnr::compute(&s, &s_inv).unwrap();
        assert!(r.is_infinite() && r > 0.0, "polarity: SI-SNR = {}", r);
    }

    #[test]
    fn test_sisnr_pure_noise() {
        let s = sine(440.0, 16000, 16000);
        let noise: Vec<f32> = (0..s.len())
            .map(|i| ((i as f32 * 1.61803).sin() + (i as f32 * 0.31415).cos()) * 0.5)
            .collect();
        let r = SiSnr::compute(&s, &noise).unwrap();
        assert!(r < 0.0, "pure noise: expected negative SI-SNR, got {}", r);
    }

    #[test]
    fn test_sisnr_known_snr() {
        let n = 16000usize;
        let s: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let noise: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 1.61803).sin() + (i as f32 * 2.71828).cos()) * 0.5)
            .collect();
        let estimate: Vec<f32> = s
            .iter()
            .zip(noise.iter())
            .map(|(&si, &ni)| si + 0.1 * ni)
            .collect();
        let r = SiSnr::compute(&s, &estimate).unwrap();
        assert!(r > 15.0, "10% noise: expected SI-SNR > 15 dB, got {:.2}", r);
    }

    #[test]
    fn test_sisnr_silent_reference() {
        let zeros = vec![0.0f32; 1000];
        let s = sine(440.0, 16000, 1000);
        let r = SiSnr::compute(&zeros, &s);
        assert!(r.is_err(), "silent reference should return Err");
    }

    #[test]
    fn test_sisnr_length_mismatch() {
        let a = vec![0.0f32; 100];
        let b = vec![0.0f32; 200];
        let r = SiSnr::compute(&a, &b);
        assert!(matches!(r, Err(EvalError::LengthMismatch { .. })));
    }

    #[test]
    fn test_sisnr_batch_mean() {
        let s = sine(440.0, 16000, 4000);
        let pairs: Vec<(&[f32], &[f32])> = vec![(&s, &s), (&s, &s)];
        let r = SiSnr::batch_mean(&pairs).unwrap();
        // Both pairs are identical → both give 100 dB (clamp in batch_mean)
        assert!(r > 50.0, "batch mean of identical: {}", r);
    }
}
