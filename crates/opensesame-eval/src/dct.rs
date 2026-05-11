//! DCT-II (normalised / orthonormal) — "scipy.fft.dct(x, type=2, norm='ortho')".
//!
//! Used for MCD mel cepstral coefficient extraction.

use std::f64::consts::PI;

/// Normalised DCT-II of an input slice.
///
/// Definition:
/// ```text
/// X[k] = w_k × sqrt(2/N) × Σ_{m=0}^{N-1} x[m] × cos(π k (2m+1) / (2N))
/// ```
/// where w_0 = 1/sqrt(2), w_{k>0} = 1.
///
/// This is the orthonormal variant — applying dct then idct (DCT-III with same
/// normalisation) recovers the original signal.
pub fn dct2_normalized(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let pi_over_2n = PI / (2.0 * n as f64);
    let mut out = vec![0.0_f64; n];
    for k in 0..n {
        let sum: f64 = x
            .iter()
            .enumerate()
            .map(|(m, &v)| v * (pi_over_2n * k as f64 * (2 * m + 1) as f64).cos())
            .sum();
        let scale = if k == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };
        out[k] = sum * scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct2_dc_only() {
        // Constant input [c; N] → output[0] = c * sqrt(N), output[k>0] = 0
        let n = 8;
        let x = vec![2.0_f64; n];
        let y = dct2_normalized(&x);
        let expected_dc = 2.0 * (n as f64).sqrt();
        assert!((y[0] - expected_dc).abs() < 1e-10, "DC = {}", y[0]);
        for k in 1..n {
            assert!(y[k].abs() < 1e-10, "AC component at k={}: {}", k, y[k]);
        }
    }

    #[test]
    fn test_dct2_length() {
        let x = vec![1.0_f64; 13];
        let y = dct2_normalized(&x);
        assert_eq!(y.len(), 13);
    }

    #[test]
    fn test_dct2_empty() {
        let y = dct2_normalized(&[]);
        assert!(y.is_empty());
    }
}
