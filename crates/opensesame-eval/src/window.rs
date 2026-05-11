//! Window functions for spectral analysis.

use std::f64::consts::PI;

/// Symmetric Hann window: w[n] = 0.5 × (1 − cos(2πn / (N−1)))
pub fn hann(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Periodic Hann window (for STFT): w[n] = 0.5 × (1 − cos(2πn / N))
///
/// Avoids the endpoint discontinuity and satisfies the COLA condition.
pub fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Hamming window: w[n] = 0.54 − 0.46 × cos(2πn / (N−1))
pub fn hamming(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}
