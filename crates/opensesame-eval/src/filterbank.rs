//! Mel filterbank construction.
//!
//! Builds a triangular mel filterbank matrix of shape `[n_mels][n_fft/2+1]`.

/// Convert Hz to mel scale: m = 2595 × log10(1 + f/700)
#[inline]
pub fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Convert mel to Hz: f = 700 × (10^(m/2595) − 1)
#[inline]
pub fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
}

/// Build a mel filterbank matrix: shape `[n_mels][n_fft/2+1]`.
///
/// Each row is a triangular filter centred at a mel-spaced frequency.
/// All values are non-negative.
///
/// # Arguments
/// * `n_mels`   — number of mel filter channels
/// * `freq_min` — lowest frequency in Hz
/// * `freq_max` — highest frequency in Hz (typically `sr/2`)
/// * `n_fft`    — FFT size (must be even; matrix width = n_fft/2+1)
/// * `sr`       — sample rate in Hz
pub fn mel_filterbank(
    n_mels: usize,
    freq_min: f64,
    freq_max: f64,
    n_fft: usize,
    sr: u32,
) -> Vec<Vec<f64>> {
    let mel_min = hz_to_mel(freq_min);
    let mel_max = hz_to_mel(freq_max);

    // n_mels + 2 mel-spaced centre points (including edges)
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let freq_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let n_bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sr as f64 / n_fft as f64)
        .collect();

    let mut matrix = vec![vec![0.0_f64; n_bins]; n_mels];
    for m in 0..n_mels {
        let f_left = freq_points[m];
        let f_center = freq_points[m + 1];
        let f_right = freq_points[m + 2];
        for (k, &f) in fft_freqs.iter().enumerate() {
            matrix[m][k] = if f >= f_left && f <= f_center {
                if (f_center - f_left).abs() < 1e-12 {
                    0.0
                } else {
                    (f - f_left) / (f_center - f_left)
                }
            } else if f > f_center && f <= f_right {
                if (f_right - f_center).abs() < 1e-12 {
                    0.0
                } else {
                    (f_right - f) / (f_right - f_center)
                }
            } else {
                0.0
            };
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(80, 0.0, 8000.0, 1024, 16000);
        assert_eq!(fb.len(), 80, "n_mels rows");
        assert_eq!(fb[0].len(), 513, "n_fft/2+1 cols");
    }

    #[test]
    fn test_hz_to_mel_roundtrip() {
        for &f in &[100.0, 500.0, 1000.0, 4000.0, 8000.0_f64] {
            let m = hz_to_mel(f);
            let f2 = mel_to_hz(m);
            assert!((f - f2).abs() < 1e-8, "roundtrip error at {}Hz", f);
        }
    }

    #[test]
    fn test_filterbank_non_negative() {
        let fb = mel_filterbank(24, 50.0, 8000.0, 512, 16000);
        for row in &fb {
            for &v in row {
                assert!(v >= 0.0, "filterbank value must be non-negative: {}", v);
            }
        }
    }
}
