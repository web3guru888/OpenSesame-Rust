//! Radix-2 Cooley-Tukey DIT FFT — pure Rust, no external dependencies.
//!
//! Supports in-place complex FFT on power-of-2 length buffers, stored as
//! interleaved [re0, im0, re1, im1, …] f64 arrays.

use std::f64::consts::PI;

/// In-place Cooley-Tukey radix-2 DIT FFT.
///
/// `buf` is treated as complex samples: buf[2k] = re, buf[2k+1] = im.
/// Length of `buf` must be 2 × (power of 2).
/// Set `inverse = true` for IFFT (includes 1/N normalisation).
pub fn fft_inplace(buf: &mut [f64], inverse: bool) {
    let n = buf.len() / 2;
    assert!(n.is_power_of_two() && n >= 1, "FFT size must be a power of 2");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(2 * i, 2 * j);
            buf.swap(2 * i + 1, 2 * j + 1);
        }
    }

    // Butterfly passes
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let wre = ang.cos();
        let wim = ang.sin();
        let mut i = 0;
        while i < n {
            let mut wr = 1.0_f64;
            let mut wi = 0.0_f64;
            for k in 0..len / 2 {
                let (ur, ui) = (buf[2 * (i + k)], buf[2 * (i + k) + 1]);
                let (vr, vi) = (buf[2 * (i + k + len / 2)], buf[2 * (i + k + len / 2) + 1]);
                let tr = wr * vr - wi * vi;
                let ti = wr * vi + wi * vr;
                buf[2 * (i + k)] = ur + tr;
                buf[2 * (i + k) + 1] = ui + ti;
                buf[2 * (i + k + len / 2)] = ur - tr;
                buf[2 * (i + k + len / 2) + 1] = ui - ti;
                let new_wr = wr * wre - wi * wim;
                let new_wi = wr * wim + wi * wre;
                wr = new_wr;
                wi = new_wi;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for x in buf.iter_mut() {
            *x *= scale;
        }
    }
}

/// Compute the magnitude spectrum |X[k]| for k = 0..n_fft/2+1.
///
/// Input: real signal (f32). Zero-padded or truncated to `n_fft`.
/// `n_fft` must be a power of 2.
pub fn rfft_mag(signal: &[f32], n_fft: usize) -> Vec<f64> {
    assert!(n_fft.is_power_of_two(), "n_fft must be power of 2");
    let mut buf = vec![0.0_f64; n_fft * 2];
    for (i, &s) in signal.iter().take(n_fft).enumerate() {
        buf[2 * i] = s as f64;
    }
    fft_inplace(&mut buf, false);
    (0..n_fft / 2 + 1)
        .map(|k| {
            let re = buf[2 * k];
            let im = buf[2 * k + 1];
            (re * re + im * im).sqrt()
        })
        .collect()
}

/// Compute complex output (re, im) for k = 0..n_fft/2+1 from a real signal.
///
/// `n_fft` must be a power of 2.
pub fn rfft_complex(signal: &[f32], n_fft: usize) -> Vec<(f64, f64)> {
    assert!(n_fft.is_power_of_two(), "n_fft must be power of 2");
    let mut buf = vec![0.0_f64; n_fft * 2];
    for (i, &s) in signal.iter().take(n_fft).enumerate() {
        buf[2 * i] = s as f64;
    }
    fft_inplace(&mut buf, false);
    (0..n_fft / 2 + 1)
        .map(|k| (buf[2 * k], buf[2 * k + 1]))
        .collect()
}

/// Power spectrum: |X[k]|² for k = 0..n_fft/2+1.
///
/// `n_fft` must be a power of 2.
pub fn power_spectrum(signal: &[f32], n_fft: usize) -> Vec<f64> {
    rfft_mag(signal, n_fft).into_iter().map(|m| m * m).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_dc_component() {
        // DC input [1, 1, 1, 1] → FFT[0] = 4+0i, FFT[k>0] = 0
        let mut buf = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        fft_inplace(&mut buf, false);
        assert!((buf[0] - 4.0).abs() < 1e-10, "FFT[0].re = {}", buf[0]);
        assert!(buf[1].abs() < 1e-10, "FFT[0].im = {}", buf[1]);
        // All other bins should be ~0
        for k in 1..4 {
            let mag = (buf[2 * k] * buf[2 * k] + buf[2 * k + 1] * buf[2 * k + 1]).sqrt();
            assert!(mag < 1e-10, "bin {} mag = {}", k, mag);
        }
    }

    #[test]
    fn test_fft_sine_peak() {
        // sin(2π * k/N) → peak at bin k
        let n = 16usize;
        let k_peak = 3;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * k_peak as f64 * i as f64 / n as f64).sin() as f32)
            .collect();
        let mags = rfft_mag(&signal, n);
        // Find bin with max magnitude
        let max_bin = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_bin, k_peak, "Expected peak at bin {}, got {}", k_peak, max_bin);
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = signal.len();
        let mut buf = vec![0.0f64; n * 2];
        for (i, &s) in signal.iter().enumerate() {
            buf[2 * i] = s;
        }
        fft_inplace(&mut buf, false);
        fft_inplace(&mut buf, true);
        for (i, &orig) in signal.iter().enumerate() {
            let recovered = buf[2 * i];
            assert!(
                (recovered - orig).abs() < 1e-10,
                "Roundtrip error at {}: {} vs {}",
                i,
                recovered,
                orig
            );
        }
    }

    #[test]
    fn test_power_spectrum_shape() {
        let signal: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
        let ps = power_spectrum(&signal, 512);
        assert_eq!(ps.len(), 257, "Power spectrum length = n_fft/2+1 = 257");
        for &p in &ps {
            assert!(p >= 0.0, "Power spectrum must be non-negative");
        }
    }
}
