//! Short-Time Objective Intelligibility (STOI).
//!
//! Reference: Taal et al. "An Algorithm for Intelligibility Prediction of
//! Time-Frequency Weighted Noisy Speech", IEEE Trans. Audio 2011.
//!
//! Algorithm summary:
//! 1. Resample to 10 kHz.
//! 2. Remove silent frames.
//! 3. Compute STFT (Hann, N=256, hop=128, FFT=512).
//! 4. Apply 15-band 1/3-octave H matrix.
//! 5. Segment-based normalised correlation (β-clipping).
//! 6. Average over segments.

use crate::resample::resample_linear;
use crate::fft::fft_inplace;
use crate::window::hann_periodic;
use crate::{EvalError, EvalResult};

/// STOI algorithm configuration.
pub struct StoiConfig {
    /// Target sample rate (always 10 000 Hz for standard STOI).
    pub fs_target: u32,
    /// Analysis frame length in samples at `fs_target`.
    pub n_frame: usize,
    /// FFT size (must be power of 2, ≥ n_frame).
    pub n_fft: usize,
    /// Frame hop (50% overlap → 128).
    pub n_overlap: usize,
    /// Number of 1/3-octave bands.
    pub j_bands: usize,
    /// Segment length in frames for short-time correlation.
    pub seg_len: usize,
    /// SDR bound in dB (β = 10^(−beta_db/20)).
    pub beta_db: f32,
    /// Dynamic range threshold for silent-frame removal.
    pub dyn_range_db: f32,
}

impl Default for StoiConfig {
    fn default() -> Self {
        Self {
            fs_target: 10_000,
            n_frame: 256,
            n_fft: 512,
            n_overlap: 128,
            j_bands: 15,
            seg_len: 30,
            beta_db: -15.0,
            dyn_range_db: 40.0,
        }
    }
}

/// STOI metric.
pub struct Stoi {
    config: StoiConfig,
    /// Band centre frequencies at 10 kHz, shape [15].
    band_cf: [f64; 15],
    /// H matrix: [15][n_fft/2+1] — 1 if bin is in band, else 0.
    h_matrix: Vec<Vec<bool>>,
}

impl Default for Stoi {
    fn default() -> Self {
        Self::new()
    }
}

impl Stoi {
    /// Create a STOI instance with default parameters.
    pub fn new() -> Self {
        let config = StoiConfig::default();
        let band_cf = Self::band_centers();
        let h_matrix = Self::build_h_matrix(&config, &band_cf);
        Self { config, band_cf, h_matrix }
    }

    /// 1/3-octave band centre frequencies: cf[k] = 150 × 2^(k/3) for k=0..14
    fn band_centers() -> [f64; 15] {
        std::array::from_fn(|k| 150.0 * 2.0_f64.powf(k as f64 / 3.0))
    }

    /// Lower and upper band-edge frequencies.
    fn band_edges(cf: &[f64; 15]) -> (Vec<f64>, Vec<f64>) {
        let fl: Vec<f64> = (0..15)
            .map(|k| {
                if k == 0 {
                    cf[0] / 2.0_f64.powf(1.0 / 6.0)
                } else {
                    (cf[k] * cf[k - 1]).sqrt()
                }
            })
            .collect();
        let fr: Vec<f64> = (0..15)
            .map(|k| {
                if k == 14 {
                    cf[14] * 2.0_f64.powf(1.0 / 6.0)
                } else {
                    (cf[k] * cf[k + 1]).sqrt()
                }
            })
            .collect();
        (fl, fr)
    }

    /// Build binary H matrix: H[j][b] = true iff bin b falls in band j.
    fn build_h_matrix(cfg: &StoiConfig, cf: &[f64; 15]) -> Vec<Vec<bool>> {
        let (fl, fr) = Self::band_edges(cf);
        let n_bins = cfg.n_fft / 2 + 1;
        let mut h = vec![vec![false; n_bins]; 15];
        for j in 0..15 {
            for b in 0..n_bins {
                let f = b as f64 * cfg.fs_target as f64 / cfg.n_fft as f64;
                h[j][b] = f >= fl[j] && f <= fr[j];
            }
        }
        h
    }

    /// Remove silent frames from both signals simultaneously.
    ///
    /// Returns `(x_out, y_out)` with only active-frame segments concatenated.
    fn remove_silent_frames(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = self.config.n_frame;
        let hop = n / 2;
        let win = hann_periodic(n);

        if x.len() < n {
            return (x.to_vec(), y.to_vec());
        }

        let num_frames = (x.len() - n) / hop + 1;
        let mut energies: Vec<f64> = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let start = i * hop;
            let energy: f64 = x[start..start + n]
                .iter()
                .zip(win.iter())
                .map(|(&s, &w)| {
                    let v = s as f64 * w;
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            energies.push(20.0 * (energy + 1e-10_f64).log10());
        }

        let max_e = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_e - self.config.dyn_range_db as f64;

        let mut x_out = Vec::new();
        let mut y_out = Vec::new();
        for (i, &e) in energies.iter().enumerate() {
            if e >= threshold {
                let start = i * hop;
                x_out.extend_from_slice(&x[start..start + n]);
                y_out.extend_from_slice(&y[start..start + n]);
            }
        }
        (x_out, y_out)
    }

    /// Compute the STFT magnitude matrix, shape [n_bins][T].
    fn stft(&self, signal: &[f32]) -> Vec<Vec<f64>> {
        let n = self.config.n_frame;
        let n_fft = self.config.n_fft;
        let hop = self.config.n_overlap; // hop = n_overlap (= n/2)
        let n_bins = n_fft / 2 + 1;
        let win = hann_periodic(n);

        let num_frames = if signal.len() >= n {
            (signal.len() - n) / hop + 1
        } else {
            0
        };

        let mut out = vec![vec![0.0_f64; num_frames]; n_bins];
        for t in 0..num_frames {
            let start = t * hop;
            let mut buf = vec![0.0_f64; n_fft * 2];
            for k in 0..n.min(signal.len() - start) {
                buf[2 * k] = signal[start + k] as f64 * win[k];
            }
            fft_inplace(&mut buf, false);
            for b in 0..n_bins {
                let re = buf[2 * b];
                let im = buf[2 * b + 1];
                out[b][t] = (re * re + im * im).sqrt();
            }
        }
        out
    }

    /// Apply H matrix: band energies X[j][t] = sqrt(sum_b H[j,b] * |STFT[b,t]|²)
    fn apply_h(&self, stft_mag: &[Vec<f64>], t_frames: usize) -> Vec<Vec<f64>> {
        let j = self.config.j_bands;
        let n_bins = self.config.n_fft / 2 + 1;
        let mut out = vec![vec![0.0_f64; t_frames]; j];
        for jj in 0..j {
            for t in 0..t_frames {
                let energy: f64 = (0..n_bins)
                    .filter(|&b| self.h_matrix[jj][b])
                    .map(|b| stft_mag[b][t] * stft_mag[b][t])
                    .sum();
                out[jj][t] = energy.sqrt();
            }
        }
        out
    }

    /// Short-time normalised correlation between two length-N vectors.
    fn taa_corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx: f64 = x.iter().sum::<f64>() / n;
        let my: f64 = y.iter().sum::<f64>() / n;
        let xc: Vec<f64> = x.iter().map(|&v| v - mx).collect();
        let yc: Vec<f64> = y.iter().map(|&v| v - my).collect();
        let nx: f64 = xc.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let ny: f64 = yc.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if nx < 1e-10 || ny < 1e-10 {
            return 0.0;
        }
        xc.iter().zip(yc.iter()).map(|(&a, &b)| a * b).sum::<f64>() / (nx * ny)
    }

    /// Compute STOI score ∈ [−1, 1] (typically [0, 1]).
    ///
    /// Both signals must have the same length. They will be resampled from
    /// `fs_in` Hz to 10 kHz internally.
    pub fn compute(&self, reference: &[f32], estimate: &[f32], fs_in: u32) -> EvalResult<f32> {
        if reference.len() != estimate.len() {
            return Err(EvalError::LengthMismatch {
                expected: reference.len(),
                got: estimate.len(),
            });
        }
        if reference.is_empty() {
            return Err(EvalError::EmptySignal);
        }

        // Step 1: resample to 10 kHz
        let x = resample_linear(reference, fs_in, self.config.fs_target);
        let y = resample_linear(estimate, fs_in, self.config.fs_target);

        // Step 2: remove silent frames
        let (x, y) = self.remove_silent_frames(&x, &y);
        if x.len() < self.config.n_frame {
            return Ok(0.0); // signal too short after silence removal
        }

        // Step 3: STFT
        let x_stft = self.stft(&x);
        let y_stft = self.stft(&y);
        let t_frames = if x_stft.is_empty() { 0 } else { x_stft[0].len() };
        if t_frames == 0 {
            return Ok(0.0);
        }

        // Step 4: apply H matrix
        let x_bands = self.apply_h(&x_stft, t_frames);
        let y_bands = self.apply_h(&y_stft, t_frames);

        // Step 5: segment-based processing
        let seg = self.config.seg_len;
        let c = 10.0_f64.powf(-self.config.beta_db as f64 / 20.0); // = 5.6234

        if t_frames < seg {
            return Ok(0.0);
        }

        let mut d_vals: Vec<f64> = Vec::new();

        for m in seg..=t_frames {
            let mut d_m = 0.0_f64;
            for j in 0..self.config.j_bands {
                let x_seg: Vec<f64> = x_bands[j][m - seg..m].to_vec();
                let y_seg: Vec<f64> = y_bands[j][m - seg..m].to_vec();

                let norm_x: f64 = x_seg.iter().map(|&v| v * v).sum::<f64>().sqrt();
                let norm_y: f64 = y_seg.iter().map(|&v| v * v).sum::<f64>().sqrt();

                let alpha = if norm_y < 1e-10 { 0.0 } else { norm_x / norm_y };

                // β-clip: Y' = min(α·Y, (1+c)·X) element-wise
                let y_prime: Vec<f64> = x_seg
                    .iter()
                    .zip(y_seg.iter())
                    .map(|(&xi, &yi)| (alpha * yi).min(xi * (1.0 + c)))
                    .collect();

                d_m += Self::taa_corr(&x_seg, &y_prime);
            }
            d_vals.push(d_m / self.config.j_bands as f64);
        }

        if d_vals.is_empty() {
            return Ok(0.0);
        }
        let mean_d = d_vals.iter().sum::<f64>() / d_vals.len() as f64;
        Ok(mean_d.clamp(-1.0, 1.0) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_stoi_identical() {
        let signal = sine(440.0, 16000, 16000);
        let stoi = Stoi::new();
        let r = stoi.compute(&signal, &signal, 16000).unwrap();
        assert!(r > 0.95, "identical: STOI = {}", r);
    }

    #[test]
    fn test_stoi_pure_noise_vs_speech() {
        // Reference: clean speech (tonal). Estimate: broadband LCG white noise.
        // White noise is uncorrelated with speech across all 1/3-octave bands → low STOI.
        let speech: Vec<f32> = (0..16000)
            .map(|i| {
                (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5
                    + (2.0 * PI * 880.0 * i as f32 / 16000.0).sin() * 0.3
            })
            .collect();
        // LCG white noise — broadband, statistically uncorrelated with speech
        let mut state = 12345u64;
        let noise: Vec<f32> = (0..16000)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as f32 / (0x7FFF_FFFFu32 as f32) - 1.0
            })
            .collect();
        let stoi = Stoi::new();
        let r = stoi.compute(&speech, &noise, 16000).unwrap();
        assert!(r < 0.6, "speech vs white noise: STOI = {}", r);
    }

    #[test]
    fn test_stoi_range() {
        let s: Vec<f32> = (0..8000).map(|i| (i as f32 / 100.0).sin()).collect();
        let n: Vec<f32> = (0..8000).map(|i| (i as f32 / 37.0).cos()).collect();
        let stoi = Stoi::new();
        let r = stoi.compute(&s, &n, 16000).unwrap();
        assert!(r >= -1.0 && r <= 1.0, "STOI out of range: {}", r);
    }

    #[test]
    fn test_stoi_length_mismatch() {
        let a = vec![0.0f32; 8000];
        let b = vec![0.0f32; 9000];
        let stoi = Stoi::new();
        assert!(stoi.compute(&a, &b, 16000).is_err());
    }

    #[test]
    fn test_stoi_silent_input() {
        let silence = vec![0.0f32; 16000];
        let speech = sine(440.0, 16000, 16000);
        let stoi = Stoi::new();
        // Must not panic; result is graceful (0 or ok)
        let _ = stoi.compute(&silence, &speech, 16000);
    }

    #[test]
    fn test_stoi_band_centers() {
        let cf = Stoi::band_centers();
        assert_eq!(cf.len(), 15);
        assert!((cf[0] - 150.0).abs() < 1e-6);
        // cf[3] ≈ 300 Hz
        assert!((cf[3] - 300.0).abs() < 1.0, "cf[3] = {}", cf[3]);
    }
}
