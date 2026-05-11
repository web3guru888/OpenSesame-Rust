//! Simplified ViSQOL v3 — mel-NSIM based MOS estimation.
//!
//! Reference: Chinen et al. "ViSQOL v3: An Open Source Production Ready
//! Objective Speech and Audio Metric", INTERSPEECH 2020.
//!
//! This implementation uses a mel spectrogram (linear energy) as an
//! approximation to ViSQOL's gammatone neurogram, and NSIM patch
//! similarity mapped to MOS via a calibrated polynomial.

use crate::fft::rfft_mag;
use crate::filterbank::mel_filterbank;
use crate::window::hann;
use crate::{EvalError, EvalResult};

/// ViSQOL-Lite configuration.
pub struct ViSQOLConfig {
    /// Number of mel bands (gammatone approximation).
    pub n_mels: usize,
    /// Lowest frequency in Hz.
    pub freq_min: f64,
    /// Highest frequency in Hz.
    pub freq_max: f64,
    /// Spectrogram hop length in ms.
    pub hop_ms: f64,
    /// Spectrogram window length in ms.
    pub window_ms: f64,
    /// Patch length in frames.
    pub patch_frames: usize,
    /// Patch step (overlap stride) in frames.
    pub patch_step: usize,
    /// Temporal alignment search range in frames (±).
    pub search_range: i32,
    /// NSIM stabilisation constant C1.
    pub c1: f64,
    /// NSIM stabilisation constant C2.
    pub c2: f64,
    /// Polynomial coefficient a (NSIM² term).
    pub poly_a: f64,
    /// Polynomial coefficient b (NSIM term).
    pub poly_b: f64,
    /// Polynomial constant c.
    pub poly_c: f64,
}

impl Default for ViSQOLConfig {
    fn default() -> Self {
        Self {
            n_mels: 24,
            freq_min: 50.0,
            freq_max: 8000.0,
            hop_ms: 10.0,
            window_ms: 20.0,
            patch_frames: 30,
            patch_step: 15,
            search_range: 10,
            c1: 0.0001,
            c2: 0.0009,
            poly_a: -0.0028,
            poly_b: 4.9651,
            poly_c: 0.0,
        }
    }
}

/// ViSQOL metric.
pub struct ViSQOL {
    config: ViSQOLConfig,
}

impl Default for ViSQOL {
    fn default() -> Self {
        Self::new()
    }
}

impl ViSQOL {
    /// Create ViSQOL with default parameters.
    pub fn new() -> Self {
        Self {
            config: ViSQOLConfig::default(),
        }
    }

    /// Create ViSQOL with custom configuration.
    pub fn with_config(config: ViSQOLConfig) -> Self {
        Self { config }
    }

    fn ms_to_samples(ms: f64, sr: u32) -> usize {
        (ms / 1000.0 * sr as f64).round() as usize
    }

    /// Compute mel spectrogram (linear energy, not log).
    ///
    /// Returns matrix of shape `[n_mels][T]`.
    fn mel_spectrogram(&self, signal: &[f32], sr: u32) -> Vec<Vec<f64>> {
        let n_win = Self::ms_to_samples(self.config.window_ms, sr).max(2);
        let hop = Self::ms_to_samples(self.config.hop_ms, sr).max(1);
        let n_fft = n_win.next_power_of_two();
        let win = hann(n_win);

        let fb = mel_filterbank(
            self.config.n_mels,
            self.config.freq_min,
            self.config.freq_max.min(sr as f64 / 2.0),
            n_fft,
            sr,
        );

        let n_frames = if signal.len() >= n_win {
            (signal.len() - n_win) / hop + 1
        } else {
            0
        };

        let mut spec = vec![vec![0.0_f64; n_frames]; self.config.n_mels];
        for t in 0..n_frames {
            let start = t * hop;
            let mut frame = vec![0.0f32; n_win];
            for (k, &s) in signal.iter().skip(start).take(n_win).enumerate() {
                frame[k] = (s as f64 * win[k]) as f32;
            }
            let mag = rfft_mag(&frame, n_fft);
            for (m, fb_row) in fb.iter().enumerate() {
                let energy: f64 = fb_row
                    .iter()
                    .zip(mag.iter())
                    .map(|(&h, &mg)| h * mg * mg)
                    .sum();
                spec[m][t] = energy;
            }
        }
        spec
    }

    /// Compute NSIM between two flattened patches (both length B×L).
    ///
    /// NSIM is analogous to SSIM but for audio spectrograms.
    pub fn nsim(x: &[f64], y: &[f64], c1: f64, c2: f64) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 1.0;
        }
        let mu_x = x.iter().sum::<f64>() / n;
        let mu_y = y.iter().sum::<f64>() / n;
        let var_x: f64 = x.iter().map(|&v| (v - mu_x) * (v - mu_x)).sum::<f64>() / n;
        let var_y: f64 = y.iter().map(|&v| (v - mu_y) * (v - mu_y)).sum::<f64>() / n;
        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - mu_x) * (b - mu_y))
            .sum::<f64>()
            / n;
        let num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2);
        let den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);
        if den.abs() < 1e-20 {
            return 1.0;
        }
        num / den
    }

    /// Map mean NSIM to MOS using the calibrated polynomial, clamped to [1, 5].
    pub fn mos_from_nsim(&self, nsim: f64) -> f32 {
        let mos = self.config.poly_a * nsim * nsim
            + self.config.poly_b * nsim
            + self.config.poly_c;
        mos.clamp(1.0, 5.0) as f32
    }

    /// Find the best frame shift in `[-search, +search]` by normalised cross-correlation.
    fn find_best_shift(
        patch_ref: &[Vec<f64>],   // [n_mels][patch_frames]
        s_est: &[Vec<f64>],        // [n_mels][T_est]
        center: usize,
        patch_frames: usize,
        search: i32,
    ) -> i32 {
        let t_est = if s_est.is_empty() { 0 } else { s_est[0].len() };
        let n_mels = patch_ref.len();

        // Flatten reference patch
        let ref_flat: Vec<f64> = patch_ref
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let ref_norm: f64 = ref_flat.iter().map(|&v| v * v).sum::<f64>().sqrt();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_shift = 0i32;

        for delta in -search..=search {
            let est_start = center as i32 + delta;
            if est_start < 0 || (est_start as usize) + patch_frames > t_est {
                continue;
            }
            let est_start = est_start as usize;

            // Flatten estimate patch
            let est_flat: Vec<f64> = s_est
                .iter()
                .take(n_mels)
                .flat_map(|row| row[est_start..est_start + patch_frames].iter().copied())
                .collect();
            let est_norm: f64 = est_flat.iter().map(|&v| v * v).sum::<f64>().sqrt();

            let dot: f64 = ref_flat.iter().zip(est_flat.iter()).map(|(&a, &b)| a * b).sum();
            let score = dot / (ref_norm * est_norm + 1e-10);

            if score > best_score {
                best_score = score;
                best_shift = delta;
            }
        }
        best_shift
    }

    /// Compute ViSQOL MOS score ∈ [1.0, 5.0].
    ///
    /// Higher is better.
    pub fn compute(&self, reference: &[f32], estimate: &[f32], sr: u32) -> EvalResult<f32> {
        if reference.is_empty() || estimate.is_empty() {
            return Err(EvalError::EmptySignal);
        }

        // Step 1: mel spectrograms (linear energy)
        let s_ref = self.mel_spectrogram(reference, sr);
        let s_est = self.mel_spectrogram(estimate, sr);

        let t_ref = if s_ref.is_empty() { 0 } else { s_ref[0].len() };
        let t_est = if s_est.is_empty() { 0 } else { s_est[0].len() };
        if t_ref < self.config.patch_frames || t_est < self.config.patch_frames {
            return Ok(1.0); // too short for patch extraction
        }

        // Step 2: normalise both to [0,1] for NSIM stability
        let max_ref = s_ref
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_est = s_est
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_val = max_ref.max(max_est) + 1e-10;

        let s_ref_norm: Vec<Vec<f64>> = s_ref
            .iter()
            .map(|row| row.iter().map(|&v| v / max_val).collect())
            .collect();
        let s_est_norm: Vec<Vec<f64>> = s_est
            .iter()
            .map(|row| row.iter().map(|&v| v / max_val).collect())
            .collect();

        // Step 3: extract patches and compute NSIM
        let patch_l = self.config.patch_frames;
        let step = self.config.patch_step;
        let n_mels = self.config.n_mels;
        let c1 = self.config.c1;
        let c2 = self.config.c2;

        let mut nsim_scores: Vec<f64> = Vec::new();
        let mut start = 0usize;
        while start + patch_l <= t_ref {
            // Extract reference patch: [n_mels][patch_l]
            let patch_ref: Vec<Vec<f64>> = s_ref_norm
                .iter()
                .take(n_mels)
                .map(|row| row[start..start + patch_l].to_vec())
                .collect();

            // Find best shift in estimate
            let shift = Self::find_best_shift(
                &patch_ref,
                &s_est_norm,
                start,
                patch_l,
                self.config.search_range,
            );
            let est_start =
                (start as i32 + shift).clamp(0, (t_est - patch_l) as i32) as usize;

            let patch_est_flat: Vec<f64> = s_est_norm
                .iter()
                .take(n_mels)
                .flat_map(|row| row[est_start..est_start + patch_l].iter().copied())
                .collect();
            let patch_ref_flat: Vec<f64> = patch_ref
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect();

            let score = Self::nsim(&patch_ref_flat, &patch_est_flat, c1, c2);
            nsim_scores.push(score);
            start += step;
        }

        if nsim_scores.is_empty() {
            return Ok(1.0);
        }
        let nsim_mean = nsim_scores.iter().sum::<f64>() / nsim_scores.len() as f64;
        Ok(self.mos_from_nsim(nsim_mean))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_multi(freqs: &[f32], amps: &[f32], sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                freqs
                    .iter()
                    .zip(amps.iter())
                    .map(|(&f, &a)| a * (2.0 * PI * f * i as f32 / sr as f32).sin())
                    .sum::<f32>()
            })
            .collect()
    }

    #[test]
    fn test_visqol_identical_signals() {
        let signal =
            sine_multi(&[440.0, 880.0], &[0.5, 0.3], 24000, 24000 * 2);
        let visqol = ViSQOL::new();
        let r = visqol.compute(&signal, &signal, 24000).unwrap();
        assert!(r > 4.5, "identical: ViSQOL = {:.3}", r);
        assert!(r <= 5.0, "ViSQOL must be ≤ 5.0");
    }

    #[test]
    fn test_visqol_mos_range() {
        let n = 48000usize;
        let speech: Vec<f32> = (0..n).map(|i| (i as f32 / 100.0).sin() * 0.5).collect();
        let noise: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 31337.0).sin()) * 0.5)
            .collect();
        let visqol = ViSQOL::new();
        let r = visqol.compute(&speech, &noise, 24000).unwrap();
        assert!(r >= 1.0 && r <= 5.0, "MOS out of range: {}", r);
    }

    #[test]
    fn test_nsim_identical_patches() {
        let patch: Vec<f64> = (0..24 * 30)
            .map(|i| (i as f64 / 10.0).sin().abs() + 0.1)
            .collect();
        let score = ViSQOL::nsim(&patch, &patch, 0.0001, 0.0009);
        assert!((score - 1.0).abs() < 1e-5, "NSIM of identical patches = {}", score);
    }

    #[test]
    fn test_nsim_dissimilar_patches() {
        let n = 24 * 30;
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let b: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let score = ViSQOL::nsim(&a, &b, 0.0001, 0.0009);
        assert!(score < 0.5, "dissimilar patches: NSIM = {}", score);
    }

    #[test]
    fn test_visqol_mos_poly_mapping() {
        let visqol = ViSQOL::new();
        let mos_at_1 = visqol.mos_from_nsim(1.0);
        assert!((mos_at_1 - 4.9623).abs() < 0.001, "MOS at NSIM=1: {}", mos_at_1);
        let mos_at_0 = visqol.mos_from_nsim(0.0);
        assert!((mos_at_0 - 1.0).abs() < 0.001, "MOS at NSIM=0: {}", mos_at_0);
        let mos_at_half = visqol.mos_from_nsim(0.5);
        assert!(
            (mos_at_half - 2.482).abs() < 0.01,
            "MOS at NSIM=0.5: {}",
            mos_at_half
        );
    }

    #[test]
    fn test_visqol_degraded_lower() {
        let n = 24000 * 3usize;
        let clean = sine_multi(&[200.0, 600.0, 1200.0], &[0.3, 0.3, 0.2], 24000, n);
        let noisy: Vec<f32> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 0.5 * (i as f32 * 0.999).sin())
            .collect();
        let visqol = ViSQOL::new();
        let mos_clean = visqol.compute(&clean, &clean, 24000).unwrap();
        let mos_noisy = visqol.compute(&clean, &noisy, 24000).unwrap();
        assert!(
            mos_noisy < mos_clean,
            "noisy ({:.3}) should score < clean ({:.3})",
            mos_noisy,
            mos_clean
        );
    }
}
