//! Mel Cepstral Distortion (MCD).
//!
//! Reference: Kubichek (1993) "Mel-cepstral distance measure for objective
//! speech quality assessment".
//!
//! MCD_t = (10 / ln 10) × sqrt(2 × Σ_{k=1}^{K} (c_ref[k] − c_est[k])²)
//!
//! where K = 13 cepstral coefficients (k=1..13; k=0 is the energy term, excluded).

use crate::fft::rfft_mag;
use crate::filterbank::mel_filterbank;
use crate::dct::dct2_normalized;
use crate::window::hann;
use crate::{EvalError, EvalResult};

/// MCD algorithm configuration.
pub struct McdConfig {
    /// Frame analysis window length in ms.
    pub window_ms: f32,
    /// Frame hop in ms.
    pub hop_ms: f32,
    /// Number of mel filter channels.
    pub n_mels: usize,
    /// Number of mel cepstral coefficients (k=1..n_mcc; k=0 excluded).
    pub n_mcc: usize,
    /// Minimum frequency in Hz for mel filterbank.
    pub freq_min: f32,
}

impl Default for McdConfig {
    fn default() -> Self {
        Self {
            window_ms: 25.0,
            hop_ms: 5.0,
            n_mels: 80,
            n_mcc: 13,
            freq_min: 0.0,
        }
    }
}

/// MCD metric.
pub struct Mcd {
    config: McdConfig,
}

impl Default for Mcd {
    fn default() -> Self {
        Self::new()
    }
}

impl Mcd {
    /// Create Mcd with default parameters.
    pub fn new() -> Self {
        Self {
            config: McdConfig::default(),
        }
    }

    /// Create Mcd with custom configuration.
    pub fn with_config(config: McdConfig) -> Self {
        Self { config }
    }

    fn window_samples(ms: f32, sr: u32) -> usize {
        ((ms / 1000.0) * sr as f32).round() as usize
    }

    /// Extract per-frame mel cepstral coefficients.
    ///
    /// Returns a `Vec<Vec<f64>>` of length `n_frames`, each inner vec has `n_mels` elements
    /// (the full DCT output; caller selects k=1..=n_mcc).
    fn signal_to_mcc(&self, signal: &[f32], sr: u32) -> Vec<Vec<f64>> {
        let n_window = Self::window_samples(self.config.window_ms, sr).max(1);
        let hop = Self::window_samples(self.config.hop_ms, sr).max(1);
        let n_fft = n_window.next_power_of_two();
        let freq_max = sr as f64 / 2.0;

        let mel_fb =
            mel_filterbank(self.config.n_mels, self.config.freq_min as f64, freq_max, n_fft, sr);
        let win = hann(n_window);

        let n_frames = if signal.len() >= n_window {
            (signal.len() - n_window) / hop + 1
        } else {
            0
        };

        (0..n_frames)
            .map(|t| {
                let start = t * hop;
                // Apply window
                let mut frame = vec![0.0f32; n_window];
                for (k, s) in signal.iter().skip(start).take(n_window).enumerate() {
                    frame[k] = (*s as f64 * win[k]) as f32;
                }
                // STFT magnitude
                let mag = rfft_mag(&frame, n_fft);

                // Mel energies: sum over fb weights × mag², then log
                let log_mel: Vec<f64> = mel_fb
                    .iter()
                    .map(|fb| {
                        let energy: f64 = fb
                            .iter()
                            .zip(mag.iter())
                            .map(|(&h, &m)| h * m * m)
                            .sum();
                        (energy + 1e-10).ln()
                    })
                    .collect();

                // DCT-II → mel cepstrum
                dct2_normalized(&log_mel)
            })
            .collect()
    }

    /// Compute MCD in dB between `reference` and `estimate` at `sr` Hz.
    ///
    /// Lower is better. Returns 0.0 for identical signals.
    /// Frame pairs are aligned by index (truncated to shorter signal).
    pub fn compute(&self, reference: &[f32], estimate: &[f32], sr: u32) -> EvalResult<f32> {
        if reference.is_empty() || estimate.is_empty() {
            return Err(EvalError::EmptySignal);
        }

        let mcc_ref = self.signal_to_mcc(reference, sr);
        let mcc_est = self.signal_to_mcc(estimate, sr);

        let t = mcc_ref.len().min(mcc_est.len());
        if t == 0 {
            return Err(EvalError::EmptySignal);
        }

        let factor = 10.0_f64 / 10.0_f64.ln(); // ≈ 4.3429
        let mcd_sum: f64 = (0..t)
            .map(|i| {
                let sq_sum: f64 = (1..=self.config.n_mcc)
                    .map(|k| {
                        // DCT output has n_mels elements; guard bounds
                        let cr = if k < mcc_ref[i].len() { mcc_ref[i][k] } else { 0.0 };
                        let ce = if k < mcc_est[i].len() { mcc_est[i][k] } else { 0.0 };
                        let d = cr - ce;
                        d * d
                    })
                    .sum();
                factor * (2.0 * sq_sum).sqrt()
            })
            .sum();

        Ok((mcd_sum / t as f64) as f32)
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
    fn test_mcd_identical() {
        let signal = sine(500.0, 24000, 24000);
        let mcd = Mcd::new();
        let r = mcd.compute(&signal, &signal, 24000).unwrap();
        assert!(r.abs() < 1e-4, "MCD of identical signals = {}", r);
    }

    #[test]
    fn test_mcd_different_tones() {
        let n = 24000usize;
        let s1 = sine(440.0, 24000, n);
        let s2 = sine(880.0, 24000, n);
        let mcd = Mcd::new();
        let r = mcd.compute(&s1, &s2, 24000).unwrap();
        assert!(r > 1.0, "different tones: MCD = {}", r);
    }

    #[test]
    fn test_mcd_non_negative() {
        let s1: Vec<f32> = (0..8000).map(|i| (i as f32 / 80.0).sin()).collect();
        let s2: Vec<f32> = (0..8000).map(|i| (i as f32 / 160.0).sin()).collect();
        let mcd = Mcd::new();
        let r = mcd.compute(&s1, &s2, 16000).unwrap();
        assert!(r >= 0.0, "MCD must be non-negative: {}", r);
    }

    #[test]
    fn test_mcd_scaled_signal() {
        // MCD is NOT scale-invariant
        let s: Vec<f32> = (0..16000).map(|i| (i as f32 / 100.0).sin() * 0.1).collect();
        let s_loud: Vec<f32> = s.iter().map(|&x| x * 10.0).collect();
        let mcd = Mcd::new();
        let r = mcd.compute(&s, &s_loud, 16000).unwrap();
        assert!(r > 0.1, "scaled signal should have non-zero MCD: {}", r);
    }

    #[test]
    fn test_mcd_factor_check() {
        // (10 / ln10) × sqrt(2 × 13 × 1) ≈ 22.14 dB
        let factor = 10.0_f32 / 10.0_f32.ln();
        let expected = factor * (2.0_f32 * 13.0).sqrt();
        assert!((expected - 22.14).abs() < 0.1, "factor check: {}", expected);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = crate::filterbank::mel_filterbank(80, 0.0, 8000.0, 1024, 16000);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), 513);
    }

    #[test]
    fn test_hz_to_mel_roundtrip() {
        use crate::filterbank::{hz_to_mel, mel_to_hz};
        let f = 1000.0_f64;
        let m = hz_to_mel(f);
        let f2 = mel_to_hz(m);
        assert!((f - f2).abs() < 1e-6, "roundtrip error: {}", f2);
    }
}
