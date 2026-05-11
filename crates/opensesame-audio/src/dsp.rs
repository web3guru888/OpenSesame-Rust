//! Basic DSP utility functions for audio processing.
//!
//! All functions operate on `f32` sample slices and are pure (no hidden
//! state).  They are designed to be composable with [`AudioBuffer`] and other
//! modules in this crate.
//!
//! [`AudioBuffer`]: crate::audio_buffer::AudioBuffer

use std::f32::consts::PI;

/// Compute the Root-Mean-Square (RMS) amplitude of `samples`.
///
/// Returns `0.0` for an empty slice.
///
/// # Example
/// ```
/// use opensesame_audio::dsp::rms;
/// let v = rms(&[0.0, 1.0, 0.0, -1.0]);
/// assert!((v - 0.5_f32.sqrt()).abs() < 1e-6);
/// ```
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Return the maximum absolute value in `samples`.
///
/// Returns `0.0` for an empty slice.
///
/// # Example
/// ```
/// use opensesame_audio::dsp::peak;
/// assert_eq!(peak(&[-0.5, 0.3, 0.7]), 0.7);
/// ```
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().cloned().fold(0.0_f32, |acc, x| acc.max(x.abs()))
}

/// Compute the mean (DC offset) of `samples`.
///
/// Returns `0.0` for an empty slice.
pub fn dc_offset(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f32>() / samples.len() as f32
}

/// Multiply every sample by `gain` in-place.
///
/// # Parameters
/// * `samples` – mutable sample slice
/// * `gain`    – linear gain factor
pub fn apply_gain(samples: &mut [f32], gain: f32) {
    for s in samples {
        *s *= gain;
    }
}

/// Clamp every sample to `[-ceiling, ceiling]` in-place.
///
/// # Parameters
/// * `samples` – mutable sample slice
/// * `ceiling` – maximum absolute amplitude (must be ≥ 0)
pub fn clip(samples: &mut [f32], ceiling: f32) {
    for s in samples {
        *s = s.clamp(-ceiling, ceiling);
    }
}

/// Append zero-padding to `samples` until `target_len` is reached.
///
/// If `samples.len() >= target_len` the input is returned unchanged (cloned).
///
/// # Parameters
/// * `samples`    – input samples
/// * `target_len` – desired output length
pub fn zero_pad(samples: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = samples.to_vec();
    if out.len() < target_len {
        out.resize(target_len, 0.0);
    }
    out
}

/// Generate a Hann window of length `size`.
///
/// The window is defined as:
/// ```text
/// w[n] = 0.5 * (1 − cos(2π n / (N − 1)))   for n = 0..N
/// ```
///
/// The first and last sample are (approximately) zero, which avoids spectral
/// leakage at the boundaries.
///
/// # Parameters
/// * `size` – number of window coefficients
pub fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = (size - 1) as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n).cos()))
        .collect()
}

/// Multiply `samples` element-wise by `window` in-place.
///
/// `window` must have the same length as `samples`; if `window` is shorter,
/// only the first `window.len()` samples are modified.
///
/// # Parameters
/// * `samples` – mutable sample slice
/// * `window`  – window coefficients
pub fn apply_window(samples: &mut [f32], window: &[f32]) {
    let len = samples.len().min(window.len());
    for i in 0..len {
        samples[i] *= window[i];
    }
}
