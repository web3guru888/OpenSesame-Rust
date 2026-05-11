//! Sinc interpolation resampler with Kaiser window.
//!
//! Implements a Kaiser-windowed sinc filter suitable for the common
//! audio rate conversions used in OpenSesame:
//! * 44 100 Hz → 24 000 Hz
//! * 48 000 Hz → 24 000 Hz
//! * 16 000 Hz → 24 000 Hz
//! * 22 050 Hz → 24 000 Hz
//!
//! ## Algorithm
//! For each output sample at index `n`, the corresponding continuous-time
//! position in the input stream is `pos = n * in_rate / out_rate`. The output
//! value is computed as the inner product of the input samples surrounding
//! `pos` with the Kaiser-windowed sinc filter evaluated at the fractional
//! offset of each tap:
//!
//! ```text
//! y[n] = Σ_k  x[⌊pos⌋ + k]  ·  h(k - frac)
//! ```
//!
//! where `frac = pos − ⌊pos⌋` and `h(t) = 2·fc·sinc(2·fc·t)·w(t/L)`.
//!
//! Parameters: β = 8.0, 64 filter taps (±32 taps around the centre).
//!
//! The GPU path (resample.cu) is invoked via atlas-tensor in Phase B.

use std::f64::consts::PI;

/// Kaiser-windowed sinc resampler (CPU implementation).
pub struct Resampler {
    /// Numerator of the exact rate ratio after GCD reduction (out_rate / gcd).
    pub ratio_num: u32,
    /// Denominator of the exact rate ratio after GCD reduction (in_rate / gcd).
    pub ratio_den: u32,
}

impl Resampler {
    /// Half the number of taps on each side of centre (total = 2 * HALF_TAPS).
    const HALF_TAPS: usize = 32;
    /// Kaiser window shape parameter β.
    const BETA: f64 = 8.0;

    /// Construct a `Resampler` for the given in/out rate pair.
    ///
    /// The ratio is reduced to lowest terms via GCD.
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let g = gcd(in_rate, out_rate);
        Self { ratio_num: out_rate / g, ratio_den: in_rate / g }
    }

    /// Resample `input` from `in_rate` to `out_rate`.
    ///
    /// Returns a new `Vec<f32>` with approximately
    /// `ceil(input.len() * out_rate / in_rate)` samples.
    ///
    /// If `in_rate == out_rate`, returns a clone of the input unchanged.
    pub fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
        if in_rate == out_rate {
            return input.to_vec();
        }
        if input.is_empty() {
            return Vec::new();
        }

        let r = Resampler::new(in_rate, out_rate);
        r.process(input)
    }

    /// Core resampling loop.
    fn process(&self, input: &[f32]) -> Vec<f32> {
        let in_len = input.len();
        // out/in ratio — > 1 means upsampling, < 1 means downsampling.
        let ratio = self.ratio_num as f64 / self.ratio_den as f64;
        let out_len = ((in_len as f64 * ratio).ceil() as usize).max(1);

        // Anti-aliasing cutoff: min(out_rate, in_rate) / 2, normalised to in_rate.
        // For downsampling we attenuate above the output Nyquist.
        let fc = 0.5 * ratio.min(1.0);

        let half = Self::HALF_TAPS as isize;
        let l = half as f64; // window half-length

        let mut out = Vec::with_capacity(out_len);

        for n in 0..out_len {
            // Continuous position in the input domain.
            let pos = n as f64 / ratio;
            let pos_int = pos.floor() as isize;
            let frac = pos - pos.floor(); // fractional part ∈ [0, 1)

            let mut sum = 0.0_f64;
            // Sum over taps k = -half .. +half  (symmetric, 2*half+1 taps).
            for k in -half..=half {
                let src_idx = pos_int + k;
                if src_idx < 0 || src_idx >= in_len as isize {
                    // Zero-pad implicitly.
                    continue;
                }
                // Evaluate the Kaiser-windowed sinc at (k - frac).
                let t = k as f64 - frac;
                let h = kaiser_sinc(t, fc, l, Self::BETA);
                sum += input[src_idx as usize] as f64 * h;
            }
            out.push(sum as f32);
        }

        out
    }
}

/// Evaluate a Kaiser-windowed sinc at position `t`.
///
/// `h(t) = 2·fc · sinc(2·fc·t) · kaiser(t / L, β)`
///
/// # Parameters
/// * `t`    – tap position (may be fractional)
/// * `fc`   – normalised cut-off frequency (0..0.5)
/// * `l`    – window half-length in samples
/// * `beta` – Kaiser window β
#[inline]
fn kaiser_sinc(t: f64, fc: f64, l: f64, beta: f64) -> f64 {
    // Sinc component.
    let sinc = if t.abs() < 1e-10 {
        2.0 * fc
    } else {
        (2.0 * fc * PI * t).sin() / (PI * t)
    };

    // Kaiser window component (zero outside [-L, L]).
    let r = t / l;
    if r.abs() > 1.0 {
        return 0.0;
    }
    let w = bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta);

    sinc * w
}

/// Zeroth-order modified Bessel function `I₀(x)` via series expansion.
///
/// Accurate to machine precision for `x ≤ 20` (sufficient for β ≤ 12).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let x2 = (x / 2.0) * (x / 2.0);
    for k in 1..=35 {
        term *= x2 / (k as f64 * k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Compute the greatest common divisor of `a` and `b` (Euclidean algorithm).
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bessel_i0_zero() {
        // I₀(0) = 1.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(44100, 24000), 300);
        assert_eq!(gcd(48000, 24000), 24000);
        assert_eq!(gcd(16000, 24000), 8000);
    }

    #[test]
    fn test_kaiser_sinc_centre() {
        // At t = 0 the function should equal 2·fc.
        let fc = 0.5;
        let v = kaiser_sinc(0.0, fc, 32.0, 8.0);
        assert!((v - 2.0 * fc).abs() < 1e-10, "centre: {}", v);
    }
}
