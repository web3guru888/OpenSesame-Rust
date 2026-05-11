//! Simple linear-interpolation resampler.
//!
//! Accuracy is sufficient for evaluation purposes (STOI, ViSQOL).
//! For production use, prefer the polyphase sinc resampler in `opensesame-audio`.

/// Resample `src` from `src_rate` Hz to `dst_rate` Hz using linear interpolation.
///
/// If `src_rate == dst_rate`, returns a clone of `src`.
pub fn resample_linear(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return src.to_vec();
    }
    if src.is_empty() {
        return vec![];
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let dst_len = ((src.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(dst_len);
    for i in 0..dst_len {
        let src_pos = i as f64 * ratio;
        let lo = src_pos.floor() as usize;
        let hi = (lo + 1).min(src.len() - 1);
        let frac = src_pos - lo as f64;
        let sample = src[lo] as f64 * (1.0 - frac) + src[hi] as f64 * frac;
        out.push(sample as f32);
    }
    out
}
