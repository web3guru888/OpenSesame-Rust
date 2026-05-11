//! `AudioBuffer` — interleaved PCM sample storage with common DSP helpers.
//!
//! Samples are always stored as `f32` in the normalized range `[-1.0, 1.0]`.
//! For stereo audio the samples are interleaved: `[L0, R0, L1, R1, …]`.

use crate::resample::Resampler;

/// Interleaved PCM audio buffer, normalized to `[-1.0, 1.0]`.
///
/// The `samples` field stores all channel data interleaved:
/// for `channels = 2`, index `i` refers to frame `i/2` and
/// channel `i % 2` (0 = left, 1 = right).
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved PCM samples, normalized to `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g. 44100, 48000, 24000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u8,
}

impl AudioBuffer {
    /// Create a new `AudioBuffer` from raw interleaved samples.
    ///
    /// # Parameters
    /// * `samples`     – interleaved f32 PCM data, normalized `[-1, 1]`
    /// * `sample_rate` – sample rate in Hz
    /// * `channels`    – channel count (1 or 2 are the most common values)
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u8) -> Self {
        Self { samples, sample_rate, channels }
    }

    /// Duration of the buffer in seconds.
    ///
    /// Returns `0.0` for an empty buffer regardless of `sample_rate`.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.num_frames() as f32 / self.sample_rate as f32
    }

    /// Number of audio frames (samples per channel).
    ///
    /// For mono this equals `samples.len()`; for stereo it is `samples.len() / 2`.
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Down-mix to mono by averaging all channels.
    ///
    /// If the buffer is already mono, returns a clone.
    pub fn to_mono(&self) -> AudioBuffer {
        if self.channels == 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let frames = self.num_frames();
        let mut mono = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut sum = 0.0_f32;
            for c in 0..ch {
                sum += self.samples[f * ch + c];
            }
            mono.push(sum / ch as f32);
        }
        AudioBuffer::new(mono, self.sample_rate, 1)
    }

    /// Scale all samples so the absolute peak equals `peak`.
    ///
    /// If all samples are zero, the buffer is left unchanged.
    ///
    /// # Parameters
    /// * `peak` – target absolute peak amplitude (e.g. `0.9`)
    pub fn normalize(&mut self, peak: f32) {
        let current_peak = self.samples.iter().cloned().fold(0.0_f32, |acc, x| acc.max(x.abs()));
        if current_peak < 1e-9 {
            return;
        }
        let gain = peak / current_peak;
        for s in &mut self.samples {
            *s *= gain;
        }
    }

    /// Extract a time-domain segment `[start_s, end_s)` from the buffer.
    ///
    /// Clamps indices to the buffer bounds; returns an empty `AudioBuffer`
    /// if `start_s >= end_s`.
    ///
    /// # Parameters
    /// * `start_s` – segment start in seconds (inclusive)
    /// * `end_s`   – segment end in seconds (exclusive)
    pub fn segment(&self, start_s: f32, end_s: f32) -> AudioBuffer {
        let ch = self.channels as usize;
        let sr = self.sample_rate as f32;
        let total_frames = self.num_frames();

        let start_frame = ((start_s * sr) as usize).min(total_frames);
        let end_frame = ((end_s * sr) as usize).min(total_frames);

        if start_frame >= end_frame {
            return AudioBuffer::new(Vec::new(), self.sample_rate, self.channels);
        }

        let slice = &self.samples[start_frame * ch..end_frame * ch];
        AudioBuffer::new(slice.to_vec(), self.sample_rate, self.channels)
    }

    /// Resample the buffer to `target_rate` using sinc interpolation.
    ///
    /// Calls [`Resampler::resample`] channel-by-channel and re-interleaves
    /// the result.
    ///
    /// # Parameters
    /// * `target_rate` – desired output sample rate in Hz
    pub fn resample(&self, target_rate: u32) -> AudioBuffer {
        if self.sample_rate == target_rate {
            return self.clone();
        }
        let ch = self.channels as usize;
        let frames = self.num_frames();

        // De-interleave, resample, re-interleave.
        let mut resampled_channels: Vec<Vec<f32>> = Vec::with_capacity(ch);
        for c in 0..ch {
            let channel: Vec<f32> = (0..frames).map(|f| self.samples[f * ch + c]).collect();
            let out = Resampler::resample(&channel, self.sample_rate, target_rate);
            resampled_channels.push(out);
        }

        // Re-interleave
        let out_frames = resampled_channels[0].len();
        let mut interleaved = Vec::with_capacity(out_frames * ch);
        for f in 0..out_frames {
            for c in 0..ch {
                interleaved.push(resampled_channels[c][f]);
            }
        }
        AudioBuffer::new(interleaved, target_rate, self.channels)
    }
}
