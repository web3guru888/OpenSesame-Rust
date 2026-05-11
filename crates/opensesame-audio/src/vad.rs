//! Simple Voice Activity Detection (VAD) using energy and zero-crossing rate.
//!
//! The detector classifies each audio frame as speech or silence by comparing
//! its short-term energy and zero-crossing rate against configurable
//! thresholds.  A **hangover** mechanism keeps the detector active for a
//! number of frames after energy drops below the threshold, preventing
//! spurious cuts in the middle of speech.
//!
//! ## Algorithm
//! 1. Compute frame energy: `E = mean(x²)`.
//! 2. Compute zero-crossing rate: `ZCR = (sign changes) / frame_len`.
//! 3. Mark frame as speech if `E > energy_threshold`.
//! 4. Apply hangover: stay active for `hangover_frames` frames after speech
//!    ends.

/// Voice Activity Detector state.
///
/// Call [`VadState::is_speech`] on each consecutive audio frame to obtain
/// a binary speech / silence decision.
pub struct VadState {
    /// Minimum frame energy to trigger speech detection.
    ///
    /// Default: `0.01` (corresponds to roughly -20 dB FS).
    pub energy_threshold: f32,
    /// Zero-crossing rate threshold (unused in the current decision rule but
    /// preserved for future extensions).
    ///
    /// Default: `0.3`.
    pub zcr_threshold: f32,
    /// Number of frames to stay active after energy drops below threshold.
    ///
    /// Default: `8`.
    pub hangover_frames: usize,
    /// Internal counter for the hangover period.
    active_count: usize,
}

impl VadState {
    /// Create a `VadState` with default thresholds.
    pub fn new() -> Self {
        Self {
            energy_threshold: 0.01,
            zcr_threshold: 0.3,
            hangover_frames: 8,
            active_count: 0,
        }
    }

    /// Create a `VadState` with custom energy and ZCR thresholds.
    ///
    /// # Parameters
    /// * `energy` – energy threshold (mean squared amplitude per sample)
    /// * `zcr`    – zero-crossing rate threshold `[0, 1]`
    pub fn with_thresholds(energy: f32, zcr: f32) -> Self {
        Self {
            energy_threshold: energy,
            zcr_threshold: zcr,
            hangover_frames: 8,
            active_count: 0,
        }
    }

    /// Classify `frame` as speech (`true`) or silence (`false`).
    ///
    /// Updates internal hangover state. Should be called with consecutive,
    /// non-overlapping frames of the same length.
    ///
    /// # Parameters
    /// * `frame` – a slice of PCM samples for one analysis frame
    pub fn is_speech(&mut self, frame: &[f32]) -> bool {
        if frame.is_empty() {
            return false;
        }

        let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;

        if energy > self.energy_threshold {
            // Active speech detected — reset hangover counter.
            self.active_count = self.hangover_frames;
            true
        } else if self.active_count > 0 {
            // In hangover period.
            self.active_count -= 1;
            true
        } else {
            false
        }
    }

    /// Reset the detector to its initial silent state.
    ///
    /// Clears the hangover counter.
    pub fn reset(&mut self) {
        self.active_count = 0;
    }

    /// Compute zero-crossing rate for `frame`.
    ///
    /// Returns the fraction of sample transitions where the sign changes,
    /// in `[0, 1]`.
    pub fn zero_crossing_rate(frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }
        let crossings = frame
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        crossings as f32 / (frame.len() - 1) as f32
    }
}

impl Default for VadState {
    fn default() -> Self {
        Self::new()
    }
}
