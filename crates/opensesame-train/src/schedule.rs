//! Learning-rate schedule for CSM training.
//!
//! `CosineSchedule` wraps `atlas_optim::CosineScheduler` and adds a
//! convenience constructor that accepts the `TrainConfig` fields directly.
//!
//! Schedule:
//! - **Warm-up**: linear ramp from 0 → `base_lr` over `warmup_steps`.
//! - **Decay**: cosine anneal from `base_lr` → `min_lr` over the remaining
//!   steps.

use atlas_optim::CosineScheduler;
use crate::config::TrainConfig;

/// Cosine LR schedule with linear warm-up.
///
/// ```
/// use opensesame_train::CosineSchedule;
/// let s = CosineSchedule::new(1e-4, 1e-6, 1000, 100_000);
/// assert!(s.lr_at(0) == 0.0);
/// assert!((s.lr_at(1000) - 1e-4).abs() < 1e-8);
/// ```
#[derive(Debug, Clone)]
pub struct CosineSchedule {
    inner: CosineScheduler,
}

impl CosineSchedule {
    /// Create a new schedule.
    ///
    /// - `base_lr`      — peak learning rate (after warm-up).
    /// - `min_lr`       — cosine floor.
    /// - `warmup_steps` — linear warm-up length in steps.
    /// - `total_steps`  — total training steps.
    pub fn new(base_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            inner: CosineScheduler::new(base_lr, min_lr, total_steps as u64, warmup_steps as u64),
        }
    }

    /// Build from a `TrainConfig`.
    pub fn from_config(cfg: &TrainConfig) -> Self {
        Self::new(cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
    }

    /// Compute learning rate at step `t` (0-indexed).
    ///
    /// `t = 0` returns 0.0 (start of warm-up).
    /// `t = warmup_steps` returns `base_lr` (end of warm-up).
    /// `t >= total_steps` is clamped to `min_lr`.
    pub fn lr_at(&self, t: usize) -> f32 {
        self.inner.lr(t as u64)
    }

    /// Apply this schedule to an `atlas_optim::AdamW` optimizer.
    pub fn apply_to_opt(&self, opt: &mut atlas_optim::AdamW, step: usize) {
        self.inner.apply(opt, step as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_warmup_start_zero() {
        let s = CosineSchedule::new(1e-4, 1e-6, 1000, 100_000);
        assert_eq!(s.lr_at(0), 0.0, "LR at step 0 should be 0 (warm-up start)");
    }

    #[test]
    fn test_schedule_warmup_end_base_lr() {
        // step=1000 == warmup_steps → should equal base_lr
        let s = CosineSchedule::new(1e-4, 1e-6, 1000, 100_000);
        let lr = s.lr_at(1000);
        assert!((lr - 1e-4).abs() < 1e-8,
            "LR at warmup end should be base_lr=1e-4, got {}", lr);
    }

    #[test]
    fn test_schedule_cosine_halfway() {
        // step=50500, warmup=1000, max=100000, base=1e-4, min=1e-5
        // progress = (50500-1000)/(100000-1000) = 49500/99000 = 0.5
        // lr = min + 0.5*(base-min)*(1+cos(π*0.5)) = min + 0.5*(base-min)*1 = min+0.5*(base-min)
        // = 1e-5 + 0.5*(1e-4-1e-5) = 1e-5 + 0.5*9e-5 = 1e-5 + 4.5e-5 = 5.5e-5
        let s = CosineSchedule::new(1e-4, 1e-5, 1000, 100_000);
        let lr = s.lr_at(50_500);
        let expected = 5.5e-5f32;
        assert!((lr - expected).abs() < 1e-7,
            "Cosine halfway: expected {}, got {}", expected, lr);
    }

    #[test]
    fn test_schedule_cosine_end_min_lr() {
        let s = CosineSchedule::new(1e-4, 1e-6, 1000, 100_000);
        let lr = s.lr_at(100_000);
        assert!((lr - 1e-6).abs() < 1e-8,
            "LR at total_steps should approach min_lr=1e-6, got {}", lr);
    }

    #[test]
    fn test_schedule_monotone_after_warmup() {
        let s = CosineSchedule::new(1e-3, 1e-6, 10, 1000);
        let mut prev = s.lr_at(10);
        for t in 11..=1000 {
            let cur = s.lr_at(t);
            assert!(cur <= prev + 1e-10,
                "LR increased at step {}: {} → {}", t, prev, cur);
            prev = cur;
        }
    }

    #[test]
    fn test_schedule_warmup_linear_ramp() {
        // step=500, warmup=1000, base_lr=1e-4 → lr = 1e-4 * 500/1000 = 5e-5
        let s = CosineSchedule::new(1e-4, 1e-6, 1000, 100_000);
        let lr = s.lr_at(500);
        let expected = 5e-5f32;
        assert!((lr - expected).abs() < 1e-8,
            "Warmup linear ramp: expected {}, got {}", expected, lr);
    }
}
