//! Training configuration for OpenSesame CSM.
//!
//! `TrainConfig` centralises all hyperparameters needed to reproduce a training
//! run.  A `Default` implementation reflects the Speechmatics fine-tuning
//! settings (lr=3e-5, batch_size=8, decoder_loss_weight=0.5).

use std::fmt;

/// Fraction of audio frames fed through the depth-decoder per step
/// (compute amortization: 1/16 = 0.0625).
pub const DECODER_AMORT: f32 = 1.0 / 16.0;

/// Balanced backbone / decoder loss weight (λ = 0.5).
pub const DECODER_LOSS_WEIGHT: f32 = 0.5;

/// Training hyperparameters for the OpenSesame CSM.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    // ── Learning rate ───────────────────────────────────────────────────────
    /// Peak learning rate (3e-5 for fine-tuning).
    pub lr: f32,
    /// Cosine decay floor (1e-6).
    pub min_lr: f32,
    /// Linear warm-up steps before cosine decay begins.
    pub warmup_steps: usize,
    /// Total optimiser steps.
    pub max_steps: usize,

    // ── Batch ───────────────────────────────────────────────────────────────
    /// Number of samples per gradient step.
    pub batch_size: usize,
    /// Maximum Mimi frames per sample (longer sequences are truncated).
    pub max_audio_frames: usize,

    // ── Regularisation ──────────────────────────────────────────────────────
    /// AdamW weight-decay coefficient (0.1).
    pub weight_decay: f32,
    /// Global gradient-clip norm (1.0).
    pub grad_clip: f32,
    /// Dropout probability (0.0 — disabled for CSM inference stability).
    pub dropout: f32,

    // ── Loss ────────────────────────────────────────────────────────────────
    /// λ in `loss = (1-λ)·c0_loss + λ·decoder_loss`.
    /// Confirmed 0.5 from Moshi §4.2 and Speechmatics sweep config.
    pub decoder_loss_weight: f32,
    /// Fraction of audio frames to train the depth-decoder on per step.
    /// 1/16 ≈ 0.0625 (compute amortization as described in the Moshi paper).
    pub decoder_amort: f32,

    // ── Checkpointing ───────────────────────────────────────────────────────
    /// Directory where checkpoint sub-dirs are written.
    pub checkpoint_dir: String,
    /// Save a checkpoint every N steps.
    pub checkpoint_every: usize,
    /// Run evaluation every N steps.
    pub eval_every: usize,
    /// Keep only the N most-recent checkpoints on disk.
    pub keep_last_n: usize,

    // ── Logging ─────────────────────────────────────────────────────────────
    /// Log metrics every N steps.
    pub log_every: usize,
    /// Random seed for reproducibility.
    pub seed: u64,

    // ── Freezing ────────────────────────────────────────────────────────────
    /// When true, the backbone body is frozen and only the output heads are
    /// updated (Phase J.0 fine-tuning mode).
    pub freeze_backbone: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr:                  3e-5,
            min_lr:              1e-6,
            warmup_steps:        1_000,
            max_steps:           500_000,
            batch_size:          8,
            max_audio_frames:    256,
            weight_decay:        0.1,
            grad_clip:           1.0,
            dropout:             0.0,
            decoder_loss_weight: DECODER_LOSS_WEIGHT,
            decoder_amort:       DECODER_AMORT,
            checkpoint_dir:      "./checkpoints".to_string(),
            checkpoint_every:    5_000,
            eval_every:          2_500,
            keep_last_n:         3,
            log_every:           100,
            seed:                42,
            freeze_backbone:     false,
        }
    }
}

impl TrainConfig {
    /// Default config (alias for `Default::default()`).
    pub fn default_config() -> Self { Self::default() }

    /// Fast config for unit tests: 10 steps, batch=1, warmup=2.
    pub fn fast_test() -> Self {
        Self {
            lr:                3e-4,
            max_steps:         10,
            batch_size:        1,
            warmup_steps:      2,
            checkpoint_dir:    "/tmp/opensesame_test_ckpt".to_string(),
            checkpoint_every:  5,
            eval_every:        5,
            keep_last_n:       2,
            log_every:         1,
            ..Self::default()
        }
    }

    /// Serialise to a JSON string (atlas-json free-format).
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                r#"{{"lr":{lr},"min_lr":{min_lr},"warmup_steps":{warmup},"max_steps":{max},"#,
                r#""batch_size":{bs},"max_audio_frames":{maf},"weight_decay":{wd},"#,
                r#""grad_clip":{gc},"dropout":{do_},"decoder_loss_weight":{dlw},"#,
                r#""decoder_amort":{da},"checkpoint_dir":"{cd}","checkpoint_every":{ce},"#,
                r#""eval_every":{ee},"keep_last_n":{kl},"log_every":{le},"seed":{seed},"#,
                r#""freeze_backbone":{fb}}}"#
            ),
            lr   = self.lr,
            min_lr = self.min_lr,
            warmup = self.warmup_steps,
            max  = self.max_steps,
            bs   = self.batch_size,
            maf  = self.max_audio_frames,
            wd   = self.weight_decay,
            gc   = self.grad_clip,
            do_  = self.dropout,
            dlw  = self.decoder_loss_weight,
            da   = self.decoder_amort,
            cd   = self.checkpoint_dir,
            ce   = self.checkpoint_every,
            ee   = self.eval_every,
            kl   = self.keep_last_n,
            le   = self.log_every,
            seed = self.seed,
            fb   = self.freeze_backbone,
        )
    }

    /// Deserialise from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, String> {
        use atlas_json::Json;
        let v = Json::parse(s).map_err(|e| e.to_string())?;
        let get_f32 = |key: &str, default: f32| -> f32 {
            v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(default)
        };
        let get_u = |key: &str, default: usize| -> usize {
            v.get(key).and_then(|x| x.as_usize()).unwrap_or(default)
        };
        let get_u64 = |key: &str, default: u64| -> u64 {
            v.get(key).and_then(|x| x.as_i64()).map(|x| x as u64).unwrap_or(default)
        };
        let get_bool = |key: &str, default: bool| -> bool {
            v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
        };
        let get_str = |key: &str, default: &str| -> String {
            v.get(key).and_then(|x| x.as_str()).unwrap_or(default).to_string()
        };
        Ok(Self {
            lr:                  get_f32("lr", 3e-5),
            min_lr:              get_f32("min_lr", 1e-6),
            warmup_steps:        get_u("warmup_steps", 1_000),
            max_steps:           get_u("max_steps", 500_000),
            batch_size:          get_u("batch_size", 8),
            max_audio_frames:    get_u("max_audio_frames", 256),
            weight_decay:        get_f32("weight_decay", 0.1),
            grad_clip:           get_f32("grad_clip", 1.0),
            dropout:             get_f32("dropout", 0.0),
            decoder_loss_weight: get_f32("decoder_loss_weight", DECODER_LOSS_WEIGHT),
            decoder_amort:       get_f32("decoder_amort", DECODER_AMORT),
            checkpoint_dir:      get_str("checkpoint_dir", "./checkpoints"),
            checkpoint_every:    get_u("checkpoint_every", 5_000),
            eval_every:          get_u("eval_every", 2_500),
            keep_last_n:         get_u("keep_last_n", 3),
            log_every:           get_u("log_every", 100),
            seed:                get_u64("seed", 42),
            freeze_backbone:     get_bool("freeze_backbone", false),
        })
    }
}

impl fmt::Display for TrainConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TrainConfig(lr={}, steps={}, bs={})",
            self.lr, self.max_steps, self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let cfg = TrainConfig::default();
        assert!((cfg.lr - 3e-5).abs() < 1e-8, "lr should be 3e-5, got {}", cfg.lr);
        assert!((cfg.decoder_loss_weight - 0.5).abs() < 1e-6,
            "decoder_loss_weight should be 0.5");
        assert!((cfg.decoder_amort - 0.0625).abs() < 1e-6,
            "decoder_amort should be 0.0625 (1/16)");
        assert_eq!(cfg.batch_size, 8);
        assert_eq!(cfg.max_steps, 500_000);
        assert_eq!(cfg.warmup_steps, 1_000);
        assert_eq!(cfg.keep_last_n, 3);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_config_fast_test() {
        let cfg = TrainConfig::fast_test();
        assert_eq!(cfg.max_steps, 10);
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.warmup_steps, 2);
    }

    #[test]
    fn test_config_serialization() {
        let cfg = TrainConfig::default();
        let json = cfg.to_json();
        let cfg2 = TrainConfig::from_json(&json).expect("roundtrip");
        assert!((cfg2.lr - cfg.lr).abs() < 1e-9, "lr roundtrip");
        assert!((cfg2.decoder_loss_weight - cfg.decoder_loss_weight).abs() < 1e-6);
        assert_eq!(cfg2.batch_size, cfg.batch_size);
        assert_eq!(cfg2.seed, cfg.seed);
        assert_eq!(cfg2.checkpoint_dir, cfg.checkpoint_dir);
    }
}
