//! opensesame-train — 3-stage training pipeline (codec → backbone → finetune).
//! Phase J implementation target: 20 tests.
pub mod config;
pub mod codec_trainer;
pub mod csm_trainer;
pub mod loss;
pub mod checkpoint;
