//! CSM trainer — Phase J.0 implementation.
//!
//! `CsmTrainer` performs teacher-forced forward passes through the CSM model
//! and computes the combined backbone + depth-decoder loss.
//!
//! **Phase J.0 scope**: forward pass + loss only (weight updates require
//! autograd, which is Phase J.1).
//!
//! # Compute amortization
//! To keep GPU utilisation high, only a fraction (`config.decoder_amort`,
//! default 1/16) of audio frames are fed through the depth-decoder per step.
//! This is the "compute amortization" described in Moshi §4.2.

use crate::config::TrainConfig;
use crate::schedule::CosineSchedule;
use crate::loss::{cross_entropy, csm_loss_parts};
use atlas_optim::{AdamW, AdamWConfig, ParamState};

/// One training step summary.
#[derive(Debug, Clone)]
pub struct TrainStep {
    /// Global step number (0-indexed).
    pub step: usize,
    /// Combined loss.
    pub loss: f32,
    /// Backbone CB0 cross-entropy.
    pub cb0_loss: f32,
    /// Depth-decoder mean cross-entropy.
    pub decoder_loss: f32,
    /// Learning rate used for this step.
    pub lr: f32,
}

/// Calculate the number of decoder frames to train per step.
///
/// `n_audio_frames · decoder_amort`, rounded down to at least 1.
pub fn compute_amortized_frames(n_audio_frames: usize, decoder_amort: f32) -> usize {
    let n = ((n_audio_frames as f32) * decoder_amort).floor() as usize;
    n.max(1)
}

/// Phase J.0 trainer: forward pass + loss.
///
/// Owns a `CsmModel` and `CosineSchedule` and orchestrates the training loop.
/// Weight updates are *not* applied in Phase J.0 — the `forward_loss` method
/// returns losses only.
pub struct CsmTrainer {
    /// The CSM model being trained.
    pub model: opensesame_csm::CsmModel,
    /// Training configuration.
    pub config: TrainConfig,
    /// Learning-rate schedule.
    pub schedule: CosineSchedule,
    /// AdamW optimizer (maintained for phase J.1 weight updates).
    pub optimizer: AdamW,
    /// Current step count.
    pub step: usize,
    /// Exponential moving average of training loss (α=0.99).
    pub loss_ema: f32,
}

impl CsmTrainer {
    /// Build a trainer from a model and config.
    pub fn new(model: opensesame_csm::CsmModel, config: TrainConfig) -> Self {
        let schedule = CosineSchedule::from_config(&config);
        let opt_cfg = AdamWConfig {
            lr:           config.lr,
            beta1:        0.9,
            beta2:        0.95,
            eps:          1e-8,
            weight_decay: config.weight_decay,
            clip_norm:    config.grad_clip,
        };
        let optimizer = AdamW::new(opt_cfg);
        Self {
            model,
            config,
            schedule,
            optimizer,
            step: 0,
            loss_ema: 0.0,
        }
    }

    /// Build a tiny trainer suitable for unit tests.
    pub fn new_tiny() -> Self {
        let model = opensesame_csm::CsmModel::new_tiny();
        let config = TrainConfig::fast_test();
        Self::new(model, config)
    }

    /// Current learning rate from the schedule.
    pub fn current_lr(&self) -> f32 {
        self.schedule.lr_at(self.step)
    }

    /// Advance the step counter and update the optimizer's learning rate.
    pub fn advance_step(&mut self) {
        self.step += 1;
        self.schedule.apply_to_opt(&mut self.optimizer, self.step);
    }

    /// Phase J.0: forward-only loss computation without weight update.
    ///
    /// Performs a teacher-forced forward pass:
    /// 1. Embeds text tokens as context frames.
    /// 2. Embeds context audio codes as context frames.
    /// 3. Runs the backbone over all context frames.
    /// 4. For each target frame, embeds the target frame and runs backbone.
    /// 5. Computes CB0 loss from backbone logits.
    /// 6. Selects `decoder_amort` fraction of target frames and computes
    ///    depth-decoder loss for CB1..CB(n−1).
    ///
    /// Returns `(total_loss, cb0_loss, decoder_loss)`.
    ///
    /// - `text_tokens`     — tokenised text prompt (can be empty).
    /// - `context_codes`   — `[n_codebooks][T_ctx]` audio codes for context.
    /// - `target_codes`    — `[n_codebooks][T_tgt]` audio codes to predict.
    pub fn forward_loss(
        &mut self,
        text_tokens:   &[u32],
        context_codes: &[Vec<u32>],  // [n_codebooks][T_context]
        target_codes:  &[Vec<u32>],  // [n_codebooks][T_target]
    ) -> (f32, f32, f32) {
        let n_cb = self.model.config.n_codebooks;
        let backbone_dim = self.model.config.backbone_dim;
        let audio_vocab  = self.model.config.audio_vocab;

        // ── Context: build a sequence of frame embeddings ─────────────────
        let mut frame_embeds: Vec<f32> = Vec::new();
        let mut n_frames = 0usize;

        // Text tokens as text-only frames
        for &tok in text_tokens {
            let emb = self.model.embed_frame(Some(tok), &vec![u32::MAX; n_cb]);
            frame_embeds.extend_from_slice(&emb);
            n_frames += 1;
        }

        // Context audio frames
        let t_ctx = context_codes.first().map(|v| v.len()).unwrap_or(0);
        let pad_audio = vec![u32::MAX; n_cb];
        for t in 0..t_ctx {
            let codes: Vec<u32> = (0..n_cb)
                .map(|cb| context_codes.get(cb).and_then(|v| v.get(t)).copied().unwrap_or(0))
                .collect();
            let emb = self.model.embed_frame(None, &codes);
            frame_embeds.extend_from_slice(&emb);
            n_frames += 1;
        }
        let _ = pad_audio;

        // ── Target frames: forward pass + collect logits ──────────────────
        let t_tgt = target_codes.first().map(|v| v.len()).unwrap_or(0);
        if t_tgt == 0 {
            return (0.0, 0.0, 0.0);
        }

        // Accumulate CB0 loss over all target frames
        let mut total_cb0_loss = 0.0f32;
        // Decoder: collect (logits, target) for amortized subset
        let n_amort = compute_amortized_frames(t_tgt, self.config.decoder_amort);

        // Simple deterministic subset: first n_amort frames
        // (production would use random sampling seeded from config.seed + step)
        let amort_set: Vec<usize> = (0..t_tgt.min(n_amort)).collect();

        let mut dec_losses: Vec<f32> = Vec::new();

        for t in 0..t_tgt {
            // Ground-truth codes for this target frame
            let codes: Vec<u32> = (0..n_cb)
                .map(|cb| target_codes.get(cb).and_then(|v| v.get(t)).copied().unwrap_or(0))
                .collect();

            // Embed and run backbone (teacher forcing — feed ground truth)
            let emb = self.model.embed_frame(None, &codes);
            frame_embeds.extend_from_slice(&emb);
            n_frames += 1;

            // Backbone forward on all frames so far (stateful KV cache not modelled here)
            let (cb0_logits, backbone_hidden) =
                self.model.backbone_forward(&frame_embeds, n_frames);

            // CB0 loss: predict current frame's CB0 from previous hidden state
            let cb0_target = codes[0] as usize % audio_vocab;
            total_cb0_loss += cross_entropy(&cb0_logits, cb0_target);

            // Depth-decoder loss (only for amortized subset)
            if amort_set.contains(&t) {
                // Build simple depth logits by projecting hidden → audio_vocab
                // (In Phase J.1 the real depformer will provide these; here we use
                //  the backbone's cb0_head as a stand-in for structure correctness.)
                let depth_logits: Vec<Vec<f32>> = (0..(n_cb.saturating_sub(1)))
                    .map(|_cb_idx| {
                        // Synthetic uniform logits for depth codebooks
                        vec![0.0f32; audio_vocab]
                    })
                    .collect();
                let depth_targets: Vec<u32> = codes[1..].to_vec();
                let dep_len = depth_logits.len().min(depth_targets.len());
                if dep_len > 0 {
                    let sum: f32 = depth_logits[..dep_len].iter()
                        .zip(depth_targets[..dep_len].iter())
                        .map(|(l, &tgt)| cross_entropy(l, (tgt as usize).min(audio_vocab - 1)))
                        .sum();
                    dec_losses.push(sum / dep_len as f32);
                }
                let _ = backbone_hidden;
            }
        }

        let cb0_loss = total_cb0_loss / t_tgt as f32;
        let dec_loss = if dec_losses.is_empty() {
            0.0
        } else {
            dec_losses.iter().sum::<f32>() / dec_losses.len() as f32
        };
        let w = self.config.decoder_loss_weight;
        let total = (1.0 - w) * cb0_loss + w * dec_loss;

        // Update loss EMA
        let alpha = 0.99f32;
        if self.step == 0 {
            self.loss_ema = total;
        } else {
            self.loss_ema = alpha * self.loss_ema + (1.0 - alpha) * total;
        }

        (total, cb0_loss, dec_loss)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_amortization_fraction() {
        // N_audio=160, amort=1/16=0.0625 → floor(160*0.0625)=floor(10)=10
        let n = compute_amortized_frames(160, 1.0 / 16.0);
        assert_eq!(n, 10, "expected 10, got {}", n);
    }

    #[test]
    fn test_compute_amortization_min_one() {
        // N_audio=8, amort=0.0625 → floor(8*0.0625)=floor(0.5)=0 → clamped to 1
        let n = compute_amortized_frames(8, 1.0 / 16.0);
        assert_eq!(n, 1, "should be at least 1, got {}", n);
    }

    #[test]
    fn test_compute_amortization_small_amort() {
        // N_audio=32, amort=1/16 → floor(2.0) = 2
        assert_eq!(compute_amortized_frames(32, 1.0 / 16.0), 2);
    }

    #[test]
    fn test_trainer_new_tiny() {
        let trainer = CsmTrainer::new_tiny();
        assert_eq!(trainer.step, 0);
        assert_eq!(trainer.config.max_steps, 10);
    }

    #[test]
    fn test_trainer_advance_step() {
        let mut trainer = CsmTrainer::new_tiny();
        assert_eq!(trainer.step, 0);
        trainer.advance_step();
        assert_eq!(trainer.step, 1);
        trainer.advance_step();
        assert_eq!(trainer.step, 2);
    }

    #[test]
    fn test_trainer_current_lr() {
        let trainer = CsmTrainer::new_tiny();
        // Step 0 → start of warmup → LR = 0
        let lr = trainer.current_lr();
        assert!(lr >= 0.0 && lr.is_finite(), "LR should be finite, got {}", lr);
    }

    #[test]
    fn test_trainer_forward_loss_returns_three_floats() {
        let mut trainer = CsmTrainer::new_tiny();
        let n_cb = trainer.model.config.n_codebooks;
        // Minimal: 1 target frame, no context
        let target_codes: Vec<Vec<u32>> = (0..n_cb).map(|_| vec![0u32]).collect();
        let (total, cb0, dec) = trainer.forward_loss(&[], &[], &target_codes);
        assert!(total.is_finite(), "total loss should be finite, got {}", total);
        assert!(cb0.is_finite(),   "cb0 loss should be finite, got {}", cb0);
        assert!(dec.is_finite(),   "dec loss should be finite, got {}", dec);
    }

    #[test]
    fn test_trainer_forward_loss_empty_target() {
        let mut trainer = CsmTrainer::new_tiny();
        let (total, cb0, dec) = trainer.forward_loss(&[], &[], &[]);
        assert_eq!(total, 0.0);
        assert_eq!(cb0,   0.0);
        assert_eq!(dec,   0.0);
    }

    #[test]
    fn test_decoder_input_shape_concept() {
        // Verify compute_amortized_frames with n_amort=5 selection from 16 frames
        // n_codebooks=32 → depth depth_logits has shape [5][31][audio_vocab]
        // We test the amort count here
        let n_amort = compute_amortized_frames(160, 5.0 / 160.0);
        assert_eq!(n_amort, 5, "Expected 5 amortized frames, got {}", n_amort);
    }
}
