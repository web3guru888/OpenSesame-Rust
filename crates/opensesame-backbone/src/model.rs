//! CSM backbone model — multimodal embedding + dual output heads.
//!
//! The `BackboneModel` wraps an `atlas_model::OlmoModel` (16-layer Llama-3.2-1B)
//! and adds:
//! - `CsmEmbedding` — summed text+audio embeddings injected via `forward_hidden_raw`
//! - `audio_head`   — CB0 prediction: `Linear(d_model → audio_vocab_size)`
//! - `text_head`    — Inner Monologue: `Linear(d_model → text_vocab_size)`
//!
//! During inference, call `forward()` (batch mode) or `forward_step()` (one token
//! at a time) to obtain `BackboneOutput`.

use atlas_model::OlmoModel;

use crate::{
    config::{BackboneConfig, AUDIO_VOCAB_SIZE},
    embedding::CsmEmbedding,
    heads::linear_forward,
    loader::load_weights_from_safetensors,
};

// ── BackboneOutput ────────────────────────────────────────────────────────────

/// Output from a `BackboneModel` forward pass.
pub struct BackboneOutput {
    /// CB0 audio logits: `[B × T × audio_vocab_size]` flat.
    pub audio_logits: Vec<f32>,
    /// Inner Monologue text logits: `[B × T × text_vocab_size]` flat.
    pub text_logits: Vec<f32>,
    /// Final transformer hidden states: `[B × T × d_model]` flat.
    /// Used by the Depformer to generate CB1..CB7.
    pub hidden_states: Vec<f32>,
}

// ── BackboneModel ─────────────────────────────────────────────────────────────

/// The CSM backbone transformer.
///
/// Combines a Llama-3.2-1B-compatible transformer (via `OlmoModel`) with
/// multimodal embeddings and dual output heads.
///
/// # Forward pass
///
/// For each sequence position the embedding is computed as:
/// ```text
/// embed[t] = sum(text_emb[text_token[t]], audio_emb[cb0][code0], ..., audio_emb[cb7][code7])
/// ```
/// The summed embedding is passed through the transformer via
/// `OlmoModel::forward_hidden_raw`, bypassing the standard token embedding lookup
/// and LM head.  The hidden state is then projected through both output heads.
pub struct BackboneModel {
    /// Multimodal embedding module.
    pub embedding: CsmEmbedding,
    /// 16-layer Llama-3.2-1B transformer.
    pub transformer: OlmoModel,
    /// CB0 audio head weights: `[audio_vocab_size × d_model]` row-major.
    pub audio_head: Vec<f32>,
    /// CB0 audio head bias: `[audio_vocab_size]` (zeros if none).
    pub audio_head_bias: Vec<f32>,
    /// Text (Inner Monologue) head weights: `[text_vocab_size × d_model]` row-major.
    pub text_head: Vec<f32>,
    /// Text head bias: `[text_vocab_size]` (zeros if none).
    pub text_head_bias: Vec<f32>,
    /// Model configuration.
    pub config: BackboneConfig,
}

impl BackboneModel {
    /// Create a new `BackboneModel` with random weights (for training from scratch).
    ///
    /// The transformer is initialised with `OlmoModel::new()`, the embedding
    /// tables with `CsmEmbedding::new()`, and both heads with small random weights.
    pub fn new(config: BackboneConfig) -> Self {
        let mc = config.to_model_config();
        let transformer = OlmoModel::new(mc);
        let embedding = CsmEmbedding::new(config.clone());

        let d  = config.d_model;
        let av = AUDIO_VOCAB_SIZE;
        let tv = config.text_vocab_size;

        // Audio head: Linear(d_model → audio_vocab_size), no bias
        let audio_head = random_weights(av * d, 0xdeadbeef_cafef00d);
        let audio_head_bias = vec![0.0f32; av];

        // Text head: Linear(d_model → text_vocab_size), no bias (large!)
        let text_head = random_weights(tv * d, 0xabcd1234_5678ef90);
        let text_head_bias = vec![0.0f32; tv];

        Self {
            embedding,
            transformer,
            audio_head,
            audio_head_bias,
            text_head,
            text_head_bias,
            config,
        }
    }

    /// Full forward pass over a batch of sequences.
    ///
    /// - `tokens_text`:  `[B × T]` flat, `u32::MAX` = no text token (pad).
    /// - `tokens_audio`: `[B × T × n_audio_codebooks]` flat, `u32::MAX` = pad.
    /// - Returns `BackboneOutput` with shapes:
    ///   - `audio_logits`: `[B × T × audio_vocab_size]`
    ///   - `text_logits`:  `[B × T × text_vocab_size]`
    ///   - `hidden_states`: `[B × T × d_model]`
    pub fn forward(
        &mut self,
        tokens_text:  &[u32],
        tokens_audio: &[u32],
        batch:   usize,
        seq_len: usize,
    ) -> BackboneOutput {
        let d  = self.config.d_model;
        let av = AUDIO_VOCAB_SIZE;
        let tv = self.config.text_vocab_size;
        let total = batch * seq_len;

        // Compute multimodal embeddings for the full batch
        let embeds = self.embedding.embed_sequence(
            tokens_text, tokens_audio, batch, seq_len,
        );
        // embeds: [B × T × d_model]

        // Reset transformer KV cache for a fresh sequence
        self.transformer.reset();

        // Run each position through the transformer
        let mut hidden_states = Vec::with_capacity(total * d);
        for pos in 0..total {
            let embed = embeds[pos * d..(pos + 1) * d].to_vec();
            let h = self.transformer.forward_hidden_raw(embed);
            hidden_states.extend(h);
        }

        // Apply audio head: [total × d] → [total × audio_vocab_size]
        let audio_logits = linear_forward(
            &hidden_states,
            &self.audio_head,
            &self.audio_head_bias,
            total, d, av,
        );

        // Apply text head: [total × d] → [total × text_vocab_size]
        let text_logits = linear_forward(
            &hidden_states,
            &self.text_head,
            &self.text_head_bias,
            total, d, tv,
        );

        BackboneOutput {
            audio_logits,
            text_logits,
            hidden_states,
        }
    }

    /// Single-step autoregressive forward.
    ///
    /// Feeds one position to the transformer (maintaining KV cache state from
    /// previous calls).  Useful during autoregressive inference.
    ///
    /// `text_token`:   `None` → text pad; `Some(id)` → text token.
    /// `audio_tokens`: `&[Option<u32>]` of length `n_audio_codebooks`.
    ///
    /// Returns a `BackboneOutput` with shapes `[1 × 1 × *]` (batch=1, seq=1).
    pub fn forward_step(
        &mut self,
        text_token:   Option<u32>,
        audio_tokens: &[Option<u32>],
    ) -> BackboneOutput {
        let d  = self.config.d_model;
        let av = AUDIO_VOCAB_SIZE;
        let tv = self.config.text_vocab_size;

        let embed = self.embedding.embed_position(text_token, audio_tokens);
        let h = self.transformer.forward_hidden_raw(embed);

        let audio_logits = {
            let mut logits = vec![0.0f32; av];
            for i in 0..av {
                let row = &self.audio_head[i * d..(i + 1) * d];
                logits[i] = row.iter().zip(h.iter()).map(|(&w, &x)| w * x).sum::<f32>()
                    + self.audio_head_bias[i];
            }
            logits
        };

        let text_logits = {
            let mut logits = vec![0.0f32; tv];
            for i in 0..tv {
                let row = &self.text_head[i * d..(i + 1) * d];
                logits[i] = row.iter().zip(h.iter()).map(|(&w, &x)| w * x).sum::<f32>()
                    + self.text_head_bias[i];
            }
            logits
        };

        BackboneOutput {
            audio_logits,
            text_logits,
            hidden_states: h,
        }
    }

    /// Load backbone weights from a safetensors file.
    ///
    /// Currently validates that the file can be opened; full weight injection
    /// is implemented in `loader.rs` / Phase H.
    pub fn load_weights(&mut self, path: &str) -> Result<(), String> {
        load_weights_from_safetensors(path)
    }

    /// Return the current KV-cache position (number of tokens processed since last reset).
    pub fn pos(&self) -> usize {
        self.transformer.pos()
    }

    /// Reset the KV cache and position counter.
    pub fn reset(&mut self) {
        self.transformer.reset();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Generate `n` pseudo-random f32 values in `[-scale, +scale)`.
///
/// Uses a simple xorshift64 PRNG — no external crate required.
fn random_weights(n: usize, seed: u64) -> Vec<f32> {
    // Scale by 1/sqrt(n) as a reasonable initialisation
    let scale = 1.0 / (n as f32).sqrt();
    let mut v = Vec::with_capacity(n);
    let mut s = seed | 1;
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let f = (s >> 11) as f32 / (1u64 << 53) as f32;
        v.push((f * 2.0 - 1.0) * scale);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackboneConfig;

    /// Build a tiny BackboneModel (2 layers, d=64) for fast tests.
    fn tiny_model() -> BackboneModel {
        BackboneModel::new(BackboneConfig::tiny())
    }

    // ── Shape tests ────────────────────────────────────────────────────────

    #[test]
    fn test_backbone_forward_shape() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let B = 1usize;
        let T = 2usize;
        let n_cb = cfg.n_audio_codebooks;
        let text  = vec![0u32; B * T];
        let audio = vec![0u32; B * T * n_cb];
        let out = m.forward(&text, &audio, B, T);
        assert_eq!(out.hidden_states.len(), B * T * cfg.d_model);
        assert_eq!(out.audio_logits.len(), B * T * AUDIO_VOCAB_SIZE);
        assert_eq!(out.text_logits.len(),  B * T * cfg.text_vocab_size);
    }

    #[test]
    fn test_backbone_audio_logits_shape() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let B = 1usize; let T = 2usize;
        let text  = vec![1u32; B * T];
        let audio = vec![2u32; B * T * cfg.n_audio_codebooks];
        let out = m.forward(&text, &audio, B, T);
        // audio_vocab_size stays at AUDIO_VOCAB_SIZE=2048 regardless of tiny config
        assert_eq!(out.audio_logits.len(), B * T * AUDIO_VOCAB_SIZE);
    }

    #[test]
    fn test_backbone_text_logits_shape() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let B = 1usize; let T = 2usize;
        let text  = vec![0u32; B * T];
        let audio = vec![0u32; B * T * cfg.n_audio_codebooks];
        let out = m.forward(&text, &audio, B, T);
        assert_eq!(out.text_logits.len(), B * T * cfg.text_vocab_size);
    }

    #[test]
    fn test_backbone_hidden_shape() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let B = 1usize; let T = 2usize;
        let text  = vec![0u32; B * T];
        let audio = vec![0u32; B * T * cfg.n_audio_codebooks];
        let out = m.forward(&text, &audio, B, T);
        assert_eq!(out.hidden_states.len(), B * T * cfg.d_model);
    }

    #[test]
    fn test_backbone_finite_outputs() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let B = 1usize; let T = 2usize;
        let text  = vec![0u32; B * T];
        let audio = vec![0u32; B * T * cfg.n_audio_codebooks];
        let out = m.forward(&text, &audio, B, T);
        assert!(out.hidden_states.iter().all(|v| v.is_finite()), "hidden has NaN/Inf");
        assert!(out.audio_logits.iter().all(|v| v.is_finite()), "audio_logits has NaN/Inf");
        assert!(out.text_logits.iter().all(|v| v.is_finite()), "text_logits has NaN/Inf");
    }

    #[test]
    fn test_backbone_text_only_seq() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let T = 2usize;
        // All audio = u32::MAX (padding)
        let text  = vec![5u32; T];
        let audio = vec![u32::MAX; T * cfg.n_audio_codebooks];
        let out = m.forward(&text, &audio, 1, T);
        assert_eq!(out.hidden_states.len(), T * cfg.d_model);
        assert!(out.hidden_states.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_backbone_audio_only_seq() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let T = 2usize;
        // All text = u32::MAX (padding)
        let text  = vec![u32::MAX; T];
        let audio = vec![3u32; T * cfg.n_audio_codebooks];
        let out = m.forward(&text, &audio, 1, T);
        assert_eq!(out.hidden_states.len(), T * cfg.d_model);
        assert!(out.hidden_states.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_backbone_mixed_seq() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let T = 4usize;
        // Interleave text and audio positions
        let mut text  = vec![u32::MAX; T];
        let mut audio = vec![u32::MAX; T * cfg.n_audio_codebooks];
        text[0] = 1;
        text[2] = 2;
        for cb in 0..cfg.n_audio_codebooks {
            audio[1 * cfg.n_audio_codebooks + cb] = 5;
            audio[3 * cfg.n_audio_codebooks + cb] = 7;
        }
        let out = m.forward(&text, &audio, 1, T);
        assert!(out.hidden_states.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_backbone_batch2() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let T = 2usize;
        let n_cb = cfg.n_audio_codebooks;

        let text1  = vec![1u32; T];
        let audio1 = vec![2u32; T * n_cb];
        let out1 = m.forward(&text1, &audio1, 1, T);

        m.reset();

        let text2  = vec![1u32; 2 * T];
        let audio2 = vec![2u32; 2 * T * n_cb];
        let out2 = m.forward(&text2, &audio2, 2, T);

        // B=2 produces twice as many values
        assert_eq!(out2.hidden_states.len(), 2 * out1.hidden_states.len());
        assert_eq!(out2.audio_logits.len(), 2 * out1.audio_logits.len());
        assert_eq!(out2.text_logits.len(), 2 * out1.text_logits.len());
    }

    #[test]
    fn test_backbone_deterministic() {
        let cfg = BackboneConfig::tiny();
        let T = 2usize;
        let n_cb = cfg.n_audio_codebooks;
        let text  = vec![3u32; T];
        let audio = vec![1u32; T * n_cb];

        let mut m1 = BackboneModel::new(cfg.clone());
        let out1 = m1.forward(&text, &audio, 1, T);

        let mut m2 = BackboneModel::new(cfg.clone());
        let out2 = m2.forward(&text, &audio, 1, T);

        // Same model (same random seed logic in OlmoModel::new) → same output
        assert_eq!(out1.hidden_states, out2.hidden_states);
    }

    // ── forward_step ───────────────────────────────────────────────────────

    #[test]
    fn test_forward_step_shape() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let audio: Vec<Option<u32>> = vec![Some(1); cfg.n_audio_codebooks];
        let out = m.forward_step(Some(5), &audio);
        assert_eq!(out.hidden_states.len(), cfg.d_model);
        assert_eq!(out.audio_logits.len(), AUDIO_VOCAB_SIZE);
        assert_eq!(out.text_logits.len(), cfg.text_vocab_size);
    }

    #[test]
    fn test_forward_step_finite() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let audio: Vec<Option<u32>> = vec![None; cfg.n_audio_codebooks];
        let out = m.forward_step(None, &audio);
        assert!(out.hidden_states.iter().all(|v| v.is_finite()));
        assert!(out.audio_logits.iter().all(|v| v.is_finite()));
        assert!(out.text_logits.iter().all(|v| v.is_finite()));
    }

    // ── Weight loading ─────────────────────────────────────────────────────

    #[test]
    fn test_load_weights_missing_file() {
        let mut m = tiny_model();
        let result = m.load_weights("/tmp/definitely_does_not_exist_phase_f.safetensors");
        assert!(result.is_err());
    }

    // ── Misc ───────────────────────────────────────────────────────────────

    #[test]
    fn test_backbone_pos_resets() {
        let mut m = tiny_model();
        let cfg = BackboneConfig::tiny();
        let T = 3usize;
        let text  = vec![0u32; T];
        let audio = vec![0u32; T * cfg.n_audio_codebooks];
        m.forward(&text, &audio, 1, T);
        assert_eq!(m.pos(), T);
        m.reset();
        assert_eq!(m.pos(), 0);
    }
}
