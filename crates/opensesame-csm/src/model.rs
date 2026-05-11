//! CsmModel — the full CSM assembly.
//!
//! Wires Mimi (audio codec) + Backbone (Llama-3.2-1B) + Depformer (Llama-3.2-100M)
//! into a single model capable of generating speech frame-by-frame.
//!
//! # Generation pipeline
//! ```text
//! Mimi.encode(context_pcm) → codes[n_codebooks][T_context]
//!        ↓
//! CsmEmbedding.embed_position(text, audio) × (T_context + T_text)
//!        ↓
//! BackboneModel.forward_hidden_raw(embed) × seq_len → hidden[2048]
//!        ↓
//! CB0Head.forward(hidden) → cb0_logits[2048] → sample → cb0
//! Projection.forward(hidden) → proj_hidden[1024]
//!        ↓
//! Depformer.generate_depth_sequence(proj_h, embed_fn, …) → [CB1…CB31]
//!        ↓
//! full_codes = [cb0, CB1…CB31]
//!        ↓
//! Mimi.decode(full_codes) → pcm
//! ```

use crate::config::CsmModelConfig;
use crate::projection::Projection;

use opensesame_backbone::embedding::CsmEmbedding;
use opensesame_depformer::{Depformer, sample_topk};
use opensesame_mimi::Mimi;

// ── GenerateOutput ────────────────────────────────────────────────────────────

/// Output of a full CSM generation pass.
#[derive(Debug)]
pub struct GenerateOutput {
    /// Decoded PCM audio samples (mono, 24 kHz).
    pub pcm: Vec<f32>,
    /// Generated RVQ codes, shape `[n_codebooks][n_new_frames]`.
    pub codes: Vec<Vec<u32>>,
    /// Inner-monologue text tokens (empty in inference-only mode).
    pub text_tokens: Vec<u32>,
}

// ── CsmModel ─────────────────────────────────────────────────────────────────

/// Full Conversational Speech Model assembly.
///
/// Contains the three sub-models and their shared configuration:
/// - `mimi`:       audio codec (encode PCM ↔ discrete codes)
/// - `backbone`:   Llama-3.2-1B transformer (processes multimodal context)
/// - `depformer`:  Llama-3.2-100M transformer (generates CB1..CB31 per frame)
///
/// All weights are random by default; use the Phase J loader to set pretrained weights.
pub struct CsmModel {
    /// Full model configuration.
    pub config: CsmModelConfig,
    /// Mimi audio codec.
    pub mimi: Mimi,
    /// Backbone transformer (Llama-style, multimodal KV-cached).
    pub backbone: atlas_model::OlmoModel,
    /// Multimodal embedding tables (text + audio).
    pub embedding: CsmEmbedding,
    /// Depformer depth transformer.
    pub depformer: Depformer,
    /// CB0 audio head: `Linear(backbone_dim → audio_vocab)`, no bias.
    pub cb0_head: Projection,
    /// Backbone → depformer projection: `Linear(backbone_dim → decoder_dim)`, no bias.
    pub proj: Projection,
}

impl CsmModel {
    /// Construct a new CsmModel with random weights.
    ///
    /// All sub-models are initialised with pseudo-random weights using the
    /// `new` / `new_zeroed` constructors from their respective crates.
    pub fn new(config: CsmModelConfig) -> Self {
        let mut mimi = Mimi::new(config.mimi.clone());
        mimi.set_num_codebooks(config.n_codebooks);

        let backbone_mc = config.backbone.to_model_config();
        let backbone = atlas_model::OlmoModel::new(backbone_mc);

        let embedding = CsmEmbedding::new(config.backbone.clone());
        let depformer = Depformer::new(config.depformer.clone());

        let cb0_head = Projection::new_random(config.backbone_dim, config.audio_vocab, 0x1234_5678_ABCD_EF01u64);
        let proj     = Projection::new_random(config.backbone_dim, config.decoder_dim, 0xDEAD_BEEF_CAFE_BABEu64);

        Self { config, mimi, backbone, embedding, depformer, cb0_head, proj }
    }

    /// Construct a tiny model for fast unit tests.
    ///
    /// Uses `CsmModelConfig::tiny()`: 2-layer backbone, d=64, 4 codebooks.
    pub fn new_tiny() -> Self {
        Self::new(CsmModelConfig::tiny())
    }

    // ── Audio encode/decode ───────────────────────────────────────────────────

    /// Encode PCM audio to RVQ codes using Mimi.
    ///
    /// `pcm`: raw audio samples (mono, `config.mimi.sample_rate`).
    /// Returns `[n_codebooks][n_frames]` discrete code indices.
    pub fn encode_audio(&self, pcm: &[f32]) -> Vec<Vec<u32>> {
        if pcm.is_empty() {
            return vec![vec![]; self.config.n_codebooks];
        }
        self.mimi.encode(pcm, pcm.len())
    }

    /// Decode RVQ codes to PCM audio using Mimi.
    ///
    /// `codes`: `[n_codebooks][n_frames]` code indices.
    /// Returns decoded PCM samples.
    pub fn decode_audio(&self, codes: &[Vec<u32>]) -> Vec<f32> {
        if codes.is_empty() || codes[0].is_empty() {
            return vec![];
        }
        let (pcm, _n) = self.mimi.decode(codes);
        pcm
    }

    // ── Embedding ────────────────────────────────────────────────────────────

    /// Build the multimodal embedding for one sequence frame.
    ///
    /// Sums the text embedding (if present) with all active audio codebook embeddings.
    ///
    /// - `text_token`: `None` or `u32::MAX` → text padding (zero text contribution).
    /// - `audio_codes`: length `n_codebooks`; `u32::MAX` → audio padding for that slot.
    ///
    /// Returns `[backbone_dim]` float vector.
    pub fn embed_frame(&self, text_token: Option<u32>, audio_codes: &[u32]) -> Vec<f32> {
        let n_cb = self.config.n_codebooks;
        // Pad or truncate audio_codes to exactly n_codebooks
        let audio_opts: Vec<Option<u32>> = (0..n_cb)
            .map(|i| {
                if i < audio_codes.len() {
                    let c = audio_codes[i];
                    if c == u32::MAX { None } else { Some(c) }
                } else {
                    None
                }
            })
            .collect();
        self.embedding.embed_position(text_token, &audio_opts)
    }

    // ── Backbone forward ─────────────────────────────────────────────────────

    /// Run the backbone transformer on a sequence of frame embeddings.
    ///
    /// Resets the backbone KV cache, then feeds each frame embedding through
    /// `OlmoModel::forward_hidden_raw`, returning the **last** hidden state and
    /// the CB0 logits computed from it.
    ///
    /// - `frame_embeds`: flat `[n_frames × backbone_dim]` embeddings.
    /// - `n_frames`: number of frames in the sequence.
    ///
    /// Returns `(cb0_logits [audio_vocab], backbone_hidden [backbone_dim])`.
    pub fn backbone_forward(
        &mut self,
        frame_embeds: &[f32],
        n_frames: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.config.backbone_dim;
        assert_eq!(
            frame_embeds.len(),
            n_frames * d,
            "backbone_forward: frame_embeds length mismatch"
        );
        self.backbone.reset();

        let mut hidden = vec![0.0f32; d];
        for i in 0..n_frames {
            let embed: Vec<f32> = frame_embeds[i * d..(i + 1) * d].to_vec();
            hidden = self.backbone
                .forward_hidden_raw_gpu(embed.clone())
                .unwrap_or_else(|| self.backbone.forward_hidden_raw(embed));
        }

        let cb0_logits = self.cb0_head.forward(&hidden);
        (cb0_logits, hidden)
    }

    // ── Frame code generation ─────────────────────────────────────────────────

    /// Generate one audio frame's full code sequence from backbone outputs.
    ///
    /// 1. Samples CB0 from `cb0_logits` using top-k temperature sampling.
    /// 2. Projects `backbone_hidden` from backbone_dim → decoder_dim.
    /// 3. Runs the Depformer to autoregressively generate CB1..CB(n−1).
    /// 4. Returns `[n_codebooks]` token IDs.
    ///
    /// - `backbone_hidden`: `[backbone_dim]` last hidden state from the backbone.
    /// - `cb0_logits`:      `[audio_vocab]` logits for CB0 prediction.
    pub fn generate_frame_codes(
        &mut self,
        backbone_hidden: &[f32],
        cb0_logits:      &[f32],
    ) -> Vec<u32> {
        // Sample CB0 (backbone predicts this directly)
        let cb0 = sample_topk(cb0_logits, self.config.topk, self.config.temperature);

        // Project backbone hidden → depformer space
        let proj_hidden = self.proj.forward(backbone_hidden);

        // Build the embed_fn for the depformer using raw pointers so the borrow
        // checker is satisfied (embedding and proj are disjoint from depformer).
        let audio_vocab = self.config.audio_vocab;
        let embedding_ptr = &self.embedding as *const CsmEmbedding;
        let proj_ptr      = &self.proj      as *const Projection;

        let embed_fn = move |depth: usize, code: u32| -> Vec<f32> {
            // SAFETY: embedding and proj are different struct fields from depformer;
            // the shared references are valid for the duration of this call.
            let embedding = unsafe { &*embedding_ptr };
            let proj      = unsafe { &*proj_ptr };
            // Flat index into the audio embedding table for codebook `depth`, token `code`
            let flat = depth * (audio_vocab + 1) + (code as usize).min(audio_vocab);
            let emb = embedding.audio_embed.embed(flat);
            proj.forward(emb)
        };

        // Depformer generates CB1..CB(n−1)
        let dep_codes = self.depformer.generate_depth_sequence(
            proj_hidden,
            embed_fn,
            self.config.temperature,
            self.config.topk,
            cb0,
        );

        // Assemble full frame: [CB0, CB1, …, CB(n−1)]
        let mut codes = Vec::with_capacity(1 + dep_codes.len());
        codes.push(cb0);
        codes.extend(dep_codes);
        codes
    }

    // ── Full generation ───────────────────────────────────────────────────────

    /// Generate `n_new_frames` of audio conditioned on text and context audio.
    ///
    /// # Pipeline
    /// 1. Encode `context_pcm` to RVQ codes (if non-empty).
    /// 2. Build frame embeddings for all text + context audio positions.
    /// 3. Run backbone forward pass on the full context.
    /// 4. For each new frame:
    ///    a. Generate backbone hidden → CB0 logits.
    ///    b. Run Depformer to get CB1..CBn.
    ///    c. Re-embed the new frame and step backbone forward.
    /// 5. Decode all generated codes to PCM.
    ///
    /// # Arguments
    /// - `text_tokens`:    BPE token IDs for the text prompt (may be empty).
    /// - `context_pcm`:    Context audio PCM (may be empty).
    /// - `n_new_frames`:   Number of Mimi frames to generate.
    /// - `temperature`:    Sampling temperature (overrides config for this call).
    /// - `topk`:           Top-k cutoff (overrides config for this call).
    ///
    /// # Returns
    /// [`GenerateOutput`] with decoded PCM, generated codes, and (empty) text tokens.
    pub fn generate(
        &mut self,
        text_tokens:   &[u32],
        context_pcm:   &[f32],
        n_new_frames:  usize,
        temperature:   f32,
        topk:          usize,
    ) -> GenerateOutput {
        // Save generation hyper-params
        let saved_temp = self.config.temperature;
        let saved_topk = self.config.topk;
        self.config.temperature = temperature;
        self.config.topk        = topk;

        let d   = self.config.backbone_dim;
        let n_cb = self.config.n_codebooks;

        // ── 1. Encode context audio ───────────────────────────────────────────
        let ctx_codes = if !context_pcm.is_empty() {
            self.mimi.encode(context_pcm, context_pcm.len())
        } else {
            vec![vec![]; n_cb]
        };
        let n_ctx_frames = if !ctx_codes.is_empty() { ctx_codes[0].len() } else { 0 };

        // ── 2. Build context sequence embeddings ──────────────────────────────
        // Sequence layout: [text_0, …, text_T, audio_frame_0, …, audio_frame_C]
        let n_text   = text_tokens.len();
        let seq_len  = n_text + n_ctx_frames;

        let mut frame_embeds = Vec::with_capacity(seq_len.max(1) * d);

        // Text positions (no audio)
        for &tok in text_tokens {
            let pad_audio = vec![u32::MAX; n_cb];
            let embed = self.embed_frame(Some(tok), &pad_audio);
            frame_embeds.extend(embed);
        }

        // Audio context positions (no text for non-first frames)
        for f in 0..n_ctx_frames {
            let audio_codes: Vec<u32> = (0..n_cb)
                .map(|cb| if cb < ctx_codes.len() { ctx_codes[cb][f] } else { u32::MAX })
                .collect();
            let embed = self.embed_frame(None, &audio_codes);
            frame_embeds.extend(embed);
        }

        // ── 3. Prime backbone with context ────────────────────────────────────
        let effective_seq = seq_len.max(1);
        // Pad to at least 1 token so we always get a hidden state
        if frame_embeds.is_empty() {
            let pad = self.embed_frame(None, &vec![u32::MAX; n_cb]);
            frame_embeds.extend(pad);
        }

        let (mut cb0_logits, mut backbone_hidden) =
            self.backbone_forward(&frame_embeds, effective_seq);

        // ── 4. Autoregressive frame generation ───────────────────────────────
        // codes[cb][frame] — collect generated codes column-major then transpose
        let mut gen_codes_by_frame: Vec<Vec<u32>> = Vec::with_capacity(n_new_frames);

        for _frame_idx in 0..n_new_frames {
            // Generate this frame's codes
            let frame_codes = self.generate_frame_codes(&backbone_hidden, &cb0_logits);
            gen_codes_by_frame.push(frame_codes.clone());

            // Step backbone forward with the newly generated frame's embedding
            let embed = self.embed_frame(None, &frame_codes);
            let h = self.backbone
                .forward_hidden_raw_gpu(embed.clone())
                .unwrap_or_else(|| self.backbone.forward_hidden_raw(embed));
            cb0_logits     = self.cb0_head.forward(&h);
            backbone_hidden = h;
        }

        // ── 5. Transpose to [n_codebooks][n_frames] ───────────────────────────
        let mut codes: Vec<Vec<u32>> = vec![Vec::with_capacity(n_new_frames); n_cb];
        for frame_codes in &gen_codes_by_frame {
            for (cb, &code) in frame_codes.iter().enumerate().take(n_cb) {
                codes[cb].push(code);
            }
        }

        // ── 6. Decode to PCM ─────────────────────────────────────────────────
        let pcm = if n_new_frames > 0 {
            self.decode_audio(&codes)
        } else {
            vec![]
        };

        // Restore hyper-params
        self.config.temperature = saved_temp;
        self.config.topk        = saved_topk;

        GenerateOutput { pcm, codes, text_tokens: vec![] }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_model_new_tiny() {
        // Should not panic
        let _m = CsmModel::new_tiny();
    }

    // ── Encode / Decode ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_audio_empty() {
        let m = CsmModel::new_tiny();
        let codes = m.encode_audio(&[]);
        assert_eq!(codes.len(), m.config.n_codebooks, "empty encode → n_codebooks rows");
        for row in &codes {
            assert!(row.is_empty(), "each row is empty");
        }
    }

    #[test]
    fn test_encode_audio_one_frame() {
        let m = CsmModel::new_tiny();
        // Use full hop_length = 1920 samples to ensure at least 1 Mimi frame
        let pcm = vec![0.01f32; m.config.frame_samples];
        let codes = m.encode_audio(&pcm);
        assert_eq!(codes.len(), m.config.n_codebooks, "codes has n_codebooks rows");
        // Should have at least some frames (Mimi hop = 1920 samples)
        // (exact count depends on padding)
    }

    #[test]
    fn test_decode_audio_empty() {
        let m = CsmModel::new_tiny();
        let pcm = m.decode_audio(&[]);
        assert!(pcm.is_empty(), "decode empty → empty pcm");
    }

    #[test]
    fn test_decode_audio_empty_rows() {
        let m = CsmModel::new_tiny();
        let empty_codes: Vec<Vec<u32>> = vec![vec![]; m.config.n_codebooks];
        let pcm = m.decode_audio(&empty_codes);
        assert!(pcm.is_empty(), "all-empty rows → empty pcm");
    }

    #[test]
    fn test_decode_audio_round_trip() {
        let m = CsmModel::new_tiny();
        // Encode then decode — we just check shapes are consistent
        let pcm_in = vec![0.01f32; m.config.frame_samples * 2];
        let codes = m.encode_audio(&pcm_in);
        if !codes.is_empty() && !codes[0].is_empty() {
            let pcm_out = m.decode_audio(&codes);
            // PCM out should be non-empty and same channels
            assert!(!pcm_out.is_empty(), "round-trip should produce some PCM");
        }
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    #[test]
    fn test_embed_frame_shape() {
        let m = CsmModel::new_tiny();
        let audio = vec![0u32; m.config.n_codebooks];
        let embed = m.embed_frame(None, &audio);
        assert_eq!(embed.len(), m.config.backbone_dim, "embed shape = backbone_dim");
    }

    #[test]
    fn test_embed_frame_text_only() {
        let m = CsmModel::new_tiny();
        // Audio all-zero (valid codes = 0)
        let audio = vec![0u32; m.config.n_codebooks];
        let embed = m.embed_frame(Some(1), &audio);
        assert_eq!(embed.len(), m.config.backbone_dim);
    }

    #[test]
    fn test_embed_frame_audio_only() {
        let m = CsmModel::new_tiny();
        // No text token
        let audio = vec![3u32; m.config.n_codebooks];
        let embed = m.embed_frame(None, &audio);
        assert_eq!(embed.len(), m.config.backbone_dim);
    }

    #[test]
    fn test_embed_frame_mixed() {
        let m = CsmModel::new_tiny();
        let audio: Vec<u32> = (0..m.config.n_codebooks as u32).collect();
        let embed = m.embed_frame(Some(5), &audio);
        assert_eq!(embed.len(), m.config.backbone_dim);
    }

    // ── Backbone forward ──────────────────────────────────────────────────────

    #[test]
    fn test_backbone_forward_shape() {
        let mut m = CsmModel::new_tiny();
        let d = m.config.backbone_dim;
        let embed = vec![0.0f32; d]; // 1 frame
        let (cb0_logits, hidden) = m.backbone_forward(&embed, 1);
        assert_eq!(cb0_logits.len(), m.config.audio_vocab, "cb0_logits = audio_vocab");
        assert_eq!(hidden.len(), d, "hidden = backbone_dim");
    }

    #[test]
    fn test_backbone_forward_multi_frame() {
        let mut m = CsmModel::new_tiny();
        let d = m.config.backbone_dim;
        let n = 3;
        let embeds = vec![0.0f32; d * n];
        let (cb0, h) = m.backbone_forward(&embeds, n);
        assert_eq!(cb0.len(), m.config.audio_vocab);
        assert_eq!(h.len(), d);
    }

    // ── Frame code generation ─────────────────────────────────────────────────

    #[test]
    fn test_generate_frame_codes_shape() {
        let mut m = CsmModel::new_tiny();
        let d = m.config.backbone_dim;
        let hidden = vec![0.0f32; d];
        let logits = vec![1.0f32; m.config.audio_vocab];
        let codes = m.generate_frame_codes(&hidden, &logits);
        assert_eq!(codes.len(), m.config.n_codebooks, "generate_frame_codes returns n_codebooks tokens");
    }

    #[test]
    fn test_generate_frame_codes_in_range() {
        let mut m = CsmModel::new_tiny();
        let d = m.config.backbone_dim;
        let hidden = vec![0.1f32; d];
        let logits: Vec<f32> = (0..m.config.audio_vocab).map(|i| i as f32).collect();
        let codes = m.generate_frame_codes(&hidden, &logits);
        for &c in &codes {
            assert!((c as usize) < m.config.audio_vocab, "code {} < audio_vocab {}", c, m.config.audio_vocab);
        }
    }

    // ── Full generation ───────────────────────────────────────────────────────

    #[test]
    fn test_generate_no_context() {
        let mut m = CsmModel::new_tiny();
        let out = m.generate(&[], &[], 3, 1.0, 1);
        assert_eq!(out.codes.len(), m.config.n_codebooks, "codes rows = n_codebooks");
        assert_eq!(out.codes[0].len(), 3, "codes cols = n_new_frames");
    }

    #[test]
    fn test_generate_with_context() {
        let mut m = CsmModel::new_tiny();
        let ctx = vec![0.01f32; m.config.frame_samples];
        let out = m.generate(&[], &ctx, 2, 1.0, 1);
        assert_eq!(out.codes.len(), m.config.n_codebooks);
        assert_eq!(out.codes[0].len(), 2);
    }

    #[test]
    fn test_generate_codes_shape() {
        let mut m = CsmModel::new_tiny();
        let n = 4;
        let out = m.generate(&[1, 2], &[], n, 1.0, 1);
        assert_eq!(out.codes.len(), m.config.n_codebooks);
        for row in &out.codes {
            assert_eq!(row.len(), n);
        }
    }

    #[test]
    fn test_generate_output_shape() {
        let mut m = CsmModel::new_tiny();
        let n = 2;
        let out = m.generate(&[], &[], n, 1.0, 1);
        // PCM length should be > 0 (Mimi decodes codes to samples)
        // The exact count depends on Mimi's convolution padding
        assert!(!out.pcm.is_empty() || n == 0, "PCM should be non-empty for n>0");
    }

    #[test]
    fn test_generate_deterministic_seed() {
        // Temperature=0 → greedy argmax → deterministic output
        // (Both calls use the same random model weights, same greedy sampling)
        let mut m = CsmModel::new_tiny();
        let out1 = m.generate(&[1], &[], 2, 0.0, 0);
        let out2 = m.generate(&[1], &[], 2, 0.0, 0);
        // With temperature=0 and same model weights, codes should be identical
        assert_eq!(out1.codes, out2.codes, "greedy generation is deterministic");
    }
}
