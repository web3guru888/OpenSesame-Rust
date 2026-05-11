//! Multimodal embedding for the CSM backbone.
//!
//! Each sequence position in the CSM combines:
//! - A **text embedding** (from the Llama vocabulary, 128_000 tokens)
//! - Up to 8 **audio embeddings** (one per Mimi codebook, 2_048 tokens each)
//!
//! The final embedding for a position is the **element-wise sum** of all
//! active (unmasked) slot embeddings.  Masked/padding slots contribute zero.

use crate::config::BackboneConfig;

/// Flat embedding table `[vocab_size × d_model]` stored row-major.
///
/// For text: `vocab_size = text_vocab_size + 1` (the extra row is the padding
/// token at index `text_vocab_size`).
///
/// For audio: `vocab_size = n_codebooks × (audio_vocab_size + 1)` (one padding
/// row per codebook, at index `k × (audio_vocab_size + 1) + audio_vocab_size`).
pub struct EmbeddingTable {
    /// Weight data: `[vocab_size × d_model]` f32, row-major.
    pub weight: Vec<f32>,
    /// Number of rows (tokens) in the table.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub d_model: usize,
}

impl EmbeddingTable {
    /// Construct from raw weight data.
    ///
    /// Panics if `weight.len() != vocab_size * d_model`.
    pub fn from_data(weight: Vec<f32>, vocab_size: usize, d_model: usize) -> Self {
        assert_eq!(
            weight.len(),
            vocab_size * d_model,
            "expected {} weights, got {}",
            vocab_size * d_model,
            weight.len()
        );
        Self { weight, vocab_size, d_model }
    }

    /// Allocate a table of zeros.
    pub fn zeros(vocab_size: usize, d_model: usize) -> Self {
        Self {
            weight: vec![0.0f32; vocab_size * d_model],
            vocab_size,
            d_model,
        }
    }

    /// Allocate with small random values (for initializing new embeddings).
    ///
    /// Uses a deterministic LCG seeded with the table dimensions (no external
    /// rand crate), scaled by `1/sqrt(d_model)`.
    pub fn random(vocab_size: usize, d_model: usize) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        let n = vocab_size * d_model;
        let mut weight = Vec::with_capacity(n);
        let mut s: u64 = (vocab_size as u64 * 6364136223846793005)
            ^ (d_model as u64 * 1442695040888963407);
        for _ in 0..n {
            // xorshift64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let f = (s >> 11) as f32 / (1u64 << 53) as f32; // [0,1)
            weight.push((f * 2.0 - 1.0) * scale);            // [-scale, +scale)
        }
        Self { weight, vocab_size, d_model }
    }

    /// Look up one token row.  Out-of-range indices are clamped to `vocab_size - 1`.
    #[inline]
    pub fn embed(&self, id: usize) -> &[f32] {
        let i = id.min(self.vocab_size - 1);
        &self.weight[i * self.d_model..(i + 1) * self.d_model]
    }
}

// ── CsmEmbedding ─────────────────────────────────────────────────────────────

/// Multimodal embedding module for the CSM backbone.
///
/// Holds two separate embedding tables:
/// - `text_embed`:  `[text_vocab_size + 1, d_model]` — the +1 is the padding row.
/// - `audio_embed`: `[n_codebooks × (audio_vocab_size + 1), d_model]` — one padding
///   row per codebook.
///
/// The padding row for **text** is at index `text_vocab_size` (= `TEXT_PAD_TOKEN`).
/// The padding row for **audio codebook k** is at flat index
/// `k × (audio_vocab_size + 1) + audio_vocab_size`.
pub struct CsmEmbedding {
    /// Text token embedding table: `[text_vocab_size + 1, d_model]`.
    pub text_embed:  EmbeddingTable,
    /// Audio token embedding table: `[n_codebooks × (audio_vocab_size + 1), d_model]`.
    pub audio_embed: EmbeddingTable,
    /// Configuration.
    pub config: BackboneConfig,
}

impl CsmEmbedding {
    /// Construct with random text and audio weights (for training from scratch).
    ///
    /// Both tables are initialised with small random values — audio weights are
    /// NOT zeroed, so that `embed_position` with audio tokens produces embeddings
    /// distinct from the text-only result (important for the `test_embedding_sum`
    /// invariant: text+audio ≠ text-only ≠ audio-only).
    pub fn new(config: BackboneConfig) -> Self {
        let text_vocab  = config.text_vocab_size + 1;          // +1 padding row
        let audio_vocab = config.n_audio_codebooks * (config.audio_vocab_size + 1);
        let d = config.d_model;
        Self {
            text_embed:  EmbeddingTable::random(text_vocab, d),
            audio_embed: EmbeddingTable::random(audio_vocab, d),
            config,
        }
    }

    /// Compute the embedding for **one sequence position**.
    ///
    /// - `text_token`:   `None` → use text padding row; `Some(id)` → look up text row.
    ///   Passing `u32::MAX` is treated the same as `None` (clamped to padding).
    /// - `audio_tokens`: slice of length `n_codebooks`.  `None` or `u32::MAX` → audio pad.
    ///
    /// Returns `[d_model]` — the element-wise sum of all active embeddings.
    pub fn embed_position(
        &self,
        text_token: Option<u32>,
        audio_tokens: &[Option<u32>],
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let n_cb = self.config.n_audio_codebooks;
        let av = self.config.audio_vocab_size;
        let mut result = vec![0.0f32; d];

        // ── Text embedding ────────────────────────────────────────────────
        let text_id = match text_token {
            Some(id) if id != u32::MAX => id as usize,
            _ => self.config.text_vocab_size,  // padding row
        };
        let te = self.text_embed.embed(text_id);
        for (r, &e) in result.iter_mut().zip(te.iter()) {
            *r += e;
        }

        // ── Audio embeddings (one per codebook) ───────────────────────────
        for (cb, opt) in audio_tokens.iter().enumerate().take(n_cb) {
            let code = match opt {
                Some(c) if *c != u32::MAX => *c as usize,
                _ => av,  // padding token for this codebook
            };
            // Flat index: cb × (audio_vocab_size + 1) + code
            let flat = cb * (av + 1) + code;
            let ae = self.audio_embed.embed(flat);
            for (r, &e) in result.iter_mut().zip(ae.iter()) {
                *r += e;
            }
        }

        result
    }

    /// Embed an entire sequence in batch.
    ///
    /// - `tokens_text`:  `[B × T]` flat — `u32::MAX` means "no text token" (pad).
    /// - `tokens_audio`: `[B × T × n_codebooks]` flat — `u32::MAX` means "no audio" (pad).
    /// - Returns:        `[B × T × d_model]` flat.
    pub fn embed_sequence(
        &self,
        tokens_text:  &[u32],
        tokens_audio: &[u32],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let n_cb = self.config.n_audio_codebooks;
        assert_eq!(tokens_text.len(), batch * seq_len);
        assert_eq!(tokens_audio.len(), batch * seq_len * n_cb);

        let mut out = Vec::with_capacity(batch * seq_len * d);
        for b in 0..batch {
            for t in 0..seq_len {
                let ti = b * seq_len + t;
                let text_tok = tokens_text[ti];
                let text_opt = if text_tok == u32::MAX { None } else { Some(text_tok) };

                let audio_start = (b * seq_len + t) * n_cb;
                let audio_slice = &tokens_audio[audio_start..audio_start + n_cb];
                let audio_opts: Vec<Option<u32>> = audio_slice
                    .iter()
                    .map(|&c| if c == u32::MAX { None } else { Some(c) })
                    .collect();

                let embed = self.embed_position(text_opt, &audio_opts);
                out.extend(embed);
            }
        }
        out
    }
}

/// Compute the multimodal embedding for one sequence position using two
/// separate tables (as specified in §4.4 of the implementation spec).
///
/// - `text_emb`:  text embedding table.
/// - `audio_emb`: audio embedding table (flat over all codebooks).
/// - `slots`:     `[N_CODEBOOKS + 1]` — audio CB0..CB(N-1) then text.
/// - `mask`:      same shape; `true` means the slot is active.
/// - `d_model`:   hidden dimension.
pub fn compute_position_embedding(
    text_emb:  &EmbeddingTable,
    audio_emb: &EmbeddingTable,
    slots:     &[u32],
    mask:      &[bool],
    d_model:   usize,
    n_codebooks: usize,
    audio_vocab_size: usize,
) -> Vec<f32> {
    assert_eq!(slots.len(), n_codebooks + 1);
    assert_eq!(mask.len(), slots.len());

    let mut result = vec![0.0f32; d_model];

    // Audio embeddings (slots 0..n_codebooks-1)
    for cb in 0..n_codebooks {
        if mask[cb] {
            let code = slots[cb] as usize;
            let flat = cb * (audio_vocab_size + 1) + code;
            let ae = audio_emb.embed(flat);
            for (r, &e) in result.iter_mut().zip(ae.iter()) {
                *r += e;
            }
        }
    }

    // Text embedding (last slot)
    let text_slot = n_codebooks;
    if mask[text_slot] {
        let te = text_emb.embed(slots[text_slot] as usize);
        for (r, &e) in result.iter_mut().zip(te.iter()) {
            *r += e;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackboneConfig;

    fn tiny_cfg() -> BackboneConfig {
        BackboneConfig::tiny()
    }

    #[test]
    fn test_embedding_text_only() {
        let emb = CsmEmbedding::new(tiny_cfg());
        let pos = emb.embed_position(Some(5), &[None; 4]);
        assert_eq!(pos.len(), tiny_cfg().d_model);
        assert!(pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_embedding_audio_only() {
        let emb = CsmEmbedding::new(tiny_cfg());
        let audio = [Some(3u32), None, None, None];
        let pos = emb.embed_position(None, &audio);
        assert_eq!(pos.len(), tiny_cfg().d_model);
        assert!(pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_embedding_sum() {
        let emb = CsmEmbedding::new(tiny_cfg());
        let text_only  = emb.embed_position(Some(5), &[None; 4]);
        let audio_only = emb.embed_position(None, &[Some(3), None, None, None]);
        let combined   = emb.embed_position(Some(5), &[Some(3), None, None, None]);
        // combined should differ from each component
        assert_ne!(combined, text_only);
        assert_ne!(combined, audio_only);
    }

    #[test]
    fn test_embedding_pad_text() {
        let emb = CsmEmbedding::new(tiny_cfg());
        // u32::MAX treated as padding
        let pad   = emb.embed_position(Some(u32::MAX), &[None; 4]);
        let nopad = emb.embed_position(None, &[None; 4]);
        // Both should use the text padding row — same result
        assert_eq!(pad, nopad);
        assert!(pad.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_embedding_pad_audio() {
        let emb = CsmEmbedding::new(tiny_cfg());
        let cfg = tiny_cfg();
        // Explicit audio pad token vs u32::MAX — both map to the padding row
        let explicit_pad = emb.embed_position(None, &[Some(cfg.audio_vocab_size as u32); 4]);
        let max_pad      = emb.embed_position(None, &[Some(u32::MAX); 4]);
        // Both use the same audio padding row per codebook → identical embedding
        assert_eq!(explicit_pad, max_pad);
        assert!(explicit_pad.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_embedding_sequence_shape() {
        let cfg = tiny_cfg();
        let emb = CsmEmbedding::new(cfg.clone());
        let T = 4usize;
        let tokens_text  = vec![1u32; T];
        let tokens_audio = vec![2u32; T * cfg.n_audio_codebooks];
        let out = emb.embed_sequence(&tokens_text, &tokens_audio, 1, T);
        assert_eq!(out.len(), 1 * T * cfg.d_model);
    }

    #[test]
    fn test_embedding_batch_shape() {
        let cfg = tiny_cfg();
        let emb = CsmEmbedding::new(cfg.clone());
        let T = 4usize;
        let B = 2usize;
        let tokens_text  = vec![1u32; B * T];
        let tokens_audio = vec![2u32; B * T * cfg.n_audio_codebooks];
        let out = emb.embed_sequence(&tokens_text, &tokens_audio, B, T);
        assert_eq!(out.len(), B * T * cfg.d_model);
    }

    #[test]
    fn test_embedding_audio_codebook_idx() {
        let mut cfg = tiny_cfg();
        // Make audio table non-zero so codebooks differ
        let av = cfg.audio_vocab_size + 1;
        let d  = cfg.d_model;
        let n_cb = cfg.n_audio_codebooks;
        let mut emb = CsmEmbedding::new(cfg.clone());
        // Give CB0 code=5 a non-zero, CB1 code=5 a different non-zero
        let row0 = 0 * av + 5;
        let row1 = 1 * av + 5;
        for j in 0..d {
            emb.audio_embed.weight[row0 * d + j] = 1.0;
            emb.audio_embed.weight[row1 * d + j] = 2.0;
        }
        let e0 = emb.embed_position(None, &[Some(5), None, None, None]);
        let e1 = emb.embed_position(None, &[None, Some(5), None, None]);
        assert_ne!(e0, e1, "codebook 0 and 1 should produce distinct embeddings");
    }

    #[test]
    fn test_embedding_deterministic() {
        let emb = CsmEmbedding::new(tiny_cfg());
        let a = emb.embed_position(Some(7), &[Some(2), None, None, None]);
        let b = emb.embed_position(Some(7), &[Some(2), None, None, None]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_embedding_audio_all_codebooks() {
        let cfg = tiny_cfg();
        let d   = cfg.d_model;
        let av  = cfg.audio_vocab_size + 1;
        let n_cb = cfg.n_audio_codebooks;
        let mut emb = CsmEmbedding::new(cfg.clone());
        // Give each codebook slot a distinct value
        for cb in 0..n_cb {
            let row = cb * av + 1;
            for j in 0..d {
                emb.audio_embed.weight[row * d + j] = (cb + 1) as f32;
            }
        }
        // Embed each codebook individually and check they're distinct
        let mut seen = Vec::new();
        for cb in 0..n_cb {
            let mut audio = vec![None; n_cb];
            audio[cb] = Some(1u32);
            let e = emb.embed_position(None, &audio);
            for prev in &seen {
                assert_ne!(&e, prev, "codebook {} produces duplicate embedding", cb);
            }
            seen.push(e);
        }
    }

    #[test]
    fn test_embedding_table_from_data() {
        let d = 8usize;
        let v = 4usize;
        let data = vec![1.0f32; v * d];
        let table = EmbeddingTable::from_data(data, v, d);
        let row = table.embed(2);
        assert_eq!(row.len(), d);
        assert!(row.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_compute_position_embedding() {
        let d = 8usize;
        let n_cb = 2usize;
        let av = 4usize;
        // Text table: vocab = text_vocab + 1 = 5; audio table: n_cb*(av+1) = 10
        let mut text_w = vec![0.0f32; 5 * d];
        let mut audio_w = vec![0.0f32; n_cb * (av + 1) * d];
        // text token 1 → all 1.0
        for j in 0..d { text_w[1 * d + j] = 1.0; }
        // audio CB0, code=2 → all 2.0
        let flat = 0 * (av + 1) + 2;
        for j in 0..d { audio_w[flat * d + j] = 2.0; }

        let text_emb  = EmbeddingTable::from_data(text_w, 5, d);
        let audio_emb = EmbeddingTable::from_data(audio_w, n_cb * (av + 1), d);

        // slots: [CB0=2, CB1=pad, text=1], mask: [true, false, true]
        let slots = [2u32, av as u32, 1u32];
        let mask  = [true, false, true];
        let result = compute_position_embedding(&text_emb, &audio_emb, &slots, &mask, d, n_cb, av);
        assert_eq!(result.len(), d);
        // text(1.0) + audio_CB0(2.0) = 3.0 for all dims
        for &v in &result { assert!((v - 3.0).abs() < 1e-6, "expected 3.0, got {}", v); }
    }
}
