//! External integration tests for Frame (Phase H).

use opensesame_csm::Frame;

/// TEST F01: text frame has only col 32 set in mask.
#[test]
fn text_frame_only_col32_valid() {
    let f = Frame::from_text_tokens(&[42i64]);
    assert_eq!(f.seq_len(), 1);
    for col in 0..32 {
        assert!(!f.mask[0][col], "col {col} must be invalid in text frame");
    }
    assert!(f.mask[0][32], "col 32 (text) must be valid");
    assert_eq!(f.tokens[0][32], 42);
}

/// TEST F02: audio frame has cols 0..n_codebooks set in mask, not col n_codebooks.
#[test]
fn audio_frame_audio_cols_valid() {
    let n_cb = 4;
    let audio: Vec<Vec<i64>> = (0..n_cb).map(|cb| vec![(cb + 1) as i64; 3]).collect();
    let f = Frame::from_audio_tokens(&audio, n_cb);
    for t in 0..3 {
        for col in 0..n_cb {
            assert!(f.mask[t][col], "audio col {col} must be valid");
        }
        assert!(!f.mask[t][n_cb], "text col must be invalid in audio frame");
    }
}

/// TEST F03: concat produces correct seq_len.
#[test]
fn concat_seq_len() {
    let a = Frame::from_text_tokens(&[1, 2, 3]);
    let b = Frame::from_text_tokens(&[4, 5]);
    let c = Frame::concat(a, b);
    assert_eq!(c.seq_len(), 5);
}

/// TEST F04: from_text_tokens produces correct token placement.
#[test]
fn text_frame_token_placement() {
    let toks: Vec<i64> = (10..15).collect();
    let f = Frame::from_text_tokens(&toks);
    for (t, &expected) in toks.iter().enumerate() {
        assert_eq!(f.tokens[t][32], expected, "token[{t}][32] should be {expected}");
    }
}

/// TEST F05: from_audio_tokens with 32 codebooks × T frames has correct shape.
#[test]
fn audio_frame_csm1b_shape() {
    let n_cb = 32;
    let n_frames = 7;
    let audio: Vec<Vec<i64>> = (0..n_cb).map(|_| vec![100i64; n_frames]).collect();
    let f = Frame::from_audio_tokens(&audio, n_cb);
    assert_eq!(f.seq_len(), n_frames);
    assert_eq!(f.frame_width(), 33); // 32 + 1
}

/// TEST F06: from_audio_tokens stores tokens correctly.
#[test]
fn audio_frame_tokens_correct() {
    let n_cb = 4;
    let audio: Vec<Vec<i64>> = (0..n_cb)
        .map(|cb| (0..3).map(|t| (cb * 10 + t) as i64).collect())
        .collect();
    let f = Frame::from_audio_tokens(&audio, n_cb);
    for t in 0..3 {
        for cb in 0..n_cb {
            assert_eq!(f.tokens[t][cb], (cb * 10 + t) as i64);
        }
    }
}

/// TEST F07: EOS frame = all zeros, mask all false.
#[test]
fn eos_frame_all_false() {
    let f = Frame::eos(4, 33);
    assert_eq!(f.seq_len(), 4);
    assert_eq!(f.frame_width(), 33);
    for t in 0..4 {
        for col in 0..33 {
            assert!(!f.mask[t][col], "EOS mask must be all false");
            assert_eq!(f.tokens[t][col], 0, "EOS tokens must be 0");
        }
    }
}

/// TEST F08: empty text tokens produces empty frame.
#[test]
fn empty_text_tokens_empty_frame() {
    let f = Frame::from_text_tokens(&[]);
    assert_eq!(f.seq_len(), 0);
}

/// TEST F09: concat empty + non-empty = non-empty.
#[test]
fn concat_empty_plus_nonempty() {
    let empty = Frame::from_text_tokens(&[]);
    let nonempty = Frame::from_text_tokens(&[1, 2]);
    let c = Frame::concat(empty, nonempty);
    assert_eq!(c.seq_len(), 2);
}

/// TEST F10: concat preserves mask of both halves.
#[test]
fn concat_preserves_masks() {
    let a = Frame::from_text_tokens(&[10]);  // mask[0][32] = true
    let n_cb = 4;
    let audio: Vec<Vec<i64>> = (0..n_cb).map(|_| vec![5i64]).collect();
    let b = Frame::from_audio_tokens(&audio, n_cb);
    // b.frame_width = 5, a.frame_width = 33 — these would panic on concat
    // Use same-width frames for this test
    let a2 = Frame::from_text_tokens(&[10]);
    let b2 = Frame::from_text_tokens(&[20]);
    let c = Frame::concat(a2, b2);
    assert!(c.mask[0][32]);   // from `a2`
    assert!(c.mask[1][32]);   // from `b2`
    assert_eq!(c.tokens[0][32], 10);
    assert_eq!(c.tokens[1][32], 20);
    // suppress unused warning
    let _ = a; let _ = b;
}
