//! Integration tests for opensesame-seanet (Phase D).
//! All tests use T=960 or smaller for speed. Architecture correctness is tested via shapes.

use opensesame_seanet::{
    SEANetDecoder, SEANetEncoder, StreamingEncoder,
    CausalConv1d, EncoderBlock, DecoderBlock, ResidualUnit,
};

fn sine(n: usize, freq: f32) -> Vec<f32> {
    let sr = 24000.0f32;
    (0..n).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin()).collect()
}

/// Scale-invariant SNR (dB).
fn sisnr(reference: &[f32], estimated: &[f32]) -> f32 {
    assert_eq!(reference.len(), estimated.len());
    let n = reference.len() as f32;
    let sm = reference.iter().sum::<f32>() / n;
    let em = estimated.iter().sum::<f32>() / n;
    let s: Vec<f32> = reference.iter().map(|&v| v - sm).collect();
    let e: Vec<f32> = estimated.iter().map(|&v| v - em).collect();
    let sd: f32 = s.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
    let ss: f32 = s.iter().map(|v| v * v).sum();
    if ss < 1e-9 { return -100.0; }
    let alpha = sd / ss;
    let st: Vec<f32> = s.iter().map(|v| alpha * v).collect();
    let noise: Vec<f32> = st.iter().zip(e.iter()).map(|(a, b)| a - b).collect();
    let ts: f32 = st.iter().map(|v| v * v).sum::<f32>().max(1e-9);
    let ns: f32 = noise.iter().map(|v| v * v).sum::<f32>().max(1e-9);
    10.0 * (ts / ns).log10()
}

// ─────────────────────────────────────────────────────────────────────────────
// Roundtrip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_encoder_decoder_shape_roundtrip() {
    // Encode 960 samples → 1 latent frame → decode back to 960 samples.
    let enc = SEANetEncoder::new();
    let dec = SEANetDecoder::new();
    let audio = vec![0.1f32; 960];
    let (latent, t_lat) = enc.forward(&audio, 1, 960);
    let (recon, t_out) = dec.forward(&latent, 1, t_lat);
    assert_eq!(t_out, 960, "roundtrip output length mismatch");
    assert_eq!(recon.len(), 960);
}

#[test]
fn test_encoder_decoder_rough_quality() {
    // SISNR is computable (finite) — a random untrained codec won't be > 0 dB.
    let enc = SEANetEncoder::new();
    let dec = SEANetDecoder::new();
    let audio = sine(960, 440.0);
    let (latent, t_lat) = enc.forward(&audio, 1, 960);
    let (recon, _) = dec.forward(&latent, 1, t_lat);
    let s = sisnr(&audio, &recon);
    assert!(s.is_finite(), "SISNR is not finite");
}

#[test]
fn test_encoder_decoder_batch_shapes() {
    // B=2, T=960: batch size 2.
    let enc = SEANetEncoder::new();
    let dec = SEANetDecoder::new();
    let audio = vec![0.0f32; 2 * 960];
    let (latent, t_lat) = enc.forward(&audio, 2, 960);
    assert_eq!(t_lat, 1);
    let (recon, t_out) = dec.forward(&latent, 2, t_lat);
    assert_eq!(t_out, 960);
    assert_eq!(recon.len(), 2 * 960);
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture validation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_encoder_block_strides_in_mimi_order() {
    // Encoder strides: [8, 6, 5, 4].
    let enc = SEANetEncoder::new();
    let expected = [8usize, 6, 5, 4];
    for (i, block) in enc.blocks.iter().enumerate() {
        assert_eq!(block.stride, expected[i], "encoder block {} stride", i);
    }
}

#[test]
fn test_decoder_block_strides_in_mimi_order() {
    // Decoder strides: [4, 5, 6, 8].
    let dec = SEANetDecoder::new();
    let expected = [4usize, 5, 6, 8];
    for (i, block) in dec.blocks.iter().enumerate() {
        assert_eq!(block.stride, expected[i], "decoder block {} stride", i);
    }
}

#[test]
fn test_encoder_channel_progression() {
    // 64 → 128 → 256 → 512 → 1024.
    let enc = SEANetEncoder::new();
    let expected_out = [128usize, 256, 512, 1024];
    for (i, block) in enc.blocks.iter().enumerate() {
        assert_eq!(block.out_ch, expected_out[i], "encoder block {} out_ch", i);
    }
    assert_eq!(enc.output_conv.weight_norm.in_ch, 1024);
    assert_eq!(enc.output_conv.weight_norm.out_ch, 512);
}

#[test]
fn test_decoder_channel_progression() {
    // 1024 → 512 → 256 → 128 → 64.
    let dec = SEANetDecoder::new();
    let expected_out = [512usize, 256, 128, 64];
    for (i, block) in dec.blocks.iter().enumerate() {
        assert_eq!(block.out_ch, expected_out[i], "decoder block {} out_ch", i);
    }
    assert_eq!(dec.output_conv.weight_norm.in_ch, 64);
    assert_eq!(dec.output_conv.weight_norm.out_ch, 1);
}

#[test]
fn test_residual_unit_all_channel_sizes() {
    // Verify residual unit preserves shape for all Mimi channel counts.
    for &ch in &[64usize, 128, 256, 512] {
        let ru = ResidualUnit::new(ch);
        let input: Vec<f32> = (0..ch * 4).map(|i| i as f32 * 0.001).collect();
        let out = ru.forward(&input, 1, ch, 4);
        assert_eq!(out.len(), ch * 4, "channels not preserved ch={}", ch);
    }
}

#[test]
fn test_causal_conv_output_len_mimi_strides() {
    // Verify each Mimi encoder stride with T=960 chain.
    let cases = [(960usize, 8usize, 120usize), (120, 6, 20), (20, 5, 4), (4, 4, 1)];
    for (t_in, stride, expected) in cases {
        let conv = CausalConv1d::new(1, 1, 2 * stride, stride, 1);
        assert_eq!(conv.output_len(t_in), expected, "stride={}", stride);
    }
}

#[test]
fn test_streaming_long_audio_matches_batch() {
    // 4 chunks of 960 = 3840 samples; compare each frame against batch.
    let n_chunks = 4usize;
    let audio = sine(n_chunks * 960, 220.0);
    let enc = SEANetEncoder::new();
    let (batch_out, _) = enc.forward(&audio, 1, audio.len());
    // batch_out: [512, n_chunks], batch_out[ch * n_chunks + t] = value.

    let mut se = StreamingEncoder::new();
    for (t, chunk) in audio.chunks(960).enumerate() {
        let frame = se.push_chunk(chunk); // 512 elements
        for ch in 0..512 {
            assert!(
                (frame[ch] - batch_out[ch * n_chunks + t]).abs() < 1e-4,
                "t={} ch={}: {} vs {}", t, ch, frame[ch], batch_out[ch * n_chunks + t]
            );
        }
    }
}

#[test]
fn test_encoder_decoder_output_finite() {
    let enc = SEANetEncoder::new();
    let dec = SEANetDecoder::new();
    let audio: Vec<f32> = (0..960).map(|i| (i as f32 * 0.001).cos()).collect();
    let (lat, t_lat) = enc.forward(&audio, 1, 960);
    assert!(lat.iter().all(|v| v.is_finite()));
    let (recon, _) = dec.forward(&lat, 1, t_lat);
    assert!(recon.iter().all(|v| v.is_finite()));
}

#[test]
fn test_encoder_block_3_is_512_to_1024() {
    // The 4th encoder block (stride=4) goes 512→1024.
    let enc = SEANetEncoder::new();
    assert_eq!(enc.blocks[3].in_ch, 512);
    assert_eq!(enc.blocks[3].out_ch, 1024);
    assert_eq!(enc.blocks[3].stride, 4);
}

#[test]
fn test_decoder_block_0_is_1024_to_512() {
    // The 1st decoder block (stride=4) goes 1024→512.
    let dec = SEANetDecoder::new();
    assert_eq!(dec.blocks[0].in_ch, 1024);
    assert_eq!(dec.blocks[0].out_ch, 512);
    assert_eq!(dec.blocks[0].stride, 4);
}
