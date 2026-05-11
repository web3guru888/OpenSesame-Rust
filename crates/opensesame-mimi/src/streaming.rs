//! Streaming Mimi codec for real-time audio tokenisation.

use crate::codec::Mimi;
use crate::config::MimiConfig;

/// Real-time frame-by-frame Mimi encoder/decoder.
pub struct StreamingMimi {
    /// Underlying codec.
    pub codec:          Mimi,
    /// PCM accumulation buffer.
    buffer:             Vec<f32>,
    /// Samples per codec frame.
    hop:                usize,
    /// Total frames encoded.
    pub frames_encoded: usize,
    /// Total frames decoded.
    pub frames_decoded: usize,
}

impl StreamingMimi {
    /// Create from a configured [`Mimi`] codec.
    pub fn new(codec: Mimi) -> Self {
        let hop = codec.config.hop_length();
        Self { codec, buffer: Vec::new(), hop, frames_encoded: 0, frames_decoded: 0 }
    }

    /// Create with default [`MimiConfig`].
    pub fn with_default_config() -> Self {
        Self::new(Mimi::new(MimiConfig::default()))
    }

    /// Reset all streaming state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.frames_encoded = 0;
        self.frames_decoded = 0;
    }

    /// Set number of active codebooks.
    pub fn set_num_codebooks(&mut self, n: usize) { self.codec.set_num_codebooks(n); }

    /// Return the active number of codebooks.
    pub fn num_codebooks(&self) -> usize { self.codec.num_codebooks() }

    /// Push samples; returns one frame's codes when `hop_length` samples accumulated.
    pub fn encode_step(&mut self, samples: &[f32]) -> Option<Vec<Vec<u32>>> {
        self.buffer.extend_from_slice(samples);
        if self.buffer.len() >= self.hop {
            let chunk: Vec<f32> = self.buffer.drain(..self.hop).collect();
            let codes = self.codec.encode(&chunk, self.hop);
            self.frames_encoded += 1;
            Some(codes)
        } else {
            None
        }
    }

    /// Decode one frame of codes to PCM.
    pub fn decode_step(&mut self, codes: Vec<Vec<u32>>) -> Vec<f32> {
        let (pcm, _) = self.codec.decode(&codes);
        self.frames_decoded += 1;
        pcm
    }

    /// Flush remaining samples (zero-padded to hop_length).
    pub fn flush(&mut self) -> Option<Vec<Vec<u32>>> {
        if self.buffer.is_empty() { return None; }
        self.buffer.resize(self.hop, 0.0);
        let chunk: Vec<f32> = self.buffer.drain(..).collect();
        let codes = self.codec.encode(&chunk, self.hop);
        self.frames_encoded += 1;
        Some(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_codec() -> Mimi {
        // transformer_dim must match encoder_dim=512 (SEANet hardcodes 512 output channels).
        Mimi::new(MimiConfig {
            transformer_layers: 1,
            transformer_context: 16,
            num_codebooks: 8,
            max_codebooks: 8,
            ..MimiConfig::default()
        })
    }

    #[test]
    fn test_streaming_new() {
        let s = StreamingMimi::new(small_codec());
        assert_eq!(s.hop, 1920);
        assert_eq!(s.frames_encoded, 0);
    }

    #[test]
    fn test_encode_step_partial_none() {
        let mut s = StreamingMimi::new(small_codec());
        assert!(s.encode_step(&vec![0.0_f32; 960]).is_none());
        assert_eq!(s.frames_encoded, 0);
    }

    #[test]
    fn test_encode_step_full_frame() {
        let mut s = StreamingMimi::new(small_codec());
        let result = s.encode_step(&vec![0.1_f32; 1920]);
        assert!(result.is_some());
        assert_eq!(s.frames_encoded, 1);
        let codes = result.unwrap();
        assert_eq!(codes.len(), 8);
        assert_eq!(codes[0].len(), 1);
    }

    #[test]
    fn test_encode_step_accumulates() {
        let mut s = StreamingMimi::new(small_codec());
        assert!(s.encode_step(&vec![0.0_f32; 960]).is_none());
        assert!(s.encode_step(&vec![0.0_f32; 960]).is_some());
        assert_eq!(s.frames_encoded, 1);
        assert_eq!(s.buffer.len(), 0);
    }

    #[test]
    fn test_decode_step_shape() {
        let mut s = StreamingMimi::new(small_codec());
        let codes: Vec<Vec<u32>> = (0..8).map(|_| vec![0u32]).collect();
        let pcm = s.decode_step(codes);
        assert!(!pcm.is_empty());
        assert_eq!(s.frames_decoded, 1);
    }

    #[test]
    fn test_reset() {
        let mut s = StreamingMimi::new(small_codec());
        s.encode_step(&vec![0.0_f32; 960]);
        s.reset();
        assert_eq!(s.buffer.len(), 0);
        assert_eq!(s.frames_encoded, 0);
    }

    #[test]
    fn test_flush_partial() {
        let mut s = StreamingMimi::new(small_codec());
        s.encode_step(&vec![0.0_f32; 500]);
        assert!(s.flush().is_some());
        assert_eq!(s.buffer.len(), 0);
    }

    #[test]
    fn test_flush_empty_none() {
        let mut s = StreamingMimi::new(small_codec());
        assert!(s.flush().is_none());
    }

    #[test]
    fn test_set_num_codebooks() {
        let mut s = StreamingMimi::new(small_codec());
        s.set_num_codebooks(4);
        assert_eq!(s.num_codebooks(), 4);
    }

    #[test]
    fn test_two_frame_stream() {
        let mut s = StreamingMimi::new(small_codec());
        assert!(s.encode_step(&vec![0.1_f32; 1920]).is_some());
        assert!(s.encode_step(&vec![0.2_f32; 1920]).is_some());
        assert_eq!(s.frames_encoded, 2);
    }
}
