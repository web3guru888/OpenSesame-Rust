//! Zero-dependency WAV file parser and writer.
//!
//! Supported audio formats:
//! * PCM 16-bit integer  (AudioFormat = 1, BitsPerSample = 16)
//! * PCM 24-bit integer  (AudioFormat = 1, BitsPerSample = 24)
//! * PCM 32-bit integer  (AudioFormat = 1, BitsPerSample = 32)
//! * IEEE 754 32-bit float (AudioFormat = 3, BitsPerSample = 32)
//!
//! All samples are converted to/from `f32` in the range `[-1.0, 1.0]`.

use std::fs::File;
use std::io::{Read, Write};
use crate::audio_buffer::AudioBuffer;

// ── RIFF/WAV format constants ────────────────────────────────────────────────
const RIFF_ID: &[u8; 4] = b"RIFF";
const WAVE_ID: &[u8; 4] = b"WAVE";
const FMT_ID:  &[u8; 4] = b"fmt ";
const DATA_ID: &[u8; 4] = b"data";

const AUDIO_FORMAT_PCM:   u16 = 1;
const AUDIO_FORMAT_FLOAT: u16 = 3;
const AUDIO_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

// ── WavReader ────────────────────────────────────────────────────────────────

/// Reads a WAV file from disk and returns a normalised [`AudioBuffer`].
pub struct WavReader;

impl WavReader {
    /// Open and parse a WAV file at `path`.
    ///
    /// Returns an [`AudioBuffer`] with `f32` samples in `[-1.0, 1.0]`, or
    /// a descriptive error `String` on failure.
    ///
    /// # Supported formats
    /// PCM 16/24/32-bit and IEEE float 32-bit.
    pub fn open(path: &str) -> Result<AudioBuffer, String> {
        let mut f = File::open(path)
            .map_err(|e| format!("cannot open '{}': {}", path, e))?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)
            .map_err(|e| format!("read error '{}': {}", path, e))?;
        parse_wav(&data)
    }
}

// ── WavWriter ────────────────────────────────────────────────────────────────

/// Writes an [`AudioBuffer`] to disk as a WAV file.
pub struct WavWriter;

impl WavWriter {
    /// Write `buf` as a WAV file at `path`.
    ///
    /// Samples are encoded as IEEE 754 32-bit floats (AudioFormat = 3).
    ///
    /// Returns `Ok(())` or a descriptive error `String`.
    pub fn write(buf: &AudioBuffer, path: &str) -> Result<(), String> {
        let mut f = File::create(path)
            .map_err(|e| format!("cannot create '{}': {}", path, e))?;
        let raw = encode_wav_f32(buf)?;
        f.write_all(&raw)
            .map_err(|e| format!("write error '{}': {}", path, e))?;
        Ok(())
    }

    /// Write `buf` as a PCM 16-bit WAV file at `path`.
    ///
    /// Samples are clamped to `[-1.0, 1.0]` before quantisation.
    pub fn write_pcm16(buf: &AudioBuffer, path: &str) -> Result<(), String> {
        let mut f = File::create(path)
            .map_err(|e| format!("cannot create '{}': {}", path, e))?;
        let raw = encode_wav_pcm16(buf)?;
        f.write_all(&raw)
            .map_err(|e| format!("write error '{}': {}", path, e))?;
        Ok(())
    }
}

// ── Parsing ──────────────────────────────────────────────────────────────────

/// Parse raw WAV bytes into an [`AudioBuffer`].
fn parse_wav(data: &[u8]) -> Result<AudioBuffer, String> {
    if data.len() < 12 {
        return Err("file too small to be a WAV".into());
    }

    // RIFF header
    if &data[0..4] != RIFF_ID {
        return Err("missing RIFF header".into());
    }
    if &data[8..12] != WAVE_ID {
        return Err("missing WAVE identifier".into());
    }

    // Iterate over chunks
    let mut pos = 12_usize;
    let mut audio_format:   Option<u16> = None;
    let mut num_channels:   Option<u16> = None;
    let mut sample_rate:    Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut pcm_data:       Option<&[u8]> = None;

    while pos + 8 <= data.len() {
        let chunk_id   = &data[pos..pos + 4];
        let chunk_size = read_u32_le(data, pos + 4) as usize;
        pos += 8;

        if pos + chunk_size > data.len() {
            // Truncated chunk — try what we have.
            break;
        }

        let chunk_data = &data[pos..pos + chunk_size];

        if chunk_id == FMT_ID {
            if chunk_size < 16 {
                return Err("fmt chunk too small".into());
            }
            let fmt = read_u16_le(chunk_data, 0);
            // Handle EXTENSIBLE: treat sub-format as PCM if GUID starts with 01 00
            let resolved_fmt = if fmt == AUDIO_FORMAT_EXTENSIBLE && chunk_size >= 28 {
                read_u16_le(chunk_data, 20) // SubFormat first 2 bytes
            } else {
                fmt
            };
            audio_format    = Some(resolved_fmt);
            num_channels    = Some(read_u16_le(chunk_data, 2));
            sample_rate     = Some(read_u32_le(chunk_data, 4));
            bits_per_sample = Some(read_u16_le(chunk_data, 14));
        } else if chunk_id == DATA_ID {
            pcm_data = Some(chunk_data);
        }
        // Skip unknown chunks (e.g. LIST, INFO).
        pos += chunk_size;
        // Chunks are word-aligned.
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }

    let audio_format    = audio_format.ok_or("missing fmt chunk")?;
    let num_channels    = num_channels.ok_or("missing fmt chunk")?;
    let sample_rate     = sample_rate.ok_or("missing fmt chunk")?;
    let bits_per_sample = bits_per_sample.ok_or("missing fmt chunk")?;
    let pcm_data        = pcm_data.ok_or("missing data chunk")?;

    if num_channels == 0 {
        return Err("invalid channel count (0)".into());
    }

    let samples = decode_samples(pcm_data, audio_format, bits_per_sample)?;

    Ok(AudioBuffer::new(samples, sample_rate, num_channels as u8))
}

/// Decode raw PCM bytes into normalised `f32` samples.
fn decode_samples(data: &[u8], fmt: u16, bits: u16) -> Result<Vec<f32>, String> {
    match (fmt, bits) {
        (AUDIO_FORMAT_PCM, 16) => {
            if data.len() % 2 != 0 {
                return Err("PCM16 data length not multiple of 2".into());
            }
            Ok((0..data.len() / 2)
                .map(|i| {
                    let s = read_i16_le(data, i * 2);
                    s as f32 / 32768.0
                })
                .collect())
        }
        (AUDIO_FORMAT_PCM, 24) => {
            if data.len() % 3 != 0 {
                return Err("PCM24 data length not multiple of 3".into());
            }
            Ok((0..data.len() / 3)
                .map(|i| {
                    let b0 = data[i * 3] as i32;
                    let b1 = data[i * 3 + 1] as i32;
                    let b2 = data[i * 3 + 2] as i32;
                    // Sign-extend 24-bit two's complement.
                    let val = b0 | (b1 << 8) | (b2 << 16);
                    let val = if val & 0x80_0000 != 0 {
                        val | !0xFF_FFFF_i32
                    } else {
                        val
                    };
                    val as f32 / 8_388_608.0 // 2^23
                })
                .collect())
        }
        (AUDIO_FORMAT_PCM, 32) => {
            if data.len() % 4 != 0 {
                return Err("PCM32 data length not multiple of 4".into());
            }
            Ok((0..data.len() / 4)
                .map(|i| {
                    let s = read_i32_le(data, i * 4);
                    s as f32 / 2_147_483_648.0 // 2^31
                })
                .collect())
        }
        (AUDIO_FORMAT_FLOAT, 32) => {
            if data.len() % 4 != 0 {
                return Err("float32 data length not multiple of 4".into());
            }
            Ok((0..data.len() / 4)
                .map(|i| read_f32_le(data, i * 4))
                .collect())
        }
        _ => Err(format!(
            "unsupported WAV format: AudioFormat={}, BitsPerSample={}",
            fmt, bits
        )),
    }
}

// ── Encoding ─────────────────────────────────────────────────────────────────

/// Encode an `AudioBuffer` as a WAV byte vector (IEEE float 32-bit).
fn encode_wav_f32(buf: &AudioBuffer) -> Result<Vec<u8>, String> {
    let num_channels    = buf.channels as u16;
    let sample_rate     = buf.sample_rate;
    let bits_per_sample: u16 = 32;
    let byte_rate       = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align     = num_channels * (bits_per_sample / 8);
    let data_size       = (buf.samples.len() * 4) as u32;
    let chunk_size      = 36 + data_size;

    let mut out = Vec::with_capacity(44 + buf.samples.len() * 4);
    // RIFF header
    out.extend_from_slice(RIFF_ID);
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(WAVE_ID);
    // fmt chunk
    out.extend_from_slice(FMT_ID);
    out.extend_from_slice(&16_u32.to_le_bytes()); // chunk size
    out.extend_from_slice(&(AUDIO_FORMAT_FLOAT as u16).to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data chunk
    out.extend_from_slice(DATA_ID);
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in &buf.samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}

/// Encode an `AudioBuffer` as a WAV byte vector (PCM 16-bit).
fn encode_wav_pcm16(buf: &AudioBuffer) -> Result<Vec<u8>, String> {
    let num_channels    = buf.channels as u16;
    let sample_rate     = buf.sample_rate;
    let bits_per_sample: u16 = 16;
    let byte_rate       = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align     = num_channels * (bits_per_sample / 8);
    let data_size       = (buf.samples.len() * 2) as u32;
    let chunk_size      = 36 + data_size;

    let mut out = Vec::with_capacity(44 + buf.samples.len() * 2);
    out.extend_from_slice(RIFF_ID);
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(WAVE_ID);
    out.extend_from_slice(FMT_ID);
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&(AUDIO_FORMAT_PCM as u16).to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(DATA_ID);
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in &buf.samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

// ── Byte-order helpers ───────────────────────────────────────────────────────

#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

#[inline]
fn read_i16_le(data: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([data[offset], data[offset + 1]])
}

#[inline]
fn read_i32_le(data: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

#[inline]
fn read_f32_le(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}
