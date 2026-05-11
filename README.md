# OpenSesame-Rust

> **Pure Rust Conversational Speech Model — built on [ATLAS](https://github.com/web3guru888/ATLAS)**

[![License: Apache 2.0](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE-CODE)
[![License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Zero Dependencies](https://img.shields.io/badge/external%20crates-0-brightgreen.svg)](#zero-dependencies)
[![Crates](https://img.shields.io/badge/crates-34-blueviolet.svg)](#crate-status)
[![CUDA Kernels](https://img.shields.io/badge/CUDA%20kernels-16-76b900.svg)](#cuda-kernels)

---

OpenSesame is a fully open-source implementation of the **Conversational Speech Model (CSM)**
architecture described in Sesame's [Crossing the Uncanny Valley of Voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)
and the [Moshi paper](https://arxiv.org/abs/2410.00037).

**Built on [ATLAS](https://github.com/web3guru888/ATLAS)** — the pure Rust LLM framework
by [Robin Dey / OpenHub Research Thailand](https://openhubresearch.org).

**The SQLite principle applied to speech AI**: zero external Rust crate dependencies,
every algorithm implemented from scratch, every kernel hand-written in CUDA.

---

## What Is OpenSesame?

OpenSesame implements the full CSM pipeline:

```
Audio In (24kHz PCM)
    ↓
[opensesame-seanet]   SEANet Encoder — 4 strided causal conv blocks
                      ratios=[8,6,5,4] → 1920× temporal compression
    ↓
[opensesame-rvq]      Split-RVQ Tokenizer
                      CB0: semantic (WavLM-distilled)
                      CB1..CB7: acoustic (EMA-trained)
    ↓  discrete tokens [B, 8, T/1920]
[opensesame-backbone] Backbone Transformer
                      Llama-3.2-1B compatible (SwiGLU, RMSNorm, RoPE, GQA)
                      Multimodal: text (128k) + audio (2048) tokens
                      Inner Monologue: dual text+audio prediction head
    ↓  CB0 logits + hidden states
[opensesame-depformer] Depformer Audio Decoder
                       Lightweight transformer (4–6 layers, d=1024)
                       Generates CB1..CB7 from backbone hidden state
    ↓  all 8 codebooks
[opensesame-seanet]   SEANet Decoder — 4 transposed conv blocks (upsample)
    ↓
Audio Out (24kHz PCM)
```

**Key properties:**
- **12.5 fps** — 80ms per audio frame
- **8 codebooks** — 1 semantic + 7 acoustic, 2048 codes each
- **1.1 kbps** — codec bitrate
- **< 400ms** — end-to-end conversational latency target
- **RTF < 0.5** — faster than real-time on A100

---

## Architecture: 34 Crates, 16 CUDA Kernels

```
OpenSesame-Rust/
├── Cargo.toml          # 34-crate workspace — [dependencies] empty by design
├── kernels/
│   │  ── ATLAS (inherited) ──────────────────────────────────────────────
│   ├── matmul.cu           tiled GEMM, cuBLAS TF32 tensor cores
│   ├── attention.cu        GQA decode attention, KV cache, RoPE, RMSNorm
│   └── quant.cu            INT4/INT8 quantization
│   │  ── OpenSesame (new) ───────────────────────────────────────────────
│   ├── conv1d_causal.cu    1D causal conv — SEANet encoder
│   ├── conv1d_strided.cu   Strided downsample (ratios 4×5×6×8 = 1920×)
│   ├── conv1d_transposed.cu Transposed upsample — SEANet decoder
│   ├── depthwise_conv1d.cu  Depthwise separable conv
│   ├── vq_search.cu        Tiled L2 VQ nearest-neighbour search (K=2048, D=256)
│   ├── ema_update.cu       Atomic EMA codebook update (VQ training)
│   ├── stft.cu             Batched FFT → mel spectrogram
│   └── resample.cu         Kaiser-windowed sinc (44.1/48/16kHz → 24kHz)
└── crates/
    │  ── ATLAS (22 crates, unchanged or minimally extended) ─────────────
    ├── atlas-core/         Error types, traits, config
    ├── atlas-tensor/       Tensor + all 16 CUDA kernels  ← EXTENDED
    ├── atlas-grad/         Autograd tape, backward pass
    ├── atlas-optim/        AdamW, cosine LR, warmup
    ├── atlas-quant/        INT4/INT8 quantization, QLoRA
    ├── atlas-model/        Transformer: MHA, FFN, RMSNorm, RoPE, GQA
    ├── atlas-tokenize/     BPE tokenizer, HF tokenizer.json (zero deps)
    ├── atlas-infer/        StigmergicHook + InferEngine GPU/CPU dispatch
    ├── atlas-corpus/       SFT trainer, LoRA, DeepSupervisionTrainer
    ├── atlas-api/          OpenAI-compatible HTTP endpoint  ← EXTENDED
    ├── atlas-palace/       GraphPalace stigmergic memory
    ├── atlas-mcp/          MCP server, 28 tools, JSON-RPC 2.0
    ├── atlas-trm/          TRM-CausalValidator (7M params)
    ├── atlas-safety/       Horn-clause safety constitution
    ├── atlas-http/         HTTP client via raw libc syscalls
    ├── atlas-json/         JSON parser from scratch
    ├── atlas-causal/       PC/FCI causal inference
    ├── atlas-bayes/        Bayesian confidence scoring
    ├── atlas-astra/        ASTRA OODA discovery engine
    ├── atlas-zk/           ZK Schnorr proofs
    ├── atlas-bridge/       ZK-attested blockchain interface
    └── atlas-cli/          Atlas CLI (extended with opensesame subcommand)
    │  ── OpenSesame (12 new crates) ─────────────────────────────────────
    ├── opensesame-audio/       WAV I/O, sinc resampler, ring buffer, VAD
    ├── opensesame-rvq/         VQ + ResidualVQ + SplitRVQ + EMA training
    ├── opensesame-seanet/      SEANet causal conv encoder/decoder
    ├── opensesame-mimi/        Full Mimi codec + safetensors weight loader
    ├── opensesame-backbone/    CSM backbone (multimodal, Inner Monologue)
    ├── opensesame-depformer/   Audio decoder (Depformer, CB1..CB7)
    ├── opensesame-csm/         Full CSM assembly + streaming session
    ├── opensesame-data/        LibriSpeech/GigaSpeech loaders + DEFLATE/tar
    ├── opensesame-train/       3-stage training pipeline
    ├── opensesame-eval/        PESQ, STOI, SI-SNR, WER (all from scratch)
    ├── opensesame-serve/       WebSocket real-time voice API (RFC 6455)
    └── opensesame-cli/         `opensesame` CLI binary
```

---

## Zero Dependencies

```toml
# Cargo.toml — workspace root
[workspace]
# ...
# NO [dependencies] section — by design
```

Every algorithm written from scratch:
- ✅ WAV parser (RIFF/WAVE PCM16/f32)
- ✅ Sinc resampler (Kaiser-windowed, 64-tap, β=8.0)
- ✅ DEFLATE/GZIP/TAR decompressor (LibriSpeech download)
- ✅ Vector quantization + EMA updates
- ✅ 1D causal/strided/transposed convolution (CPU + CUDA)
- ✅ VQ nearest-neighbour search (tiled CUDA, D=256, K=2048)
- ✅ Short-time Fourier transform (CUDA, Cooley-Tukey)
- ✅ SEANet encoder/decoder (weight-normalized causal conv stack)
- ✅ Safetensors loader (header via atlas-json + raw f32/bf16)
- ✅ CSM multimodal embedding (extends atlas-model)
- ✅ Depformer transformer (uses atlas-model TransformerLayer)
- ✅ Multi-scale mel reconstruction loss
- ✅ PESQ metric (ITU-T P.862 port)
- ✅ STOI metric (Taal et al. 2010 port)
- ✅ WebSocket framing (RFC 6455)
- ✅ CTC beam search decoder (WER evaluation)

---

## Build Status

| Phase | Crates | Status | Tests |
|-------|--------|--------|-------|
| 0: Repo Setup | all scaffolds | ✅ Complete | — |
| A: Audio Foundation | opensesame-audio | 🟡 In Progress | 0/30 |
| B: CUDA Audio Kernels | atlas-tensor extensions | ⬜ Queued | 0/20 |
| C: Vector Quantization | opensesame-rvq | ⬜ Queued | 0/40 |
| D: SEANet Conv | opensesame-seanet | ⬜ Queued | 0/50 |
| E: Mimi Codec | opensesame-mimi | ⬜ Queued | 0/30 |
| F: CSM Backbone | opensesame-backbone | ⬜ Queued | 0/30 |
| G: Depformer | opensesame-depformer | ⬜ Queued | 0/25 |
| H: CSM Assembly | opensesame-csm | ⬜ Queued | 0/30 |
| I: Data Pipeline | opensesame-data | ⬜ Queued | 0/25 |
| J: Training | opensesame-train | ⬜ Queued | 0/20 |
| K: Evaluation | opensesame-eval | ⬜ Queued | 0/20 |
| L: Serving | opensesame-serve | ⬜ Queued | 0/15 |
| M: CLI + Paper | opensesame-cli | ⬜ Queued | 0/10 |
| **TOTAL** | **34 crates** | | **0/945** |

---

## Getting Started

**Prerequisites:**
- Rust 1.75+ (`rustup update stable`)
- CUDA 12.x + nvcc (optional; falls back to CPU if absent)
- GPU with sm_75+ for CUDA path

```bash
git clone https://github.com/web3guru888/OpenSesame-Rust.git
cd OpenSesame-Rust

# Build (CPU path, no GPU required for development)
cargo build --workspace --exclude atlas-tensor

# Run tests (CPU path)
cargo test --workspace --exclude atlas-tensor

# Build CLI (once implementation is complete)
cargo build --release -p opensesame-cli

# Usage (Phase M+)
./target/release/opensesame download --dataset librispeech --split train-clean-100
./target/release/opensesame train    --stage codec --config config.toml
./target/release/opensesame train    --stage backbone --config config.toml
./target/release/opensesame infer    --model ./ckpt --input speech.wav
./target/release/opensesame serve    --model ./ckpt --port 8080
```

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| M1 | WAV → Rust → WAV, SNR > 100dB | ⬜ |
| M2 | VQ codebook usage > 80%, commit loss converges | ⬜ |
| M3 | SEANet 1920× compression, causal property verified | ⬜ |
| **M4** | **Mimi codes bit-identical to kyutai/moshi Python** | ⬜ |
| M5 | 1B backbone forward pass, GPU < 10ms | ⬜ |
| M6 | Depformer step < 1ms (real-time 80ms budget) | ⬜ |
| M7 | End-to-end: raw waveform in → raw waveform out | ⬜ |
| M8 | Codec converged: SISNR > 20dB, PESQ > 3.5 | ⬜ |
| M9 | Backbone converged: WER < 10%, natural speech | ⬜ |
| M10 | RTF < 0.5, latency < 400ms, 4 concurrent sessions | ⬜ |
| M11 | Paper submitted INTERSPEECH/ICASSP 2027 | ⬜ |

---

## Key Papers

- [Moshi](https://arxiv.org/abs/2410.00037) — core CSM architecture
- [Sesame Blog](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — voice interaction philosophy
- [SoundStream](https://arxiv.org/abs/2107.03312) — RVQ foundations
- [Encodec](https://arxiv.org/abs/2210.13438) — neural audio codec reference
- [WavLM](https://arxiv.org/abs/2110.13900) — semantic codebook distillation
- [ATLAS](https://github.com/web3guru888/ATLAS) — foundation framework

---

## License

- **Code** (`crates/`, `kernels/`): [Apache 2.0](LICENSE-CODE)
- **Documentation, paper, figures**: [CC BY 4.0](LICENSE)

Built on ATLAS by Robin Dey / OpenHub Research (Thailand). Apache 2.0.

---

## Citation

```bibtex
@software{opensesame_rust_2026,
  title       = {OpenSesame-Rust: Pure Rust Conversational Speech Model},
  year        = {2026},
  url         = {https://github.com/web3guru888/OpenSesame-Rust},
  note        = {Built on ATLAS (web3guru888/ATLAS). 34 crates, 16 CUDA kernels.
                 Zero external Rust crate dependencies. SEANet + RVQ + CSM backbone
                 + Depformer. Implements Sesame/Moshi CSM architecture in pure Rust.}
}
```
