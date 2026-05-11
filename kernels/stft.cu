// stft.cu — Short-Time Fourier Transform
// OpenSesame-Rust | Phase B
//
// Batched STFT for mel spectrogram computation.
// Used in: multi-scale mel reconstruction loss (training) + STOI eval metric.
//
// Algorithm: Cooley-Tukey radix-2 FFT on GPU.
//   1. Frame input: [B, T] → [B, num_frames, frame_size] with hop_size
//   2. Apply Hann window: frame *= hann_window
//   3. Zero-pad to next power of 2 if needed
//   4. FFT each frame in parallel
//   5. Compute magnitude: |X[f]|² = re² + im²
//   6. Apply mel filterbank: [B, num_frames, freq_bins] → [B, num_frames, n_mels]
//
// Multiple resolutions used in mel loss:
//   (frame_size=512,  hop=128,  n_mels=80)
//   (frame_size=1024, hop=256,  n_mels=80)
//   (frame_size=2048, hop=512,  n_mels=80)
//
// TODO (Phase B): implement Cooley-Tukey FFT + mel filterbank kernels

#include <stdint.h>
#include <math.h>

#define PI 3.14159265358979323846f

// Frame + window kernel
extern "C" __global__ void stft_frame_window_kernel(
    const float* __restrict__ audio,     // [B, T]
    const float* __restrict__ window,    // [frame_size]
    float*       __restrict__ frames,    // [B, num_frames, frame_size] (complex: real part)
    int B, int T, int frame_size, int hop_size, int num_frames
) {
    int b   = blockIdx.x;
    int fr  = blockIdx.y;
    int t   = threadIdx.x;
    if (b >= B || fr >= num_frames || t >= frame_size) return;

    int sample_idx = fr * hop_size + t;
    float val = (sample_idx < T) ? audio[b * T + sample_idx] : 0.0f;
    frames[b * num_frames * frame_size + fr * frame_size + t] = val * window[t];
}

// Hann window precomputation
extern "C" __global__ void hann_window_kernel(
    float* __restrict__ window,
    int frame_size
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= frame_size) return;
    window[t] = 0.5f * (1.0f - cosf(2.0f * PI * t / (frame_size - 1)));
}
