// resample.cu — Sinc Interpolation Audio Resampling
// OpenSesame-Rust | Phase B
//
// Kaiser-windowed sinc interpolation for audio rate conversion.
// Common ratios: 44100→24000, 48000→24000, 16000→24000
//
// Algorithm:
//   output[n] = Σ input[m] · sinc(n/L - m/M) · kaiser(n/L - m/M)
//   where L=output_rate, M=input_rate (reduced by GCD)
//
// Kaiser window: w(x) = I₀(β√(1-(2x/N)²)) / I₀(β)  β=8.0, N=64 taps
// This achieves >90dB stopband attenuation.
//
// TODO (Phase B): implement polyphase filter bank version for efficiency

#include <stdint.h>
#include <math.h>

#define PI       3.14159265358979323846f
#define TAPS     64
#define BETA     8.0f

// Zeroth-order modified Bessel function I₀(x)
__device__ float bessel_i0(float x) {
    float sum = 1.0f;
    float term = 1.0f;
    float half_x = x * 0.5f;
    for (int k = 1; k <= 20; k++) {
        term *= (half_x / k) * (half_x / k);
        sum  += term;
        if (term < 1e-10f) break;
    }
    return sum;
}

// Sinc function: sin(π·x) / (π·x)
__device__ float sinc(float x) {
    if (fabsf(x) < 1e-8f) return 1.0f;
    float px = PI * x;
    return sinf(px) / px;
}

extern "C" __global__ void resample_kernel(
    const float* __restrict__ input,   // [B, T_in]
    float*       __restrict__ output,  // [B, T_out]
    int B, int T_in, int T_out,
    float ratio,      // T_out / T_in  (e.g. 24000/44100 ≈ 0.544)
    float inv_ratio   // T_in / T_out
) {
    int b = blockIdx.x;
    int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || n >= T_out) return;

    float i0_beta = bessel_i0(BETA);
    float center  = n * inv_ratio;   // position in input space
    float acc     = 0.0f;
    float norm    = 0.0f;

    float cutoff = (ratio < 1.0f) ? ratio : 1.0f;  // anti-alias

    for (int tap = -TAPS/2; tap <= TAPS/2; tap++) {
        int m = (int)(center + 0.5f) + tap;
        if (m < 0 || m >= T_in) continue;

        float x = (center - m) * cutoff;
        // Kaiser window
        float window_pos = (float)tap / (TAPS / 2);
        float w = (fabsf(window_pos) <= 1.0f)
            ? bessel_i0(BETA * sqrtf(1.0f - window_pos * window_pos)) / i0_beta
            : 0.0f;

        float h = sinc(x) * w * cutoff;
        acc  += input[b * T_in + m] * h;
        norm += h;
    }

    output[b * T_out + n] = (norm > 1e-8f) ? acc / norm : 0.0f;
}
