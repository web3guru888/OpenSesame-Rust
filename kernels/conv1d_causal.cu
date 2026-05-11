// conv1d_causal.cu — 1D Causal Convolution CUDA Kernel
// OpenSesame-Rust | Phase B
//
// Computes 1D convolution with causal (left-only) padding.
// No future context: output[t] depends only on input[0..=t].
// Used in SEANet encoder for real-time streaming audio processing.
//
// Shapes:
//   input:  [batch, C_in,  T]
//   weight: [C_out, C_in,  kernel_size]
//   bias:   [C_out]  (optional)
//   output: [batch, C_out, T]  (causal: left-pad input by kernel_size-1 zeros)
//
// Dilation: output[t] = sum_{k=0}^{K-1} weight[o,i,k] * input[t - k*dilation]
// Stride:   for downsampling, only compute output at t % stride == 0
//
// TODO (Phase B): implement full kernel body

#include <stdint.h>
#include <math.h>

// Kernel signature — wired from atlas-tensor build.rs
extern "C" __global__ void conv1d_causal_kernel(
    const float* __restrict__ input,    // [B, C_in, T]
    const float* __restrict__ weight,   // [C_out, C_in, K]
    const float* __restrict__ bias,     // [C_out] or NULL
    float*       __restrict__ output,   // [B, C_out, T_out]
    int batch, int c_in, int c_out,
    int time_in, int time_out,
    int kernel_size, int stride, int dilation,
    int padding                          // = (kernel_size-1)*dilation for causal
) {
    // Phase B implementation goes here
    // Grid: (batch * c_out) blocks, each handles all T_out positions
    int b    = blockIdx.x / c_out;
    int ch_o = blockIdx.x % c_out;
    int t_o  = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || ch_o >= c_out || t_o >= time_out) return;

    float acc = (bias != NULL) ? bias[ch_o] : 0.0f;
    int t_i_base = t_o * stride - padding;

    for (int ci = 0; ci < c_in; ci++) {
        for (int k = 0; k < kernel_size; k++) {
            int t_i = t_i_base + k * dilation;
            if (t_i >= 0 && t_i < time_in) {
                float x = input[b * c_in * time_in + ci * time_in + t_i];
                float w = weight[ch_o * c_in * kernel_size + ci * kernel_size + k];
                acc += x * w;
            }
        }
    }
    output[b * c_out * time_out + ch_o * time_out + t_o] = acc;
}
