// depthwise_conv1d.cu — Depthwise Separable 1D Convolution
// OpenSesame-Rust | Phase B
//
// groups = C_in = C_out: each input channel has its own filter.
// Used in SEANet ResidualUnits for efficiency.
//
// Memory-efficient: weight size [C, 1, K] vs [C_out, C_in, K] for full conv.

#include <stdint.h>

extern "C" __global__ void depthwise_conv1d_kernel(
    const float* __restrict__ input,    // [B, C, T]
    const float* __restrict__ weight,   // [C, 1, K]
    const float* __restrict__ bias,     // [C] or NULL
    float*       __restrict__ output,   // [B, C, T]
    int batch, int channels, int time_len,
    int kernel_size, int dilation
) {
    int b  = blockIdx.x / channels;
    int ch = blockIdx.x % channels;
    int t  = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || ch >= channels || t >= time_len) return;

    float acc = (bias != NULL) ? bias[ch] : 0.0f;
    int padding = (kernel_size - 1) * dilation;

    for (int k = 0; k < kernel_size; k++) {
        int t_i = t - padding + k * dilation;
        if (t_i >= 0 && t_i < time_len) {
            float x = input[b * channels * time_len + ch * time_len + t_i];
            float w = weight[ch * kernel_size + k];
            acc += x * w;
        }
    }
    output[b * channels * time_len + ch * time_len + t] = acc;
}
