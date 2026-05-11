// conv1d_strided.cu — Strided 1D Causal Convolution (downsampling)
// OpenSesame-Rust | Phase B
//
// Fused causal + strided in a single pass for SEANet EncoderBlocks.
// Ratios used: stride=8, stride=6, stride=5, stride=4 → total 1920×
//
// This is a specialised version of conv1d_causal for stride>1.
// Outputs only every `stride`-th position: T_out = ceil(T_in / stride)
//
// TODO (Phase B): add tiled shared-memory optimisation for large C

#include <stdint.h>

extern "C" __global__ void conv1d_strided_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int batch, int c_in, int c_out,
    int time_in, int time_out,
    int kernel_size, int stride, int dilation
) {
    int b    = blockIdx.x / c_out;
    int ch_o = blockIdx.x % c_out;
    int t_o  = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || ch_o >= c_out || t_o >= time_out) return;

    float acc = (bias != NULL) ? bias[ch_o] : 0.0f;
    int padding = (kernel_size - 1) * dilation;   // causal left-pad
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
