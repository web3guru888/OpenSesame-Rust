// conv1d_transposed.cu — Transposed 1D Convolution (upsampling)
// OpenSesame-Rust | Phase B
//
// Used in SEANet DecoderBlocks to upsample from T/1920 back to T.
// stride=4,5,6,8 progressively restore temporal resolution.
//
// Algorithm: scatter-add (transpose of strided gather in conv1d_strided)
//   output[t_i] += weight[ci, ch_o, k] * input[t_o]
//   where t_i = t_o * stride + k - padding
//
// Boundary trimming: remove (kernel_size - stride) samples from output end
// to maintain causal alignment.

#include <stdint.h>

extern "C" __global__ void conv1d_transposed_kernel(
    const float* __restrict__ input,    // [B, C_in, T_in]
    const float* __restrict__ weight,   // [C_in, C_out, K]
    const float* __restrict__ bias,     // [C_out] or NULL
    float*       __restrict__ output,   // [B, C_out, T_out]  (zeroed before call)
    int batch, int c_in, int c_out,
    int time_in, int time_out,
    int kernel_size, int stride
) {
    int b    = blockIdx.x / c_in;
    int ch_i = blockIdx.x % c_in;
    int t_i  = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || ch_i >= c_in || t_i >= time_in) return;

    float val = input[b * c_in * time_in + ch_i * time_in + t_i];

    for (int co = 0; co < c_out; co++) {
        for (int k = 0; k < kernel_size; k++) {
            int t_o = t_i * stride + k;
            if (t_o >= 0 && t_o < time_out) {
                float w = weight[ch_i * c_out * kernel_size + co * kernel_size + k];
                atomicAdd(&output[b * c_out * time_out + co * time_out + t_o], val * w);
            }
        }
    }
    // Bias added in separate pass after all atomic-adds complete
}
