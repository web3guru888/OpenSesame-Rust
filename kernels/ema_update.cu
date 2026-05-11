// ema_update.cu — EMA Codebook Update for VQ Training
// OpenSesame-Rust | Phase B
//
// Implements the exponential moving average update for VQ codebook vectors.
// This is the training update for ResidualVQ — replaces the gradient-based
// update with a more stable EMA approach (used in SoundStream, Encodec, Mimi).
//
// EMA update equations (per codebook vector k):
//   cluster_size[k] = γ · cluster_size[k] + (1-γ) · n_k
//   embed_avg[k]    = γ · embed_avg[k]    + (1-γ) · sum_k
//   codebook[k]     = embed_avg[k] / (cluster_size[k] + ε)  (Laplace smoothing)
//
// where:
//   n_k    = number of vectors in batch assigned to codebook k
//   sum_k  = sum of all vectors assigned to codebook k
//   γ      = EMA decay (default 0.99)
//   ε      = 1e-5 (prevents division by zero)
//
// Inputs:
//   assignments: [N]       — argmin indices from vq_search
//   queries:     [N, D]    — input vectors (before quantization)
//   cluster_size:[K]       — EMA cluster counts (in-place update)
//   embed_avg:   [K, D]    — EMA embed sums (in-place update)
//   codebook:    [K, D]    — updated in-place
//   N, K, D, gamma: shapes + EMA decay

#include <stdint.h>

extern "C" __global__ void ema_scatter_kernel(
    const int32_t* __restrict__ assignments,  // [N]
    const float*   __restrict__ queries,      // [N, D]
    float*         __restrict__ cluster_size, // [K]
    float*         __restrict__ embed_avg,    // [K, D]
    int N, int K, int D
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int k = assignments[n];
    atomicAdd(&cluster_size[k], 1.0f);

    const float* q = queries + n * D;
    float* ea = embed_avg + k * D;
    for (int d = 0; d < D; d++) {
        atomicAdd(&ea[d], q[d]);
    }
}

extern "C" __global__ void ema_apply_kernel(
    float*       __restrict__ cluster_size,  // [K]  in-place
    float*       __restrict__ embed_avg,     // [K, D] in-place
    float*       __restrict__ codebook,      // [K, D] updated
    int K, int D,
    float gamma,
    float n_total   // batch size (for unbiased EMA scaling)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // EMA update for cluster size
    float n_k = cluster_size[k];   // raw scatter count from this batch
    cluster_size[k] = gamma * cluster_size[k] + (1.0f - gamma) * n_k;

    // EMA update for embed average
    float* ea  = embed_avg  + k * D;
    float* cb  = codebook   + k * D;
    float  cs  = cluster_size[k];

    for (int d = 0; d < D; d++) {
        ea[d] = gamma * ea[d] + (1.0f - gamma) * ea[d];  // embed_avg already holds scatter
        cb[d] = ea[d] / (cs + 1e-5f);
    }
}
