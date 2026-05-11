// vq_search.cu — Vector Quantization Nearest-Neighbour Search
// OpenSesame-Rust | Phase B
//
// Finds the nearest codebook vector for each input embedding.
// Used in SplitRVQ encode() for every codebook depth.
//
// Algorithm: ||q - e||² = ||q||² - 2·(q·eᵀ) + ||e||²
//   1. Precompute ||e||² for each codebook entry (once per forward pass)
//   2. Compute dot products q·eᵀ via tiled GEMM
//   3. Combine: dist[n,k] = qnorm[n] - 2*dot[n,k] + enorm[k]
//   4. argmin over k
//
// Input shapes:
//   queries:   [N, D]   N = batch*time, D = quant_dim (256 for Mimi)
//   codebook:  [K, D]   K = num_codes (2048 for Mimi)
//   enorms:    [K]      precomputed ||e_k||²
//   indices:   [N]      output: argmin_k dist[n,k]
//   distances: [N]      output: min distance

#include <stdint.h>
#include <float.h>

#define TILE 32

extern "C" __global__ void vq_search_kernel(
    const float* __restrict__ queries,    // [N, D]
    const float* __restrict__ codebook,   // [K, D]
    const float* __restrict__ enorms,     // [K]
    int32_t*     __restrict__ indices,    // [N]
    float*       __restrict__ distances,  // [N]
    int N, int K, int D
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* q = queries + n * D;

    // Compute ||q||²
    float qnorm = 0.0f;
    for (int d = 0; d < D; d++) qnorm += q[d] * q[d];

    float best_dist = FLT_MAX;
    int   best_idx  = 0;

    for (int k = 0; k < K; k++) {
        const float* e = codebook + k * D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += q[d] * e[d];
        float dist = qnorm - 2.0f * dot + enorms[k];
        if (dist < best_dist) {
            best_dist = dist;
            best_idx  = k;
        }
    }

    indices[n]   = best_idx;
    distances[n] = best_dist;
}

// Precompute ||e_k||² for all codebook entries
extern "C" __global__ void vq_compute_enorms_kernel(
    const float* __restrict__ codebook,  // [K, D]
    float*       __restrict__ enorms,    // [K]
    int K, int D
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const float* e = codebook + k * D;
    float norm = 0.0f;
    for (int d = 0; d < D; d++) norm += e[d] * e[d];
    enorms[k] = norm;
}
