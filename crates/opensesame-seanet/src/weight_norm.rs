//! Weight normalization reparameterization: W = g * v / ||v||
//! Stores g (magnitude) and v (direction) separately.
//! At inference, weight() materializes the effective weight matrix.

/// Hash a (seed, index) pair to a float in [0, 1).
fn hash_float(seed: u64, idx: usize) -> f32 {
    let mut h = seed.wrapping_add(idx as u64);
    h = h.wrapping_mul(0x9e3779b97f4a7c15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h >> 11) as f32 / (1u64 << 53) as f32
}

/// Weight normalization reparameterization: W = g · (v / ‖v‖).
///
/// Stores g ∈ ℝ^{out_ch} (magnitude per output filter) and
/// v ∈ ℝ^{out_ch × in_ch × kernel} (direction).
/// The effective weight is recomputed on each call to `weight()`.
pub struct WeightNorm {
    /// Per-output-filter magnitude scalars, shape [out_ch].
    pub g: Vec<f32>,
    /// Direction vectors, shape [out_ch * in_ch * kernel].
    pub v: Vec<f32>,
    /// Number of output channels (first dim of weight tensor).
    pub out_ch: usize,
    /// Number of input channels (second dim of weight tensor).
    pub in_ch: usize,
    /// Kernel size (third dim of weight tensor).
    pub kernel: usize,
}

impl WeightNorm {
    /// Create a new WeightNorm with Kaiming-uniform initialised v and g = ‖v_i‖.
    ///
    /// fan_in = in_ch * kernel; bound = 1/sqrt(fan_in).
    pub fn new(out_ch: usize, in_ch: usize, kernel: usize) -> Self {
        let fan_in = in_ch * kernel;
        let bound = if fan_in == 0 {
            1.0
        } else {
            1.0 / (fan_in as f32).sqrt()
        };
        let total = out_ch * in_ch * kernel;
        // Unique seed per (out_ch, in_ch, kernel) triple for reproducibility.
        let seed: u64 = (out_ch as u64)
            .wrapping_mul(1000003)
            .wrapping_add(in_ch as u64)
            .wrapping_mul(1000033)
            .wrapping_add(kernel as u64);

        let v: Vec<f32> = (0..total)
            .map(|i| hash_float(seed, i) * 2.0 * bound - bound)
            .collect();

        let row_size = in_ch * kernel;
        let g: Vec<f32> = (0..out_ch)
            .map(|i| {
                let row = &v[i * row_size..(i + 1) * row_size];
                row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
            })
            .collect();

        WeightNorm { g, v, out_ch, in_ch, kernel }
    }

    /// Materialise W = g_i * v_i / ‖v_i‖ for each output filter i.
    ///
    /// Returns a flat [out_ch * in_ch * kernel] vector.
    pub fn weight(&self) -> Vec<f32> {
        let row_size = self.in_ch * self.kernel;
        let mut w = vec![0.0f32; self.out_ch * row_size];
        for i in 0..self.out_ch {
            let row = &self.v[i * row_size..(i + 1) * row_size];
            let norm_v = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let scale = self.g[i] / norm_v;
            for j in 0..row_size {
                w[i * row_size + j] = scale * row[j];
            }
        }
        w
    }

    /// Initialise from an existing weight matrix (e.g., for loading pretrained weights).
    ///
    /// Sets v = w, g[i] = ‖w_i‖. Then weight() == w (within floating-point precision).
    pub fn from_weight(w: &[f32], out_ch: usize, in_ch: usize, kernel: usize) -> Self {
        let row_size = in_ch * kernel;
        assert_eq!(w.len(), out_ch * row_size, "weight length mismatch");
        let g: Vec<f32> = (0..out_ch)
            .map(|i| {
                let row = &w[i * row_size..(i + 1) * row_size];
                row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
            })
            .collect();
        WeightNorm { g, v: w.to_vec(), out_ch, in_ch, kernel }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_norm_magnitude() {
        // ‖w_i‖ should equal g_i for each output filter.
        let wn = WeightNorm::new(8, 4, 3);
        let w = wn.weight();
        let row_size = wn.in_ch * wn.kernel;
        for i in 0..wn.out_ch {
            let row = &w[i * row_size..(i + 1) * row_size];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let diff = (norm - wn.g[i]).abs();
            assert!(diff < 1e-5, "filter {}: norm={} g={} diff={}", i, norm, wn.g[i], diff);
        }
    }

    #[test]
    fn test_weight_norm_from_weight() {
        // WeightNorm::from_weight should reconstruct the original weight within 1e-5.
        let out_ch = 4;
        let in_ch = 2;
        let kernel = 3;
        let w: Vec<f32> = (0..(out_ch * in_ch * kernel))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let wn = WeightNorm::from_weight(&w, out_ch, in_ch, kernel);
        let w_reconstructed = wn.weight();
        for (a, b) in w.iter().zip(w_reconstructed.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "from_weight mismatch: original={} reconstructed={}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_weight_norm_nonzero_grad_g() {
        // Manually updating g changes the weight.
        let mut wn = WeightNorm::new(4, 2, 3);
        let w_before = wn.weight();
        // Double g[0] — should double the first output filter in w.
        wn.g[0] *= 2.0;
        let w_after = wn.weight();
        let row_size = wn.in_ch * wn.kernel;
        // First filter should be doubled.
        for j in 0..row_size {
            let ratio = if w_before[j].abs() > 1e-9 {
                w_after[j] / w_before[j]
            } else {
                1.0
            };
            assert!((ratio - 2.0).abs() < 1e-4, "expected 2x, got {}", ratio);
        }
        // Other filters unchanged.
        for j in row_size..w_before.len() {
            assert!((w_after[j] - w_before[j]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weight_norm_weight_returns_vec() {
        let wn = WeightNorm::new(3, 5, 7);
        let w = wn.weight();
        assert_eq!(w.len(), 3 * 5 * 7);
    }

    #[test]
    fn test_weight_norm_scale_invariant() {
        // Scaling only v (not g) should not change the effective weight:
        // W = g * (5v) / ||5v|| = g * 5v / (5||v||) = g * v / ||v|| = W.
        let wn = WeightNorm::new(4, 3, 3);
        let w_original = wn.weight();
        let mut wn2 = WeightNorm::from_weight(&w_original, 4, 3, 3);
        // Scale v for filter 0 only — g stays the same, so effective weight is unchanged.
        let row_size = wn2.in_ch * wn2.kernel;
        for j in 0..row_size {
            wn2.v[j] *= 5.0;
        }
        // g[0] stays the same — effective weight[0] = g * (5v)/(5||v||) = g * v/||v||
        let w_scaled = wn2.weight();
        // First filter should be unchanged (only v scaled, g fixed).
        for j in 0..row_size {
            assert!(
                (w_original[j] - w_scaled[j]).abs() < 1e-4,
                "scale invariance failed at j={}: {} vs {}",
                j, w_original[j], w_scaled[j]
            );
        }
    }
}
