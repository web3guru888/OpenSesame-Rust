//! Temporal down/up-sampling convolutions for Mimi.

/// 1-D strided convolution for temporal downsampling (25 fps → 12.5 fps).
///
/// Standard conv (groups=1), kernel = 2 × stride, causal left-padding.
#[derive(Debug, Clone)]
pub struct ConvDownsample1d {
    /// Weights `[C_out, C_in, k]` flattened.
    pub weight: Vec<f32>,
    /// Bias `[C_out]`. Empty = no bias.
    pub bias: Vec<f32>,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Stride (default 2).
    pub stride: usize,
    /// Kernel size (default 4).
    pub kernel_size: usize,
}

impl ConvDownsample1d {
    /// Construct zero-init [`ConvDownsample1d`].
    pub fn new(channels: usize, stride: usize) -> Self {
        let kernel_size = 2 * stride;
        Self {
            weight:       vec![0.0_f32; channels * channels * kernel_size],
            bias:         vec![0.0_f32; channels],
            in_channels:  channels,
            out_channels: channels,
            stride,
            kernel_size,
        }
    }

    /// Forward: causal strided Conv1d. Input `x` is `[T, C_in]`, returns `([T_out, C_out], T_out)`.
    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let cin  = self.in_channels;
        let cout = self.out_channels;
        let k    = self.kernel_size;
        let s    = self.stride;
        let pad  = k - 1;
        let padded_t = t + pad;
        let t_out    = (padded_t - k) / s + 1;
        let mut y = vec![0.0_f32; t_out * cout];
        for ot in 0..t_out {
            let offset = ot * s;
            for oc in 0..cout {
                let mut acc = if self.bias.is_empty() { 0.0 } else { self.bias[oc] };
                for ic in 0..cin {
                    for ki in 0..k {
                        let padded_pos = offset + ki;
                        let x_val = if padded_pos < pad {
                            0.0
                        } else {
                            let tp = padded_pos - pad;
                            if tp < t { x[tp * cin + ic] } else { 0.0 }
                        };
                        acc += self.weight[oc * cin * k + ic * k + ki] * x_val;
                    }
                }
                y[ot * cout + oc] = acc;
            }
        }
        (y, t_out)
    }
}

/// 1-D transposed depthwise convolution for temporal upsampling (12.5 fps → 25 fps).
///
/// Depthwise (groups = channels), kernel = 2 × stride, causal trim.
#[derive(Debug, Clone)]
pub struct ConvTrUpsample1d {
    /// Weights `[C, 1, k]` flattened (depthwise).
    pub weight: Vec<f32>,
    /// Bias `[C]`. Empty = no bias.
    pub bias: Vec<f32>,
    /// Number of channels.
    pub channels: usize,
    /// Stride (default 2).
    pub stride: usize,
    /// Kernel size (default 4).
    pub kernel_size: usize,
}

impl ConvTrUpsample1d {
    /// Construct zero-init [`ConvTrUpsample1d`].
    pub fn new(channels: usize, stride: usize) -> Self {
        let kernel_size = 2 * stride;
        Self {
            weight:      vec![0.0_f32; channels * kernel_size],
            bias:        vec![0.0_f32; channels],
            channels,
            stride,
            kernel_size,
        }
    }

    /// Forward: depthwise ConvTranspose1d with causal trim.
    /// Input `x` is `[T_in, C]`, returns `([T_out, C], T_out)` where `T_out = T_in * stride`.
    pub fn forward(&self, x: &[f32], t_in: usize) -> (Vec<f32>, usize) {
        let c     = self.channels;
        let k     = self.kernel_size;
        let s     = self.stride;
        let trim  = k - s;
        let raw_t = t_in * s + (k - s);
        let mut raw = vec![0.0_f32; raw_t * c];
        if !self.bias.is_empty() {
            for ot in 0..raw_t {
                for ch in 0..c { raw[ot * c + ch] = self.bias[ch]; }
            }
        }
        for ti in 0..t_in {
            for ch in 0..c {
                let x_val = x[ti * c + ch];
                for ki in 0..k {
                    let ot = ti * s + ki;
                    raw[ot * c + ch] += self.weight[ch * k + ki] * x_val;
                }
            }
        }
        let t_out = t_in * s;
        let mut y = vec![0.0_f32; t_out * c];
        for ot in 0..t_out {
            for ch in 0..c { y[ot * c + ch] = raw[(ot + trim) * c + ch]; }
        }
        (y, t_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_shape() {
        let ds = ConvDownsample1d::new(4, 2);
        let x = vec![0.0_f32; 8 * 4];
        let (y, t_out) = ds.forward(&x, 8);
        assert_eq!(t_out, 4);
        assert_eq!(y.len(), 4 * 4);
    }

    #[test]
    fn test_downsample_1frame() {
        let ds = ConvDownsample1d::new(2, 2);
        let x = vec![1.0_f32; 2 * 2];
        let (_, t_out) = ds.forward(&x, 2);
        assert_eq!(t_out, 1);
    }

    #[test]
    fn test_upsample_shape() {
        let us = ConvTrUpsample1d::new(4, 2);
        let x = vec![0.0_f32; 4 * 4];
        let (y, t_out) = us.forward(&x, 4);
        assert_eq!(t_out, 8);
        assert_eq!(y.len(), 8 * 4);
    }

    #[test]
    fn test_roundtrip_shape_960frames() {
        let ds = ConvDownsample1d::new(512, 2);
        let us = ConvTrUpsample1d::new(512, 2);
        let x = vec![0.0_f32; 960 * 512];
        let (z, t_z) = ds.forward(&x, 960);
        let (y, t_y) = us.forward(&z, t_z);
        assert_eq!(t_z, 480);
        assert_eq!(t_y, 960);
    }

    #[test]
    fn test_downsample_bias_only() {
        let mut ds = ConvDownsample1d::new(2, 2);
        ds.bias = vec![3.0, 7.0];
        let x = vec![1.0_f32; 4 * 2];
        let (y, t_out) = ds.forward(&x, 4);
        assert_eq!(t_out, 2);
        assert!((y[0] - 3.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_upsample_stride1() {
        let us = ConvTrUpsample1d::new(2, 1);
        let x = vec![0.0_f32; 4 * 2];
        let (y, t_out) = us.forward(&x, 4);
        assert_eq!(t_out, 4);
        assert_eq!(y.len(), 4 * 2);
    }
}
