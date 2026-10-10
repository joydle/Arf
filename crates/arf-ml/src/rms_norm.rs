//! Root-mean-square layer normalization (Llama-style: no mean subtraction, no
//! bias). `y = x / sqrt(mean(x²) + eps) * weight`, per row.

use crate::Tensor;

/// A single RMSNorm layer.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    /// Build from a learned `weight` vector of shape `[hidden]`.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        RmsNorm {
            weight: weight.into_vec(),
            eps: eps as f32,
        }
    }

    /// Normalize each row of `x` (`[tokens, hidden]`).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (tokens, hidden) = x.dims2();
        assert_eq!(hidden, self.weight.len(), "RMSNorm hidden mismatch");
        let mut out = vec![0.0f32; tokens * hidden];
        for t in 0..tokens {
            let row = x.row(t);
            let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            let dst = &mut out[t * hidden..(t + 1) * hidden];
            for ((d, &v), &w) in dst.iter_mut().zip(row).zip(&self.weight) {
                *d = v * inv_rms * w;
            }
        }
        Tensor::from_vec(out, vec![tokens, hidden])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_formula() {
        let norm = RmsNorm::new(Tensor::ones(vec![2]), 1e-6);
        let x = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2]);
        let y = norm.forward(&x);
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((y.as_slice()[0] - 3.0 / rms).abs() < 1e-5);
        assert!((y.as_slice()[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn weight_scales_output() {
        let norm = RmsNorm::new(Tensor::from_vec(vec![2.0, 0.5], vec![2]), 1e-6);
        let x = Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]);
        let y = norm.forward(&x);
        assert!((y.as_slice()[0] - 2.0).abs() < 1e-5);
        assert!((y.as_slice()[1] - 0.5).abs() < 1e-5);
    }
}
