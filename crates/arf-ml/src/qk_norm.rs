//! Per-head RMSNorm on the query/key projections, applied *before* RoPE.
//!
//! Qwen3 (and Gemma 3) normalize each attention head's `head_dim`-vector with a
//! learned RMSNorm — `q_norm.weight` / `k_norm.weight`, each `[head_dim]` — right
//! after the q/k projection and split into heads, before the rotary embedding.
//! Llama has no such norm. This is gated by `ModelConfig::qk_norm`.
//!
//! The op is the standard RMSNorm (`x / sqrt(mean(x²)+eps) · w`) but the row is
//! one head's `head_dim` slice, not the full hidden row — so a `[total, n_heads,
//! head_dim]` buffer holds `total * n_heads` independent rows. The CPU path here is
//! the bit-exact oracle for the GPU `qk_norm` kernel.

/// A per-head RMSNorm: one learned weight vector of length `head_dim`.
#[derive(Debug, Clone)]
pub struct QkNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl QkNorm {
    /// Build from a `[head_dim]` weight vector and the model's `rms_norm_eps`.
    pub fn new(weight: Vec<f32>, eps: f64) -> Self {
        QkNorm {
            weight,
            eps: eps as f32,
        }
    }

    /// The head dimension this norm operates over (= `weight.len()`).
    pub fn head_dim(&self) -> usize {
        self.weight.len()
    }

    /// Normalize each `head_dim`-row of a flat `[total * n_heads, head_dim]` buffer
    /// in place. `data.len()` must be a multiple of `head_dim`.
    pub fn apply(&self, data: &mut [f32]) {
        let hd = self.weight.len();
        debug_assert_eq!(
            data.len() % hd,
            0,
            "QkNorm buffer not a multiple of head_dim"
        );
        for row in data.chunks_exact_mut(hd) {
            let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / hd as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            for (v, &w) in row.iter_mut().zip(&self.weight) {
                *v = *v * inv_rms * w;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_each_head_independently() {
        // head_dim=2, two heads worth of data in one flat buffer.
        let norm = QkNorm::new(vec![1.0, 1.0], 0.0);
        let mut data = vec![3.0, 4.0, /* head 0 */ 6.0, 8.0 /* head 1 */];
        norm.apply(&mut data);
        // head 0: rms = sqrt((9+16)/2) = sqrt(12.5); 3/rms, 4/rms
        let rms0 = (12.5f32).sqrt();
        assert!((data[0] - 3.0 / rms0).abs() < 1e-5);
        assert!((data[1] - 4.0 / rms0).abs() < 1e-5);
        // head 1: rms = sqrt((36+64)/2) = sqrt(50); 6/rms, 8/rms
        let rms1 = (50.0f32).sqrt();
        assert!((data[2] - 6.0 / rms1).abs() < 1e-5);
        assert!((data[3] - 8.0 / rms1).abs() < 1e-5);
    }

    #[test]
    fn weight_scales_per_lane() {
        let norm = QkNorm::new(vec![2.0, 0.5], 0.0);
        let mut data = vec![1.0, 1.0];
        norm.apply(&mut data);
        // rms = 1, so output = weight.
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn eps_guards_zero_row() {
        let norm = QkNorm::new(vec![1.0, 1.0], 1e-6);
        let mut data = vec![0.0, 0.0];
        norm.apply(&mut data); // must not NaN
        assert_eq!(data, vec![0.0, 0.0]);
    }
}
