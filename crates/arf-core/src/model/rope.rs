//! Rotary position embeddings (RoPE), HuggingFace-Llama layout, with optional
//! Llama 3 frequency scaling.
//!
//! We precompute per-position `cos`/`sin` for the `head_dim/2` frequency pairs,
//! then rotate each `(x[i], x[i+half])` pair. Tables are indexed by absolute
//! position so ragged/continuous batches work.

use crate::config::{ModelConfig, RopeScaling};
use crate::tensor::Tensor;

/// Precomputed rotary tables of shape `[max_positions, head_dim/2]`.
#[derive(Debug, Clone)]
pub struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
    half: usize,
}

impl Rope {
    /// Build tables for `cfg` covering `max_positions` positions, using the
    /// config's `rope_theta` and any configured `rope_scaling`.
    pub fn new(cfg: &ModelConfig, max_positions: usize) -> Self {
        Self::from_inv_freq(compute_inv_freq(cfg), cfg.head_dim, max_positions)
    }

    /// Build tables with an explicit `theta`, ignoring `rope_scaling`. Gemma uses
    /// this for its two RoPE bases: a small `rope_local_base_freq` on the
    /// sliding-window (local) layers and a large `rope_theta` on the global ones.
    pub fn with_theta(cfg: &ModelConfig, max_positions: usize, theta: f64) -> Self {
        Self::from_inv_freq(
            inv_freq_for(cfg.head_dim, theta),
            cfg.head_dim,
            max_positions,
        )
    }

    /// Build tables for a GLOBAL (full-attention) layer with an explicit `theta`,
    /// an explicit `head_dim`, and PARTIAL rotary: only the first `rotary_dim` of
    /// each head's dims are rotated. The frequencies are computed over `rotary_dim`
    /// (HF: `inv_freq[i] = theta^(-2i/rotary_dim)`, `i < rotary_dim/2`), so the
    /// table row stride is `rotary_dim/2`. `head_dim` is stored for the `apply`
    /// assert; the rotated pairs are `(i, i + rotary_dim/2)` within each head.
    /// Gemma 4 global layers use this (head_dim 512, rotary_dim 128, θ 1M); full
    /// rotation (`rotary_dim == head_dim`) reduces to [`Self::with_theta`].
    pub fn with_theta_rotary(
        max_positions: usize,
        theta: f64,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Self {
        Self::from_inv_freq(inv_freq_for(rotary_dim, theta), head_dim, max_positions)
    }

    /// Precompute the `[max_positions, half]` cos/sin tables from per-pair inverse
    /// frequencies (`half = inv_freq.len()`, normally `head_dim/2`, but
    /// `rotary_dim/2` for a partial-rotary layer).
    fn from_inv_freq(inv_freq: Vec<f64>, head_dim: usize, max_positions: usize) -> Self {
        let half = inv_freq.len();
        let mut cos = vec![0.0f32; max_positions * half];
        let mut sin = vec![0.0f32; max_positions * half];
        for pos in 0..max_positions {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let (s, c) = (pos as f64 * freq).sin_cos();
                cos[pos * half + i] = c as f32;
                sin[pos * half + i] = s as f32;
            }
        }
        Rope {
            cos,
            sin,
            head_dim,
            half,
        }
    }

    /// Precomputed cosine table, `[max_positions, head_dim/2]` row-major (for
    /// uploading to the GPU RoPE kernel).
    pub fn cos_table(&self) -> &[f32] {
        &self.cos
    }

    /// Precomputed sine table, `[max_positions, head_dim/2]` row-major.
    pub fn sin_table(&self) -> &[f32] {
        &self.sin
    }

    /// Apply RoPE to `x` of shape `[tokens, heads, head_dim]` using absolute
    /// `positions` (`[tokens]`). Returns a new tensor of the same shape.
    pub fn apply(&self, x: &Tensor, positions: &[u32]) -> Tensor {
        let (tokens, heads, hd) = x.dims3();
        assert_eq!(hd, self.head_dim, "RoPE head_dim mismatch");
        assert_eq!(positions.len(), tokens, "one position per token");
        let half = self.half;
        let src = x.as_slice();
        // Start from a COPY so a PARTIAL-rotary head (rotary_dim < head_dim, i.e.
        // `2*half < hd`, Gemma-4 global layers) passes its non-rotated tail through
        // unchanged instead of zeroing it. For full rotary (`2*half == hd`) every
        // element is overwritten below, so the copy is harmless.
        let mut out = src.to_vec();
        let rot = 2 * half; // count of rotated dims = rotary_dim
        for (t, &pos) in positions.iter().enumerate() {
            let p = pos as usize;
            let cos = &self.cos[p * half..(p + 1) * half];
            let sin = &self.sin[p * half..(p + 1) * half];
            for h in 0..heads {
                let base = (t * heads + h) * hd;
                for i in 0..half {
                    let (c, s) = (cos[i], sin[i]);
                    let a = src[base + i];
                    let b = src[base + i + half];
                    out[base + i] = a * c - b * s;
                    out[base + i + half] = b * c + a * s;
                }
                // dims [rot..hd] are the non-rotated tail — already copied from src.
                debug_assert!(rot <= hd, "rotary_dim must not exceed head_dim");
            }
        }
        Tensor::from_vec(out, vec![tokens, heads, hd])
    }
}

/// Plain inverse frequencies `theta^(-2i/d)` for a given base, no scaling.
fn inv_freq_for(head_dim: usize, theta: f64) -> Vec<f64> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect()
}

/// Base inverse frequencies with Llama 3 scaling applied (if configured).
fn compute_inv_freq(cfg: &ModelConfig) -> Vec<f64> {
    let base = inv_freq_for(cfg.head_dim, cfg.rope_theta);

    match &cfg.rope_scaling {
        RopeScaling::None => base,
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } => {
            let orig = *original_max_position as f64;
            let low_wavelen = orig / low_freq_factor;
            let high_wavelen = orig / high_freq_factor;
            base.into_iter()
                .map(|freq| {
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen > low_wavelen {
                        freq / factor
                    } else if wavelen < high_wavelen {
                        freq
                    } else {
                        let smooth = (orig / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * freq / factor + smooth * freq
                    }
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeScaling;

    fn tiny_cfg() -> ModelConfig {
        let mut cfg = ModelConfig::llama_3_2_1b();
        cfg.head_dim = 4;
        cfg.rope_theta = 10000.0;
        cfg.rope_scaling = RopeScaling::None;
        cfg
    }

    #[test]
    fn position_zero_is_identity() {
        let rope = Rope::new(&tiny_cfg(), 8);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let y = rope.apply(&x, &[0]);
        for (a, b) in y.as_slice().iter().zip(&[1.0, 2.0, 3.0, 4.0]) {
            assert!((a - b).abs() < 1e-5, "position 0 must be identity");
        }
    }

    #[test]
    fn partial_rotary_passes_tail_through() {
        // Gemma-4 global layer shape: head_dim 8, rotary_dim 4 → only dims 0..4 rotate,
        // dims 4..8 must pass through UNCHANGED (the bug was zeroing them).
        let rope = Rope::with_theta_rotary(16, 1_000_000.0, 8, 4);
        let head = vec![0.5, -1.0, 2.0, 0.25, 9.0, -8.0, 7.0, -6.0];
        let x = Tensor::from_vec(head.clone(), vec![1, 1, 8]);
        let y = rope.apply(&x, &[5]); // non-zero pos → rotation is active
        let out = y.as_slice();
        // Tail (dims 4..8) unchanged.
        assert_eq!(
            &out[4..8],
            &head[4..8],
            "non-rotated tail must pass through"
        );
        // Rotated head (dims 0..4) changed, but norm of the rotated pair preserved.
        let in_norm: f32 = head[0..4].iter().map(|v| v * v).sum();
        let out_norm: f32 = out[0..4].iter().map(|v| v * v).sum();
        assert!(
            (in_norm - out_norm).abs() < 1e-4,
            "rotated block preserves norm"
        );
        assert_ne!(
            &out[0..4],
            &head[0..4],
            "rotated dims should change at pos 5"
        );
    }

    #[test]
    fn rotation_preserves_norm() {
        let rope = Rope::new(&tiny_cfg(), 64);
        let x = Tensor::from_vec(vec![0.3, -1.2, 0.7, 2.1], vec![1, 1, 4]);
        let n_in: f32 = x.as_slice().iter().map(|v| v * v).sum();
        for p in [1u32, 5, 37] {
            let y = rope.apply(&x, &[p]);
            let n_out: f32 = y.as_slice().iter().map(|v| v * v).sum();
            assert!((n_in - n_out).abs() < 1e-4, "RoPE is a rotation");
        }
    }
}
