//! DFlash 2 draft — a CPU REFERENCE of the block forward. GATES ONLY.
//!
//! Written from the published reference implementation (z-lab/dflash, MIT: `dflash/model.py`),
//! not from the GPU code, and deliberately plain: f64 norms, rotations and softmax, the CPU Q4
//! matmul, no fusion. It multiplies by the SAME Q4_K_S weights the GPU holds (the checkpoint
//! through the same deterministic transcode), so a mismatch is an implementation error and not
//! the ~5-8% quantisation error. `examples/dflash2_block -- ref` compares the two.

use super::dflash2::Dflash2Config;
use arf_core::model::weights::read_weights_lazy;
use arf_core::tensor::{matmul_nt_q4ks, Q4KSMatrix};
use std::path::Path;

/// A projection as the GPU holds it (Q4_K_S) or as PUBLISHED (bf16 -> f32). The second exists to
/// measure what the 4-bit transcode of the draft costs its OUTPUT, not just its weights.
pub enum RefMat {
    Q4(Q4KSMatrix),
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
}

pub struct RefLayer {
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    /// `[2][kernel][hidden]` = `[stage][tap][channel]`, as published.
    pub attention_conv_base: Vec<f32>,
    pub mlp_conv_base: Vec<f32>,
    pub attention_conv_dynamic: RefMat,
    pub mlp_conv_dynamic: RefMat,
    pub q_proj: RefMat,
    pub k_proj: RefMat,
    pub v_proj: RefMat,
    pub o_proj: RefMat,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate_proj: RefMat,
    pub up_proj: RefMat,
    pub down_proj: RefMat,
}

pub struct Dflash2BlockReference {
    pub cfg: Dflash2Config,
    pub layers: Vec<RefLayer>,
    pub norm: Vec<f32>,
}

fn mm(m: &RefMat, a: &[f32], rows: usize) -> Vec<f32> {
    let q = match m {
        RefMat::Q4(q) => q,
        RefMat::F32 {
            w,
            rows: n,
            cols: k,
        } => {
            let mut y = vec![0.0f32; rows * n];
            for r in 0..rows {
                let x = &a[r * k..][..*k];
                for o in 0..*n {
                    let wr = &w[o * k..][..*k];
                    y[r * n + o] = x.iter().zip(wr).map(|(p, q)| p * q).sum();
                }
            }
            return y;
        }
    };
    matmul_nt_q4ks(
        a,
        rows,
        q.cols(),
        q.codes(),
        q.scales(),
        q.mins(),
        q.d(),
        q.dmin(),
        q.rows(),
    )
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = w.len();
    let mut y = vec![0.0f32; x.len()];
    for (xr, yr) in x.chunks(n).zip(y.chunks_mut(n)) {
        let ss: f64 = xr.iter().map(|a| *a as f64 * *a as f64).sum();
        let inv = 1.0 / (ss / n as f64 + eps as f64).sqrt();
        for i in 0..n {
            yr[i] = (xr[i] as f64 * inv * w[i] as f64) as f32;
        }
    }
    y
}

impl Dflash2BlockReference {
    /// `published = false`: the Q4_K_S weights the GPU holds. `true`: the bf16 originals
    /// (~6 GB of f32 on the host).
    pub fn load(dir: &Path, cfg: &Dflash2Config, published: bool) -> Result<Self, String> {
        Self::load_mixed(dir, cfg, &|_| published)
    }

    /// Per-tensor choice: `published(name)` keeps that projection at bf16, the rest are Q4_K_S —
    /// for finding WHICH tensors the 4-bit transcode hurts.
    pub fn load_mixed(
        dir: &Path,
        cfg: &Dflash2Config,
        published: &dyn Fn(&str) -> bool,
    ) -> Result<Self, String> {
        let w = read_weights_lazy(&[dir.join("model.safetensors")]).map_err(|e| format!("{e}"))?;
        let (h, qd, kvd) = (
            cfg.hidden,
            cfg.heads * cfg.head_dim,
            cfg.kv_heads * cfg.head_dim,
        );
        let q = |name: &str, rows: usize, cols: usize| -> Result<RefMat, String> {
            let m = w
                .bf16_matrix(name, rows, cols)
                .map_err(|e| format!("{name}: {e}"))?;
            Ok(if published(name) {
                RefMat::F32 {
                    w: m.bits()
                        .iter()
                        .map(|b| f32::from_bits((*b as u32) << 16))
                        .collect(),
                    rows,
                    cols,
                }
            } else {
                RefMat::Q4(super::dflash2::quantize(m.bits(), rows, cols))
            })
        };
        let f = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            w.get_f32_shaped(name, shape)
                .map(|v| v.to_vec())
                .map_err(|e| format!("{name}: {e}"))
        };
        let mut layers = vec![];
        for l in 0..cfg.layers {
            let p = format!("layers.{l}");
            let base = [2, 2, h];
            layers.push(RefLayer {
                input_norm: f(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_attention_norm: f(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                attention_conv_base: f(&format!("{p}.attention_conv.base_kernel"), &base)?,
                mlp_conv_base: f(&format!("{p}.mlp_conv.base_kernel"), &base)?,
                attention_conv_dynamic: q(
                    &format!("{p}.attention_conv.kernel_projection.weight"),
                    cfg.dynamic_size(),
                    h,
                )?,
                mlp_conv_dynamic: q(
                    &format!("{p}.mlp_conv.kernel_projection.weight"),
                    cfg.dynamic_size(),
                    h,
                )?,
                q_proj: q(&format!("{p}.self_attn.q_proj.weight"), qd, h)?,
                k_proj: q(&format!("{p}.self_attn.k_proj.weight"), kvd, h)?,
                v_proj: q(&format!("{p}.self_attn.v_proj.weight"), kvd, h)?,
                o_proj: q(&format!("{p}.self_attn.o_proj.weight"), h, qd)?,
                q_norm: f(&format!("{p}.self_attn.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: f(&format!("{p}.self_attn.k_norm.weight"), &[cfg.head_dim])?,
                gate_proj: q(&format!("{p}.mlp.gate_proj.weight"), cfg.intermediate, h)?,
                up_proj: q(&format!("{p}.mlp.up_proj.weight"), cfg.intermediate, h)?,
                down_proj: q(&format!("{p}.mlp.down_proj.weight"), h, cfg.intermediate)?,
            });
        }
        Ok(Self {
            cfg: cfg.clone(),
            layers,
            norm: f("norm.weight", &[h])?,
        })
    }

    /// `_grouped_dynamic_convolve` of the reference, one stage: `dynamic` is this stage's
    /// `[rows][kernel][groups]`, `base` its `[kernel][hidden]`.
    fn conv(&self, x: &[f32], dynamic: &[f32], base: &[f32], stage: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (h, g, ks) = (c.hidden, c.hidden / c.conv_group_size, 2usize);
        let rows = x.len() / h;
        let mut y = vec![0.0f32; x.len()];
        for r in 0..rows {
            for off in 0..ks {
                if off > r {
                    continue; // F.pad: rows before the block are zeros
                }
                let v = &x[(r - off) * h..][..h];
                for ch in 0..h {
                    let kernel = base[(stage * ks + off) * h + ch];
                    let dynw =
                        dynamic[r * 2 * ks * g + (stage * ks + off) * g + ch / c.conv_group_size];
                    y[r * h + ch] += (kernel + dynw) * v[ch];
                }
            }
        }
        y
    }

    fn head_norm_rope(&self, x: &mut [f32], w: &[f32], heads: usize, start: usize) {
        let c = &self.cfg;
        let hd = c.head_dim;
        let rows = x.len() / (heads * hd);
        for r in 0..rows {
            for hh in 0..heads {
                let p = &mut x[(r * heads + hh) * hd..][..hd];
                let ss: f64 = p.iter().map(|a| *a as f64 * *a as f64).sum();
                let inv = 1.0 / (ss / hd as f64 + c.rms_eps as f64).sqrt();
                let n: Vec<f64> = (0..hd).map(|i| p[i] as f64 * inv * w[i] as f64).collect();
                for i in 0..hd / 2 {
                    let ang = (start + r) as f64
                        * (c.rope_theta as f64).powf(-2.0 * i as f64 / hd as f64);
                    let (s, co) = ang.sin_cos();
                    // rotate_half: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)
                    p[i] = (n[i] * co - n[i + hd / 2] * s) as f32;
                    p[i + hd / 2] = (n[i + hd / 2] * co + n[i] * s) as f32;
                }
            }
        }
    }

    /// One block: `embeds` `[rows, hidden]` (anchor + masks, the target's raw embeddings),
    /// `ring_k/v[l]` `[kv_heads, window, head_dim]` holding positions `[0, ctx_len)` at slot
    /// `p % window`. Returns the final NORMED hidden `[rows, hidden]` — the lm_head's input.
    pub fn forward(
        &self,
        embeds: &[f32],
        ring_k: &[Vec<f32>],
        ring_v: &[Vec<f32>],
        ctx_len: usize,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (h, hd, nh, nkv, win) = (c.hidden, c.head_dim, c.heads, c.kv_heads, c.sliding_window);
        let rows = embeds.len() / h;
        let mut x = embeds.to_vec();
        for (l, ly) in self.layers.iter().enumerate() {
            // ---- attention ----
            let n = rms(&x, &ly.input_norm, c.rms_eps);
            let dynw = mm(&ly.attention_conv_dynamic, &n, rows);
            let cv = self.conv(&n, &dynw, &ly.attention_conv_base, 0);
            let mut q = mm(&ly.q_proj, &cv, rows);
            let mut k = mm(&ly.k_proj, &cv, rows);
            let v = mm(&ly.v_proj, &cv, rows);
            self.head_norm_rope(&mut q, &ly.q_norm, nh, ctx_len);
            self.head_norm_rope(&mut k, &ly.k_norm, nkv, ctx_len);
            let mut att = vec![0.0f32; rows * nh * hd];
            for r in 0..rows {
                let p = ctx_len + r;
                let lo = (p + 1).saturating_sub(win);
                for hh in 0..nh {
                    let kvh = hh / (nh / nkv);
                    let qp = &q[(r * nh + hh) * hd..][..hd];
                    let mut keys: Vec<(&[f32], &[f32])> = vec![];
                    for pos in lo..ctx_len {
                        let o = (kvh * win + pos % win) * hd;
                        keys.push((&ring_k[l][o..o + hd], &ring_v[l][o..o + hd]));
                    }
                    for j in 0..rows {
                        let o = (j * nkv + kvh) * hd;
                        keys.push((&k[o..o + hd], &v[o..o + hd]));
                    }
                    let sc: Vec<f64> = keys
                        .iter()
                        .map(|(kk, _)| {
                            qp.iter()
                                .zip(*kk)
                                .map(|(a, b)| *a as f64 * *b as f64)
                                .sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    let out = &mut att[(r * nh + hh) * hd..][..hd];
                    for (wgt, (_, vv)) in e.iter().zip(&keys) {
                        for i in 0..hd {
                            out[i] += (wgt / z * vv[i] as f64) as f32;
                        }
                    }
                }
            }
            let o = mm(&ly.o_proj, &att, rows);
            let fin = self.conv(&o, &dynw, &ly.attention_conv_base, 1);
            let r1: Vec<f32> = x.iter().zip(&fin).map(|(a, b)| a + b).collect();
            // ---- MLP ----
            let n2 = rms(&r1, &ly.post_attention_norm, c.rms_eps);
            let dyn2 = mm(&ly.mlp_conv_dynamic, &n2, rows);
            let c2 = self.conv(&n2, &dyn2, &ly.mlp_conv_base, 0);
            let (gt, up) = (mm(&ly.gate_proj, &c2, rows), mm(&ly.up_proj, &c2, rows));
            let act: Vec<f32> = gt
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let dn = mm(&ly.down_proj, &act, rows);
            let fin2 = self.conv(&dn, &dyn2, &ly.mlp_conv_base, 1);
            x = r1.iter().zip(&fin2).map(|(a, b)| a + b).collect();
        }
        rms(&x, &self.norm, c.rms_eps)
    }
}
