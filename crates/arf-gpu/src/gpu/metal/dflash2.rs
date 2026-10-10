//! DFlash 2 — the block-diffusion DRAFT model for speculative decoding. **Stage 1: the loader.**
//!
//! The plan, the checkpoint's tensor table, the computation and the gated stages are in
//! In one paragraph: single-stream decode is pinned
//! at the bandwidth roofline (one token per ~48 ms pass over the 27B), an 8-row verify pass
//! costs ~124 ms, so everything rests on how many of 7 drafted tokens are ACCEPTED. The in-model
//! MTP head lands ~1 (deeper chains and a KV catch-up were both built and measured out,
//! 2026-09-20); prompt-lookup lands ~6 but only where the answer copies the conversation. This
//! draft — `incoai/Qwen3.8-27B-DFlash2`, Apache-2.0, 81 tensors, 3.58 GB bf16 — lands ~4.75 on
//! free-form text at 4-bit (measured here, in Splash, whose Apache-2.0 implementation this port
//! was read from and which `NOTICE` credits).
//!
//! It has NO embedding and NO lm_head: it borrows the target's. What it owns is `fc` (five tapped
//! target hiddens -> its width), five layers — each an attention and an MLP wrapped in a two-tap
//! DYNAMIC convolution over the block's rows — a final norm, and a candidate selector.
//!
//! Every projection is transcoded bf16 -> Q4_K_S, the format every trunk matmul uses, so the
//! block forward (stage 4) runs on the MPP matmuls that already exist. All the shapes are whole
//! 256-row tiles and whole 256-wide super-blocks, which the MPP path requires — asserted at load.
//! Norms and conv base kernels are f32; the two 248320x256 selector codebooks stay bf16 bits
//! (127 MB each) until stage 5 decides how they are read.

use super::island::{MtlBuf, Q4ksMtl};
use crate::gpu::GpuContext;
use arf_core::model::weights::{read_weights_lazy, Weights};
use arf_core::tensor::Q4KSMatrix;
use objc2_metal::{MTLBuffer as _, MTLDevice as _};
use std::path::Path;
use wgpu::hal::api::Metal;

/// `config.json`, the fields the computation needs. Defaults are the published checkpoint's.
#[derive(Debug, Clone, PartialEq)]
pub struct Dflash2Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    /// Target layers whose output hidden is tapped, in concatenation order.
    pub target_layer_ids: Vec<usize>,
    /// Positions per drafted block: the anchor plus `block_size - 1` proposals.
    pub block_size: usize,
    pub mask_token_id: u32,
    /// The first special token id — `eos_token_id` (Qwen's control tokens are one block at the
    /// top of the vocabulary). The adaptive draft candidate range ignores ids at or above it.
    /// Falls back to the mask token when the config has no eos.
    pub special_floor: u32,
    /// Channels that share one dynamic conv weight.
    pub conv_group_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl Dflash2Config {
    /// Width of the per-row dynamic conv weights: 2 stages x 2 taps x (hidden / group).
    pub fn dynamic_size(&self) -> usize {
        4 * self.hidden / self.conv_group_size
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json: {e}"))?;
        let u = |o: &serde_json::Value, k: &str| -> Result<usize, String> {
            o.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json: missing or non-integer `{k}`"))
        };
        let d = v
            .get("dflash_config")
            .ok_or("config.json: no `dflash_config`")?;
        let cfg = Self {
            hidden: u(&v, "hidden_size")?,
            layers: u(&v, "num_hidden_layers")?,
            heads: u(&v, "num_attention_heads")?,
            kv_heads: u(&v, "num_key_value_heads")?,
            head_dim: u(&v, "head_dim")?,
            intermediate: u(&v, "intermediate_size")?,
            vocab: u(&v, "vocab_size")?,
            target_layer_ids: d
                .get("target_layer_ids")
                .and_then(|x| x.as_array())
                .ok_or("config.json: no `target_layer_ids`")?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|x| x as usize)
                        .ok_or("target_layer_ids: not an integer")
                })
                .collect::<Result<_, _>>()?,
            block_size: u(d, "block_size")?,
            mask_token_id: u(d, "mask_token_id")? as u32,
            special_floor: v
                .get("eos_token_id")
                .and_then(|x| x.as_u64())
                .or_else(|| d.get("mask_token_id").and_then(|x| x.as_u64()))
                .unwrap_or(u64::MAX) as u32,
            conv_group_size: u(d, "conv_group_size")?,
            selector_rank: u(d, "selector_rank")?,
            selector_top_k: u(d, "selector_top_k")?,
            sliding_window: u(&v, "sliding_window")?,
            rope_theta: v
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .ok_or("config.json: no `rope_parameters.rope_theta`")?
                as f32,
            rms_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .ok_or("config.json: no `rms_norm_eps`")? as f32,
        };
        if u(d, "conv_kernel_size")? != 2 {
            return Err("only the two-tap conv (conv_kernel_size = 2) is implemented".into());
        }
        if !cfg.hidden.is_multiple_of(cfg.conv_group_size) {
            return Err("hidden_size is not a multiple of conv_group_size".into());
        }
        Ok(cfg)
    }
}

/// One of the draft's five layers.
pub struct Dflash2Layer {
    pub input_norm: MtlBuf,
    pub post_attention_norm: MtlBuf,
    /// `[2 stages][2 taps][hidden]` f32 — stage 0 = prepare, 1 = finish; tap 0 = this row, 1 = previous.
    pub attention_conv_base: MtlBuf,
    pub mlp_conv_base: MtlBuf,
    /// `[dynamic_size, hidden]` — the per-row dynamic conv weights.
    pub attention_conv_dynamic: Q4ksMtl,
    pub mlp_conv_dynamic: Q4ksMtl,
    pub q_proj: Q4ksMtl,
    pub k_proj: Q4ksMtl,
    pub v_proj: Q4ksMtl,
    pub o_proj: Q4ksMtl,
    pub q_norm: MtlBuf,
    pub k_norm: MtlBuf,
    pub gate_proj: Q4ksMtl,
    pub up_proj: Q4ksMtl,
    pub down_proj: Q4ksMtl,
}

pub struct Dflash2Draft {
    pub cfg: Dflash2Config,
    /// `[hidden, taps * hidden]` — the context projection over the concatenated target hiddens.
    pub fc: Q4ksMtl,
    pub hidden_norm: MtlBuf,
    pub layers: Vec<Dflash2Layer>,
    /// Final RMSNorm, applied before the TARGET's lm_head.
    pub norm: MtlBuf,
    pub selector_hidden: Q4ksMtl,
    /// `[vocab, selector_rank]`, raw bf16 bits.
    pub predecessor_codebook: MtlBuf,
    pub successor_codebook: MtlBuf,
    /// Worst relative RMS error of the bf16 -> Q4_K_S transcode over every projection, and which.
    pub worst_quant_error: (f64, String),
    gpu_bytes: usize,
}

impl Dflash2Draft {
    pub fn gpu_bytes(&self) -> usize {
        self.gpu_bytes
    }

    /// Every buffer this draft holds, summed from the buffers themselves.
    fn count_bytes(&self) -> usize {
        let plain = |b: &MtlBuf| b.0.length();
        let mut n = self.fc.bytes()
            + self.selector_hidden.bytes()
            + plain(&self.hidden_norm)
            + plain(&self.norm)
            + plain(&self.predecessor_codebook)
            + plain(&self.successor_codebook);
        for l in &self.layers {
            n += [
                &l.attention_conv_dynamic,
                &l.mlp_conv_dynamic,
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.gate_proj,
                &l.up_proj,
                &l.down_proj,
            ]
            .iter()
            .map(|w| w.bytes())
            .sum::<usize>();
            n += [
                &l.input_norm,
                &l.post_attention_norm,
                &l.attention_conv_base,
                &l.mlp_conv_base,
                &l.q_norm,
                &l.k_norm,
            ]
            .iter()
            .map(|b| plain(b))
            .sum::<usize>();
        }
        n
    }

    /// Load `dir/config.json` + `dir/model.safetensors`. Every tensor is looked up by its
    /// published name with its published shape; anything missing or mis-shaped is an error that
    /// names it — a draft that loads wrong would still "work", just never be accepted.
    pub fn load(ctx: &GpuContext, dir: &Path) -> Result<Self, String> {
        let cfg = Dflash2Config::from_json(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| format!("config.json: {e}"))?,
        )?;
        let w = read_weights_lazy(&[dir.join("model.safetensors")]).map_err(|e| format!("{e}"))?;
        let h = cfg.hidden;
        let (q_dim, kv_dim) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let mut worst = (0.0f64, String::new());
        let mut cache = QuantCache::open(&dir.join("model.safetensors"));

        let mut proj = |name: &str, rows: usize, cols: usize| -> Result<Q4ksMtl, String> {
            // What the MPP matmul needs: whole 256-row tiles, whole 256-wide super-blocks.
            if !rows.is_multiple_of(256) || !cols.is_multiple_of(256) {
                return Err(format!(
                    "{name}: [{rows}, {cols}] is not whole 256 tiles/super-blocks"
                ));
            }
            // QUANTIZE CACHE: a hit skips the bf16 read and the searched fit below.
            if let Some((q, err)) = cache.next(name, rows, cols) {
                if err > worst.0 {
                    worst = (err, name.to_string());
                }
                return Q4ksMtl::from_matrix(ctx, &q);
            }
            let m = w
                .bf16_matrix(name, rows, cols)
                .map_err(|e| format!("{name}: {e}"))?;
            // ARF_DFLASH_G64=1 (2026-09-25): the draft on the group-64 kernels — a searched
            // group-64 fit instead of Q4_K_S.
            // ⛔ MEASURED-OUT (2026-09-25, 6 prompts x 2 interleaved rounds, identical target text):
            // draft GPU 6.2 -> 5.5 ms a cycle, but acceptance 3.156 -> 3.108 tokens a cycle (-1.5%,
            // deterministic) — net 45.8 / 45.6 -> 44.6 / 45.1 tok/s. The draft's agreement is
            // sensitive to its precision (4.75-bit two-level Q4_K_S beats 4.5-bit group-64); the
            // time saved is smaller than the acceptance lost. Opt-in instrument only. DO NOT
            // RE-CHASE without a more precise draft format, not a faster one.
            if std::env::var_os("ARF_DFLASH_G64").is_some() {
                let f: Vec<f32> = m.bits().iter().map(|&b| bf16_to_f32(b)).collect();
                if let Some((gw, gs, gb)) =
                    arf_core::model::weights::g64_quantize_searched(&f, rows, cols)
                {
                    let (mut num, mut den) = (0f64, 0f64);
                    let groups = cols / 64;
                    for r in 0..rows.min(64) {
                        for c in 0..cols {
                            let p = ((r / 256) * groups + c / 64) * 256 + r % 256;
                            let q = (gw[p * 32 + (c % 64) / 2] >> (4 * (c % 2))) & 15;
                            let v = arf_core::model::gguf::f16_to_f32_pub(gs[p]) * q as f32
                                + arf_core::model::gguf::f16_to_f32_pub(gb[p]);
                            let a = f[r * cols + c] as f64;
                            num += (a - v as f64).powi(2);
                            den += a * a;
                        }
                    }
                    let err = (num / den.max(1e-30)).sqrt();
                    if err > worst.0 {
                        worst = (err, name.to_string());
                    }
                    return Q4ksMtl::from_g64(ctx, rows, cols, &gw, &gs, &gb);
                }
            }
            let q = quantize(m.bits(), rows, cols);
            let err = quant_rel_rms(m.bits(), &q, rows.min(64), cols);
            cache.put(name, err, &q);
            if err > worst.0 {
                worst = (err, name.to_string());
            }
            Q4ksMtl::from_matrix(ctx, &q)
        };
        let fc = proj("fc.weight", h, cfg.target_layer_ids.len() * h)?;
        let selector_hidden = proj(
            "candidate_selector.hidden_projection.weight",
            cfg.selector_rank,
            h,
        )?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}");
            let dynamic = cfg.dynamic_size();
            let attention_conv_dynamic = proj(
                &format!("{p}.attention_conv.kernel_projection.weight"),
                dynamic,
                h,
            )?;
            let mlp_conv_dynamic = proj(
                &format!("{p}.mlp_conv.kernel_projection.weight"),
                dynamic,
                h,
            )?;
            let q_proj = proj(&format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            let k_proj = proj(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            let v_proj = proj(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            let o_proj = proj(&format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            let gate_proj = proj(&format!("{p}.mlp.gate_proj.weight"), cfg.intermediate, h)?;
            let up_proj = proj(&format!("{p}.mlp.up_proj.weight"), cfg.intermediate, h)?;
            let down_proj = proj(&format!("{p}.mlp.down_proj.weight"), h, cfg.intermediate)?;
            layers.push(Dflash2Layer {
                input_norm: f32_buf(ctx, &w, &format!("{p}.input_layernorm.weight"), &[h])?,
                post_attention_norm: f32_buf(
                    ctx,
                    &w,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[h],
                )?,
                attention_conv_base: f32_buf(
                    ctx,
                    &w,
                    &format!("{p}.attention_conv.base_kernel"),
                    &[2, 2, h],
                )?,
                mlp_conv_base: f32_buf(ctx, &w, &format!("{p}.mlp_conv.base_kernel"), &[2, 2, h])?,
                attention_conv_dynamic,
                mlp_conv_dynamic,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: f32_buf(
                    ctx,
                    &w,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[cfg.head_dim],
                )?,
                k_norm: f32_buf(
                    ctx,
                    &w,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[cfg.head_dim],
                )?,
                gate_proj,
                up_proj,
                down_proj,
            });
        }
        cache.finish();
        let mut draft = Self {
            hidden_norm: f32_buf(ctx, &w, "hidden_norm.weight", &[h])?,
            norm: f32_buf(ctx, &w, "norm.weight", &[h])?,
            predecessor_codebook: bf16_buf(
                ctx,
                &w,
                "candidate_selector.predecessor_codebook",
                cfg.vocab,
                cfg.selector_rank,
            )?,
            successor_codebook: bf16_buf(
                ctx,
                &w,
                "candidate_selector.successor_codebook",
                cfg.vocab,
                cfg.selector_rank,
            )?,
            fc,
            selector_hidden,
            worst_quant_error: worst,
            gpu_bytes: 0,
            layers,
            cfg,
        };
        draft.gpu_bytes = draft.count_bytes();
        Ok(draft)
    }
}

fn shared(ctx: &GpuContext, bytes: &[u8]) -> Result<MtlBuf, String> {
    let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
    let b = guard
        .raw_device()
        .newBufferWithLength_options(
            bytes.len().max(16),
            objc2_metal::MTLResourceOptions::StorageModeShared,
        )
        .ok_or("dflash2 buffer alloc")?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            b.contents().as_ptr() as *mut u8,
            bytes.len(),
        )
    };
    Ok(MtlBuf(b))
}

/// A norm weight or conv base kernel as f32, read with its PUBLISHED shape (the base kernels are
/// 3-D, `[2, 2, hidden]`, and the 1-D reader rightly refuses them).
fn f32_buf(ctx: &GpuContext, w: &Weights, name: &str, shape: &[usize]) -> Result<MtlBuf, String> {
    let v = w
        .get_f32_shaped(name, shape)
        .map_err(|e| format!("{name}: {e}"))?;
    shared(ctx, unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    })
}

fn bf16_buf(
    ctx: &GpuContext,
    w: &Weights,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<MtlBuf, String> {
    let m = w
        .bf16_matrix(name, rows, cols)
        .map_err(|e| format!("{name}: {e}"))?;
    shared(ctx, unsafe {
        std::slice::from_raw_parts(m.bits().as_ptr() as *const u8, m.bits().len() * 2)
    })
}

/// GATES ONLY — CPU copies of what the context commit multiplies by: the same checkpoint bytes
/// through the same deterministic bf16 -> Q4_K_S transcode the loader ran, so a CPU reference
/// uses the weights that are on the GPU (not the bf16 originals, which differ by the ~5-8%
/// quantisation error and would hide a real bug under it).
pub struct Dflash2ContextReference {
    pub fc: Q4KSMatrix,
    pub hidden_norm: Vec<f32>,
    pub k_proj: Vec<Q4KSMatrix>,
    pub v_proj: Vec<Q4KSMatrix>,
    pub k_norm: Vec<Vec<f32>>,
}

impl Dflash2ContextReference {
    /// `fc.weight` as PUBLISHED (bf16 -> f32, `[hidden, taps*hidden]`, 524 MB) — to measure what
    /// the 4-bit transcode of this one matrix costs on REAL inputs. `fc` is the only projection
    /// that reads the raw residual stream, massive-activation channels included, so a weight
    /// error that is ~7% everywhere is not ~7% of its output.
    pub fn fc_published(dir: &Path, cfg: &Dflash2Config) -> Result<Vec<f32>, String> {
        let w = read_weights_lazy(&[dir.join("model.safetensors")]).map_err(|e| format!("{e}"))?;
        let (rows, cols) = (cfg.hidden, cfg.target_layer_ids.len() * cfg.hidden);
        let m = w
            .bf16_matrix("fc.weight", rows, cols)
            .map_err(|e| format!("fc.weight: {e}"))?;
        Ok(m.bits().iter().map(|b| bf16_to_f32(*b)).collect())
    }

    pub fn load(dir: &Path, cfg: &Dflash2Config) -> Result<Self, String> {
        let w = read_weights_lazy(&[dir.join("model.safetensors")]).map_err(|e| format!("{e}"))?;
        let (h, kv_dim) = (cfg.hidden, cfg.kv_heads * cfg.head_dim);
        let q = |name: &str, rows: usize, cols: usize| -> Result<Q4KSMatrix, String> {
            let m = w
                .bf16_matrix(name, rows, cols)
                .map_err(|e| format!("{name}: {e}"))?;
            Ok(quantize(m.bits(), rows, cols))
        };
        let f = |name: &str, n: usize| -> Result<Vec<f32>, String> {
            w.get_f32_shaped(name, &[n])
                .map(|v| v.to_vec())
                .map_err(|e| format!("{name}: {e}"))
        };
        let (mut k_proj, mut v_proj, mut k_norm) = (vec![], vec![], vec![]);
        for l in 0..cfg.layers {
            k_proj.push(q(
                &format!("layers.{l}.self_attn.k_proj.weight"),
                kv_dim,
                h,
            )?);
            v_proj.push(q(
                &format!("layers.{l}.self_attn.v_proj.weight"),
                kv_dim,
                h,
            )?);
            k_norm.push(f(
                &format!("layers.{l}.self_attn.k_norm.weight"),
                cfg.head_dim,
            )?);
        }
        Ok(Self {
            fc: q("fc.weight", h, cfg.target_layer_ids.len() * h)?,
            hidden_norm: f("hidden_norm.weight", h)?,
            k_proj,
            v_proj,
            k_norm,
        })
    }
}

/// QUANTIZE CACHE (2026-10-07). The draft's projections were read as bf16 (3.8 GB) and fitted to
/// Q4_K_S with the searched scales at EVERY start: 16.6 s of a 29.6 s start of the 27B (one M4
/// Max, 2026-10-06). The fit is deterministic, so its result is written once beside the
/// checkpoint — `model.safetensors.arf-draft-cache` — and read back on later starts.
///
/// The file is the matrices in load order, each with its name, shape and weight error, behind a
/// header keyed by the checkpoint's `(len, mtime)` and [`QUANT_CACHE_VERSION`]. A read follows
/// the load: a record whose name or shape is not the one asked for ends the cache for this load
/// (the rest is fitted as before, nothing is rewritten). A build writes `.building` and renames it
/// when the load has every matrix, so a killed start leaves no half file under the real name.
/// `ARF_NO_DRAFT_CACHE=1` turns it off; so does `ARF_DFLASH_G64` (another format).
enum QuantCache {
    Off,
    Read(std::io::BufReader<std::fs::File>),
    Build {
        out: std::io::BufWriter<std::fs::File>,
        tmp: std::path::PathBuf,
        dst: std::path::PathBuf,
        ok: bool,
    },
}

/// Bump when [`quantize`] or the record layout changes: a file from another version is rebuilt.
const QUANT_CACHE_VERSION: u32 = 1;
const QUANT_CACHE_MAGIC: &[u8; 8] = b"ARFDRFT\0";

impl QuantCache {
    fn header(src: &Path) -> Option<Vec<u8>> {
        let md = std::fs::metadata(src).ok()?;
        let mtime = md
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        let mut h = QUANT_CACHE_MAGIC.to_vec();
        h.extend_from_slice(&QUANT_CACHE_VERSION.to_le_bytes());
        h.extend_from_slice(&md.len().to_le_bytes());
        h.extend_from_slice(&mtime.as_nanos().to_le_bytes());
        Some(h)
    }

    fn open(src: &Path) -> Self {
        use std::io::{Read as _, Write as _};
        if std::env::var_os("ARF_NO_DRAFT_CACHE").is_some()
            || std::env::var_os("ARF_DFLASH_G64").is_some()
        {
            return Self::Off;
        }
        let Some(header) = Self::header(src) else {
            return Self::Off;
        };
        let with = |suffix: &str| {
            let mut s = src.as_os_str().to_os_string();
            s.push(suffix);
            std::path::PathBuf::from(s)
        };
        let (dst, tmp) = (with(".arf-draft-cache"), with(".arf-draft-cache.building"));
        if let Ok(f) = std::fs::File::open(&dst) {
            let mut r = std::io::BufReader::with_capacity(1 << 20, f);
            let mut got = vec![0u8; header.len()];
            if r.read_exact(&mut got).is_ok() && got == header {
                return Self::Read(r);
            }
        }
        // A folder that cannot be written (a read-only bundle) simply has no cache.
        match std::fs::File::create(&tmp) {
            Ok(f) => {
                let mut out = std::io::BufWriter::with_capacity(1 << 20, f);
                let ok = out.write_all(&header).is_ok();
                Self::Build { out, tmp, dst, ok }
            }
            Err(_) => Self::Off,
        }
    }

    /// The next record, when it is `name` at `[rows, cols]`; anything else ends the cache.
    fn next(&mut self, name: &str, rows: usize, cols: usize) -> Option<(Q4KSMatrix, f64)> {
        use std::io::Read as _;
        let Self::Read(r) = self else {
            return None;
        };
        let mut read = || -> Option<(Q4KSMatrix, f64)> {
            let mut n = [0u8; 4];
            r.read_exact(&mut n).ok()?;
            let mut got = vec![0u8; u32::from_le_bytes(n) as usize];
            r.read_exact(&mut got).ok()?;
            if got != name.as_bytes() {
                return None;
            }
            let mut f = [0u8; 8];
            r.read_exact(&mut f).ok()?;
            let err = f64::from_le_bytes(f);
            r.read_exact(&mut f).ok()?;
            let len = usize::try_from(u64::from_le_bytes(f)).ok()?;
            // What a matrix of this shape serializes to — never trust the file's length.
            let n = rows.checked_mul(cols)?;
            if len != 16 + n / 8 * 4 + n / 32 * 2 + n / 256 * 8 {
                return None;
            }
            let mut bytes = vec![0u8; len];
            r.read_exact(&mut bytes).ok()?;
            let q = Q4KSMatrix::from_bytes(&bytes)?;
            (q.rows() == rows && q.cols() == cols).then_some((q, err))
        };
        let hit = read();
        if hit.is_none() {
            eprintln!(
                "[dflash2] quantize cache: no record for {name} [{rows}, {cols}] where the load \
                 expects it — fitting the rest (delete the .arf-draft-cache file to rebuild it)"
            );
            *self = Self::Off;
        }
        hit
    }

    fn put(&mut self, name: &str, err: f64, q: &Q4KSMatrix) {
        use std::io::Write as _;
        let Self::Build { out, ok, .. } = self else {
            return;
        };
        let bytes = q.to_bytes();
        let mut w = || -> std::io::Result<()> {
            out.write_all(&(name.len() as u32).to_le_bytes())?;
            out.write_all(name.as_bytes())?;
            out.write_all(&err.to_le_bytes())?;
            out.write_all(&(bytes.len() as u64).to_le_bytes())?;
            out.write_all(&bytes)
        };
        *ok = *ok && w().is_ok();
    }

    /// The load has every matrix: promote a build. A failed write removes the partial file.
    fn finish(&mut self) {
        use std::io::Write as _;
        if let Self::Build { out, tmp, dst, ok } = std::mem::replace(self, Self::Off) {
            let mut out = out;
            if ok && out.flush().is_ok() && std::fs::rename(&tmp, &dst).is_ok() {
                eprintln!("[dflash2] quantize cache written: {}", dst.display());
            } else {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
}

/// THE draft's bf16 -> Q4_K_S transcode — one function, because the loader and both CPU
/// references must store the SAME numbers (a reference that quantised differently would gate
/// nothing). The error-minimising fit, not min/max: measured 2026-09-20 on the daemon, same
/// prompt, interleaved — tokens per cycle 3.02 -> 3.14, decode 20.8 -> 21.1-21.8 tok/s, worst
/// weight error 7.98% -> 7.15%. The min/max arm lost and its switch is deleted.
pub fn quantize(bits: &[u16], rows: usize, cols: usize) -> Q4KSMatrix {
    Q4KSMatrix::from_bf16_searched(bits, rows, cols)
}

fn bf16_to_f32(b: u16) -> f32 {
    // bfloat16 is the top 16 bits of an IEEE f32.
    f32::from_bits((b as u32) << 16)
}

/// Relative RMS error of the Q4_K_S transcode over the first `rows` rows: sqrt(sum (w - w_q)^2 /
/// sum w^2). The draft is only useful if its predictions survive 4-bit, so the loader reports it.
fn quant_rel_rms(bits: &[u16], q: &Q4KSMatrix, rows: usize, cols: usize) -> f64 {
    let (subs, supers) = (cols / 32, cols / 256);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for r in 0..rows {
        for sub in 0..subs {
            let s = q.d()[r * supers + sub / 8] * q.scales()[r * subs + sub] as f32;
            let lo = q.dmin()[r * supers + sub / 8] * q.mins()[r * subs + sub] as f32;
            for i in 0..32 {
                let pos = r * cols + sub * 32 + i;
                let code = (q.codes()[pos / 8] >> ((pos % 8) * 4)) & 0xF;
                let (a, b) = (bf16_to_f32(bits[pos]) as f64, (code as f32 * s + lo) as f64);
                num += (a - b) * (a - b);
                den += a * a;
            }
        }
    }
    (num / den.max(1e-30)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{quantize, Dflash2Config, QuantCache};

    /// The published `config.json` of incoai/Qwen3.8-27B-DFlash2, trimmed to what is read.
    const PUBLISHED: &str = r#"{
        "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]},
        "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
        "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 2048, "vocab_size": 248320}"#;

    #[test]
    fn parses_the_published_config() {
        let c = Dflash2Config::from_json(PUBLISHED).unwrap();
        assert_eq!(c.target_layer_ids, [5, 19, 33, 47, 61]);
        assert_eq!(
            (c.layers, c.hidden, c.heads, c.kv_heads, c.head_dim),
            (5, 5120, 32, 8, 128)
        );
        assert_eq!((c.block_size, c.mask_token_id), (8, 248070));
        // 2 stages x 2 taps x (5120 / 16) — the width of `kernel_projection`.
        assert_eq!(c.dynamic_size(), 1280);
        assert_eq!(
            (c.selector_rank, c.selector_top_k, c.sliding_window),
            (256, 16, 2048)
        );
    }

    /// A wider conv is a different computation, not a parameter: refuse it rather than mis-run it.
    #[test]
    fn refuses_a_conv_it_does_not_implement() {
        let three_tap = PUBLISHED.replace("\"conv_kernel_size\": 2", "\"conv_kernel_size\": 3");
        assert!(Dflash2Config::from_json(&three_tap).is_err());
        assert!(Dflash2Config::from_json("{}").is_err());
    }
    /// The quantize cache gives back, in load order, exactly what was put; a record asked for out
    /// of order ends it; a changed checkpoint is not read.
    #[test]
    fn quant_cache_round_trips_in_load_order_and_rejects_another_checkpoint() {
        let dir = std::env::temp_dir().join(format!("arf-draft-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("model.safetensors");
        std::fs::write(&src, b"checkpoint").unwrap();
        let bits = |seed: u16| -> Vec<u16> {
            (0..256 * 256u32)
                .map(|i| ((0.01 * ((i as f32 * 0.37 + seed as f32).sin())).to_bits() >> 16) as u16)
                .collect()
        };
        let (a, b) = (quantize(&bits(1), 256, 256), quantize(&bits(2), 256, 256));

        let mut c = QuantCache::open(&src);
        assert!(matches!(c, QuantCache::Build { .. }));
        assert!(c.next("a", 256, 256).is_none(), "a build serves nothing");
        c.put("a", 0.07, &a);
        c.put("b", 0.08, &b);
        c.finish();

        let mut c = QuantCache::open(&src);
        assert!(matches!(c, QuantCache::Read(_)));
        let (qa, ea) = c.next("a", 256, 256).expect("a");
        let (qb, eb) = c.next("b", 256, 256).expect("b");
        assert_eq!((qa.to_bytes(), ea), (a.to_bytes(), 0.07));
        assert_eq!((qb.to_bytes(), eb), (b.to_bytes(), 0.08));
        assert!(c.next("c", 256, 256).is_none(), "past the end");

        // Out of order, and a wrong shape: the cache ends for this load.
        let mut c = QuantCache::open(&src);
        assert!(c.next("b", 256, 256).is_none());
        assert!(matches!(c, QuantCache::Off));
        let mut c = QuantCache::open(&src);
        assert!(c.next("a", 512, 256).is_none());

        // Another checkpoint under the same name: rebuilt, not read.
        std::fs::write(&src, b"another checkpoint").unwrap();
        assert!(matches!(QuantCache::open(&src), QuantCache::Build { .. }));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
