//! Resident GPU model types: weights, layers, the `GpuKernels` pipeline set,
//! scratch/arena buffers, the KV pool, and sampling params.
//! Split out of gpu/mod.rs; behavior unchanged.

use super::*;

/// A resident matmul weight, either bf16-packed (the lossless default) or per-row
/// int8 with a scale buffer. The decode/batched matmul helpers pick the kernel
/// and bindings from the kind, so the rest of the pipeline is precision-agnostic.
///
/// `Clone` is cheap: every variant holds only `wgpu::Buffer` handles, which are
/// `Arc`-backed — cloning shares the same resident GPU allocation, it does NOT
/// copy bytes (used e.g. to clone K as V on Gemma 4 global layers that ship no attn_v).
#[derive(Clone)]
pub enum GpuMatWeight {
    /// bf16-packed `u32` weights (see [`GpuContext::storage_init_bf16`]).
    Bf16(wgpu::Buffer),
    /// Per-row symmetric int8: `data` packed 4-per-`u32`, one f32 `scale` per row.
    Int8 {
        data: wgpu::Buffer,
        scale: wgpu::Buffer,
    },
    /// Q4_0 block-32 4-bit: `codes` packed 8-per-`u32`, one bf16 `scale` per
    /// 32-weight block (so `scale` is a `u32` buffer holding bf16-in-low-16-bits,
    /// read 2-per-word; see the q4 GEMV). Half the int8 bytes on the big matmuls.
    Q4 {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
    },
    /// Q4_K-lite: super-block-256, UNSIGNED 4-bit `codes` (8/u32), one bf16 `scales`
    /// AND one bf16 `mins` per 32-weight sub-block (each a `u32` buffer, bf16 in low
    /// 16 bits). The asymmetric min-offset matches ollama's Q4_K accuracy.
    Q4K {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
        mins: wgpu::Buffer,
    },
    /// Q4_K_S: like Q4K but the sub-block scales/mins are u8/i8 (packed 4/u32)
    /// against a per-super-block f32 `dd` pair `[d, dmin]` — half the scale bytes.
    Q4KS {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer, // u8, 4/u32
        mins: wgpu::Buffer,   // i8, 4/u32
        dd: wgpu::Buffer,     // f32 [d, dmin] per super-block
        /// Raw-Metal views of the SAME buffers (macOS + ARF_MSL_GEMV) so the decode
        /// GEMV can run the hand-MSL `gemv_q4ks` kernel (86-97% roofline) instead of the
        /// WGSL one (~40% in-frame). `None` everywhere else — wgpu stays the default.
        #[cfg(target_os = "macos")]
        mtl: Option<std::sync::Arc<crate::gpu::concurrent_metal::Q4ksMtl>>,
    },
    /// Q3_K: 3-bit split-plane (`ql` 2 low bits 16/u32, `qh` 1 high bit 32/u32) +
    /// Q4_K_S's u8/i8 two-level scales. ~25% fewer code bytes than Q4 — the real
    /// decode-speed cut, at a 3-bit accuracy cost.
    Q3K {
        ql: wgpu::Buffer,
        qh: wgpu::Buffer,
        scales: wgpu::Buffer,
        mins: wgpu::Buffer,
        dd: wgpu::Buffer,
    },
}

impl GpuMatWeight {
    /// Bytes of weight read per element — the units the matmul streams from GPU memory.
    /// Used to report effective GB/s in the profiling label. Q4 is half a byte of
    /// codes + 1/16 byte of block scale ≈ 0.5625; report as scaled-by-16 then /16
    /// is overkill — use the dominant 0.5 (codes) since that is what streams.
    #[cfg(feature = "profiling")]
    pub(crate) fn bytes_num(&self, n: usize, kdim: usize) -> u64 {
        match self {
            GpuMatWeight::Bf16(_) => (n * kdim) as u64 * 2,
            GpuMatWeight::Int8 { .. } => (n * kdim) as u64,
            // codes: n*k/2 bytes; scales: n*k/32 * 2 = n*k/16 bytes.
            GpuMatWeight::Q4 { .. } => (n * kdim) as u64 / 2 + (n * kdim) as u64 / 16,
            // codes n*k/2 + scales+mins each n*k/32*2 = n*k/16 → +n*k/8.
            GpuMatWeight::Q4K { .. } => (n * kdim) as u64 / 2 + (n * kdim) as u64 / 8,
            // codes n*k/2 + scales+mins u8/i8 each n*k/32 → +n*k/16; dd negligible.
            GpuMatWeight::Q4KS { .. } => (n * kdim) as u64 / 2 + (n * kdim) as u64 / 16,
            // ql n*k/4 (2 bits) + qh n*k/8 (1 bit) = n*k*3/8 codes + n*k/16 scales.
            GpuMatWeight::Q3K { .. } => (n * kdim) as u64 * 3 / 8 + (n * kdim) as u64 / 16,
        }
    }
}

/// One projection's PACKED experts (all experts' `[out, in]` matrices back-to-back,
/// indexed by `expert*out` offset). bf16 (parity path) or Q4_0 (fits + runs the
/// 30B). The matching MoE GEMV kernel (`matmul_vec_moe` / `matmul_vec_moe_q4`)
/// resolves the expert on-GPU and reads only the routed rows.
pub enum PackedExperts {
    Bf16(wgpu::Buffer),
    Q4 {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
    },
    /// Q4_K-lite (super-block-256, per-32 asymmetric scale+min) — keeps the GGUF
    /// disk format's accuracy class (vs Q4_0's symmetric codes), still fits the 30B.
    Q4K {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
        mins: wgpu::Buffer,
    },
    /// Q4_K_S (super-block-256, two-level u8/i8 scales against a per-super f32
    /// d/dmin) — ~16% fewer scale bytes than Q4_K-lite's bf16 scale+min, same
    /// codes. `scales` are u8 (4/word), `mins` i8 (4/word), `dd` interleaved
    /// [d, dmin] f32 per super-block. Same dense Q4_K_S numeric class.
    Q4KS {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
        mins: wgpu::Buffer,
        dd: wgpu::Buffer,
    },
}

/// One MoE block's GPU weights: a router projection plus packed expert buffers for
/// the routed experts and the always-on shared experts. Only the routed experts'
/// bytes are streamed (the batch-1 MoE bandwidth win), in the merged pass.
pub struct GpuMoe {
    /// Router: `hidden → num_experts` logits.
    pub router: GpuMatWeight,
    /// Packed routed experts (`num_experts` of each).
    pub gate: PackedExperts,
    pub up: PackedExperts,
    pub down: PackedExperts,
    /// Packed shared experts (`shared_experts` of each); only present if shared > 0.
    pub shared_gate: Option<PackedExperts>,
    pub shared_up: Option<PackedExperts>,
    pub shared_down: Option<PackedExperts>,
    pub num_experts: usize,
    pub top_k: usize,
    pub shared_experts: usize,
    pub moe_inter: usize,
    pub norm_topk: bool,
    /// All-experts raw-Metal views of gate/up/down (macOS + ARF_MSL_GEMV) so the whole-token
    /// megakernel runs the INDIRECT GEMV (`gemv_q4ks_id`) — qwen MoE decode at the dense kernel's
    /// roofline instead of the serial wgpu MoE path (13 tok/s). `None` everywhere else (the wgpu
    /// `add_moe_block` path is the default + the oracle). gate/up have `n = moe_inter` cols/expert,
    /// down has `n = hidden`. The router's `.mtl` lives on `router` (a `GpuMatWeight::Q4KS`).
    #[cfg(target_os = "macos")]
    pub gate_mtl: Option<std::sync::Arc<crate::gpu::concurrent_metal::MoeMtl>>,
    #[cfg(target_os = "macos")]
    pub up_mtl: Option<std::sync::Arc<crate::gpu::concurrent_metal::MoeMtl>>,
    #[cfg(target_os = "macos")]
    pub down_mtl: Option<std::sync::Arc<crate::gpu::concurrent_metal::MoeMtl>>,
}

/// A dense SwiGLU MLP's three GPU projections (Llama/Gemma).
pub struct GpuDenseMlp {
    pub gate_proj: GpuMatWeight,
    pub up_proj: GpuMatWeight,
    pub down_proj: GpuMatWeight,
    /// Q3_K (3-bit) MTL views of gate/up/down for the megakernel island (ARF_Q3K_FFN) — the
    /// bandwidth-floor breaker. `None` = the island uses the Q4KS .mtl. The wgpu path is
    /// unaffected (always the Q4KS gate/up/down_proj above = byte-identical oracle).
    #[cfg(target_os = "macos")]
    pub q3k: Option<
        std::sync::Arc<(
            crate::gpu::concurrent_metal::Q3ksMtl,
            crate::gpu::concurrent_metal::Q3ksMtl,
            crate::gpu::concurrent_metal::Q3ksMtl,
        )>,
    >,
    /// EXPERIMENT (ARF_Q8_DOWN, 2026-09-22): `down_proj` served at Q8 on the island instead of
    /// the Q4_K_S the loader requantises it to. The GGUF ships `ffn_down` at Q6_K in every
    /// layer and `--quant q4ks` drops it to 4-bit; this keeps it near the file's precision to
    /// measure what the served target's quantisation does to the draft's ACCEPTANCE (2.09 per
    /// window against another engine's 2.70 on the same prompt). `None` = the shipped Q4KS view.
    #[cfg(target_os = "macos")]
    pub down_q8: Option<std::sync::Arc<crate::gpu::concurrent_metal::Q8Mtl>>,
}

/// The feed-forward weights for a layer: dense SwiGLU (Llama/Gemma) or MoE (Qwen3).
/// Both variants are boxed so the enum (and the per-layer `GpuLayer` vec) stays
/// small regardless of which is larger — `GpuMoe` grows with the packed experts,
/// `GpuDenseMlp` holds three full `GpuMatWeight`s.
pub enum GpuMlp {
    Dense(Box<GpuDenseMlp>),
    Moe(Box<GpuMoe>),
}

/// Model-level raw-Metal views for the whole-token megakernel (rope tables + final_norm).
#[cfg(target_os = "macos")]
#[derive(Default)]
pub struct MegakernelBufs {
    pub rope_cos: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub rope_sin: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub rope_cos_local: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub rope_sin_local: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub rope_cos_global: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub rope_sin_global: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub final_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    /// Muse Glimmer's all-ones pre-trunk embedding norm, for the island's own embed (op 0).
    /// Without it the megakernel feeds RAW embeddings to all 52 layers.
    pub embed_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    /// lm_head on the island, so it runs there instead of the slow bf16 GEMV — the biggest
    /// attackable per-token cost. Either the GGUF's own Q4_K blocks (bit-exact, 0.53 GB) or a
    /// bf16→Q8 transcode (1.01 GB) when the file has no native blocks; see [`IslandLmHead`].
    ///
    /// [`IslandLmHead`]: crate::gpu::concurrent_metal::IslandLmHead
    pub lm_head: Option<std::sync::Arc<crate::gpu::concurrent_metal::IslandLmHead>>,
    /// bf16 embed table MTL view — so the island runs embed first (whole token single-queue).
    pub embed: Option<crate::gpu::concurrent_metal::MtlBuf>,
}

/// Raw-Metal views of a layer's 6 norm weights (megakernel S1b) — so the island norms read
/// the SAME (1+w-folded) bytes the wgpu path uses. Each `None` when the weight isn't present
/// (pre/post-ffn = gemma only, q/k_norm = qk_norm models) or the island is off.
#[cfg(target_os = "macos")]
#[derive(Default)]
pub struct NormMtls {
    pub input_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub post_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub pre_ffn_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub post_ffn_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub q_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub k_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
    /// L138 — gated-delta-net per-head statics, as raw-Metal views so the megakernel GDN branch
    /// can bind them directly: `ssm_dt.bias` `nvh`, `ssm_a` `nvh`, `ssm_norm.weight` `head_v_dim`.
    /// `None` on every non-GDN layer (and every other arch).
    pub ssm_dt: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub ssm_a: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub ssm_norm: Option<crate::gpu::concurrent_metal::MtlBuf>,
}

/// Gated-delta-net weights for one qwen35 layer. Shapes (Qwen3.8-27B, read from the GGUF):
///   in_proj   [8192, 5120]  fused — ssm_inner(6144) + 2*1024, row-sliced at load
///   alpha     [5120, 48]    hidden x dt_rank
///   beta      [5120, 48]    hidden x dt_rank
///   conv1d    [4, 10240]    conv_kernel x 2*inner (depthwise causal)
///   out_proj  [6144, 5120]  inner x hidden
///   a_log     `48`          per-head decay log
///   dt_bias   `48`          delta bias
///   norm      `128`         state_size RMSNorm
pub struct GpuGdn {
    /// L137 — the packed SSM input projection `[h -> key_dim*2 + value_dim]`, held WHOLE.
    /// Prefer this over `in_q`/`in_k`/`in_v` below: on a GDN layer the GGUF packs the tensor
    /// `[6144 | 2048 | 2048]` while `gdn_tile` reads `[q 2048 | k 2048 | v 6144]`, so the
    /// slice *named* q_proj is actually the SSM's v. One fused GEMV avoids that trap entirely
    /// (see the note in gguf.rs's `attn_qkv.weight` arm).
    pub qkv: GpuMatWeight,
    /// The output gate `z` `[h -> value_dim]` (`attn_gate.weight` on a GDN layer) used by
    /// `out = rmsnorm(readout, ssm_norm) * silu(z)`.
    pub qkv_gate: GpuMatWeight,
    pub in_q: GpuMatWeight,
    pub in_k: GpuMatWeight,
    pub in_v: GpuMatWeight,
    pub alpha: GpuMatWeight,
    pub beta: GpuMatWeight,
    /// Depthwise causal conv, [2*inner, kernel] = 40960 f32 for qwen35. NOT a quantized
    /// matrix — kernel=4 is far below the 256-element Q4_K block, so it ships f32 and loads
    /// as a flat buffer, like the norm vectors.
    pub conv1d: wgpu::Buffer,
    /// L138 — the SAME conv kernel as a host `Vec<f32>` (`[d_conv * conv_dim]`, a few tens of KB
    /// per layer). `gdn_ensure` uploads it into the layer's resident island scratch ONCE, and it
    /// needs a CPU slice to do that. Kept here so the megakernel setup never has to read a GPU
    /// buffer back — a per-layer readback on the decode path would be far worse than this copy.
    pub conv1d_host: std::sync::Arc<Vec<f32>>,
    pub out_proj: GpuMatWeight,
    pub a_log: wgpu::Buffer,
    pub dt_bias: wgpu::Buffer,
    pub norm: wgpu::Buffer,
}

/// L241 — the trained multi-token-prediction ("nextn") draft head, `blk.64` on qwen35.
/// Sits AFTER the trunk and outside the forward pass: the model is correct without it. Held in
/// its own slot rather than in `layers`, because appending it there would break the 4-layer
/// attention schedule AND run a draft head inside the main forward pass.
pub struct GpuMtpHead {
    // ---- blk.64's own attention + FFN (named like a trunk block, loaded the same way) ----
    pub q_proj: GpuMatWeight,
    pub k_proj: GpuMatWeight,
    pub v_proj: GpuMatWeight,
    pub o_proj: GpuMatWeight,
    /// L252 — both views. The wgpu buffer for parity/oracle paths, the Metal one because the
    /// island record binds MTL buffers directly and `up_vec` hands back both anyway.
    pub q_norm: wgpu::Buffer,
    pub k_norm: wgpu::Buffer,
    pub attn_norm: wgpu::Buffer,
    pub ffn_norm: wgpu::Buffer,
    pub enorm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub hnorm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub attn_norm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub ffn_norm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub q_norm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub k_norm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub gate_proj: GpuMatWeight,
    pub up_proj: GpuMatWeight,
    pub down_proj: GpuMatWeight,
    // ---- the four nextn.* tensors ----
    /// `[2*hidden -> hidden]` — consumes `[enorm(embed(tok_t)) ‖ hnorm(h_t)]`.
    pub eh_proj: GpuMatWeight,
    /// RMS weight for the token-embedding half of the concatenation.
    pub enorm: wgpu::Buffer,
    /// RMS weight for the trunk-hidden half.
    pub hnorm: wgpu::Buffer,
    /// Final RMS before the (shared) lm_head.
    pub shared_head_norm: wgpu::Buffer,
    /// Metal view of `shared_head_norm` — the draft record's final norm. It is NOT
    /// `model.norm.weight`: on the shipped GGUF the two differ by up to 0.99 per channel, and
    /// llama.cpp's `graph_mtp` norms with this one before the shared lm_head.
    pub shared_head_norm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
}

pub struct GpuLayer {
    pub q_proj: GpuMatWeight,
    pub k_proj: GpuMatWeight,
    pub v_proj: GpuMatWeight,
    pub o_proj: GpuMatWeight,
    /// Muse Glimmer only: gates the attention output before o_proj
    /// (`attn *= sigmoid(attn_gate @ attn_input)`). `None` for every other arch, and the
    /// dispatch is skipped when it is None — presence of the tensor IS the feature flag.
    pub attn_gate: Option<GpuMatWeight>,
    /// GATED-DELTA-NET block (qwen35 family). `Some` on the 3-of-4 layers that have NO
    /// attention at all: q/k/v/o above are then placeholders and the forward pass takes the GDN
    /// path instead. Presence IS the switch, exactly like `attn_gate` — no arch enum threaded
    /// through the layer loop.
    pub gdn: Option<Box<GpuGdn>>,
    pub mlp: GpuMlp,
    pub input_norm: wgpu::Buffer,
    /// `post_attention_layernorm`. For Llama this is the pre-MLP norm (fused via
    /// `add_norm`); for Gemma it is the post-attention norm applied before the
    /// residual add.
    pub post_norm: wgpu::Buffer,
    /// Gemma only: `pre_feedforward_layernorm` (norm before the MLP) and
    /// `post_feedforward_layernorm` (norm on the MLP output before the residual).
    /// `None` for Llama/Qwen, which use the two-norm fused path.
    pub pre_ffn_norm: Option<wgpu::Buffer>,
    pub post_ffn_norm: Option<wgpu::Buffer>,
    /// Per-head q/k RMSNorm weights (`[head_dim]`), present for Qwen3/Gemma only.
    /// When `Some`, the `qk_norm` kernel runs on q,k before RoPE.
    pub q_norm: Option<wgpu::Buffer>,
    pub k_norm: Option<wgpu::Buffer>,
    /// Raw-Metal views of the 6 norm weights (megakernel S1b; macOS island only).
    #[cfg(target_os = "macos")]
    pub norm_mtls: crate::gpu::NormMtls,
    /// Gemma sliding-window *local* layer size (`Some(window)`) or global (`None`).
    /// Local layers also rotate with the local RoPE base (see `GpuModel`).
    pub window: Option<usize>,
    /// Per-layer attention geometry. For most models every layer is identical
    /// (`== cfg.head_dim` / `cfg.num_kv_heads`, full rotation). Gemma 4 gives its
    /// GLOBAL (full-attention, every-6th) layers a DIFFERENT profile from the base
    /// SLIDING layers: `head_dim` 512 vs 256, `kv_heads` 4 vs 16, `rotary_dim`
    /// (the count of head dims RoPE rotates) 128 (= 512·0.25) vs the full 256.
    /// These drive the per-layer q/k/v projection out-dims, the o_proj in-dim, the
    /// GQA group size, the attention `head_dim`/`kv_heads`/`group` uniforms, the
    /// `rope_qk` `rotary_dim` uniform, and the KV-pool row stride.
    pub head_dim: usize,
    pub kv_heads: usize,
    pub rotary_dim: usize,
    /// Gemma 4 only: a learned per-layer scalar multiplying the WHOLE layer output
    /// at the very end — `hidden *= layer_scalar` AFTER both sublayers, residuals,
    /// and post-norms (verified vs HF `Gemma4TextDecoderLayer.forward`). `None` for
    /// every other model (no-op).
    pub layer_scalar: Option<f32>,
}

impl GpuLayer {
    /// Query-projection out-dim (`num_attention_heads · head_dim`) for THIS layer.
    /// `num_attention_heads` is uniform across layers (32 for Gemma 4); only
    /// `head_dim`/`kv_heads` vary by layer type, so the q rows scale with head_dim.
    pub fn q_dim(&self, num_attention_heads: usize) -> usize {
        num_attention_heads * self.head_dim
    }
    /// K/V-projection out-dim (`kv_heads · head_dim`) for THIS layer — also the
    /// KV-pool row stride (floats per cached token) the scatter/attention use.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

impl GpuMatWeight {
    /// Every backing buffer (for residency commits / bookkeeping).
    pub fn buffers<'a>(&'a self, out: &mut Vec<&'a wgpu::Buffer>) {
        match self {
            GpuMatWeight::Bf16(b) => out.push(b),
            GpuMatWeight::Int8 { data, scale } => out.extend([data, scale]),
            GpuMatWeight::Q4 { codes, scales } => out.extend([codes, scales]),
            GpuMatWeight::Q4K {
                codes,
                scales,
                mins,
            } => out.extend([codes, scales, mins]),
            GpuMatWeight::Q4KS {
                codes,
                scales,
                mins,
                dd,
                ..
            } => out.extend([codes, scales, mins, dd]),
            GpuMatWeight::Q3K {
                ql,
                qh,
                scales,
                mins,
                dd,
            } => out.extend([ql, qh, scales, mins, dd]),
        }
    }
}

impl PackedExperts {
    /// Every backing buffer (for residency commits / bookkeeping).
    pub fn buffers<'a>(&'a self, out: &mut Vec<&'a wgpu::Buffer>) {
        match self {
            PackedExperts::Bf16(b) => out.push(b),
            PackedExperts::Q4 { codes, scales } => out.extend([codes, scales]),
            PackedExperts::Q4K {
                codes,
                scales,
                mins,
            } => out.extend([codes, scales, mins]),
            PackedExperts::Q4KS {
                codes,
                scales,
                mins,
                dd,
            } => out.extend([codes, scales, mins, dd]),
        }
    }
}

impl GpuLayer {
    /// Every weight buffer this layer owns (attention projections, norms, MLP/MoE).
    pub fn buffers<'a>(&'a self, out: &mut Vec<&'a wgpu::Buffer>) {
        for m in [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj] {
            m.buffers(out);
        }
        out.push(&self.input_norm);
        out.push(&self.post_norm);
        if let Some(pre) = &self.pre_ffn_norm {
            out.push(pre);
        }
        if let Some(post) = &self.post_ffn_norm {
            out.push(post);
        }
        if let Some(q) = &self.q_norm {
            out.push(q);
        }
        if let Some(k) = &self.k_norm {
            out.push(k);
        }
        match &self.mlp {
            GpuMlp::Dense(d) => {
                d.gate_proj.buffers(out);
                d.up_proj.buffers(out);
                d.down_proj.buffers(out);
            }
            GpuMlp::Moe(m) => {
                m.router.buffers(out);
                m.gate.buffers(out);
                m.up.buffers(out);
                m.down.buffers(out);
                for s in [&m.shared_gate, &m.shared_up, &m.shared_down]
                    .into_iter()
                    .flatten()
                {
                    s.buffers(out);
                }
            }
        }
    }
}

/// Reused scratch buffers — allocated once, overwritten every token (zero
/// per-token allocation, the game-engine discipline).
pub struct GpuScratch {
    pub hidden: wgpu::Buffer,     // [hidden]
    pub normed: wgpu::Buffer,     // [hidden]
    pub q: wgpu::Buffer,          // [q_dim]
    pub k: wgpu::Buffer,          // [kv_dim]
    pub v: wgpu::Buffer,          // [kv_dim]
    pub attn_out: wgpu::Buffer,   // [hidden] (o_proj maps q_dim -> hidden)
    pub gate: wgpu::Buffer,       // [max(intermediate, moe_inter)]
    pub up: wgpu::Buffer,         // [max(intermediate, moe_inter)]
    pub mlp_down: wgpu::Buffer,   // [hidden]
    pub logits: wgpu::Buffer,     // [vocab]
    pub next_token: wgpu::Buffer, // [1] u32 — GPU argmax of `logits`, read back as one word
    /// `[max_tokens]` u32 — each decode step copies its emitted id here, so a
    /// whole generation drains to the CPU once instead of once per token.
    pub out_tokens: wgpu::Buffer,
    /// MoE decode scratch (present only for MoE models): router logits, and the
    /// routed expert ids/weights written by `moe_route`.
    pub router_logits: Option<wgpu::Buffer>, // [num_experts]
    pub moe_ids: Option<wgpu::Buffer>, // [top_k] u32
    pub moe_wts: Option<wgpu::Buffer>, // [top_k] f32
    /// Fused-MoE per-slot gate/up outputs, [top_k * moe_inter] (slot-major). The
    /// fused gate/up kernels write all routed experts here in one dispatch; the
    /// fused down reduces over them. Present only for MoE models.
    pub moe_gate_all: Option<wgpu::Buffer>,
    pub moe_up_all: Option<wgpu::Buffer>,
    /// Constant ids `[0,1,..]` and weights `[1.0,..]` for the always-on shared
    /// experts (no router weight). Present only when `shared_experts > 0`.
    pub shared_ids: Option<wgpu::Buffer>,
    pub shared_wts: Option<wgpu::Buffer>,
    /// Raw-Metal views of the FFN-block scratch buffers (macOS + ARF_MSL_GEMV), so the
    /// island's `ffn_dispatch` reads/writes the SAME memory the wgpu path uses (via
    /// `SharedBuffer`). ADDITIVE — the wgpu `normed/gate/up/mlp_down/hidden` fields above
    /// stay the default for every existing binding + readback; these are `None` otherwise.
    /// Each aliases its same-named wgpu buffer (one allocation, two views).
    #[cfg(target_os = "macos")]
    pub normed_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub gate_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub up_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub mlp_down_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub hidden_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    // Attention scratch MTL views (megakernel S1).
    #[cfg(target_os = "macos")]
    pub q_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub k_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub v_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub attn_out_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub logits_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub next_token_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub out_tokens_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    /// SECOND out_tokens bank for the depth-2 ping-pong decode pipeline. Sized
    /// identically to `out_tokens_mtl` (`max_positions*4` bytes). Step N writes bank A while
    /// step N+1 reads bank A as its token source (`TokenSrc::GpuBank`) and writes bank B; the
    /// two banks alternate so a submitted-but-not-yet-read step's bank is never overwritten by
    /// the next submit. `None` when the shared-buffer alloc fails (depth-2 then unavailable →
    /// the loop stays depth-1). See `GpuModel::out_bank`.
    #[cfg(target_os = "macos")]
    pub out_tokens_mtl2: Option<crate::gpu::concurrent_metal::MtlBuf>,
}

/// The decode-pipeline compute kernels, compiled once at load.
pub struct GpuKernels {
    /// Force the Q4_K sub-block GEMV (`matmul_vec_q4k_batch`) over the coop-matrix
    /// GEMM for m≥8 batched prefill. The coop GEMM folds the per-sub-block `min`
    /// into each staged weight element (`W = scale·nib + min`), so its f32
    /// accumulation order differs from the GEMV oracle's `scale·Σ(nib·a) + min·Σa`.
    /// That reassociation is benign for normal activations but, on Gemma-4's
    /// massive-activation sink channels, compounds over layers and flips tokens
    /// (the batched-gemma4 garbage). Set for
    /// Gemma models; the coop path stays the throughput win for Llama/Qwen.
    pub q4k_prefer_gemv: bool,
    pub embed: ComputeKernel,
    /// Embed with the source row read from a resident buffer (on-GPU token
    /// feedback in the decode loop) rather than a uniform.
    pub embed_tok: ComputeKernel,
    /// bf16-table variants of `embed`/`embed_tok`: the token-embedding table is
    /// stored bf16 (native checkpoint dtype) so a large vocab×hidden table fits
    /// Metal's ~4 GB single-buffer-binding limit (Gemma 4: 262144×5376 = 5.6 GB f32,
    /// 2.8 GB bf16). Used ONLY for the token-table gather; the f32 `embed` kernel
    /// still serves the hidden-state gathers (which read f32 scratch).
    pub embed_bf16: ComputeKernel,
    pub embed_tok_bf16: ComputeKernel,
    pub embed_q4k: ComputeKernel,
    pub embed_tok_q4k: ComputeKernel,
    pub rmsnorm: ComputeKernel,
    pub matmul: ComputeKernel,
    /// Cooperative-matrix tiled GEMM (Apple `simdgroup_matrix` / tensor cores) for
    /// the batched/prefill bf16 path (m ≥ 8). `Some` only when the device advertises
    /// `EXPERIMENTAL_COOPERATIVE_MATRIX`; the dispatch falls back to the scalar
    /// `matmul`/`matmul_vec_batch` otherwise. Measured ~3.5× the scalar tiled GEMM.
    pub matmul_coop: Option<ComputeKernel>,
    /// Cooperative-matrix GEMM, M register-blocked RB=2 (bf16): one workgroup does
    /// two stacked 8×8 tiles, staging the weight once for both — used for m ≥ 16 to
    /// raise the batched plateau (~+15%). `Some` only with the feature.
    pub matmul_coop_rb2: Option<ComputeKernel>,
    /// RB=2×CB=4 bf16 coop GEMM (m ≥ 16): a 16×32 tile, 8 mma/barrier. `Some` w/ feature.
    pub matmul_coop_rb2_cb4: Option<ComputeKernel>,
    /// RB=2×CB=4×DK=2 bf16 coop GEMM (m ≥ 16): the CB=4 tile with two 8-deep K-slices
    /// per round (16 mma / barrier, same 8 accumulators). Default over CB=4; ARF_DK=0
    /// forces plain CB=4. Same grid as CB=4. `Some` with the feature.
    pub matmul_coop_rb2_cb4_dk2: Option<ComputeKernel>,
    /// Cooperative-matrix GEMM for int8 weights (m ≥ 8): the int8 analog of
    /// `matmul_coop`. `Some` only when the device has the feature.
    pub matmul_coop_q8: Option<ComputeKernel>,
    /// RB=2 register-blocked int8 coop GEMM (m ≥ 16): weight tile staged once for two
    /// stacked output tiles. `Some` only with the feature.
    pub matmul_coop_q8_rb2: Option<ComputeKernel>,
    /// RB=2×CB=4 int8 coop GEMM (m ≥ 16): a 16×32 tile, 8 mma/barrier. `Some` w/ feature.
    pub matmul_coop_q8_rb2_cb4: Option<ComputeKernel>,
    /// RB=2×CB=4×DK=2 int8 coop GEMM (m ≥ 16): deeper-K twin of CB=4 (16 mma / barrier,
    /// same 8 accumulators). Default over CB=4; ARF_DK=0 forces plain CB=4. `Some` w/ feature.
    pub matmul_coop_q8_rb2_cb4_dk2: Option<ComputeKernel>,
    /// Cooperative-matrix GEMM for Q4_0 (block-32 4-bit) weights (m ≥ 8): unpacks +
    /// dequantizes into f32 staging before the mma. `Some` only with the feature.
    pub matmul_coop_q4: Option<ComputeKernel>,
    /// RB=2 register-blocked Q4_0 coop GEMM (m ≥ 16): unpack+dequant shared across two
    /// stacked output tiles. `Some` only with the feature.
    pub matmul_coop_q4_rb2: Option<ComputeKernel>,
    /// RB=2×CB=4 Q4_0 coop GEMM (m ≥ 16): a 16×32 tile, 8 mma/barrier. `Some` w/ feature.
    pub matmul_coop_q4_rb2_cb4: Option<ComputeKernel>,
    /// RB=2×CB=4×DK=2 Q4_0 coop GEMM (m ≥ 16): deeper-K twin of CB=4 (16 mma / barrier,
    /// same 8 accumulators). Default over CB=4; ARF_DK=0 forces plain CB=4. `Some` w/ feature.
    pub matmul_coop_q4_rb2_cb4_dk2: Option<ComputeKernel>,
    /// Cooperative-matrix GEMM for Q4_K-lite weights (m ≥ 8): unpacks the unsigned
    /// nibble + folds per-sub-block scale·nib+min into f32 staging. `Some` w/ feature.
    pub matmul_coop_q4k: Option<ComputeKernel>,
    /// Cooperative-matrix tiled GEMM for Q4_K_S dense weights (q/k/v/o/lm_head) at m ≥ 16.
    /// Stages the weight tile in threadgroup mem, reused across all B rows — the batched
    /// concurrency lever for the model's actual quant (the per-row GEMV `matmul_vec_q4ks_batch`
    /// re-reads weights per row). `Some` only with the coop feature. The b32 advantage for Q4KS.
    pub matmul_coop_q4ks: Option<ComputeKernel>,
    /// RB=2 register-blocked Q4_K-lite coop GEMM (m ≥ 16). `Some` only with the feature.
    pub matmul_coop_q4k_rb2: Option<ComputeKernel>,
    /// RB=2×CB=2 register-blocked Q4_K-lite coop GEMM (m ≥ 16): a 16×16 tile with
    /// 4 mma/barrier (vs 2 in `_rb2`), lifting the matrix-unit occupancy that capped
    /// batched decode at ~15% of roofline. `Some` only with the cooperative feature.
    pub matmul_coop_q4k_rb2_cb2: Option<ComputeKernel>,
    /// RB=2×CB=8 Q4_K-lite coop GEMM (m ≥ 16): a 16×64 tile, 16 mma/barrier. Opt-in
    /// via `ARF_CB=8` (register-pressure A/B candidate). `Some` w/ the feature.
    pub matmul_coop_q4k_rb2_cb8: Option<ComputeKernel>,
    /// RB=2×CB=4×DK=2 Q4_K-lite coop GEMM (m ≥ 16): the CB=4 16×32 tile but with
    /// two 8-deep K-slices per round (16 mma between barriers, same 8 accumulators)
    /// — halves the barrier count without CB=8's spill. Same grid as CB=4. Opt-in
    /// via `ARF_DK=2` (A/B candidate). `Some` with the coop feature.
    pub matmul_coop_q4k_rb2_cb4_dk2: Option<ComputeKernel>,
    /// RB=2×CB=4 register-blocked Q4_K-lite coop GEMM (m ≥ 16): a 16×32 tile with
    /// 8 mma/barrier, halving the barrier count again vs CB=2. `Some` w/ the feature.
    pub matmul_coop_q4k_rb2_cb4: Option<ComputeKernel>,
    /// Matrix·vector path for decode (m == 1); full-occupancy, weight-bandwidth.
    pub matmul_vec: ComputeKernel,
    /// m==1 GEMV with a subgroup reduction (Some only when the device has SUBGROUP).
    pub matmul_vec_sg: Option<ComputeKernel>,
    /// Batched matrix·vector for small m (continuous-decode batch); streams each
    /// weight row once and applies it to all m activation rows.
    pub matmul_vec_batch: ComputeKernel,
    /// int8 matrix·vector path for decode (m == 1); a quarter of f32 bytes.
    pub matmul_vec_q8: ComputeKernel,
    /// int8 m==1 GEMV with a subgroup reduction (Some only when device has SUBGROUP).
    pub matmul_vec_q8_sg: Option<ComputeKernel>,
    /// Batched int8 matrix·vector (small m): int8 bandwidth + batched amortization.
    pub matmul_vec_q8_batch: ComputeKernel,
    /// Q4_0 block-32 4-bit GEMV (m == 1): half the int8 bytes on the big matmuls.
    pub matmul_vec_q4: ComputeKernel,
    /// Multi-COLUMN Q4_0 GEMV (4 cols/workgroup): occupancy lever for the medium-n
    /// FFN matmuls (n=15360) that under-fill the GPU with 1-col-per-workgroup.
    /// Opt-in via ARF_Q4_MC=1 (default-off until the A/B win is confirmed).
    pub matmul_vec_q4_mc: ComputeKernel,
    /// Batched Q4_0 GEMV (small m): Q4 bytes + read-each-block-once amortization.
    /// Unblocks Q4 batched throughput and the spec-decode verify.
    pub matmul_vec_q4_batch: ComputeKernel,
    /// Q4_K-lite GEMV (m == 1): super-block-256, asymmetric per-32 scale+min —
    /// ollama-Q4_K accuracy at ~Q4_0 speed.
    pub matmul_vec_q4k: ComputeKernel,
    /// Q4_K-lite GEMV with a subgroup reduction (Some only when the device granted
    /// SUBGROUP); the dispatch falls back to `matmul_vec_q4k` otherwise.
    pub matmul_vec_q4k_sg: Option<ComputeKernel>,
    /// Q4_K-lite GEMV, ggml-style: each 32-lane SIMDGROUP produces NR0 output columns
    /// (activation register-resident, reused across columns), NSG simdgroups/workgroup,
    /// `subgroupAdd` reduce. Microbench-measured +15% on q_proj-class shapes (n=4096)
    /// vs the tree-reduce `matmul_vec_q4k`. `Some` only when the device granted
    /// SUBGROUP. The decode m=1 Q4K arm prefers it (the single-stream GEMV win).
    pub matmul_vec_q4k_ggml: Option<ComputeKernel>,
    /// Q4_K-lite GEMV, MULTI-ROW (lever B): one workgroup computes 4 output columns,
    /// loading each activation sub-block once and reusing it across the 4 weight rows
    /// (≈4× less activation traffic). `Some` only when `ARF_MR=1`; defaults off
    /// until its speed win is verified on a quiet machine. Tree reduction (no simd).
    pub matmul_vec_q4k_mr: Option<ComputeKernel>,
    /// Batched Q4_K-lite GEMV (small m) — Q4_K prefill.
    pub matmul_vec_q4k_batch: ComputeKernel,
    /// Q4_K_S GEMV (m == 1): u8/i8 two-level scales — ~5% fewer bytes than Q4_K.
    pub matmul_vec_q4ks: ComputeKernel,
    /// Batched Q4_K_S GEMV (small m) — Q4_K_S prefill.
    pub matmul_vec_q4ks_batch: ComputeKernel,
    /// Q3_K GEMV (m == 1): 3-bit split-plane — ~25% fewer code bytes than Q4.
    pub matmul_vec_q3k: ComputeKernel,
    /// Batched Q3_K GEMV (small m) — Q3_K prefill.
    pub matmul_vec_q3k_batch: ComputeKernel,
    pub rope: ComputeKernel,
    /// INTERLEAVED (ggml NORM) rope twin of `rope_qk`, for archs whose GGUF stores q/k in
    /// interleaved order (muse-glimmer). Selected by `ModelConfig::rope_interleaved()`.
    pub rope_qk_interleaved: ComputeKernel,
    /// Fused RoPE over q and k in one dispatch (replaces two `rope` dispatches).
    pub rope_qk: ComputeKernel,
    /// Per-head RMSNorm on q,k before RoPE (Qwen3/Gemma). Only dispatched when the
    /// layer carries q/k norm weights; Llama skips it.
    pub qk_norm: ComputeKernel,
    /// Per-head WEIGHTLESS RMSNorm on V (Gemma 4 normalizes the value projection per
    /// head, no learned gain). Dispatched after v_proj, before attention; gated on
    /// `cfg.value_norm`.
    pub v_norm: ComputeKernel,
    /// Fused per-head RMSNorm on q+k+v in ONE dispatch (Gemma 4: q/k weighted, v
    /// weightless) — replaces the qk_norm + v_norm pair to cut a decode dispatch.
    pub qkv_norm: ComputeKernel,
    /// MoE router top-k selection (softmax + top_k → ids/weights). Qwen3 MoE only.
    pub moe_route: ComputeKernel,
    /// MoE expert GEMV (bf16): one expert selected on-GPU from a packed buffer.
    pub matmul_vec_moe: ComputeKernel,
    /// MoE expert GEMV (Q4_0): the quantized variant — fits + runs the 30B.
    pub matmul_vec_moe_q4: ComputeKernel,
    /// MoE expert GEMV (Q4_K-lite): keeps the GGUF disk accuracy class (per-32
    /// asymmetric scale+min), still fits the 30B.
    pub matmul_vec_moe_q4k: ComputeKernel,
    /// MoE expert GEMV (Q4_K_S): two-level u8/i8 scales — ~16% fewer scale bytes
    /// than Q4_K-lite. Used for the per-expert shared-MLP path (non-batched).
    pub matmul_vec_moe_q4ks: ComputeKernel,
    /// Fused MoE gate/up: all top_k routed experts' projection in ONE dispatch
    /// (output [top_k, mi]), per quant format. Removes the per-expert serialization.
    pub matmul_vec_moe_batch_gu: ComputeKernel,
    pub matmul_vec_moe_q4_batch_gu: ComputeKernel,
    pub matmul_vec_moe_q4k_batch_gu: ComputeKernel,
    /// Fused Q4_K_S gate/up (tree reduction; no subgroup variant — additive path).
    pub matmul_vec_moe_q4ks_batch_gu: ComputeKernel,
    /// Fused MoE down: weight-sums all top_k experts into mlp_down in ONE dispatch
    /// (reduces over experts inside the kernel), per quant format.
    pub matmul_vec_moe_down_reduce: ComputeKernel,
    pub matmul_vec_moe_q4_down_reduce: ComputeKernel,
    pub matmul_vec_moe_q4k_down_reduce: ComputeKernel,
    /// Fused Q4_K_S down (tree reduction, WG=32; no subgroup variant — additive).
    pub matmul_vec_moe_q4ks_down_reduce: ComputeKernel,
    /// BATCHED (B rows) MoE router top-k: one workgroup per batch row, B dispatched.
    pub moe_route_b: ComputeKernel,
    /// BATCHED (B rows) MoE gate/up: output [B, top_k, n] (row,slot,col)-major.
    /// bf16 + Q4_K_S only (the batch-tested + 30B paths); other quants fall back to
    /// the per-row loop in the `GpuMlp::Moe` arm.
    pub matmul_vec_moe_batch_gu_b: ComputeKernel,
    pub matmul_vec_moe_q4ks_batch_gu_b: ComputeKernel,
    /// v2 nr0/NSG-layout variant of the batched Q4_K_S gate/up GEMV (GROUPS=2
    /// reducer groups × NR0 columns/group, 32-lane reduces). Gated default-off by
    /// ARF_MOE_GU_V2 — byte-identical arithmetic to the v1 kernel above.
    pub matmul_vec_moe_q4ks_batch_gu_b_v2: ComputeKernel,
    /// BATCHED (B rows) fused MoE down: weight-sums each row's top_k experts into
    /// `mlp_down[B*h]`, one workgroup per (row, h_j). bf16 + Q4_K_S only.
    pub matmul_vec_moe_down_reduce_b: ComputeKernel,
    pub matmul_vec_moe_q4ks_down_reduce_b: ComputeKernel,
    pub kv_scatter: ComputeKernel,
    /// TurboQuant pack-on-write scatter (normalize + Hadamard + quantize + pack).
    /// Dispatched instead of `kv_scatter` when the KV pool is `KvQuant::Tq`.
    pub kv_scatter_tq: ComputeKernel,
    /// **M11** Qwen3.6 gated-delta-net (linear-attention) kernels — WGSL twins of the Metal-island
    /// MSL kernels, so the whole recurrence records onto the wgpu CommandPass (single queue, no
    /// cross-queue wait). Bit-exact ports; Paris-gated. Only built on the qwen35 path.
    pub gdn_conv1d: ComputeKernel,
    pub gdn_l2_norm: ComputeKernel,
    /// Muse Glimmer attention-output gate: `attn *= sigmoid(gate)` before o_proj.
    pub attn_gate_mul: ComputeKernel,
    pub gdn_sigmoid_out: ComputeKernel,
    pub gdn_g_decay: ComputeKernel,
    pub gdn_tile: ComputeKernel,
    pub gdn_recurrence: ComputeKernel,
    /// **B-PARALLEL (concurrency)** gated-delta-net twins — decode B sequences in lockstep. Each
    /// carries its own O(1) ssm_state/conv_state (b-outer layout); the static weights are shared.
    /// b=0 slice is byte-identical to the single-stream kernel (the "B=1 ≡ M11" parity gate). The
    /// O(1) state is the structural win at conc8-32 where llama's recurrence stalls.
    pub gdn_conv1d_b: ComputeKernel,
    pub gdn_l2_norm_b: ComputeKernel,
    pub gdn_sigmoid_out_b: ComputeKernel,
    pub gdn_g_decay_b: ComputeKernel,
    pub gdn_tile_b: ComputeKernel,
    pub gdn_recurrence_b: ComputeKernel,
    pub attention: ComputeKernel,
    /// Batched (ragged) attention: all sequences in the continuous-batch step are
    /// processed in ONE dispatch, keyed by per-row `(slot_base, last)` metadata.
    /// Replaces the per-sequence loop over `attention` in `forward_batch` — the
    /// lever that was capping concurrent-decode throughput.
    pub attention_batched: ComputeKernel,
    /// GQA-grouped batched attention: one workgroup per (kv_head, row) serves all
    /// `group` query heads sharing that KV head, reading each K/V slot ONCE instead
    /// of `group` times. Dispatched instead of `attention_batched` when group>1 and
    /// head_dim/group are within the kernel's shared-memory budget.
    pub attention_gqa: ComputeKernel,
    /// TurboQuant fused attention (inline dequant + rotate-q + Rᵀ-output).
    /// Dispatched instead of `attention` when the KV pool is `KvQuant::Tq`.
    pub attention_tq: ComputeKernel,
    /// Split-KV (flash-decoding) attention: two-pass split+combine, dispatched
    /// instead of `attention` when ctx is long enough to risk the per-dispatch GPU
    /// watchdog (f32 KV path). `attention_splitk` writes partials, the combine
    /// kernel merges them.
    pub attention_splitk: ComputeKernel,
    pub attention_splitk_combine: ComputeKernel,
    pub swiglu: ComputeKernel,
    /// GeGLU (`gelu_pytorch_tanh` gate) — dispatched instead of `swiglu` when the
    /// model's `gate_act` is `GeluTanh` (Gemma). Same bindings as `swiglu`.
    pub geglu: ComputeKernel,
    pub add: ComputeKernel,
    /// Fused residual-add + RMSNorm (replaces an `add` then the following norm).
    pub add_norm: ComputeKernel,
    /// Fused RMSNorm + residual-add (Gemma post-norm: normalize a sub-block output,
    /// then add it back — replaces a `rmsnorm` then an `add`).
    pub rmsnorm_add: ComputeKernel,
    pub sample: ComputeKernel,
    /// BATCHED greedy argmax: one workgroup per output row → out`row` token id, in ONE dispatch.
    /// Used by the serving `step()` override to sample on-GPU and read back only B token-ids
    /// instead of B×vocab logits — removing the per-token blocking full-vocab readback stall.
    pub sample_batched: ComputeKernel,
    /// Stochastic sampling (temperature + top-k + top-p + deterministic RNG):
    /// reduces `logits[vocab]` to one sampled token id in `next_token`, fully on
    /// the GPU (no logits readback). Dispatched in place of `sample` when
    /// `temperature > 0`; greedy (`temperature == 0`) keeps the `sample` fast path.
    pub sample_stochastic: ComputeKernel,
    /// Final-logit soft-capping `logit = cap·tanh(logit/cap)`, applied in place to
    /// the lm_head output before sampling. Dispatched only when
    /// `ModelConfig.final_logit_softcap` is `Some` (Gemma 4); a no-op otherwise.
    pub softcap: ComputeKernel,
    /// In-place elementwise scalar multiply — Gemma 4's per-layer `layer_scalar`
    /// applied to the whole layer output. Only dispatched when a layer has one.
    pub scale_inplace: ComputeKernel,
}

/// Per-layer resident KV pools.
///
/// For `KvQuant::None` (the default), `keys`/`values` are f32 buffers of
/// `[num_blocks*block_size, kv_heads*head_dim]` — the exact path. For
/// `KvQuant::Tq`, `keys`/`values` instead hold the **packed `bits`-wide codes**
/// (a u32 bitstream over `[num_blocks*block_size*kv_heads, head_dim]` coords) and
/// `key_norms`/`value_norms` hold one f32 norm per head-vector; the codebook
/// `levels` live on [`GpuModel::kv_levels`]. (A `wgpu::Buffer` is untyped bytes,
/// so the same `keys` slot is bound as `array<f32>` by `attention` or `array<u32>`
/// by `attention_tq` — the shader picks the interpretation.)
pub struct GpuKvPool {
    pub keys: Vec<wgpu::Buffer>,
    pub values: Vec<wgpu::Buffer>,
    /// Per-layer f32 norms (one per head-vector); empty unless `KvQuant::Tq`.
    pub key_norms: Vec<wgpu::Buffer>,
    pub value_norms: Vec<wgpu::Buffer>,
    /// Raw-Metal views of `keys`/`values` (megakernel S1c; macOS island only, KvQuant::None).
    /// Empty when the island is off or the pool is quantized. Same memory as the wgpu views.
    #[cfg(target_os = "macos")]
    pub keys_mtl: Vec<Option<crate::gpu::concurrent_metal::MtlBuf>>,
    #[cfg(target_os = "macos")]
    pub values_mtl: Vec<Option<crate::gpu::concurrent_metal::MtlBuf>>,
    /// PARALLEL f16 KV pool (the 3rd leg: bandwidth). Allocated ONLY when ARF_KV_F16
    /// is set (row_floats*2 bytes/row = HALF the f32 pool). Orthogonal to kv_quant (kv_quant stays
    /// None — f16 is a separate pool FORMAT, not a lossy KvQuant). Empty (all-None) unless the flag
    /// is on; the f32 pool above stays the default + correctness oracle and is untouched.
    #[cfg(target_os = "macos")]
    pub keys_mtl_f16: Vec<Option<crate::gpu::concurrent_metal::MtlBuf>>,
    #[cfg(target_os = "macos")]
    pub values_mtl_f16: Vec<Option<crate::gpu::concurrent_metal::MtlBuf>>,
    /// 8-BIT KV (`kv_q8_for`, 2026-09-23): per attention layer `[k8, ks, v8, vs]` — int8
    /// `[slot][kv_heads*hd]` and f32 scales `[slot][kv_heads]`. When present the f32 pool above is
    /// a 16-byte placeholder: this IS the cache (4x the context in the same memory).
    #[cfg(target_os = "macos")]
    pub q8: Vec<Option<[crate::gpu::concurrent_metal::MtlBuf; 4]>>,
    /// When the `q8` buffers are placement-sparse: the mapper that backs them with memory as the
    /// highest written slot climbs (`SparseKv::ensure`, called by the batched step). `None` =
    /// ordinary fully-committed buffers.
    #[cfg(target_os = "macos")]
    pub q8_sparse: Option<std::sync::Mutex<crate::gpu::metal::sparse_kv::SparseKv>>,
    /// The MTP head's own K/V: one attention layer's worth, f32, indexed by the same slot ids as
    /// the trunk. blk.64 is an attention block OUTSIDE the trunk schedule — it attends over its
    /// own history and must never scatter into a trunk layer's pool. `None` without a head.
    #[cfg(target_os = "macos")]
    pub mtp_keys_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    #[cfg(target_os = "macos")]
    pub mtp_values_mtl: Option<crate::gpu::concurrent_metal::MtlBuf>,
    pub block_size: usize,
    pub num_blocks: usize,
    pub kv_quant: arf_core::config::KvQuant,
}

/// A Llama model resident on the GPU: every weight, the paged KV pool, the RoPE
/// tables, and reused scratch live in device buffers for the model's lifetime.
/// Each token's forward pass is recorded as a single [`CommandPass`] and reads
/// back one token id.
///
/// Built via [`crate::weights::build_gpu`]. The CPU
/// [`arf_core::model::Llama`] is the correctness oracle.
/// The four per-token decode arenas, allocated ONCE (sized for the max context)
/// and `reset()` at the start of each step instead of recreated. Recreating them
/// per token meant four `wgpu::Buffer` allocations every token — pure CPU/driver
/// overhead on the critical path. `RefCell` because `decode_token` takes `&self`.
///
/// Decision: hoist-and-reset over a fresh arena per token — tradeoff: the
/// arenas hold max-context memory for the model's life vs. 4 device allocations
/// every decode step. Decode was ~36% CPU-side after the readback was killed; the
/// allocations were a chunk of it.
pub struct DecodeArenas {
    pub idx: GpuArena,
    pub out: GpuArena,
    pub uni: GpuArena,
}

impl DecodeArenas {
    /// Allocate the decode arenas sized for the worst case (`max_ctx` context).
    /// `decode_token` resets and re-sub-allocates them each step. (The old
    /// k_all/v_all `kv` arena is gone: attention reads the paged KV pool directly
    /// by slot, so no contiguous gather scratch is needed.)
    pub fn new(ctx: &Arc<GpuContext>, cfg: &arf_core::config::ModelConfig, max_ctx: usize) -> Self {
        let f32b = std::mem::size_of::<f32>() as u64;
        let u32b = std::mem::size_of::<u32>() as u64;
        // q_dim is per-layer for Gemma 4 (global layers have a larger head_dim); the
        // `attn` region (which holds the attention output before o_proj) must fit the
        // LARGEST layer's q_dim. `max_attn_dims().0` is the max head_dim over layers.
        // L154 — `max_q_dim()` also covers the hybrid SSM family's 2×-packed attention q, which
        // `num_attention_heads * max_attn_dims().0` misses by a factor of two.
        let q_dim = cfg.max_q_dim();
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        // The `silu` region holds the SwiGLU output: [intermediate_size] for dense,
        // but [top_k * moe_inter] for the FUSED MoE path (all routed experts' silu
        // at once). Size the `out` arena for the larger so the fused down's
        // silu_all read/write stays in bounds.
        let silu_len = match &cfg.mlp {
            arf_core::config::MlpKind::Moe {
                top_k,
                moe_intermediate,
                ..
            } => (top_k * moe_intermediate).max(cfg.intermediate_size),
            arf_core::config::MlpKind::Dense => cfg.intermediate_size,
        };
        // Per-layer uniform writes: ~16 for the attention+dense-MLP dispatches, +1
        // for the qk_norm dispatch (Qwen3/Gemma). MoE replaces the 3 dense-MLP
        // uniforms with router(1) + route(1) + 4 per routed/shared expert
        // (gate+up+swiglu+down), so budget that explicitly.
        let mlp_uni = match &cfg.mlp {
            arf_core::config::MlpKind::Moe {
                top_k,
                shared_experts,
                ..
            } => 2 + 4 * (*top_k as u64 + *shared_experts as u64),
            arf_core::config::MlpKind::Dense => 3,
        };
        let per_layer = 14
            + if cfg.qk_norm { 1 } else { 0 }
            + if cfg.value_norm { 1 } else { 0 } // Gemma 4 weightless V-norm
            + mlp_uni;
        let uni_count = 1 + per_layer * cfg.num_layers as u64 + 2;
        DecodeArenas {
            // The `idx` arena holds, per token: pos_buf (1 elem) + new_slot (1 elem) +
            // slots (ctx_len elems, GROWS to max_ctx). Each alloc rounds UP to a whole
            // ALIGN unit and needs a CONTIGUOUS free run, so the budget must cover
            // `slots` rounded up to a unit PLUS a full unit each for pos_buf/new_slot.
            // The old `(max_ctx+2)*4 + 4*ALIGN` under-counted that per-alloc rounding
            // and OOM'd the slots alloc near max_ctx (ctx ~8192 → 129-unit slots vs a
            // 128-unit free run). Round slots to a whole unit explicitly, then add
            // 8 full units of headroom for pos_buf/new_slot + safety.
            idx: GpuArena::new(
                ctx,
                "decode-idx",
                ((max_ctx as u64 + 1) * u32b).div_ceil(ALIGN) * ALIGN + 8 * ALIGN,
                storage_usage,
            )
            .expect("decode idx arena"),
            // The `out` arena holds attn (q_dim) + silu, and — at long ctx — the
            // split-KV `partials` buffer (nh · SPLITK · (max_hd+2) floats). SPLITK=8
            // matches decode.rs; size for it so the long-ctx alloc fits.
            out: GpuArena::new(
                ctx,
                "decode-out",
                arena_bytes(&[
                    q_dim as u64 * f32b,
                    silu_len as u64 * f32b,
                    cfg.num_attention_heads as u64 * 8 * (cfg.max_attn_dims().0 as u64 + 2) * f32b,
                ]),
                storage_usage,
            )
            .expect("decode out arena"),
            uni: GpuArena::new(
                ctx,
                "decode-uniforms",
                (uni_count + 4) * ALIGN,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            )
            .expect("decode uniform arena"),
        }
    }

    /// Free all sub-allocations — the "new generation" reset at the start of a step.
    pub(crate) fn reset(&mut self) {
        self.idx.reset();
        self.out.reset();
        self.uni.reset();
    }
}

/// Lazily-grown, reused arena cache for `forward_batch` (batched prefill + decode).
///
/// `forward_batch` used to `GpuArena::new` ~11 trunk arenas plus a fresh per-seq
/// idx/kv arena for every (layer, sequence) — `13 + 2·nseq·num_layers` device
/// buffer allocations EVERY call, which dominated continuous-batched decode (it's
/// called once per step). This cache holds each arena across calls: a slot grows
/// (reallocates) only when a call needs more capacity than it currently holds,
/// otherwise it's `reset()` and reused. `forward_batch`'s `total` varies (prefill
/// = prompt_len, decode = nseq), so the capacity ratchets to the high-water mark
/// and then every later call is pure reset — zero allocations on the decode path.
///
/// Decision: lazily-grown (vs. pre-sized-for-max like [`DecodeArenas`]) —
/// tradeoff: a capacity check per slot per call vs. guessing a max batch/context
/// up front and over-committing GPU memory. The per-seq scratch is kept as DISTINCT
/// buffers per sequence (a `Vec` of pairs), not one shared buffer: sharing one
/// kv buffer across seqs within a single command pass would re-introduce the
/// read+read_write aliasing the per-buffer layout was built to avoid.
pub struct BatchArenas {
    pub(crate) hidden: Option<GpuArena>,
    pub(crate) normed: Option<GpuArena>,
    pub(crate) tmp: Option<GpuArena>,
    pub(crate) attn: Option<GpuArena>,
    pub(crate) q: Option<GpuArena>,
    pub(crate) k: Option<GpuArena>,
    pub(crate) v: Option<GpuArena>,
    pub(crate) gate: Option<GpuArena>,
    pub(crate) up: Option<GpuArena>,
    pub(crate) pos: Option<GpuArena>,
    pub(crate) uni: Option<GpuArena>,
    pub(crate) logits: Option<GpuArena>,
    /// BATCHED MoE scratch arena "B": the routed ids/wts, the swiglu output, and the
    /// constant shared ids/wts ([B·top_k] + [B·top_k·mi] + [B·shared]). Paired with
    /// `moe_gu` ("A": logits + gate_all + up_all) so that NO single dispatch ever
    /// binds the same physical buffer as both read and read_write (wgpu rejects that
    /// even at disjoint offsets) — the route reads logits(A)/writes ids,wts(B); the
    /// swiglu reads gate_all,up_all(A)/writes silu(B); the down reads silu,ids,wts(B)/
    /// writes the trunk's tmp_h. Only sized for MoE models; `None` for dense ones.
    pub(crate) moe_silu: Option<GpuArena>,
    /// BATCHED MoE scratch arena "A": router logits + the fused gate/up outputs
    /// ([B·num_experts] + 2×[B·top_k·mi]). See `moe_silu` for the A/B split rationale.
    pub(crate) moe_gu: Option<GpuArena>,
    /// One (idx, kv) pair per sequence slot — distinct buffers, so seq i never
    /// aliases seq j within a command pass. Grown to the max nseq seen; each pair
    /// sized to the high-water ctx_len.
    pub(crate) seq: Vec<(GpuArena, GpuArena)>,
    /// Batched-attention metadata, built once per `forward_batch` and shared by
    /// every layer's single batched scatter + attention dispatch. Holds three
    /// regions: the flattened per-seq `slots` tables, the per-row `row_meta`
    /// (slot_base, last), and the per-row `write_slots` (scatter destinations).
    pub(crate) attn_meta: Option<GpuArena>,
}

impl BatchArenas {
    /// All-empty; nothing is allocated until the first `forward_batch` sizes it.
    pub fn new() -> Self {
        BatchArenas {
            hidden: None,
            normed: None,
            tmp: None,
            attn: None,
            q: None,
            k: None,
            v: None,
            gate: None,
            up: None,
            pos: None,
            uni: None,
            logits: None,
            moe_silu: None,
            moe_gu: None,
            seq: Vec::new(),
            attn_meta: None,
        }
    }

    /// Ensure `slot` holds an arena of at least `need` bytes (with `usage`), then
    /// return it freshly `reset()` (empty free-list, ready to `alloc`). Grows
    /// (reallocates) only when the current capacity is too small; `need` must
    /// already include whatever slack/`arena_bytes` wrapping the call site wants.
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<GpuArena>,
        ctx: &Arc<GpuContext>,
        label: &str,
        need: u64,
        usage: wgpu::BufferUsages,
    ) -> &'a mut GpuArena {
        let grow = match slot {
            // `capacity()` is the pow2/ALIGN-rounded backing size actually seatable.
            Some(a) => a.capacity() < need,
            None => true,
        };
        if grow {
            *slot = None; // drop the old buffer before allocating the larger one
            *slot = Some(GpuArena::new(ctx, label, need, usage).expect("batch arena"));
        } else {
            slot.as_mut().unwrap().reset();
        }
        slot.as_mut().unwrap()
    }
}

impl Default for BatchArenas {
    fn default() -> Self {
        Self::new()
    }
}

/// The ×B regions for the BATCHED MoE block, all sub-allocated from one arena and
/// simultaneously live for a `forward_batch_impl` call. The batched `GpuMlp::Moe`
/// arm routes & runs all `total` rows in one block (router GEMV → batched route →
/// batched gate/up → swiglu → batched down) instead of the old per-row loop, so
/// every intermediate is sized ×B: `logits` [B·num_experts], `ids`/`wts` [B·top_k],
/// `gate_all`/`up_all`/`silu` [B·top_k·mi]. `sh_ids`/`sh_wts` are the constant
/// always-on shared-expert routing ([0,1,..,sh-1] per row, weight 1.0), written once.
pub(crate) struct MoeBatchScratch {
    pub(crate) ne: usize, // num_experts
    pub(crate) tk: usize, // top_k
    pub(crate) mi: usize, // moe_intermediate
    pub(crate) logits: Region,
    pub(crate) ids: Region,
    pub(crate) wts: Region,
    pub(crate) gate_all: Region,
    pub(crate) up_all: Region,
    pub(crate) silu: Region,
    pub(crate) sh_ids: Region,
    pub(crate) sh_wts: Region,
}

/// On-GPU stochastic sampling spec: the temperature/top-k/top-p/seed knobs that
/// drive `sample_stochastic.wgsl`. Greedy decode (temperature == 0) uses the
/// `sample` argmax fast path and ignores this entirely — see
/// [`GpuSampling::from_params`], which returns `None` for greedy.
///
/// Derived once per request from [`arf_core::sampling::SamplingParams`]; the
/// per-step `position` (which feeds the RNG counter) is supplied at dispatch, so
/// this struct is `Copy` and is carried unchanged through the whole decode loop.
#[derive(Debug, Clone, Copy)]
pub struct GpuSampling {
    /// `1.0 / temperature` (temperature > 0 guaranteed; greedy never gets here).
    pub inv_temp: f32,
    /// Keep the `k` highest-logit tokens; `0` disables top-k (matches the shader).
    pub top_k: u32,
    /// Nucleus mass in `(0, 1]`; `>= 1.0` disables top-p (matches the shader).
    pub top_p: f32,
    /// RNG seed; combined with the decode position into the counter-based PCG.
    pub seed: u64,
}

impl GpuSampling {
    /// Build a GPU sampling spec from CPU [`SamplingParams`], or `None` when the
    /// request is greedy (`temperature <= 0`) — the caller then uses the `sample`
    /// argmax fast path, preserving zero-overhead greedy decode.
    ///
    /// [`SamplingParams`]: arf_core::sampling::SamplingParams
    pub fn from_params(p: &arf_core::sampling::SamplingParams) -> Option<Self> {
        if p.is_greedy() {
            return None;
        }
        Some(GpuSampling {
            inv_temp: 1.0 / p.temperature,
            // top_k == Some(0) is rejected by SamplingParams::validate; clamp
            // defensively so a stray 0 disables rather than keeping nothing.
            top_k: p.top_k.map(|k| k.max(1) as u32).unwrap_or(0),
            top_p: p.top_p.unwrap_or(1.0),
            seed: p.seed,
        })
    }

    /// Pack into the `sample_stochastic` `Dims` uniform: `[vocab, inv_temp_bits,
    /// top_k, top_p_bits, seed_lo, seed_hi, position, _pad]`. Floats travel as
    /// their bit patterns (the shader reads the fields as `f32`).
    pub(crate) fn dims(&self, vocab: usize, position: usize) -> [u32; 8] {
        [
            vocab as u32,
            self.inv_temp.to_bits(),
            self.top_k,
            self.top_p.to_bits(),
            self.seed as u32,
            (self.seed >> 32) as u32,
            position as u32,
            0,
        ]
    }
}
