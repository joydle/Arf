# GPU kernels — the shader map

This document maps every GPU shader in the engine: what it computes, whether it's
live in a serving path, and how the pieces fit into the two-tier kernel strategy.
It is generated from the shader headers and cross-checked against the Rust that
loads them; where a shader's status could not be established with confidence it is
marked **unverified** rather than guessed.

## Counts

| | WGSL | Metal (MSL) | total |
|---|---|---|---|
| Shader files | 121 | 26 | **147** |
| **LIVE** (in a serving/inference path) | 120 | 25 | **145** |
| **TEST/bench** (examples and tests only) | 1 | 1 | **2** |
| **DEAD** (no reference anywhere) | 0 | 0 | **0** |

> **WHAT THIS TABLE DOES NOT SAY (2026-09-19).** `LIVE` means a loader REFERENCES the file. It
> does not mean the kernel is dispatched BY DEFAULT, and one file holds several kernels: the
> shared-prefix verify kernel (opt-in; the record says it loses the prompt) and the column-major
> GEMV (a function constant, default off, measured 19% slower) both sit inside `LIVE` files. Which
> kernels run by default, which are opt-in, which lost, and WHY is scattered across code comments and 366 env levers against a guard
> ceiling of 218; an audit that puts it in one place is planned.

The two TEST shaders never compile into the engine's serving path. `matmul_vec_q4k_msl.metal` is the
superseded Q4_K-lite MSL GEMV, kept so `examples/gemv_msl_ab.rs` can still reproduce the
shootout it lost; `double_test.wgsl` (`out[i] = 2 * in[i]`) is the fixture the
GPU-foundation and `CommandPass` smoke tests in `tests/gpu.rs` dispatch. Three earlier WGSL
GEMV experiments and their harness (`matmul_vec_q4k_splitk.wgsl`,
`matmul_vec_q4k_splitk_reduce.wgsl`, `matmul_vec_q4k_unroll.wgsl`, `examples/gemv_q4k_bench.rs`)
were deleted in L47 (2026-08-06) as a dead bench.

## Two tiers: the portable path and the Metal island

Every shader belongs to one of two tiers.

**WGSL — the portable correctness path.** 121 shaders written in WebGPU Shading
Language, run through `wgpu`/`naga`. This is the path that works on any WebGPU
adapter (Apple, and Vulkan) and it is the
reference implementation of every operation. WGSL shaders are loaded by the
`kern!("name")` macro in `crates/arf-gpu/src/weights.rs` (which `include_str!`s
`shaders/<name>.wgsl`), or by a direct `include_str!` in the FLUX/vision modules.

**MSL — the hand-tuned Metal island.** 24 shaders written in raw Metal Shading
Language, compiled directly via `objc2-metal` and dispatched on a native Metal command
queue, bypassing `wgpu`. This "island" exists because `wgpu` can't express the two
things that get Apple Silicon to roofline: `simd_sum` one-op simdgroup reductions
(naga lowers these ~4× slower) and `simdgroup_matrix` tiled GEMM against
dequant-on-load quantized weights. The island is the **batched megakernel** — it
folds a whole decode frame (norms, RoPE, attention, MoE, sampling) into a small set
of native kernels on one command queue to cut per-token dispatch count and cross-queue
waits. It is loaded and driven from `crates/arf-gpu/src/gpu/metal/island.rs` (still addressed in code as
`concurrent_metal::`, through a re-export in `gpu/mod.rs`).
Several MSL files pack multiple named kernels (`decode_ops_msl.metal` alone holds 14),
so 24 files expose far more than 24 entry points.

Every MSL kernel has a WGSL (or CPU) twin it is asserted **bit-faithful** to. The
island is a performance specialization, never a divergent implementation.

## The parity rule

Every GPU kernel here — WGSL or MSL — is checked against a **CPU f32 oracle** of the
same math. This is Layer 1 of the correctness ladder and the contract we gate every
kernel change on: the CPU `Llama`/`Gemma`/MoE path is ground truth; the GPU quantized
path must match it within floating-point tolerance. Quantized GEMVs cite their exact
oracle in their header (e.g. `matmul_nt_q4ks`); MSL kernels cite the WGSL twin they
reproduce (bit-exact where reduction order is preserved, or "reassoc below a logit
tie, parity-gated" where `simd_sum` reorders the reduction). New kernels ship with
their parity test — it's a house rule. See [CONTRIBUTING.md](../CONTRIBUTING.md) (the
parity commands; the original `docs/CORRECTNESS_VERIFICATION.md` ladder was removed in the
doc cleanup, 2026-08-28) for the full ladder (Layer 0 determinism through Layer 5 perplexity, including the result
that the bf16 forward pass is bit-exact to HF `transformers` f32).

Two conventions recur in the WGSL headers and are worth naming once:

- **2D-safe flat index.** WebGPU caps a dispatch at 65535 workgroups per dimension.
  Elementwise kernels that can exceed it (vision MLP activations reach 17.6M elements)
  spread work over the Y grid and reconstruct the linear index — backward-compatible
  with the single-row text path.
- **256B storage-offset alignment.** macOS+`wgpu` hardcodes a 256-byte storage-buffer
  offset alignment, so kernels avoid binding sub-ranges and instead read by element
  offset into a whole buffer (see `adaln_modulate.wgsl`).

---

## Matmul / GEMV — the compute floor

Weight×activation is where the time goes. Decode (one token, `m==1`) is
**bandwidth-bound**: the cost is streaming the resident weight once, so these run at
weight-bandwidth, which is the decode roofline. Prefill and batched decode (`m>1`)
amortize each weight read across all rows. The quantized variants (Q4_0, Q4_K-lite,
Q4_K_S, Q3_K, int8) exist because fewer weight bytes = faster on a bandwidth-bound
kernel — Q4_K_S is the real gemma-4/coder-30B decode format. The `_batch` variants
hoist the nibble-unpack out of the row loop (the "batch CSE"). The `_sg` variants
swap the shared-memory reduction tree for a one-op `subgroupAdd`. All quantized GEMVs
are bit-identical to their `matmul_nt_*` CPU oracle.

| shader | kind | status | purpose |
|---|---|---|---|
| `matmul.wgsl` | WGSL | LIVE | Tiled 16×16 GEMM `C=A·Bᵀ`, the resident-weight Linear (prefill/batched f32/bf16). |
| `matmul_vec.wgsl` | WGSL | LIVE | bf16 decode GEMV (`m==1`); one workgroup per output column, tree reduce. |
| `matmul_vec_sg.wgsl` | WGSL | LIVE | `matmul_vec` with `subgroupAdd` reduction (when adapter has subgroups). |
| `matmul_vec_batch.wgsl` | WGSL | LIVE | Batched bf16 decode GEMV, one weight read reused across m=2..32 rows. |
| `matmul_vec_q8.wgsl` | WGSL | LIVE | int8 decode GEMV, per-row symmetric scale — narrowest resident-weight kernel. |
| `matmul_vec_q8_batch.wgsl` | WGSL | LIVE | Batched int8 GEMV (weight row read once for all m rows). |
| `matmul_vec_q8_sg.wgsl` | WGSL | LIVE | int8 decode GEMV, `subgroupAdd` reduction variant. |
| `matmul_vec_q4.wgsl` | WGSL | LIVE | Q4_0 (block-32) decode GEMV — half the int8 bytes, the ollama-parity precision class. |
| `matmul_vec_q4_batch.wgsl` | WGSL | LIVE | Batched Q4_0 GEMV; nibble-unpack hoisted out of the row loop. |
| `matmul_vec_q4_mc.wgsl` | WGSL | LIVE | Multi-column Q4_0 GEMV (4 cols/workgroup) — fills the GPU on medium-n FFN matmuls. |
| `matmul_vec_q4k.wgsl` | WGSL | LIVE | Q4_K-lite decode GEMV (per-32 asymmetric scale+min) — the Q4_K accuracy class. |
| `matmul_vec_q4k_batch.wgsl` | WGSL | LIVE | Batched Q4_K-lite GEMV (sub-block unpacked once, reused across m rows). |
| `matmul_vec_q4k_mr.wgsl` | WGSL | LIVE | Multi-row Q4_K-lite GEMV variant — same math, changed memory access pattern. |
| `matmul_vec_q4k_sg.wgsl` | WGSL | LIVE | Q4_K-lite GEMV with `subgroupAdd` + cross-subgroup combine. |
| `matmul_vec_q4k_ggml.wgsl` | WGSL | LIVE | Q4_K-lite GEMV, faithful WGSL port of ggml's `mul_mv_q4_K` simdgroup structure. |
| `matmul_vec_q4ks.wgsl` | WGSL | LIVE | Q4_K_S decode GEMV (two-level u8/i8 scales) — the real gemma-4-12B decode format. |
| `matmul_vec_q4ks_batch.wgsl` | WGSL | LIVE | Batched Q4_K_S GEMV (batch CSE), powers Q4_K_S prefill. |
| `matmul_vec_q3k.wgsl` | WGSL | LIVE | Q3_K (3-bit split-plane) decode GEMV — the bandwidth-floor breaker (~3.4 bits/weight). |
| `matmul_vec_q3k_batch.wgsl` | WGSL | LIVE | Batched Q3_K GEMV; decodes each sub-block's codes once, reuses across m rows. |
| `matmul_vec_q4k_msl.metal` | Metal | TEST | Hand-tuned MSL Q4_K-lite GEMV — superseded by the Q4_K_S island; kept in `gemv_msl_ab`. |
| `matmul_mpp_q4ks_msl.metal` | Metal | LIVE | Batched Q4 matmul via Metal 4 MPP `matmul2d`, reading the shipped row-major Q4KS through a strided view (no copy of the weights) + three f32→scaled-half passes. DEFAULT ON for 5..=16 live rows; `ARF_NO_MPP_Q4=1` opts out. |
| `dflash2_msl.metal` | Metal | GATED | The DFlash 2 draft's own kernels `dflash_tap` copies the target's residual stream after a tapped layer into its slot of `[row][slot][hidden]`, the input of the draft's `fc` (row count is the grid height, never a uniform; gate `dflash2_taps`). `dflash_rmsnorm` (plain-weight RMS norm, uniforms by `setBytes`) and `dflash_ctx_commit` (per-head k-norm + half-split RoPE at the logical position, K/V into the layer's 2048-slot ring, only the committed rows) build the draft's context; gate `dflash2_context`. The block forward (gates `spec_identity_check.py`, `dflash2_block -- … --ref`): `dflash_conv` (two-tap grouped dynamic conv, prepare / finish+residual), `dflash_head_norm_rope` (per-head norm + RoPE in place), `dflash_attn_scores` / `dflash_attn_softmax` / `dflash_attn_mix` (block queries over ring ++ block, not causal, one thread per output element — the one-thread-per-query first build is MEASURED-OUT in the shader), `dflash_swiglu`. Dispatched only after `dflash_attach` / `dflash_enable_taps` (a draft is loaded). |
| `matmul_vec_q4ks_msl.metal` | Metal | LIVE | Hand-tuned MSL Q4_K_S decode GEMV — 2 cols/simdgroup, `simd_sum`; the island decode floor. |
| `matmul_vec_q4ks_batch_msl.metal` | Metal | LIVE | B-row MSL Q4_K_S GEMV; unpack amortized across B rows (B=1 ≡ the m==1 kernel). |
| `matmul_vec_q3k_msl.metal` | Metal | LIVE | Hand-tuned MSL Q3_K GEMV for the island (2 cols/simdgroup, `simd_sum`). |
| `mtp_eh_proj_msl.metal` | Metal | LIVE | MTP ("nextn") input projection for the Qwen3.8 draft head: RMS-norm the trunk hidden and the emitted token's embedding, concatenate, project through `nextn.eh_proj` (L241; loaded in `weights.rs`). |

*(Three bench-only Q4_K GEMV experiments — `matmul_vec_q4k_splitk.wgsl`,
`matmul_vec_q4k_splitk_reduce.wgsl`, `matmul_vec_q4k_unroll.wgsl` — had rows here until they
were deleted with their harness in L47, 2026-08-06.)*

## Cooperative-matrix GEMM — the batched fast path

When the device exposes cooperative matrix (`ctx.has_coop()` — Apple `simdgroup_matrix`
/ tensor cores), the batched/prefill path (`m≥8`) runs the inner product on the
hardware matrix units instead of scalar FMAs — measured ~3.5× the scalar tiled GEMM.
Each has a bf16, int8 (Q8), Q4_0, Q4_K-lite, and Q4_K_S flavor (the quant is
dequantized into workgroup staging, then fed to the matrix units). The `_rb2` / `_cb4`
/ `_dk2` suffixes are tiling knobs found by profiling batched decode: **RB2**
register-blocks along M (weight tile read once per 16 rows instead of 8), **CB4**
widens the column block (4× matrix-unit work per barrier — the occupancy lever that
capped batched decode at ~15% roofline), **DK2** stages two K-slices per round to
halve the barrier count without CB8's register spill. Every variant shares the math and
boundary contract of its base and holds the same `matmul_nt_*` oracle.

| shader | kind | status | purpose |
|---|---|---|---|
| `matmul_coop.wgsl` | WGSL | LIVE | Coop bf16 GEMM, one 8×8 tile/workgroup — drop-in for `matmul` when coop is available. |
| `matmul_coop_rb2.wgsl` | WGSL | LIVE | bf16 coop GEMM, RB2 (two stacked 8×8 tiles, weight tile reused across 16 rows). |
| `matmul_coop_rb2_cb4.wgsl` | WGSL | LIVE | bf16 coop GEMM, RB2×CB4 (16×32 output, 8 mma/K-step — the occupancy lever). |
| `matmul_coop_rb2_cb4_dk2.wgsl` | WGSL | LIVE | bf16 coop GEMM, RB2×CB4×DK2 (16 mma between barriers, half the barrier count). |
| `matmul_coop_q8.wgsl` | WGSL | LIVE | int8 coop GEMM; per-column dequant scale applied at the guarded store. |
| `matmul_coop_q8_rb2.wgsl` | WGSL | LIVE | int8 coop GEMM, RB2. |
| `matmul_coop_q8_rb2_cb4.wgsl` | WGSL | LIVE | int8 coop GEMM, RB2×CB4. |
| `matmul_coop_q8_rb2_cb4_dk2.wgsl` | WGSL | LIVE | int8 coop GEMM, RB2×CB4×DK2. |
| `matmul_coop_q8_0.wgsl` | WGSL | LIVE | Native Q8_0 (32-block, f16 scale) coop GEMM — the T5-XXL / FLUX quant path. |
| `matmul_coop_q4.wgsl` | WGSL | LIVE | Q4_0 coop GEMM; block scale folded into the staged element. |
| `matmul_coop_q4_rb2.wgsl` | WGSL | LIVE | Q4_0 coop GEMM, RB2 (weight read + nibble-unpack shared across 16 rows). |
| `matmul_coop_q4_rb2_cb4.wgsl` | WGSL | LIVE | Q4_0 coop GEMM, RB2×CB4. |
| `matmul_coop_q4_rb2_cb4_dk2.wgsl` | WGSL | LIVE | Q4_0 coop GEMM, RB2×CB4×DK2. |
| `matmul_coop_q4k.wgsl` | WGSL | LIVE | Q4_K-lite coop GEMM; (scale·nib+min) folded per staged element. |
| `matmul_coop_q4k_rb2.wgsl` | WGSL | LIVE | Q4_K-lite coop GEMM, RB2. |
| `matmul_coop_q4k_rb2_cb2.wgsl` | WGSL | LIVE | Q4_K-lite coop GEMM, RB2×CB2 (16×16, 4 mma/K-step). |
| `matmul_coop_q4k_rb2_cb4.wgsl` | WGSL | LIVE | Q4_K-lite coop GEMM, RB2×CB4. |
| `matmul_coop_q4k_rb2_cb4_dk2.wgsl` | WGSL | LIVE | Q4_K-lite coop GEMM, RB2×CB4×DK2. |
| `matmul_coop_q4k_rb2_cb8.wgsl` | WGSL | LIVE | Q4_K-lite coop GEMM, RB2×CB8 (16×64, widest column block; register-pressure A/B). |
| `matmul_coop_q4ks.wgsl` | WGSL | LIVE | Q4_K_S coop GEMM; two-level dequant folded per staged element. |
| `matmul_mm_q4ks.metal` | Metal | LIVE | Tiled MSL Q4_K_S GEMM (dequant-on-load, `simdgroup_float8x8`) — island dense q/k/v/o, m≥8. |
| `matmul_mm_q4ks_v2.metal` | Metal | LIVE | Q4_K_S GEMM v2 (KC=64, half-storage) — fixes v1's K-loop barrier serialization (`ARF_MM_V2`). |
| `matmul_mm_q8.metal` | Metal | LIVE | Tiled MSL Q8 GEMM for the batched lm_head — feeds the vocab matmul to the matrix units. |

## MoE — expert routing and grouped expert GEMM

The 30B-class models (Qwen3-Coder-30B) are sparse: a router picks `top_k` of N experts
per token, and only the routed experts' weights stream. The router does a full softmax
top-k on GPU (the single-thread version was ~37% of decode time). The `_batch_gu`
kernels fuse all `top_k` experts' gate/up into one dispatch; `_down_reduce` fuses the
weight-summed down-projection over all slots into one dispatch (ascending slot order to
stay bit-exact with the old sequential accumulate). The `_b` / `_b_v2` suffixes are the
concurrency-advantage generalization: B sequences in lockstep, one dispatch. On the island,
the MoE-decode trick is `mul_mv_id` — an expert-indirection wrapper that reuses the
*exact* dense GEMV body pointed at the routed expert (no special slow MoE kernel), and
`mul_mm_id` — the tiled per-expert GEMM that loads each expert's weight tile once and
reuses it across the expert's grouped rows (closes the conc≥32 expert-traffic gap).

| shader | kind | status | purpose |
|---|---|---|---|
| `moe_route.wgsl` | WGSL | LIVE | MoE router top-k (batch-1): full softmax, pick top_k, optional Qwen3 renorm. |
| `moe_route_b.wgsl` | WGSL | LIVE | Batched MoE router top-k, one workgroup per row (B sequences). |
| `matmul_vec_moe.wgsl` | WGSL | LIVE | bf16 MoE expert GEMV; expert selected at GPU runtime from a packed all-experts buffer. |
| `matmul_vec_moe_batch_gu.wgsl` | WGSL | LIVE | Fused bf16 MoE gate/up — all top_k experts in one dispatch. |
| `matmul_vec_moe_batch_gu_b.wgsl` | WGSL | LIVE | B-row fused bf16 MoE gate/up (`[B, top_k, n]`). |
| `matmul_vec_moe_down_reduce.wgsl` | WGSL | LIVE | Fused bf16 MoE down — weight-summed over all top_k slots in one dispatch. |
| `matmul_vec_moe_down_reduce_b.wgsl` | WGSL | LIVE | B-row fused bf16 MoE down. |
| `matmul_vec_moe_q4.wgsl` | WGSL | LIVE | Q4_0 MoE expert GEMV (packed all-experts codes + block scales). |
| `matmul_vec_moe_q4_batch_gu.wgsl` | WGSL | LIVE | Fused Q4_0 MoE gate/up. |
| `matmul_vec_moe_q4_down_reduce.wgsl` | WGSL | LIVE | Fused Q4_0 MoE down. |
| `matmul_vec_moe_q4k.wgsl` | WGSL | LIVE | Q4_K-lite MoE expert GEMV (asymmetric per-32 scale+min, the Q4_K_M accuracy idea). |
| `matmul_vec_moe_q4k_batch_gu.wgsl` | WGSL | LIVE | Fused Q4_K-lite MoE gate/up (vec4-partial ALU de-stall). |
| `matmul_vec_moe_q4k_down_reduce.wgsl` | WGSL | LIVE | Fused Q4_K-lite MoE down (WG=32 to match Qwen3-30B sub-block count). |
| `matmul_vec_moe_q4ks.wgsl` | WGSL | LIVE | Q4_K_S MoE expert GEMV — ~16% smaller than Q4_K-lite; the coder-30B decode path. |
| `matmul_vec_moe_q4ks_batch_gu.wgsl` | WGSL | LIVE | Fused Q4_K_S MoE gate/up. |
| `matmul_vec_moe_q4ks_batch_gu_b.wgsl` | WGSL | LIVE | B-row fused Q4_K_S MoE gate/up (`[B, top_k, n]`). |
| `matmul_vec_moe_q4ks_batch_gu_b_v2.wgsl` | WGSL | LIVE | B-row Q4_K_S gate/up, v2 nr0/NSG thread layout (llama `mul_mv_id` ported to plain WGSL). |
| `matmul_vec_moe_q4ks_down_reduce.wgsl` | WGSL | LIVE | Fused Q4_K_S MoE down. |
| `matmul_vec_moe_q4ks_down_reduce_b.wgsl` | WGSL | LIVE | B-row fused Q4_K_S MoE down. |
| `moe_route_msl.metal` | Metal | LIVE | Island MoE router top-k (B=1), faithful port of `moe_route_b`'s B=1 path. |
| `moe_swiglu_msl.metal` | Metal | LIVE | Island MoE SwiGLU (silu-gated, Qwen MoE) over the routed-slot gate/up outputs. |
| `gemv_q4ks_id_msl.metal` | Metal | LIVE | Island indirect (`mul_mv_id`) Q4_K_S MoE GEMV — dense body pointed at the routed expert. |
| `gemv_q4ks_id_down_msl.metal` | Metal | LIVE | Island indirect MoE down GEMV with router-weight reduce over top_k slots. |
| `moe_mm_id_q4ks.metal` | Metal | LIVE | Island tiled per-expert MoE GEMM (`mul_mm_id`) — expert tile loaded once, reused across rows. |
| `moe_batch_msl.metal` | Metal | LIVE | B-row island MoE kernels (route/gate-up/down/swiglu) for the batched megakernel. |

## Attention — decode, prefill, batched, GQA, split-KV

All decode attention is **streamed (online / flash) softmax** over a paged KV pool:
keys are read directly from the pool by slot (FlashInfer-style in-kernel gather), and a
running max/denominator is rescaled tile-by-tile so shared memory is O(tile), not
O(context) — context is bounded by the KV pool, not a fixed score row. `attention` is
the single-sequence base; `attention_batched` serves a ragged batch in one dispatch
(the continuous-batching lever); `attention_gqa` fans one KV read across the whole
query-head group (the KV-bandwidth lever — decode attention is KV-bandwidth-bound);
`attention_splitk`(+`_combine`) is flash-decoding, splitting the key range across
workgroups so no single dispatch trips Metal's ~10s watchdog at long context. The `_tq`
pair adds inline TurboQuant dequant with the Walsh-Hadamard rotation trick (score in
R-space); `_tq_par` parallelizes the butterfly across the workgroup.

| shader | kind | status | purpose |
|---|---|---|---|
| `attention.wgsl` | WGSL | LIVE | GQA causal attention, single sequence, streamed softmax over the paged pool. |
| `attention_batched.wgsl` | WGSL | LIVE | Ragged-batch GQA attention in one dispatch (continuous-batched decode). |
| `attention_gqa.wgsl` | WGSL | LIVE | GQA attention reading each K/V slot once, fanned across the query-head group. |
| `attention_splitk.wgsl` | WGSL | LIVE | Split-KV (flash-decoding) pass 1 — partial online-softmax over a key chunk. |
| `attention_splitk_combine.wgsl` | WGSL | LIVE | Split-KV pass 2 — merge the split partials into the final output. |
| `attention_tq.wgsl` | WGSL | LIVE | Attention over a TurboQuant-packed KV pool, inline dequant + R-space rotation trick. |
| `attention_tq_par.wgsl` | WGSL | LIVE | `attention_tq` with the Walsh-Hadamard rotation parallelized across the workgroup. |
| `kv_scatter.wgsl` | WGSL | LIVE | Write new K/V into the physical pool at the block-table slots (`PagedKvCache::write`). |
| `kv_scatter_tq.wgsl` | WGSL | LIVE | Quantize + write new K/V packed into the pool under TurboQuant. |
| `attention_msl.metal` | Metal | LIVE | Island decode attention (q_len==1), flash online-softmax; port of `attention`'s tile shape. |
| `attention_fused_msl.metal` | Metal | LIVE | Island attention with the new-token KV scatter folded in (kv_scatter + attention → 1 dispatch). |

## Norms & RoPE

RMSNorm (Llama-style), LayerNorm (with/without affine, for vision/FLUX), the
per-head Q/K/V norms (Qwen3/Gemma normalize per head before RoPE — V weightless on
Gemma 4), and the fused-pair kernels that cut dispatch count (`add_norm` = residual-add
+ RMSNorm; `rmsnorm_add` = Gemma's post-sublayer norm-and-add; `qkv_norm` fuses Q/K/V
into one dispatch; `rope_qk` fuses RoPE over Q and K). RoPE has a GPT-NeoX rotate-half
convention (`rope`), an interleaved-pair fused q/k variant for muse-glimmer (`rope_qk_interleaved`,
ggml "NORM" pairing), and a 3-axis interleaved-pair convention for FLUX (`flux_rope_3axis`).
Fused-pair reductions are why the megakernel can beat a graph engine that can't merge
distinct graph nodes — see `gemma_normpair_msl`.

| shader | kind | status | purpose |
|---|---|---|---|
| `rmsnorm.wgsl` | WGSL | LIVE | RMSNorm (no mean-subtraction, no bias), one workgroup per row. |
| `add_norm.wgsl` | WGSL | LIVE | Fused residual-add + RMSNorm (Llama/Qwen), single dispatch, no DRAM round-trip. |
| `rmsnorm_add.wgsl` | WGSL | LIVE | Fused RMSNorm + residual-add (Gemma post-attention / post-FFN norm). |
| `qk_norm.wgsl` | WGSL | LIVE | Per-head RMSNorm on Q and K in one dispatch, before RoPE (Qwen3/Gemma). |
| `qkv_norm.wgsl` | WGSL | LIVE | Per-head RMSNorm on Q, K, and V in one dispatch (Gemma 4; V weightless). |
| `v_norm.wgsl` | WGSL | LIVE | Per-head weightless RMSNorm on V (Gemma 4 `Vcur_normed`). |
| `layernorm_bias.wgsl` | WGSL | LIVE | LayerNorm with weight+bias (SigLIP / ViT / CLIP). |
| `layernorm_plain.wgsl` | WGSL | LIVE | LayerNorm with no affine (FLUX DiT norms; affine comes from adaLN). |
| `rope.wgsl` | WGSL | LIVE | RoPE, GPT-NeoX rotate-half convention, applied in place. |
| `rope_qk.wgsl` | WGSL | LIVE | Fused RoPE over Q and K in one dispatch (replaces two `rope` dispatches). |
| `rope_qk_interleaved.wgsl` | WGSL | LIVE | Fused RoPE over Q and K, interleaved (ggml "NORM") pairing — muse-glimmer; twin of `rope_qk` (split-half / NEOX). |
| `flux_rope_3axis.wgsl` | WGSL | LIVE | FLUX 3-axis RoPE, interleaved-pair convention, per-position cos/sin tables with the [16,56,56] axis split baked in (`gpu/flux/dit.rs`). *`rope_interleaved.wgsl`, which had this row, was the one dead shader in 146 and was deleted in L288 (2026-08-28).* |
| `attn_gate_mul.wgsl` | WGSL | LIVE | muse-glimmer's attention output gate: `attn[i] *= sigmoid(gate[i])` after SDPA, before o_proj — fused because the gate is dead afterwards. |
| `rmsnorm_msl.metal` | Metal | LIVE | Island RMSNorm (`simd_sum` reduction idiom), bit-close to `rmsnorm`. |
| `add_norm_msl.metal` | Metal | LIVE | Island fused residual-add + RMSNorm (Llama/Qwen), port of `add_norm`. |
| `gemma_normpair_msl.metal` | Metal | LIVE | Fused Gemma post-attn-add + pre-FFN norm in ONE dispatch — a structural beat-llama win. |

## Sampling & embedding

On-GPU greedy/stochastic sampling and the token-embed feedback path close the decode
loop without a CPU round-trip between tokens: the sample kernel writes the emitted
token id into a resident buffer, and `embed_tok` embeds that row directly into the next
step. `sample_batched` removes the per-token full-vocab readback stall (reads back only
`rows` u32s). bf16 embed variants exist because a large vocab×hidden table (Gemma 4:
262144×5376 = 5.6 GB at f32) exceeds Metal's ~4 GB single-buffer-binding limit.

| shader | kind | status | purpose |
|---|---|---|---|
| `sample.wgsl` | WGSL | LIVE | Greedy argmax over `logits[vocab]` → one token id (ties to lowest index). |
| `sample_batched.wgsl` | WGSL | LIVE | Batched greedy argmax, one workgroup per row — no full-vocab CPU readback. |
| `sample_stochastic.wgsl` | WGSL | LIVE | Stochastic sampling (temperature/top-k/top-p, threshold method, no host sort). |
| `softcap.wgsl` | WGSL | LIVE | Final-logit soft-capping `cap·tanh(logit/cap)` (Gemma 4, before sampling). |
| `embed.wgsl` | WGSL | LIVE | Row gather (embed a token / gather the last hidden row); optional Gemma √hidden scale. |
| `embed_bf16.wgsl` | WGSL | LIVE | bf16-table row gather (large Gemma embedding that overflows the f32 binding limit). |
| `embed_tok.wgsl` | WGSL | LIVE | Token embed with source row from a resident buffer — the on-GPU decode feedback path. |
| `embed_tok_bf16.wgsl` | WGSL | LIVE | bf16 token embed from a resident buffer (feedback path + bf16 table). |

## SSM / gated-delta-net — the Qwen3.5/3.6 hybrid

The Qwen3.6-27B "beast" is a hybrid: 3/4 of its layers are gated-delta-net (linear
attention with an O(1)-per-token recurrent state) and 1/4 GQA. The recurrence is a
bit-exact port of llama.cpp's `build_delta_net_autoregressive`. The M11 work put the
whole linear layer on one command queue: a causal depthwise conv1d, the delta-rule
state update, and its prologue (L2-norm, softplus g-decay, sigmoid gate, tile/repeat).
Each op has a single-stream WGSL twin **and** a `_b` B-parallel twin — the recurrent
state grows with batch B but *not* with context length, which is the concurrency advantage.
The `b=0` slice of every `_b` kernel is byte-identical to its single-stream twin ("B=1
≡ M11" parity gate). The MSL versions are the island originals these were ported from.

| shader | kind | status | purpose |
|---|---|---|---|
| `gated_delta_net.wgsl` | WGSL | LIVE | Delta-net 1-token recurrence (single-queue port), one workgroup per v-head. |
| `gated_delta_net_b.wgsl` | WGSL | LIVE | B-parallel delta-net recurrence — per-seq state, the concurrency advantage. |
| `ssm_conv1d.wgsl` | WGSL | LIVE | Causal depthwise conv1d + SiLU, one invocation per conv channel. |
| `ssm_conv1d_b.wgsl` | WGSL | LIVE | B-parallel causal depthwise conv1d (per-seq conv ring). |
| `gdn_g_decay.wgsl` | WGSL | LIVE | Fused g-decay `softplus(alpha+dt)·a` over the v-heads. |
| `gdn_g_decay_b.wgsl` | WGSL | LIVE | B-parallel g-decay (shared static dt/a, per-seq alpha). |
| `gdn_l2_norm.wgsl` | WGSL | LIVE | Per-head L2 normalize (ggml `l2_norm`, no learned gain). |
| `gdn_l2_norm_b.wgsl` | WGSL | LIVE | B-parallel per-head L2 normalize. |
| `gdn_sigmoid_out.wgsl` | WGSL | LIVE | Non-in-place sigmoid for the beta gate. |
| `gdn_sigmoid_out_b.wgsl` | WGSL | LIVE | B-parallel sigmoid gate. |
| `gdn_tile.wgsl` | WGSL | LIVE | Split conv_out into q/k/v and tile q,k across v-head groups (`repeat_4d`). |
| `gdn_tile_b.wgsl` | WGSL | LIVE | B-parallel tile/repeat. |
| `gated_delta_net_msl.metal` | Metal | LIVE | Island delta-net autoregressive recurrence (M3), the original of the WGSL twin. |
| `gdn_prologue_msl.metal` | Metal | LIVE | Island delta-net prologue (l2_norm / softplus / sigmoid / tile), 6 kernels. |
| `ssm_conv1d_msl.metal` | Metal | LIVE | Island causal depthwise conv1d (M2), one thread per conv channel. |

## Batched-decode megakernel ops (island)

The remaining island files bundle the small per-token decode ops (norms, activations,
RoPE, sampling) into the megakernel so the whole frame is a handful of native dispatches
on one queue. Each op is bit-close to its WGSL twin, parity-gated against the CPU oracle.
`_batch` is the B-row generalization for the batched (concurrency) megakernel.

| shader | kind | status | purpose |
|---|---|---|---|
| `decode_ops_msl.metal` | Metal | LIVE | 14 per-token decode ops (geglu, qk_norm, rope_qk, rmsnorm_add, batched lm_head, …). |
| `decode_ops_batch_msl.metal` | Metal | LIVE | B-row generalizations of the `decode_ops_msl` per-row ops (B=1 ≡ the m==1 sibling). |

## Vision — SigLIP encoder

Gemma-3's image path is a 27-layer SigLIP ViT: bidirectional (unmasked) multi-head
self-attention with a workgroup-resident score cache (one q·k per key, reused by the
softmax and V-weighting) to stay under Metal's per-submission watchdog, plus the
elementwise ops the encoder needs. GELU and bias-add here are the "2D-safe flat index"
kernels — vision activations reach 17.6M elements and overflow the workgroup-per-dim cap.

| shader | kind | status | purpose |
|---|---|---|---|
| `vision_attn.wgsl` | WGSL | LIVE | Bidirectional multi-head self-attention for the SigLIP ViT (score-cached). |
| `gelu_tanh.wgsl` | WGSL | LIVE | Elementwise GELU (tanh approx), in place, 2D-safe (vision MLP). |
| `add_bias.wgsl` | WGSL | LIVE | Per-column bias add over a `[rows, cols]` matrix (SigLIP Q/K/V/O + MLP linears). |

## FLUX / VAE — image generation

FLUX.1-schnell text→image, all on GPU. The DiT is a joint txt+img transformer
(256+4096 tokens) with adaLN-Zero modulation and 3-axis interleaved RoPE; the text
encoders are CLIP-L (causal) and T5-XXL (bidirectional + relative-position bias, run at
Q8_0); the VAE decoder lowers 2D convs to im2col→matmul, with GroupNorm, nearest-neighbor
upsample, and a full-resolution mid-block self-attention. `matmul_coop_q8_0` (listed under
coop GEMM) is the T5 quant path. Metric of record for these is `phys_footprint`, not RSS.

| shader | kind | status | purpose |
|---|---|---|---|
| `dit_attn.wgsl` | WGSL | LIVE | Full (non-causal) MHA over FLUX's joint DiT sequence (4352 tokens), score-cached. |
| `adaln_modulate.wgsl` | WGSL | LIVE | FLUX adaLN-Zero modulation (scale/shift/gate by element offset, no sub-range bind). |
| `text_attn.wgsl` | WGSL | LIVE | Text-encoder MHA shared by CLIP-L (causal) and T5-XXL (full + rel-position bias). |
| `quick_gelu.wgsl` | WGSL | LIVE | Quick-GELU `x·sigmoid(1.702x)` (CLIP text-encoder MLP). |
| `im2col.wgsl` | WGSL | LIVE | Lower a 2D conv to a matmul (patch matrix), tiled/streamable for the VAE. |
| `conv_epilogue.wgsl` | WGSL | LIVE | Transpose matmul output to channel-major + add per-channel conv bias, in one pass. |
| `group_norm.wgsl` | WGSL | LIVE | GroupNorm over channel-major `[C,H,W]` activations (VAE, 32 groups). |
| `upsample_nearest.wgsl` | WGSL | LIVE | Nearest-neighbor 2× upsample `[C,H,W]→[C,2H,2W]` (VAE Upsample2D). |
| `vae_attn.wgsl` | WGSL | LIVE | Single-head full self-attention for the VAE mid-block (streamed, unbounded seq). |
| `row_bias.wgsl` | WGSL | LIVE | Per-column bias add over a pixel-major `[seq, c_out]` activation (VAE 1×1 conv bias). |

## Elementwise & misc

The small building blocks: residual add, elementwise multiply, the GLU activations
(SwiGLU for Llama/Qwen, GeGLU for Gemma), SiLU, and the column-slice / scale utilities
the fused-projection and per-layer-scalar paths need.

| shader | kind | status | purpose |
|---|---|---|---|
| `add.wgsl` | WGSL | LIVE | Residual add in place `a[i]+=b[i]`, 2D-safe. |
| `mul.wgsl` | WGSL | LIVE | Elementwise multiply in place `a[i]*=b[i]` (T5 gated-GELU). |
| `swiglu.wgsl` | WGSL | LIVE | SwiGLU `silu(gate)·up` (Llama/Qwen FFN). |
| `geglu.wgsl` | WGSL | LIVE | GeGLU `gelu_tanh(gate)·up` (Gemma FFN). |
| `silu.wgsl` | WGSL | LIVE | Elementwise SiLU in place (FLUX adaLN MLP, single-stream activation). |
| `add_bias.wgsl` | WGSL | LIVE | (also vision) per-column bias add. |
| `scale_inplace.wgsl` | WGSL | LIVE | In-place scalar multiply `x[i]*=scale` (Gemma 4 per-layer output scalar). |
| `slice_cols.wgsl` | WGSL | LIVE | Gather/scatter a column-block between a fused-projection matrix and a dense buffer. |

---

*Sibling docs: [CONTRIBUTING.md](../CONTRIBUTING.md) (the parity commands — the original
`docs/CORRECTNESS_VERIFICATION.md` ladder was removed in the doc cleanup, 2026-08-28),
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) (the engine architecture).*

---

## `gemv_q4ks_batch` — the batched Q4_K_S decode GEMV (L220 shape)

The kernel every daemon token goes through. Two structural properties matter, and both were
established by measurement rather than design intent:

**1. It is LATENCY-bound on outstanding loads, not bandwidth- or ALU-bound.**
Three separate attempts to trade arithmetic for bytes all LOST:
- L201 COLS=2 at b=1 (traded threadgroup parallelism for reuse): -25%
- L204 hoisting the column-independent `a_sum` into a prepass: **-31%**, after removing
  99.994% of that computation
- L216 Q3_K FFN (fewer bits, more unpack work): -4%

`a_sum`'s 24 adds were free — they ran in the shadow of memory the loop already awaited.
Adding ALU to save bytes is a losing trade here. Adding *requests in flight* wins (L167,
NSGB=4, +11.6%).

**2. Activations dominate the byte stream, and the loop order decides whether that matters.**
Per output column at b=3: 4480 B of weights against 61440 B of activations. Across
n=17408 columns that is 78 MB of weights and 1070 MB of activations — **93% of all bytes**.
The activation slab is only 60 KB; the pre-L220 kernel re-read it once per column.

### The L220 loop order (row-outer, column-inner)

```
for sub-block b:
    for row r in 0..BR:                    <- OUTER
        load a0..a7 into NAMED registers   <- ONCE per (row, sub-block)
        compute a_sum
        for cc in 0..COLS_B:               <- INNER
            unpack this (column, sub-block)'s nibbles
            acc[cc][r] += s*dot(...) + lo*a_sum
```

Each row's activations are loaded once and dotted against `COLS_B` columns. Effect on the
marginal cost of one extra batch row (GPU-hardware time):

| shape | b=1 | b=3 | marginal |
|---|---|---|---|
| column-outer (pre-L220) | 55.0 ms | 160.5 ms | 0.96 |
| row-outer, COLS_B=1 | 55.8 ms | 103.2 ms | 0.42 |
| row-outer, COLS_B=4 (**default**) | 51.1 ms | 81.0 ms | **0.29** |

A weight-stream-bound decode predicts a marginal near 0 — weights are read once, all rows
reuse them. 0.96 meant activations were re-streamed per row; 0.29 means they largely are not.

### ⚠️ Do not stage activations in an array

L197 implemented this reuse as `float4 av[MAXB*8]` — 512 dynamically-indexed slots. It
SPILLED to memory and cost 34% **even at COLS=1**, which L200 found only by bisecting the
binary. Named registers with the column loop inside get the same reuse for free. If you
extend this kernel, keep every activation register statically indexed.

### Function constants
| id | env | default | effect |
|---|---|---|---|
| 3 | `ARF_GEMV_B_NSG` | 4 | simdgroups per threadgroup; 6 and 8 fall off sharply |
| 4 | `ARF_GEMV_ACC_CAP` | MAXB | probe only — rows >= cap are NEVER written |
| 5 | `ARF_GEMV_B_COLS` | 4 | output columns per threadgroup; host grid must match |

`ARF_GEMV_B_COLS` and the host's `gemv_b_cols` **must agree** — the grid is
`n.div_ceil(COLS_B)`, so a mismatch silently computes the wrong columns.
