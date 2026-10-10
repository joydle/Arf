# FLUX text→image on Arf — module layout

GPU-native FLUX.1-schnell, built the `vision.rs` way: small self-contained GPU modules that
own their `ComputeKernel`s and reuse Arf's existing kernel library. No CPU compute on the
hot path (CPU only orchestrates the step loop, uploads GGUF→GPU, and PNG-encodes the readback).

## Module map (`crates/arf-gpu/src/gpu/flux/`)

- `mod.rs`        — re-exports + the `FluxPipeline` façade (composes the four modules + the
                    flow-match Euler host loop). Built last (P3).
- `clip.rs`       — `GpuClipText`: CLIP-L text encoder → pooled `[768]`. (P0)
- `t5.rs`         — `GpuT5`: T5-XXL encoder → sequence `[seq, 4096]`. (P0)
- `dit.rs`        — `GpuDiT`: the FLUX MMDiT denoiser (19 double + 38 single blocks). (P1)
- `vae.rs`        — `GpuVae`: the conv decoder, latent `[16,128,128]` → image `[3,1024,1024]`. (P2)

Each module: `new(ctx, &LazyGguf or safetensors)` uploads resident bf16 weights + compiles its
kernels; `forward(...)` runs a fixed-shape multi-dispatch GPU pass and reads back only the final
result. Exactly `GpuVisionEncoder`'s shape.

## New WGSL kernels (added to `crates/arf-gpu/src/shaders/`)

P0: `quick_gelu.wgsl` (CLIP), a T5-relative-position-bias attention variant. Everything else
(LayerNorm-bias, RMSNorm, Q/K/V/O matmul+bias, MLP, causal + full attention, GELU) REUSES
existing shaders.

P1: `adaln_modulate.wgsl`, 3-axis RoPE table, `timestep_embed`. Reuse matmul / qk_norm /
vision_attn.

P2 (the new family, shared with the 3D texture pipeline): `im2col.wgsl` (+ reuse matmul.wgsl
for the conv GEMM), `groupnorm.wgsl`, `silu.wgsl`, `upsample_nearest.wgsl`.

## Weights (`models/flux-schnell/`)

- `t5xxl-Q8_0.gguf`     (~5 GB, city96)            — T5-XXL encoder
- `clip_l.safetensors`  (~250 MB, comfyanonymous)  — CLIP-L
- `flux1-schnell-Q4_K_S.gguf` (~6.8 GB, city96)    — the MMDiT transformer  [fetch at P1]
- `ae.safetensors`      (~335 MB, BFL)             — the VAE                [fetch at P2]

## Parity discipline

Every module is parity-gated against an oracle (ComfyUI / diffusers / city96) via per-stage
numerical diffing — the method (`MTMD_DEBUG_GRAPH sum=` style) that caught the vision
projector-transpose bug. No "looks right" merges.
