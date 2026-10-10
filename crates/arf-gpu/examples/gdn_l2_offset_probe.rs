//! Focused probe for the rec_gdn_chain_b L2 handling. The B-parallel conv_out is `[B*conv_dim]`
//! b-outer, where each seq's conv_out = [q_conv(key_dim) | k_conv(key_dim) | v_conv(value_dim)].
//! The q/k regions of seq b are NOT a contiguous `[B*nkh*s]` array (stride between seqs is
//! conv_dim, not key_dim), so `gdn_l2_norm_b` (which assumes lane*dim packing) CANNOT be applied
//! directly. This probe proves the chosen correct approach: run the SINGLE-STREAM `gdn_l2_norm`
//! per (seq, region) with an OFFSET binding at `(b*conv_dim + region_off)` over key_dim, matching
//! exactly what the single-stream chain does per-region — and that the result equals decoding
//! each seq's conv_out alone through the single-stream chain's l2 step.
//!
//! Run: cargo run --release -p arf-gpu --example gdn_l2_offset_probe

use std::sync::Arc;

use arf_gpu::gpu::{CommandPass, GpuContext};
use arf_gpu::weights::build_kernels;

fn gen(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
            lo + (hi - lo) * (u % 1.0)
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

fn main() {
    let s: usize = 128;
    let nkh: usize = 16;
    let nvh: usize = 48;
    let key_dim = s * nkh; // 2048
    let value_dim = s * nvh; // 6144
    let conv_dim = key_dim * 2 + value_dim; // 10240
    let eps = 1e-6f32;
    let batch: usize = 4;

    // sanity: byte offsets we will bind must be 256-aligned (macOS wgpu storage offset align).
    assert_eq!(
        (conv_dim * 4) % 256,
        0,
        "conv_dim*4 must be 256-aligned for per-seq offset bindings"
    );
    assert_eq!(
        (key_dim * 4) % 256,
        0,
        "key_dim*4 must be 256-aligned for the k-region offset"
    );

    let ctx = Arc::new(GpuContext::new().expect("gpu context"));
    let k = build_kernels(&ctx, false);

    // b-outer conv_out: each seq's [q|k|v], distinct random data per seq.
    let mut conv_out_all = Vec::with_capacity(batch * conv_dim);
    for b in 0..batch {
        conv_out_all.extend(gen(900 + b as u64 * 31, conv_dim, -0.8, 1.6));
    }

    // ---- Batched: ONE buffer [B*conv_dim]; run gdn_l2_norm per (seq, region) via OFFSET binding.
    let co_b = ctx.storage_init("pb.l2o.co", &conv_out_all);
    {
        let mut cp = CommandPass::new(&ctx);
        for b in 0..batch {
            for region_off in [0usize, key_dim] {
                let dd = ctx.uniform_of("pb.l2o.d", &[s as u32, nkh as u32, eps.to_bits(), 0u32]);
                let byte_off = ((b * conv_dim + region_off) * 4) as u64;
                let xr = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &co_b,
                    offset: byte_off,
                    size: wgpu::BufferSize::new((key_dim * 4) as u64),
                });
                cp.add_bound(
                    &k.gdn_l2_norm,
                    "l2_off",
                    &[xr, dd.as_entire_binding()],
                    [nkh as u32, 1, 1],
                );
            }
        }
        cp.submit();
    }
    let gpu = ctx.read_f32(&co_b, batch * conv_dim);

    // ---- Reference: each seq's conv_out decoded ALONE through the single-stream l2 (q then k).
    let mut worst = 0.0f32;
    for b in 0..batch {
        let co = conv_out_all[b * conv_dim..(b + 1) * conv_dim].to_vec();
        let sco = ctx.storage_init("pb.l2o.sco", &co);
        let mut cp = CommandPass::new(&ctx);
        for region_off in [0usize, key_dim] {
            let dd = ctx.uniform_of("pb.l2o.sd", &[s as u32, nkh as u32, eps.to_bits(), 0u32]);
            let xr = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &sco,
                offset: (region_off * 4) as u64,
                size: wgpu::BufferSize::new((key_dim * 4) as u64),
            });
            cp.add_bound(
                &k.gdn_l2_norm,
                "l2_s",
                &[xr, dd.as_entire_binding()],
                [nkh as u32, 1, 1],
            );
        }
        cp.submit();
        let single = ctx.read_f32(&sco, conv_dim);
        worst = worst.max(max_abs(&gpu[b * conv_dim..(b + 1) * conv_dim], &single));
    }
    let ok = worst < 1e-7;
    println!(
        "rec_gdn_chain_b L2 offset-binding: max_abs(batched_offset vs single) = {worst:.3e}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok, "L2 offset-binding approach diverged from single-stream");
}
