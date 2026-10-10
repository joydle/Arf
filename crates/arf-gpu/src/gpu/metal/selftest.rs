//! Self-test / parity / micro-benchmark methods for [`MetalIsland`], split out of
//! `concurrent_metal.rs` to keep the engine core focused. These are driven ONLY by the
//! `examples/` harness binaries (parity gates + perf probes) and the CPU-reference path;
//! they are not on any production decode path. Split as a second `impl MetalIsland` block
//! in the same module tree (Rust permits inherent impls across files), so behavior is
//! byte-identical to when these lived inline.
// Raw objc2/Metal FFI — see concurrent_metal.rs for why unused_unsafe is allowed file-wide.
#![allow(unused_unsafe)]

use super::super::concurrent_metal::*;
use super::super::context::GpuContext;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use wgpu::hal::api::Metal;

impl MetalIsland {
    /// GO/NO-GO microbenchmark: does `MTLDispatchTypeConcurrent` actually overlap
    /// independent bandwidth-bound kernels on THIS hardware? Allocates `n` private buffers
    /// of `bytes_each`, compiles a memory-streaming kernel (read+write every element, like a
    /// GEMV's weight read), then runs all `n` dispatches SERIAL vs CONCURRENT (`iters` times,
    /// taking the min GPU time of each). Returns `(serial_ms, concurrent_ms)`. If concurrent
    /// is NOT meaningfully faster, the whole concurrent-encoder island premise is refuted —
    /// the kernels are bandwidth-bound and overlapping them saturates the same DRAM bus.
    pub fn microbench(
        &mut self,
        ctx: &GpuContext,
        n: usize,
        bytes_each: usize,
        iters: usize,
    ) -> Result<(f64, f64), String> {
        // A bandwidth-bound kernel: out[i] = in[i]*1.0001 + 0.0 over a big f32 array.
        // Each invocation streams `bytes_each` from DRAM — the GEMV weight-read analog.
        const KERN: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) { dst[i] = src[i] * 1.0001; }
}
"#;
        self.compile(ctx, "bw_stream", KERN)?;
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let n_elems = bytes_each / 4;
        // n src + n dst buffers (private/shared storage).
        let mut bufs: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> = Vec::new();
        for _ in 0..(2 * n) {
            let b = dev
                .newBufferWithLength_options(bytes_each, objc2_metal::MTLResourceOptions::empty())
                .ok_or("buffer alloc")?;
            bufs.push(b);
        }
        drop(guard);
        let pso = &self.pipelines.get("bw_stream").unwrap().pso;
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: n_elems.div_ceil(256),
            height: 1,
            depth: 1,
        };

        let run = |dtype: MTLDispatchType| -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoderWithDispatchType(dtype).unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for j in 0..n {
                    let s: &ProtocolObject<dyn MTLResource> =
                        ProtocolObject::from_ref(&*bufs[2 * j]);
                    let d: &ProtocolObject<dyn MTLResource> =
                        ProtocolObject::from_ref(&*bufs[2 * j + 1]);
                    enc.useResource_usage(s, MTLResourceUsage::Read);
                    enc.useResource_usage(d, MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&*bufs[2 * j]), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&*bufs[2 * j + 1]), 0, 1);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                }
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0 // ms
        };

        // Warm, then min-of-iters for each dispatch type.
        let _ = run(MTLDispatchType::Serial);
        let _ = run(MTLDispatchType::Concurrent);
        let mut serial = f64::INFINITY;
        let mut concur = f64::INFINITY;
        for _ in 0..iters {
            serial = serial.min(run(MTLDispatchType::Serial));
            concur = concur.min(run(MTLDispatchType::Concurrent));
        }
        Ok((serial, concur))
    }
    /// M2 GATE — standalone (NO model) self-test of the `ssm_conv1d` MSL kernel (the
    /// Qwen3.6-27B gated-delta-net causal depthwise conv1d). Compiles the kernel, uploads the
    /// caller's SYNTHETIC `conv_in[conv_dim]`, `conv_kernel[d_conv*conv_dim]`, `conv_state`
    /// `[conv_dim*(d_conv-1)]` into raw shared MTLBuffers, dispatches one thread per channel
    /// (grid = conv_dim), then reads back BOTH `conv_out[conv_dim]` (post-silu) and the
    /// updated `conv_state` (the advanced ring). The caller computes the SAME conv on the CPU
    /// and asserts parity. Tiny buffers → runs in milliseconds. Returns `(conv_out,
    /// new_conv_state)`.
    pub fn ssm_conv1d_selftest(
        &mut self,
        ctx: &GpuContext,
        conv_in: &[f32],
        conv_kernel: &[f32],
        conv_state: &[f32],
        conv_dim: usize,
        d_conv: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        assert_eq!(conv_in.len(), conv_dim, "conv_in len");
        assert_eq!(conv_kernel.len(), d_conv * conv_dim, "conv_kernel len");
        assert_eq!(conv_state.len(), conv_dim * (d_conv - 1), "conv_state len");

        let msl = include_str!("../../shaders/metal/ssm_conv1d_msl.metal");
        // ARF_SSM_CONV_BATCHED (default off): use the float4-over-channels variant
        // (4 channels/thread, coalesced float4 loads/stores). Bit-exact: each lane carries the
        // identical per-channel IEEE arithmetic in identical order as the scalar kernel.
        let batched = std::env::var("ARF_SSM_CONV_BATCHED")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        let (fn_name, pso_key) = if batched {
            ("ssm_conv1d_f4", "ssm_conv1d_f4")
        } else {
            ("ssm_conv1d", "ssm_conv1d")
        };
        self.compile_msl(ctx, pso_key, msl, fn_name)?;

        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        // SHARED storage so the CPU can write inputs + read outputs through `.contents()`.
        let opts = objc2_metal::MTLResourceOptions::StorageModeShared;
        let mk = |bytes: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            dev.newBufferWithLength_options(bytes.max(16), opts)
                .ok_or_else(|| "conv buffer alloc".to_string())
        };
        let b_in = mk(conv_in.len() * 4)?;
        let b_k = mk(conv_kernel.len() * 4)?;
        let b_state = mk(conv_state.len() * 4)?;
        let b_out = mk(conv_dim * 4)?;
        let b_dims = mk(16)?;
        // Upload inputs through the shared pointers.
        unsafe {
            let p = b_in.contents().as_ptr() as *mut f32;
            for (i, v) in conv_in.iter().enumerate() {
                *p.add(i) = *v;
            }
            let p = b_k.contents().as_ptr() as *mut f32;
            for (i, v) in conv_kernel.iter().enumerate() {
                *p.add(i) = *v;
            }
            let p = b_state.contents().as_ptr() as *mut f32;
            for (i, v) in conv_state.iter().enumerate() {
                *p.add(i) = *v;
            }
            let p = b_dims.contents().as_ptr() as *mut u32;
            *p = conv_dim as u32;
            *p.add(1) = d_conv as u32;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        drop(guard);

        let pso = &self.pipelines.get(pso_key).ok_or("pso ssm_conv1d")?.pso;
        // Scalar: one thread per channel (grid = conv_dim). float4: one thread per 4 channels
        // (grid = ceil(conv_dim/4)); the kernel guards the tail. Tile of 256 threads either way.
        let n_threads = if batched {
            conv_dim.div_ceil(4)
        } else {
            conv_dim
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: n_threads.div_ceil(256),
            height: 1,
            depth: 1,
        };
        let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("no encoder")?;
        let bufs: [&ProtocolObject<dyn MTLBuffer>; 5] = [&b_in, &b_k, &b_state, &b_out, &b_dims];
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(*b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(*b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back conv_out + the advanced conv_state.
        let out: Vec<f32> = unsafe {
            let p = b_out.contents().as_ptr() as *const f32;
            (0..conv_dim).map(|i| *p.add(i)).collect()
        };
        let new_state: Vec<f32> = unsafe {
            let p = b_state.contents().as_ptr() as *const f32;
            (0..conv_state.len()).map(|i| *p.add(i)).collect()
        };
        Ok((out, new_state))
    }
    pub fn gated_delta_net_selftest(
        &mut self,
        ctx: &GpuContext,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state: &[f32],
        s_v: usize,
        n_heads: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let s_k = s_v; // delta-net asserts S_k == S_v
        assert_eq!(q.len(), s_k * n_heads, "q len");
        assert_eq!(k.len(), s_k * n_heads, "k len");
        assert_eq!(v.len(), s_v * n_heads, "v len");
        assert_eq!(g.len(), n_heads, "g len");
        assert_eq!(beta.len(), n_heads, "beta len");
        assert_eq!(state.len(), s_v * s_v * n_heads, "state len");

        let msl = include_str!("../../shaders/metal/gated_delta_net_msl.metal");
        self.compile_msl(ctx, "gated_delta_net", msl, "gated_delta_net")?;

        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let opts = objc2_metal::MTLResourceOptions::StorageModeShared;
        let mk = |bytes: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            dev.newBufferWithLength_options(bytes.max(16), opts)
                .ok_or_else(|| "gdn buffer alloc".to_string())
        };
        let b_q = mk(q.len() * 4)?;
        let b_k = mk(k.len() * 4)?;
        let b_v = mk(v.len() * 4)?;
        let b_g = mk(g.len() * 4)?;
        let b_beta = mk(beta.len() * 4)?;
        let b_s = mk(state.len() * 4)?;
        let b_o = mk(s_v * n_heads * 4)?;
        let b_dims = mk(16)?;
        let up = |buf: &Retained<ProtocolObject<dyn MTLBuffer>>, data: &[f32]| unsafe {
            let p = buf.contents().as_ptr() as *mut f32;
            for (i, v) in data.iter().enumerate() {
                *p.add(i) = *v;
            }
        };
        up(&b_q, q);
        up(&b_k, k);
        up(&b_v, v);
        up(&b_g, g);
        up(&b_beta, beta);
        up(&b_s, state);
        unsafe {
            let p = b_dims.contents().as_ptr() as *mut u32;
            *p = s_v as u32;
            *p.add(1) = s_k as u32;
            *p.add(2) = n_heads as u32;
            *p.add(3) = 0;
        }
        drop(guard);

        let pso = &self
            .pipelines
            .get("gated_delta_net")
            .ok_or("pso gated_delta_net")?
            .pso;
        // One threadgroup per v-head; S_v threads (thread j owns column j of S).
        let tg = MTLSize {
            width: s_v,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: n_heads,
            height: 1,
            depth: 1,
        };
        let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("no encoder")?;
        let bufs: [&ProtocolObject<dyn MTLBuffer>; 8] =
            [&b_q, &b_k, &b_v, &b_g, &b_beta, &b_s, &b_o, &b_dims];
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(*b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(*b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        let out: Vec<f32> = unsafe {
            let p = b_o.contents().as_ptr() as *const f32;
            (0..s_v * n_heads).map(|i| *p.add(i)).collect()
        };
        let new_state: Vec<f32> = unsafe {
            let p = b_s.contents().as_ptr() as *const f32;
            (0..state.len()).map(|i| *p.add(i)).collect()
        };
        Ok((out, new_state))
    }
    /// Self-test the SharedBuffer bridge: write f32s through the wgpu view, read them back
    /// through the raw MTLBuffer view — proves both views alias the SAME memory (one
    /// allocation, no copy). Returns the read-back values for the caller to verify.
    pub fn bridge_selftest(ctx: &GpuContext) -> Option<Vec<f32>> {
        let sb = SharedBuffer::alloc(
            ctx,
            64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "bridge_selftest",
        )?;
        let vals: [f32; 7] = [1.0, -2.5, 3.25, 0.0, 42.0, -7.0, 99.5];
        ctx.queue
            .write_buffer(&sb.wgpu, 0, bytemuck::cast_slice(&vals));
        ctx.queue.submit(std::iter::empty());
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let p = sb.mtl.contents().as_ptr() as *const f32;
        Some((0..7).map(|i| unsafe { *p.add(i) }).collect())
    }
    /// SYNC-COST PROBE — the architectural decision for in-frame wiring. Measures the
    /// per-dispatch cost of INTERLEAVING island Metal work with wgpu work (which the real
    /// decode would do: flush wgpu → island GEMV → resume). Runs `reps` island GEMVs at a
    /// real shape, each preceded by a tiny wgpu submit+poll (simulating the flush boundary),
    /// vs the same GEMVs back-to-back on the island queue. Returns
    /// `(interleaved_ms_per, island_only_ms_per, overhead_ratio)`. If the ratio is ~1.0 the
    /// flush is free (per-GEMV routing viable); if >>1 the cross-queue sync dominates and we
    /// must batch per-LAYER or go full megakernel (one island dispatch per layer/token).
    pub fn sync_cost_probe(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        k: usize,
        n: usize,
        reps: usize,
    ) -> Result<(f64, f64, f64), String> {
        // reuse the Q4KS bench buffers via a throwaway shared dummy + the compiled kernel.
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32;
        let supers = k / 256;
        let alloc = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let bufs = [
            alloc(k * 4),
            alloc(n * subs * 16),
            alloc(n * 4),
            alloc(16),
            alloc((n * subs).div_ceil(4) * 4),
            alloc((n * subs).div_ceil(4) * 4),
            alloc(n * supers * 4), // dd = packed half2/pair
        ];
        unsafe {
            let d = bufs[3].contents().as_ptr() as *mut u32;
            *d = 1;
            *d.add(1) = k as u32;
            *d.add(2) = n as u32;
            *d.add(3) = 0;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("pso")?.pso;
        let groups = MTLSize {
            width: n.div_ceil(2),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let one_gemv = || {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**b);
                    enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&**b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
        };
        // warm
        one_gemv();
        // (a) island-only: reps GEMVs back-to-back.
        let t = std::time::Instant::now();
        for _ in 0..reps {
            one_gemv();
        }
        let island_only = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        // (b) interleaved: each GEMV preceded by a wgpu submit+poll (the flush boundary).
        let t = std::time::Instant::now();
        for _ in 0..reps {
            ctx.queue.submit(std::iter::empty());
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            one_gemv();
        }
        let interleaved = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        // Decompose: isolate the wgpu poll(wait) cost alone (no submit, no island) — is
        // THAT the 1.4ms, or is it the cross-queue/submit? This decides the fix.
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let poll_only = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        eprintln!(
            "  [decompose] island-only {island_only:.4} | poll-only {poll_only:.4} | submit+poll+island {interleaved:.4} ms (each /GEMV)"
        );
        // (c) BATCHED: M island GEMVs in ONE command buffer per cross-queue transition.
        // If the ~1.4ms cross-queue cost amortizes across M, per-LAYER batching (7 GEMVs)
        // is viable without the full megakernel. Measures per-GEMV cost at M = {1,7,48}.
        let batched = |m: usize| -> f64 {
            let t = std::time::Instant::now();
            let outer = reps / m.max(1);
            for _ in 0..outer {
                ctx.queue.submit(std::iter::empty());
                let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
                // M dispatches in ONE island command buffer, ONE waitUntilCompleted.
                let cmd = self.queue.commandBuffer().unwrap();
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                    .unwrap();
                unsafe {
                    enc.setComputePipelineState(pso);
                    for _ in 0..m {
                        for (i, b) in bufs.iter().enumerate() {
                            let r: &ProtocolObject<dyn MTLResource> =
                                ProtocolObject::from_ref(&**b);
                            enc.useResource_usage(
                                r,
                                MTLResourceUsage::Read | MTLResourceUsage::Write,
                            );
                            enc.setBuffer_offset_atIndex(Some(&**b), 0, i);
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                    }
                    enc.endEncoding();
                }
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            t.elapsed().as_secs_f64() * 1000.0 / (outer * m) as f64
        };
        let b7 = batched(7);
        let b48 = batched(48);
        eprintln!(
            "  [batched/transition] 1-per {:.4} | 7-per (layer) {b7:.4} | 48-per (token) {b48:.4} ms/GEMV — amortizes the cross-queue cost",
            interleaved
        );
        Ok((
            interleaved,
            island_only,
            interleaved / island_only.max(1e-9),
        ))
    }
    /// FFN-BLOCK SELFTEST — the first per-token-encoder proof. Runs the whole gemma FFN
    /// (gate=gemv, up=gemv, geglu, down=gemv, rmsnorm_add into hidden) in ONE island command
    /// buffer (one cross-queue transition for 5 ops, not 5), with deterministic Q4KS weights,
    /// and parity-checks the resulting `hidden` against a CPU oracle. Proves the chained-
    /// encoder pattern that the per-token megakernel uses. Requires gemv_q4ks/geglu/
    /// rmsnorm_add compiled. Returns (max_rel, one_encoder_us).
    pub fn ffn_block_selftest(
        &mut self,
        ctx: &GpuContext,
        h: usize,
        inter: usize,
        eps: f32,
        iters: usize,
    ) -> Result<(f32, f64), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs_h = h / 32;
        let subs_i = inter / 32;
        let supers_h = h / 256;
        let supers_i = inter / 256;
        let al = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        // scratch
        let normed = al(h * 4);
        let gate = al(inter * 4);
        let up = al(inter * 4);
        let silu = al(inter * 4); // geglu output (down input)
        let mlp_down = al(h * 4);
        let hidden = al(h * 4);
        let post_w = al(h * 4); // rmsnorm_add weight
                                // weights: gate[inter,h], up[inter,h], down[h,inter] — Q4KS layout each
        let mk_w = |n: usize, k: usize| -> [Retained<ProtocolObject<dyn MTLBuffer>>; 5] {
            let subs = k / 32;
            let supers = k / 256;
            let codes = al(n * subs * 16);
            let sc = al((n * subs).div_ceil(4) * 4);
            let mn = al((n * subs).div_ceil(4) * 4);
            let dd = al(n * supers * 4); // packed half2 {d,dmin}, one u32/super-block
            let dims = al(16);
            unsafe {
                let cp = codes.contents().as_ptr() as *mut u32;
                for i in 0..(n * subs * 4) {
                    *cp.add(i) = (i.wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
                }
                let sp = sc.contents().as_ptr() as *mut u8;
                let mp = mn.contents().as_ptr() as *mut i8;
                for i in 0..(n * subs) {
                    *sp.add(i) = ((i % 31) + 1) as u8;
                    *mp.add(i) = (((i % 17) as i32) - 8) as i8;
                }
                // Scale d/dmin so dequant·activation stays O(1) over the long k (realistic
                // decode magnitudes — pathologically large d makes down ~1e6 and exposes
                // f32 sum precision, not a kernel bug).
                let scl = 1.0 / (k as f32).sqrt();
                let dp = dd.contents().as_ptr() as *mut u32;
                for i in 0..(n * supers) {
                    let dv = (0.02 + (i % 5) as f32 * 0.003) * scl;
                    let dmv = (-0.01 - (i % 3) as f32 * 0.002) * scl;
                    *dp.add(i) =
                        (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
                }
                let d = dims.contents().as_ptr() as *mut u32;
                *d = 1;
                *d.add(1) = k as u32;
                *d.add(2) = n as u32;
                *d.add(3) = 0;
            }
            [codes, sc, mn, dd, dims]
        };
        let gw = mk_w(inter, h);
        let uw = mk_w(inter, h);
        let dw = mk_w(h, inter);
        let (geglu_dims, rmsadd_dims) = (al(16), al(16));
        unsafe {
            let np = normed.contents().as_ptr() as *mut f32;
            for i in 0..h {
                *np.add(i) = ((i % 19) as f32 - 9.0) * 0.04;
            }
            let hp = hidden.contents().as_ptr() as *mut f32;
            for i in 0..h {
                *hp.add(i) = ((i % 7) as f32 - 3.0) * 0.3;
            }
            let wp = post_w.contents().as_ptr() as *mut f32;
            for i in 0..h {
                *wp.add(i) = 1.0 + ((i % 11) as f32 - 5.0) * 0.02;
            }
            let gd = geglu_dims.contents().as_ptr() as *mut u32;
            *gd = inter as u32;
            let rd = rmsadd_dims.contents().as_ptr() as *mut u32;
            *rd = 1;
            *rd.add(1) = h as u32;
            *(rd.add(2) as *mut f32) = eps;
            *rd.add(3) = 0;
        }
        let hid_before: Vec<f32> = unsafe {
            let p = hidden.contents().as_ptr() as *const f32;
            (0..h).map(|i| *p.add(i)).collect()
        };
        drop(guard);
        let pso_gemv = self
            .pipelines
            .get("gemv_q4ks")
            .ok_or("gemv pso")?
            .pso
            .clone();
        let pso_geglu = self.pipelines.get("geglu").ok_or("geglu pso")?.pso.clone();
        let pso_rmsadd = self
            .pipelines
            .get("rmsnorm_add")
            .ok_or("rmsadd pso")?
            .pso
            .clone();
        // record the WHOLE FFN in ONE command buffer.
        let run = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            let gemv = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
                        w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                        a: &Retained<ProtocolObject<dyn MTLBuffer>>,
                        out: &Retained<ProtocolObject<dyn MTLBuffer>>,
                        n: usize| unsafe {
                enc.setComputePipelineState(&pso_gemv);
                let bufs: [&ProtocolObject<dyn MTLBuffer>; 7] =
                    [a, &w[0], out, &w[4], &w[1], &w[2], &w[3]];
                for (i, b) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*b),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n.div_ceil(2),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            };
            unsafe {
                // gate + up read `normed` (no dep between them); barrier before geglu reads both.
                gemv(&enc, &gw, &normed, &gate, inter);
                gemv(&enc, &uw, &normed, &up, inter);
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers); // gate,up → geglu
                                                                      // geglu(gate, up) → silu
                enc.setComputePipelineState(&pso_geglu);
                for (i, b) in [&gate, &up, &silu, &geglu_dims].iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&***b),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: inter.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers); // silu → down
                gemv(&enc, &dw, &silu, &mlp_down, h);
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers); // mlp_down → rmsnorm_add
                                                                      // rmsnorm_add(mlp_down, post_w) → hidden
                enc.setComputePipelineState(&pso_rmsadd);
                for (i, b) in [&mlp_down, &post_w, &hidden, &rmsadd_dims]
                    .iter()
                    .enumerate()
                {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&***b),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
        };
        let _ = run();
        let mut us = f64::INFINITY;
        for _ in 0..iters {
            us = us.min(run());
        }
        // CPU oracle: gate=Q4KS(normed,gw), up=Q4KS(normed,uw), s=geglu(gate,up),
        // down=Q4KS(s,dw), hidden += rmsnorm(down)·post_w.
        let q4ks = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                    a: &[f32],
                    n: usize,
                    k: usize|
         -> Vec<f32> {
            let subs = k / 32;
            let supers = k / 256;
            let cd = unsafe { w[0].contents().as_ptr() as *const u32 };
            let sc = unsafe { w[1].contents().as_ptr() as *const u8 };
            let mn = unsafe { w[2].contents().as_ptr() as *const i8 };
            let dp = unsafe { w[3].contents().as_ptr() as *const u32 };
            (0..n)
                .map(|col| {
                    let mut acc = 0.0f32;
                    for b in 0..subs {
                        let sblk = b / 8;
                        let pair = unsafe { *dp.add(col * supers + sblk) };
                        let dv = f16_bits_to_f32(pair as u16);
                        let dmv = f16_bits_to_f32((pair >> 16) as u16);
                        let su8 = unsafe { *sc.add(col * subs + b) } as f32;
                        let mi8 = unsafe { *mn.add(col * subs + b) } as f32;
                        let (s, lo) = (dv * su8, dmv * mi8);
                        let ab = 32 * b;
                        let (mut cs, mut asum) = (0.0f32, 0.0f32);
                        for wi in 0..4 {
                            let word = unsafe { *cd.add((col * subs + b) * 4 + wi) };
                            for e in 0..8u32 {
                                let av = a[ab + wi * 8 + e as usize];
                                cs += av * ((word >> (4 * e)) & 0xF) as f32;
                                asum += av;
                            }
                        }
                        acc += s * cs + lo * asum;
                    }
                    acc
                })
                .collect()
        };
        let nrm: Vec<f32> = (0..h)
            .map(|i| unsafe { *(normed.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let g = q4ks(&gw, &nrm, inter, h);
        let u = q4ks(&uw, &nrm, inter, h);
        const C: f32 = 0.797_884_6;
        let s: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&x, &uu)| {
                let inner = (C * (x + 0.044715 * x * x * x)).clamp(-15.0, 15.0);
                0.5 * x * (1.0 + inner.tanh()) * uu
            })
            .collect();
        let dn = q4ks(&dw, &s, h, inter);
        let mut ss = 0.0f32;
        for &x in &dn {
            ss += x * x;
        }
        let inv = 1.0 / (ss / h as f32 + eps).sqrt();
        let postw: Vec<f32> = (0..h)
            .map(|i| unsafe { *(post_w.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let mut max_rel = 0.0f32;
        let hp = unsafe { hidden.contents().as_ptr() as *const f32 };
        for i in 0..h {
            let want = hid_before[i] + dn[i] * inv * postw[i];
            let got = unsafe { *hp.add(i) };
            max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-3));
        }
        if std::env::var("ARF_FFN_DUMP").is_ok() {
            let gpu_gate = unsafe {
                let p = gate.contents().as_ptr() as *const f32;
                (0..4).map(|i| *p.add(i)).collect::<Vec<_>>()
            };
            let gpu_silu = unsafe {
                let p = silu.contents().as_ptr() as *const f32;
                (0..4).map(|i| *p.add(i)).collect::<Vec<_>>()
            };
            let gpu_down = unsafe {
                let p = mlp_down.contents().as_ptr() as *const f32;
                (0..4).map(|i| *p.add(i)).collect::<Vec<_>>()
            };
            eprintln!("  [ffn dbg] gate gpu={gpu_gate:?} cpu={:?}", &g[..4]);
            eprintln!("  [ffn dbg] silu gpu={gpu_silu:?} cpu={:?}", &s[..4]);
            eprintln!("  [ffn dbg] down gpu={gpu_down:?} cpu={:?}", &dn[..4]);
            eprintln!(
                "  [ffn dbg] hidden gpu={:?} want={:?}",
                unsafe { (0..4).map(|i| *hp.add(i)).collect::<Vec<_>>() },
                (0..4)
                    .map(|i| hid_before[i] + dn[i] * inv * postw[i])
                    .collect::<Vec<_>>()
            );
        }
        let _ = (subs_h, subs_i, supers_h, supers_i);
        Ok((max_rel, us))
    }
    pub fn batched_attn_block_selftest(
        &mut self,
        ctx: &GpuContext,
        b: usize,       // batch rows (sequences)
        h: usize,       // hidden size
        nh: usize,      // attention heads
        nkv: usize,     // kv heads
        hd: usize,      // head dim
        ctx_len: usize, // per-seq KV-pool capacity (rows)
        eps: f32,
    ) -> Result<(f32, f32), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let rotary_dim = hd; // rotate the full head (faithful to qwen decode)
        let rot_half = rotary_dim / 2;
        // each seq attends `past_len[r]+1` keys; vary past_len per row so the B-row attention
        // really uses per-seq pools (a shared-pool bug would diverge).
        let past_len: Vec<usize> = (0..b)
            .map(|r| 2 + r % (ctx_len.saturating_sub(2)).max(1))
            .collect();
        let positions: Vec<usize> = past_len.clone(); // decode: pos == past_len (token at index past_len)

        // ---- B-row scratch ----
        let hidden = al(b * h * 4)?; // [B,h] (the residual stream; rw)
        let normed = al(b * h * 4)?; // [B,h]
        let qb = al(b * q_dim * 4)?; // [B,q_dim]
        let kb = al(b * kv_dim * 4)?; // [B,kv_dim]
        let vb = al(b * kv_dim * 4)?; // [B,kv_dim]
        let attn = al(b * q_dim * 4)?; // [B,q_dim] (attention output, o_proj input)
        let attn_out = al(b * h * 4)?; // [B,h] (o_proj output)
                                       // per-seq KV pools + slots (each seq independent).
        let mut kpool = Vec::with_capacity(b);
        let mut vpool = Vec::with_capacity(b);
        let mut slots = Vec::with_capacity(b);
        for _r in 0..b {
            kpool.push(al(ctx_len * kv_dim * 4)?);
            vpool.push(al(ctx_len * kv_dim * 4)?);
            let sl = al(ctx_len * 4)?;
            unsafe {
                let p = sl.contents().as_ptr() as *mut u32;
                for s in 0..ctx_len {
                    *p.add(s) = s as u32;
                }
            }
            slots.push(sl);
        }
        // per-head q/k norm weights ([hd]).
        let qnw = al(hd * 4)?;
        let knw = al(hd * 4)?;
        let input_norm = al(h * 4)?;
        // rope tables [max_pos, rot_half] — fill cos/sin deterministically.
        let max_pos = ctx_len + 1;
        let rope_cos = al(max_pos * rot_half * 4)?;
        let rope_sin = al(max_pos * rot_half * 4)?;
        // positions buffer [B] (per-seq pos for rope).
        let pos_buf = al(b * 4)?;

        // ---- deterministic Q4KS weight builder (same packing as ffn_block_selftest) ----
        let supers_of = |k: usize| k / 256;
        let subs_of = |k: usize| k / 32;
        let mk_w =
            |n: usize, k: usize, seed: usize| -> [Retained<ProtocolObject<dyn MTLBuffer>>; 5] {
                let subs = subs_of(k);
                let supers = supers_of(k);
                let codes = al(n * subs * 16).unwrap();
                let sc = al((n * subs).div_ceil(4) * 4).unwrap();
                let mn = al((n * subs).div_ceil(4) * 4).unwrap();
                let dd = al(n * supers * 4).unwrap(); // ddh: packed half2 {d,dmin}, one u32/super-block
                let dims = al(16).unwrap();
                unsafe {
                    let cp = codes.contents().as_ptr() as *mut u32;
                    for i in 0..(n * subs * 4) {
                        *cp.add(i) = (i
                            .wrapping_mul(2654435761)
                            .wrapping_add(seed.wrapping_mul(40503))
                            & 0xFFFF_FFFF) as u32;
                    }
                    let sp = sc.contents().as_ptr() as *mut u8;
                    let mp = mn.contents().as_ptr() as *mut i8;
                    for i in 0..(n * subs) {
                        *sp.add(i) = (((i + seed) % 31) + 1) as u8;
                        *mp.add(i) = ((((i + seed) % 17) as i32) - 8) as i8;
                    }
                    let scl = 1.0 / (k as f32).sqrt();
                    let dp = dd.contents().as_ptr() as *mut u32;
                    for i in 0..(n * supers) {
                        let dv = (0.02 + (i % 5) as f32 * 0.003) * scl;
                        let dmv = (-0.01 - (i % 3) as f32 * 0.002) * scl;
                        *dp.add(i) =
                            (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
                    }
                    let d = dims.contents().as_ptr() as *mut u32;
                    *d = 1;
                    *d.add(1) = k as u32;
                    *d.add(2) = n as u32;
                    *d.add(3) = 0;
                }
                [codes, sc, mn, dd, dims]
            };
        let qw = mk_w(q_dim, h, 1); // q_proj [q_dim, h]
        let kw = mk_w(kv_dim, h, 2); // k_proj [kv_dim, h]
        let vw = mk_w(kv_dim, h, 3); // v_proj [kv_dim, h]
        let ow = mk_w(h, q_dim, 4); // o_proj [h, q_dim]

        // dims uniforms.
        let norm_dims = al(16)?; // {tokens=B,h,eps,_}
        let qk_dims = al(16)?; // {q_rows=nh,k_rows=nkv,hd,eps}
        let rope_dims = al(32)?; // {tokens=B,q_heads=nh,k_heads=nkv,hd,rotary_dim,_,_,_}
        let resid_dims = al(16)?; // {tokens=B,h,_,_}
        unsafe {
            let p = norm_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *(p.add(2) as *mut f32) = eps;
            *p.add(3) = 0;
            let p = qk_dims.contents().as_ptr() as *mut u32;
            *p = nh as u32;
            *p.add(1) = nkv as u32;
            *p.add(2) = hd as u32;
            *(p.add(3) as *mut f32) = eps;
            let p = rope_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = nh as u32;
            *p.add(2) = nkv as u32;
            *p.add(3) = hd as u32;
            *p.add(4) = rotary_dim as u32;
            *p.add(5) = 0;
            *p.add(6) = 0;
            *p.add(7) = 0;
            let p = resid_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = 0;
            *p.add(3) = 0;
            let pp = pos_buf.contents().as_ptr() as *mut u32;
            for r in 0..b {
                *pp.add(r) = positions[r] as u32;
            }
        }
        // Per-weight B-row GEMV dims {m=B, k, n, _}: the WEIGHT's own dims buffer carries m=1
        // (fixed per weight) — the B-row GEMV needs m=B (else the kernel clamps B=min(d.m,MAXB)=1
        // and only row 0 is written; that was the row-1-zero bug). n differs (q_dim/kv_dim/h), so
        // one each. k = the input width (h for q/k/v reading normed; q_dim for o reading attn).
        let mk_gdims =
            |k: usize, n: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
                let bd = al(16)?;
                unsafe {
                    let p = bd.contents().as_ptr() as *mut u32;
                    *p = b as u32;
                    *p.add(1) = k as u32;
                    *p.add(2) = n as u32;
                    *p.add(3) = 0;
                }
                Ok(bd)
            };
        let gd_q = mk_gdims(h, q_dim)?; // q_proj: k=h, n=q_dim
        let gd_k = mk_gdims(h, kv_dim)?; // k_proj: k=h, n=kv_dim
        let gd_v = mk_gdims(h, kv_dim)?; // v_proj: k=h, n=kv_dim
        let gd_o = mk_gdims(q_dim, h)?; // o_proj: k=q_dim, n=h

        // ---- fill inputs deterministically ----
        unsafe {
            let hp = hidden.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..h {
                    *hp.add(r * h + i) = (((r * 13 + i) % 23) as f32 - 11.0) * 0.05;
                }
            }
            let qp = qnw.contents().as_ptr() as *mut f32;
            let kp = knw.contents().as_ptr() as *mut f32;
            for i in 0..hd {
                *qp.add(i) = 1.0 + ((i % 7) as f32 - 3.0) * 0.03;
                *kp.add(i) = 1.0 + ((i % 5) as f32 - 2.0) * 0.04;
            }
            let ip = input_norm.contents().as_ptr() as *mut f32;
            for i in 0..h {
                *ip.add(i) = 1.0 + ((i % 11) as f32 - 5.0) * 0.02;
            }
            // rope tables: cos/sin of a per-(pos,i) angle (deterministic, not the real freqs but
            // self-consistent — the oracle reads the SAME tables).
            let cp = rope_cos.contents().as_ptr() as *mut f32;
            let sp = rope_sin.contents().as_ptr() as *mut f32;
            for pos in 0..max_pos {
                for i in 0..rot_half {
                    let ang =
                        (pos as f32) * (1.0 / (10000f32).powf(2.0 * i as f32 / rotary_dim as f32));
                    *cp.add(pos * rot_half + i) = ang.cos();
                    *sp.add(pos * rot_half + i) = ang.sin();
                }
            }
            // KV pools: per-seq, only rows [0..=past_len[r]] are valid (filled); attention reads
            // slots[0..=past_len[r]]. Fill all ctx_len rows deterministically per seq.
            for r in 0..b {
                let k = kpool[r].contents().as_ptr() as *mut f32;
                let v = vpool[r].contents().as_ptr() as *mut f32;
                for row in 0..ctx_len {
                    for j in 0..kv_dim {
                        *k.add(row * kv_dim + j) = (((r * 5 + row + j) % 19) as f32 - 9.0) * 0.03;
                        *v.add(row * kv_dim + j) =
                            (((r * 7 + row * 3 + j) % 13) as f32 - 6.0) * 0.04;
                    }
                }
            }
        }
        drop(guard);

        // ---- pipelines (B-row + per-seq m=1) ----
        let p_gemv_b = self
            .pipelines
            .get("gemv_q4ks_batch")
            .ok_or("gemv_q4ks_batch pso (compile decode_ops_batch first)")?
            .pso
            .clone();
        let p_rms_b = self
            .pipelines
            .get("rmsnorm_b")
            .ok_or("rmsnorm_b pso")?
            .pso
            .clone();
        let p_qk_b = self
            .pipelines
            .get("qk_norm_b")
            .ok_or("qk_norm_b pso")?
            .pso
            .clone();
        let p_rope_b = self
            .pipelines
            .get("rope_qk_b")
            .ok_or("rope_qk_b pso")?
            .pso
            .clone();
        let p_resid_b = self
            .pipelines
            .get("add_residual_b")
            .ok_or("add_residual_b pso")?
            .pso
            .clone();
        let p_scatter = self
            .pipelines
            .get("kv_scatter")
            .ok_or("kv_scatter pso")?
            .pso
            .clone();
        let p_attn = self
            .pipelines
            .get("attention_decode")
            .ok_or("attention_decode pso")?
            .pso
            .clone();

        // per-seq attn dims (12 u32) — one buffer per row (ctx/past_len differ per seq).
        let mut attn_dims = Vec::with_capacity(b);
        let mut scatter_dims = Vec::with_capacity(b);
        let mut pos_one = Vec::with_capacity(b); // [slots[pos]] = [pos] for scatter
        {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            for r in 0..b {
                let ad = al2(48);
                unsafe {
                    let p = ad.contents().as_ptr() as *mut u32;
                    p.add(0).write(1); // q_len
                    p.add(1).write((past_len[r] + 1) as u32); // ctx
                    p.add(2).write(nh as u32);
                    p.add(3).write(nkv as u32);
                    p.add(4).write(hd as u32);
                    p.add(5).write(past_len[r] as u32); // past_len
                    (p.add(6) as *mut f32).write(scale);
                    p.add(7).write(group as u32);
                    p.add(8).write(0); // q_start
                    p.add(9).write(0); // window (global)
                    p.add(10).write(0);
                    p.add(11).write(0);
                }
                attn_dims.push(ad);
                let sd = al2(16);
                unsafe {
                    let p = sd.contents().as_ptr() as *mut u32;
                    *p = 1;
                    *p.add(1) = kv_dim as u32;
                    *p.add(2) = 0;
                    *p.add(3) = 0;
                }
                scatter_dims.push(sd);
                let po = al2(16);
                unsafe {
                    *(po.contents().as_ptr() as *mut u32) = positions[r] as u32;
                }
                pos_one.push(po);
            }
            drop(guard);
        }
        // snapshot hidden BEFORE (the residual add target).
        let hid_before: Vec<f32> = unsafe {
            let p = hidden.contents().as_ptr() as *const f32;
            (0..b * h).map(|i| *p.add(i)).collect()
        };

        // ===== record the B-row attention block in ONE concurrent command buffer =====
        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        let tg32 = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let tg64 = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        unsafe {
            let bind = |bufs: &[&ProtocolObject<dyn MTLBuffer>]| {
                for (i, bf) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
            };
            let barrier = || enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            // B-row GEMV: one simdgroup (32 lanes) per output column; grid.x = n. Binds the
            // B-ROW dims (m=B) — NOT the weight's own dims (w[4], which carries m=1, fixed per
            // weight). That m=1 was the row-1-zero bug: the kernel clamps B=min(d.m,MAXB).
            let gemv_b = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                          a: &ProtocolObject<dyn MTLBuffer>,
                          c: &ProtocolObject<dyn MTLBuffer>,
                          gd: &ProtocolObject<dyn MTLBuffer>,
                          n: usize| {
                enc.setComputePipelineState(&p_gemv_b);
                bind(&[a, &w[0], c, gd, &w[1], &w[2], &w[3]]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: 1,
                        depth: 1,
                    },
                    tg32,
                );
            };
            // 1. in_norm: hidden[B,h] → normed[B,h]  (grid.x = B threadgroups)
            enc.setComputePipelineState(&p_rms_b);
            bind(&[&hidden, &input_norm, &normed, &norm_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier(); // normed → q/k/v
                       // 2-4. q/k/v proj (read normed, write disjoint q/k/v).
            gemv_b(&qw, &normed, &qb, &gd_q, q_dim);
            gemv_b(&kw, &normed, &kb, &gd_k, kv_dim);
            gemv_b(&vw, &normed, &vb, &gd_v, kv_dim);
            barrier(); // q/k written → qk_norm
                       // 5. qk_norm (q/k only): grid.x = B*(nh+nkv).
            enc.setComputePipelineState(&p_qk_b);
            bind(&[&qb, &kb, &qnw, &knw, &qk_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b * (nh + nkv),
                    height: 1,
                    depth: 1,
                },
                tg64,
            );
            barrier(); // qk_norm → rope
                       // 6. rope_qk (in-place q,k): grid covers B*(q_dim+kv_dim) elements.
            enc.setComputePipelineState(&p_rope_b);
            bind(&[&qb, &kb, &rope_cos, &rope_sin, &pos_buf, &rope_dims]);
            let rope_threads = b * (q_dim + kv_dim);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rope_threads.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier(); // k/v ready → scatter; q ready → attention
                       // 7. kv_scatter PER-SEQ: each row scatters its own k/v row into its own pool at slot pos.
            for r in 0..b {
                enc.setComputePipelineState(&p_scatter);
                // bind k/v at this row's offset (row-major [B,kv_dim]); src_row=0 in the dims, so the
                // kernel reads new_k[0..kv_dim] from the bound offset base.
                let koff = r * kv_dim * 4;
                let voff = r * kv_dim * 4;
                enc.useResource_usage(ProtocolObject::from_ref(&*kb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kb), koff as _, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*vb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vb), voff as _, 1);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*pos_one[r]),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&pos_one[r]), 0, 2);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*kpool[r]),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&kpool[r]), 0, 3);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*vpool[r]),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&vpool[r]), 0, 4);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*scatter_dims[r]),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&scatter_dims[r]), 0, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: kv_dim.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
            barrier(); // pools updated → attention
                       // 8. attention PER-SEQ: each row attends its own pool, writes its own attn[r,:].
            for r in 0..b {
                enc.setComputePipelineState(&p_attn);
                let qoff = r * q_dim * 4;
                let aoff = r * q_dim * 4;
                enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&qb), qoff as _, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*kpool[r]), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kpool[r]), 0, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&*vpool[r]), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vpool[r]), 0, 2);
                enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&attn), aoff as _, 3);
                enc.useResource_usage(ProtocolObject::from_ref(&*slots[r]), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&slots[r]), 0, 4);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*attn_dims[r]),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&attn_dims[r]), 0, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: nh,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
            barrier(); // attn → o_proj
                       // 9. o_proj: attn[B,q_dim] → attn_out[B,h].
            gemv_b(&ow, &attn, &attn_out, &gd_o, h);
            barrier(); // attn_out → residual
                       // 10. residual: hidden[B,h] += attn_out[B,h].
            enc.setComputePipelineState(&p_resid_b);
            bind(&[&hidden, &attn_out, &resid_dims]);
            let resid_threads = b * h;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: resid_threads.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ===== CPU oracle (per row): the EXACT same attention-block math =====
        // q4ks dequant-GEMV (same as ffn_block_selftest oracle).
        let q4ks = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                    a: &[f32],
                    n: usize,
                    k: usize|
         -> Vec<f32> {
            let subs = k / 32;
            let supers = k / 256;
            let cd = unsafe { w[0].contents().as_ptr() as *const u32 };
            let sc = unsafe { w[1].contents().as_ptr() as *const u8 };
            let mn = unsafe { w[2].contents().as_ptr() as *const i8 };
            let dp = unsafe { w[3].contents().as_ptr() as *const u32 };
            (0..n)
                .map(|col| {
                    let mut acc = 0.0f32;
                    for bb in 0..subs {
                        let sblk = bb / 8;
                        let pair = unsafe { *dp.add(col * supers + sblk) };
                        let dv = f16_bits_to_f32(pair as u16);
                        let dmv = f16_bits_to_f32((pair >> 16) as u16);
                        let su8 = unsafe { *sc.add(col * subs + bb) } as f32;
                        let mi8 = unsafe { *mn.add(col * subs + bb) } as f32;
                        let (s, lo) = (dv * su8, dmv * mi8);
                        let ab = 32 * bb;
                        let (mut cs, mut asum) = (0.0f32, 0.0f32);
                        for wi in 0..4 {
                            let word = unsafe { *cd.add((col * subs + bb) * 4 + wi) };
                            for e in 0..8u32 {
                                let av = a[ab + wi * 8 + e as usize];
                                cs += av * ((word >> (4 * e)) & 0xF) as f32;
                                asum += av;
                            }
                        }
                        acc += s * cs + lo * asum;
                    }
                    acc
                })
                .collect()
        };
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let n = x.len();
            let mut ss = 0.0f32;
            for &v in x {
                ss += v * v;
            }
            let inv = 1.0 / (ss / n as f32 + eps).sqrt();
            (0..n).map(|i| x[i] * inv * w[i]).collect()
        };
        let qnw_v: Vec<f32> = (0..hd)
            .map(|i| unsafe { *(qnw.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let knw_v: Vec<f32> = (0..hd)
            .map(|i| unsafe { *(knw.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let innw_v: Vec<f32> = (0..h)
            .map(|i| unsafe { *(input_norm.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let cos_v: Vec<f32> = (0..max_pos * rot_half)
            .map(|i| unsafe { *(rope_cos.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let sin_v: Vec<f32> = (0..max_pos * rot_half)
            .map(|i| unsafe { *(rope_sin.contents().as_ptr() as *const f32).add(i) })
            .collect();

        let mut want = vec![0.0f32; b * h];
        for r in 0..b {
            let hin = &hid_before[r * h..r * h + h];
            let normed_r = rms(hin, &innw_v);
            let mut q = q4ks(&qw, &normed_r, q_dim, h);
            let mut k = q4ks(&kw, &normed_r, kv_dim, h);
            let vv = q4ks(&vw, &normed_r, kv_dim, h);
            // qk_norm per head (q/k only)
            for hh in 0..nh {
                let seg = rms(&q[hh * hd..hh * hd + hd], &qnw_v);
                q[hh * hd..hh * hd + hd].copy_from_slice(&seg);
            }
            for hh in 0..nkv {
                let seg = rms(&k[hh * hd..hh * hd + hd], &knw_v);
                k[hh * hd..hh * hd + hd].copy_from_slice(&seg);
            }
            // rope on q (nh heads) and k (nkv heads)
            let pos = positions[r];
            let rope = |buf: &mut [f32], heads: usize| {
                for hh in 0..heads {
                    let base = hh * hd;
                    for i in 0..rot_half {
                        let c = cos_v[pos * rot_half + i];
                        let s = sin_v[pos * rot_half + i];
                        let a = buf[base + i];
                        let bv = buf[base + i + rot_half];
                        buf[base + i] = a * c - bv * s;
                        buf[base + i + rot_half] = bv * c + a * s;
                    }
                }
            };
            rope(&mut q, nh);
            rope(&mut k, nkv);
            // scatter: pool[pos] = k,v for this seq; build the effective pool view (rows 0..=past_len)
            // by reading the original pool then overwriting row `pos`.
            let kp_ptr = kpool[r].contents().as_ptr() as *const f32;
            let vp_ptr = vpool[r].contents().as_ptr() as *const f32;
            let pl = past_len[r];
            // attention per head (GQA): each head attends keys 0..=past_len of its kv_head, scale, softmax, ·V
            let mut attn_r = vec![0.0f32; q_dim];
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * hd;
                let mut scs = vec![0.0f32; pl + 1];
                let mut mx = f32::MIN;
                for key in 0..=pl {
                    // key row in pool: scatter wrote row `pos`(==pl) with this token's k; other rows are
                    // the prefilled pool. The CPU pool view = original pool, with row pos replaced by k.
                    let mut s = 0.0f32;
                    for i in 0..hd {
                        let kval = if key == pos {
                            k[kv_head * hd + i]
                        } else {
                            unsafe { *kp_ptr.add(key * kv_dim + hcol + i) }
                        };
                        s += q[hh * hd + i] * kval;
                    }
                    scs[key] = s * scale;
                    mx = mx.max(scs[key]);
                }
                let mut den = 0.0f32;
                for key in 0..=pl {
                    scs[key] = (scs[key] - mx).exp();
                    den += scs[key];
                }
                for dim in 0..hd {
                    let mut num = 0.0f32;
                    for key in 0..=pl {
                        let vval = if key == pos {
                            vv[kv_head * hd + dim]
                        } else {
                            unsafe { *vp_ptr.add(key * kv_dim + hcol + dim) }
                        };
                        num += scs[key] * vval;
                    }
                    attn_r[hh * hd + dim] = num / den;
                }
            }
            // o_proj: attn[q_dim] → attn_out[h]; hidden += attn_out
            let aout = q4ks(&ow, &attn_r, h, q_dim);
            for i in 0..h {
                want[r * h + i] = hin[i] + aout[i];
            }
        }

        // compare GPU hidden vs oracle. Track the worst element + its magnitude so a near-zero
        // (where a tiny abs diff inflates the ratio) is distinguishable from a real divergence.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        unsafe {
            let hp = hidden.contents().as_ptr() as *const f32;
            for i in 0..b * h {
                let got = *hp.add(i);
                let w = want[i];
                let abs = (got - w).abs();
                max_abs = max_abs.max(abs);
                let rel = abs / w.abs().max(1e-3);
                if rel > max_rel {
                    max_rel = rel;
                    worst = (i, got, w);
                }
            }
        }
        if std::env::var("ARF_BATCH_MEGA_DUMP").is_ok() {
            eprintln!(
                "  [batch dbg] worst rel @ idx {} (row {}): gpu={} want={} (|want|={:.2e})",
                worst.0,
                worst.0 / h,
                worst.1,
                worst.2,
                worst.2.abs()
            );
        }
        if std::env::var("ARF_BATCH_MEGA_DUMP").is_ok() {
            unsafe {
                let dump = |name: &str, buf: &ProtocolObject<dyn MTLBuffer>, stride: usize| {
                    let p = buf.contents().as_ptr() as *const f32;
                    for r in 0..b.min(2) {
                        let g: Vec<f32> = (0..4).map(|i| *p.add(r * stride + i)).collect();
                        eprintln!("  [batch dbg] row{r} {name}={g:?}");
                    }
                };
                dump("normed", &normed, h);
                dump("q", &qb, q_dim);
                dump("attn", &attn, q_dim);
                dump("attn_out", &attn_out, h);
                let hp = hidden.contents().as_ptr() as *const f32;
                for r in 0..b.min(2) {
                    let g: Vec<f32> = (0..4).map(|i| *hp.add(r * h + i)).collect();
                    let w: Vec<f32> = (0..4).map(|i| want[r * h + i]).collect();
                    eprintln!("  [batch dbg] row{r} hidden gpu={g:?} want={w:?}");
                }
            }
        }
        Ok((max_abs, max_rel))
    }
    pub fn batched_moe_block_selftest(
        &mut self,
        ctx: &GpuContext,
        b: usize,     // batch rows (sequences)
        h: usize,     // hidden size
        inter: usize, // moe per-expert intermediate (= moe_inter)
        num_experts: usize,
        top_k: usize,
        norm_topk: bool,
        eps: f32,
    ) -> Result<(f32, f32), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };

        // ---- B-row scratch ----
        let hidden = al(b * h * 4)?; // [B,h] residual stream (rw)
        let attn_res = al(b * h * 4)?; // [B,h] the attention output added by add_norm
        let normed = al(b * h * 4)?; // [B,h]
        let logits = al(b * num_experts * 4)?; // [B,num_experts]
        let ids = al(b * top_k * 4)?; // [B,top_k] u32
        let wts = al(b * top_k * 4)?; // [B,top_k] f32
        let gate_all = al(b * top_k * inter * 4)?; // [B,top_k,inter]
        let up_all = al(b * top_k * inter * 4)?;
        let silu_all = al(b * top_k * inter * 4)?;
        let mlp_down = al(b * h * 4)?; // [B,h]
        let down_partials = al(b * top_k * h * 4)?; // [B,top_k,h] for ARF_DOWN_PARALLEL 2-pass
                                                    // Sort-by-expert CSR scratch (ARF_MOE_SORTED). down_slots REUSES down_partials.
        let csr_offsets = al((num_experts + 1) * 4)?;
        let csr_row_slot = al(b * top_k * 4)?;
        let csr_active = al(num_experts * 4)?;
        let csr_meta = al(16)?;
        let csr_dims = al(16)?; // {n=B*top_k, ne, 0, 0}
        let csr_max_tiles = num_experts + (b * top_k).div_ceil(moe_mm_nr1()) + 1; // NR1 (32 or 8) must match the kernel
        let csr_tile_list = al(csr_max_tiles * 4)?; // precise tile-list (expert<<16)|tile_idx
        unsafe {
            let p = csr_dims.contents().as_ptr() as *mut u32;
            *p = (b * top_k) as u32;
            *p.add(1) = num_experts as u32;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let pre_ffn_norm = al(h * 4)?; // add_norm weight

        // ---- deterministic Q4KS weight builder (dense + all-experts MoE) ----
        let subs_of = |k: usize| k / 32;
        let supers_of = |k: usize| k / 256;
        // `extra` shrinks the dequant scale so the residual stream stays O(1) (the down GEMV
        // accumulates over top_k slots × inter, which otherwise blows magnitudes to ~75 and the
        // f32 accumulation noise floor to ~2e-4 — masking the strict abs bar). Keeps the test on
        // the SAME max_abs<1e-4 bar as the first parity gate.
        let mk_w = |n: usize,
                    k: usize,
                    seed: usize,
                    extra: f32|
         -> [Retained<ProtocolObject<dyn MTLBuffer>>; 5] {
            let subs = subs_of(k);
            let supers = supers_of(k);
            let codes = al(n * subs * 16).unwrap();
            let sc = al((n * subs).div_ceil(4) * 4).unwrap();
            let mn = al((n * subs).div_ceil(4) * 4).unwrap();
            let dd = al(n * supers * 4).unwrap(); // ddh: packed half2 {d,dmin}, one u32/super-block
            let dims = al(16).unwrap();
            unsafe {
                let cp = codes.contents().as_ptr() as *mut u32;
                for i in 0..(n * subs * 4) {
                    *cp.add(i) = (i
                        .wrapping_mul(2654435761)
                        .wrapping_add(seed.wrapping_mul(40503))
                        & 0xFFFF_FFFF) as u32;
                }
                let sp = sc.contents().as_ptr() as *mut u8;
                let mp = mn.contents().as_ptr() as *mut i8;
                for i in 0..(n * subs) {
                    *sp.add(i) = (((i + seed) % 31) + 1) as u8;
                    *mp.add(i) = ((((i + seed) % 17) as i32) - 8) as i8;
                }
                let scl = extra / (k as f32).sqrt();
                let dp = dd.contents().as_ptr() as *mut u32;
                for i in 0..(n * supers) {
                    let dv = (0.02 + (i % 5) as f32 * 0.003) * scl;
                    let dmv = (-0.01 - (i % 3) as f32 * 0.002) * scl;
                    *dp.add(i) =
                        (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
                }
                let d = dims.contents().as_ptr() as *mut u32;
                *d = 1;
                *d.add(1) = k as u32;
                *d.add(2) = n as u32;
                *d.add(3) = 0;
            }
            [codes, sc, mn, dd, dims]
        };
        // router: dense Q4KS [num_experts, h]. gate/up: all-experts [num_experts*inter, h].
        // down: all-experts [num_experts*h, inter]. (n = experts concatenated × per-expert cols.)
        // gate/up/down shrunk so silu(gate)*up and the top_k-summed down stay O(1) (small abs floor).
        let router_w = mk_w(num_experts, h, 11, 1.0);
        let gate_w = mk_w(num_experts * inter, h, 21, 0.5);
        let up_w = mk_w(num_experts * inter, h, 31, 0.5);
        let down_w = mk_w(num_experts * h, inter, 41, 0.2);

        // ---- dims uniforms ----
        let norm_dims = al(16)?; // add_norm {tokens=B,h,eps,_}
        let router_gdims = al(16)?; // dense B-row GEMV {m=B,k=h,n=num_experts,_}
        let gu_dims = al(16)?; // id GEMV {m=B,k=h,n=inter,top_k}
        let down_dims = al(16)?; // id down {m=B,k=inter,n=h,top_k}
        let route_dims = al(16)?; // {num_experts,top_k,norm_topk,_}
        let swiglu_dims = al(16)?; // {B*top_k*inter,_,_,_}
        let resid_dims = al(16)?; // add_residual_b {tokens=B,h,_,_}
        unsafe {
            let p = norm_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *(p.add(2) as *mut f32) = eps;
            *p.add(3) = 0;
            let p = router_gdims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = num_experts as u32;
            *p.add(3) = 0;
            let p = gu_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = inter as u32;
            *p.add(3) = top_k as u32;
            let p = down_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = inter as u32;
            *p.add(2) = h as u32;
            *p.add(3) = top_k as u32;
            let p = route_dims.contents().as_ptr() as *mut u32;
            *p = num_experts as u32;
            *p.add(1) = top_k as u32;
            *p.add(2) = norm_topk as u32;
            *p.add(3) = 0;
            let p = swiglu_dims.contents().as_ptr() as *mut u32;
            *p = (b * top_k * inter) as u32;
            *p.add(1) = 0;
            *p.add(2) = 0;
            *p.add(3) = 0;
            let p = resid_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }

        // ---- fill inputs ----
        unsafe {
            let hp = hidden.contents().as_ptr() as *mut f32;
            let ap = attn_res.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..h {
                    *hp.add(r * h + i) = (((r * 13 + i) % 23) as f32 - 11.0) * 0.05;
                    *ap.add(r * h + i) = (((r * 7 + i * 3) % 17) as f32 - 8.0) * 0.03;
                }
            }
            let wp = pre_ffn_norm.contents().as_ptr() as *mut f32;
            for i in 0..h {
                *wp.add(i) = 1.0 + ((i % 11) as f32 - 5.0) * 0.02;
            }
        }
        drop(guard);

        // snapshot hidden BEFORE (residual target before add_norm's in-place += attn_res).
        let hid_before: Vec<f32> = unsafe {
            let p = hidden.contents().as_ptr() as *const f32;
            (0..b * h).map(|i| *p.add(i)).collect()
        };

        // ---- pipelines (B-row) ----
        let p_gemv_b = self
            .pipelines
            .get("gemv_q4ks_batch")
            .ok_or("gemv_q4ks_batch pso")?
            .pso
            .clone();
        let p_add_norm = self
            .pipelines
            .get("add_norm")
            .ok_or("add_norm pso")?
            .pso
            .clone();
        let p_resid_b = self
            .pipelines
            .get("add_residual_b")
            .ok_or("add_residual_b pso")?
            .pso
            .clone();
        let p_route_b = self
            .pipelines
            .get("moe_route_b")
            .ok_or("moe_route_b pso")?
            .pso
            .clone();
        let p_id_b = self
            .pipelines
            .get("gemv_q4ks_id_b")
            .ok_or("gemv_q4ks_id_b pso")?
            .pso
            .clone();
        let p_swiglu_b = self
            .pipelines
            .get("moe_swiglu_b")
            .ok_or("moe_swiglu_b pso")?
            .pso
            .clone();
        let p_down_b = self
            .pipelines
            .get("gemv_q4ks_id_down_b")
            .ok_or("gemv_q4ks_id_down_b pso")?
            .pso
            .clone();
        let down_parallel = std::env::var_os("ARF_DOWN_PARALLEL").is_some();
        let p_down_partial_b = self
            .pipelines
            .get("gemv_q4ks_id_down_partial_b")
            .map(|p| p.pso.clone());
        let p_down_reduce_b = self
            .pipelines
            .get("moe_down_reduce_b")
            .map(|p| p.pso.clone());
        // Sort-by-expert path (ARF_MOE_SORTED) — mirrors the production splice; the SAME CPU
        // oracle (below) validates it. In the selftest the b>=32 production gate is dropped so B=1/2/8
        // exercise it (B=1 reduces the CSR to top_k singleton groups = still exact).
        let moe_sorted = std::env::var_os("ARF_MOE_SORTED").is_some();
        let p_csr = self
            .pipelines
            .get("moe_expert_csr_msl")
            .map(|p| p.pso.clone());
        let p_mm_gu = self
            .pipelines
            .get("moe_mm_id_gu_q4ks")
            .map(|p| p.pso.clone());
        let p_mm_down = self
            .pipelines
            .get("moe_mm_id_down_q4ks")
            .map(|p| p.pso.clone());
        let p_reduce_slots = self
            .pipelines
            .get("moe_reduce_slots_b")
            .map(|p| p.pso.clone());
        let use_moe_sorted = moe_sorted
            && p_csr.is_some()
            && p_mm_gu.is_some()
            && p_mm_down.is_some()
            && p_reduce_slots.is_some();
        // HARD GUARD (the lm_head/GQA trap): if the block-dequant hoist is compiled in
        // (ARF_MOE_MM_HOIST) the tiled mm_id kernels MUST be the path actually exercised — a
        // green run that secretly validated the GEMV path would be a LIE. The hoist lives ONLY in
        // moe_mm_id_gu/down_q4ks, which run only on the sorted path. Refuse to proceed otherwise.
        if std::env::var_os("ARF_MOE_MM_HOIST").is_some() && !use_moe_sorted {
            return Err(format!(
                "ARF_MOE_MM_HOIST set but the tiled mm_id path is NOT exercised (use_moe_sorted={use_moe_sorted}). \
                 Set ARF_MOE_SORTED=1 so the hoisted moe_mm_id_gu/down_q4ks kernels actually run — \
                 otherwise this selftest validates the un-hoisted GEMV path (a false PASS)."
            ));
        }
        // SAME trap for the NR1 shrink: ARF_MOE_NR1_8 only changes the tiled mm_id kernels (NR1 +
        // CSR_MM_BM). If the sorted path isn't exercised, a green run validated the GEMV path and the
        // NR1=8 fragment re-partition was never run — a false PASS. Refuse.
        if std::env::var_os("ARF_MOE_NR1_8").is_some() && !use_moe_sorted {
            return Err(format!(
                "ARF_MOE_NR1_8 set but the tiled mm_id path is NOT exercised (use_moe_sorted={use_moe_sorted}). \
                 Set ARF_MOE_SORTED=1 so the NR1=8 moe_mm_id_gu/down_q4ks kernels actually run."
            ));
        }

        // ===== record the B-row MoE FFN block in ONE concurrent command buffer =====
        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        let tg32 = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let tg128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        unsafe {
            let bind = |bufs: &[&ProtocolObject<dyn MTLBuffer>]| {
                for (i, bf) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
            };
            let barrier = || enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            // dense B-row GEMV (router): one simdgroup per output col; grid.x = n.
            let gemv_b = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                          a: &ProtocolObject<dyn MTLBuffer>,
                          c: &ProtocolObject<dyn MTLBuffer>,
                          gd: &ProtocolObject<dyn MTLBuffer>,
                          n: usize| {
                enc.setComputePipelineState(&p_gemv_b);
                bind(&[a, &w[0], c, gd, &w[1], &w[2], &w[3]]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: 1,
                        depth: 1,
                    },
                    tg32,
                );
            };
            // 1. add_norm: hidden += attn_res; normed = rmsnorm(hidden)·pre_ffn_norm. B threadgroups.
            enc.setComputePipelineState(&p_add_norm);
            bind(&[&hidden, &attn_res, &pre_ffn_norm, &normed, &norm_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier(); // normed → router + gate/up
                       // 2. router GEMV: normed[B,h] → logits[B,num_experts].
            gemv_b(&router_w, &normed, &logits, &router_gdims, num_experts);
            barrier(); // logits → route
                       // 3. moe_route_b: logits → ids[B,top_k] + wts[B,top_k]. B threadgroups (one per row).
            enc.setComputePipelineState(&p_route_b);
            bind(&[&logits, &ids, &wts, &route_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg128,
            );
            barrier(); // ids → gate/up indirect
            if use_moe_sorted {
                // ===== SORT-BY-EXPERT path (ARF_MOE_SORTED) — mirrors the production splice.
                // route(done) → CSR scatter → grouped tiled mm_id (gate/up) → swiglu → grouped down
                // (PLACE into down_partials==down_slots) → slot-reduce+router fold → mlp_down. Same
                // (row,slot,col) layout the GEMV writes → the SAME CPU oracle validates it. =====
                let (p_csr, p_mm_gu, p_mm_down, p_reduce) = (
                    p_csr.as_ref().unwrap(),
                    p_mm_gu.as_ref().unwrap(),
                    p_mm_down.as_ref().unwrap(),
                    p_reduce_slots.as_ref().unwrap(),
                );
                // (1) CSR scatter: ids → offsets/row_slot/active/meta + precise tile-list.
                enc.setComputePipelineState(p_csr);
                bind(&[
                    &ids,
                    &csr_offsets,
                    &csr_row_slot,
                    &csr_active,
                    &csr_meta,
                    &csr_dims,
                    &csr_tile_list,
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
                barrier(); // csr → grouped GEMM
                           // grid.x = max_tiles (precise tile-list indexed by grid.x; over-shoots n_tiles → early-out).
                let mm_grid = |n: usize| MTLSize {
                    width: csr_max_tiles,
                    height: n.div_ceil(64),
                    depth: 1,
                };
                // (2) grouped gate GEMM → gate_all.
                enc.setComputePipelineState(p_mm_gu);
                enc.setThreadgroupMemoryLength_atIndex(12288, 0);
                bind(&[
                    &normed,
                    &gate_w[0],
                    &gate_all,
                    &gu_dims,
                    &gate_w[1],
                    &gate_w[2],
                    &gate_w[3],
                    &csr_offsets,
                    &csr_row_slot,
                    &csr_active,
                    &csr_meta,
                    &csr_tile_list,
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(mm_grid(inter), tg128);
                // (3) grouped up GEMM → up_all.
                enc.setComputePipelineState(p_mm_gu);
                enc.setThreadgroupMemoryLength_atIndex(12288, 0);
                bind(&[
                    &normed,
                    &up_w[0],
                    &up_all,
                    &gu_dims,
                    &up_w[1],
                    &up_w[2],
                    &up_w[3],
                    &csr_offsets,
                    &csr_row_slot,
                    &csr_active,
                    &csr_meta,
                    &csr_tile_list,
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(mm_grid(inter), tg128);
                barrier(); // gate/up → swiglu
                           // (4) swiglu (UNCHANGED).
                enc.setComputePipelineState(&p_swiglu_b);
                bind(&[&gate_all, &up_all, &silu_all, &swiglu_dims]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (b * top_k * inter).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
                barrier(); // silu → down
                           // (5) grouped down GEMM → down_partials[B,top_k,h] (PLACE, router-weight-agnostic).
                enc.setComputePipelineState(p_mm_down);
                enc.setThreadgroupMemoryLength_atIndex(12288, 0);
                bind(&[
                    &silu_all,
                    &down_w[0],
                    &down_partials,
                    &down_dims,
                    &down_w[1],
                    &down_w[2],
                    &down_w[3],
                    &csr_offsets,
                    &csr_row_slot,
                    &csr_active,
                    &csr_meta,
                    &csr_tile_list,
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(mm_grid(h), tg128);
                barrier(); // down_slots → reduce
                           // (6) slot-reduce + router fold → mlp_down[B,h].
                enc.setComputePipelineState(p_reduce);
                bind(&[&down_partials, &wts, &mlp_down, &down_dims]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (b * h).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
                barrier(); // mlp_down → residual
            } else {
                // 4. gate/up indirect (B-row): grid = (ceil(n/2), B, top_k). Disjoint outputs → no
                // barrier between them (concurrent overlap).
                let gu_grid = |n: usize| MTLSize {
                    width: n.div_ceil(8),
                    height: b,
                    depth: top_k,
                }; // COLS=8 in gemv_q4ks_id_b
                let tg_gu = tg32; // L29: the gate/up NSG=2 lever was deleted (measured -5% conc16)
                let (p_gu, gu_g, gu_tg) = (&p_id_b, gu_grid(inter), tg_gu);
                enc.setComputePipelineState(p_gu);
                bind(&[
                    &normed, &gate_w[0], &gate_all, &gu_dims, &gate_w[1], &ids, &gate_w[2],
                    &gate_w[3],
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(gu_g, gu_tg);
                enc.setComputePipelineState(p_gu);
                bind(&[
                    &normed, &up_w[0], &up_all, &gu_dims, &up_w[1], &ids, &up_w[2], &up_w[3],
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(gu_g, gu_tg);
                barrier(); // gate/up → swiglu
                           // 5. swiglu: silu(gate_all)*up_all → silu_all over B*top_k*inter.
                enc.setComputePipelineState(&p_swiglu_b);
                bind(&[&gate_all, &up_all, &silu_all, &swiglu_dims]);
                let sw_n = b * top_k * inter;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: sw_n.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
                barrier(); // silu → down
                           // 6. down indirect + router-weight reduce (B-row).
                if let (true, Some(p_part), Some(p_red)) = (
                    down_parallel,
                    p_down_partial_b.as_ref(),
                    p_down_reduce_b.as_ref(),
                ) {
                    // top_k-PARALLEL: per-slot partials (grid Z = top_k) → reduce over slots.
                    enc.setComputePipelineState(p_part);
                    bind(&[
                        &silu_all,
                        &down_w[0],
                        &down_partials,
                        &down_dims,
                        &down_w[1],
                        &ids,
                        &wts,
                        &down_w[2],
                        &down_w[3],
                    ]);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: h.div_ceil(8),
                            height: b,
                            depth: top_k,
                        },
                        tg32,
                    );
                    barrier(); // partials → reduce
                    enc.setComputePipelineState(p_red);
                    bind(&[&down_partials, &mlp_down, &down_dims]);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: (b * h).div_ceil(256),
                            height: 1,
                            depth: 1,
                        },
                        tg256,
                    );
                } else {
                    // SERIAL (default oracle): grid = (h, B, 1), loop top_k inside the simdgroup.
                    enc.setComputePipelineState(&p_down_b);
                    bind(&[
                        &silu_all, &down_w[0], &mlp_down, &down_dims, &down_w[1], &ids, &wts,
                        &down_w[2], &down_w[3],
                    ]);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: h.div_ceil(8),
                            height: b,
                            depth: 1,
                        },
                        tg32,
                    ); // DCOLS=8 in gemv_q4ks_id_down_b
                }
                barrier(); // mlp_down → residual
            } // end else (GEMV MoE selftest path)
              // 7. add_residual_b: hidden += mlp_down over B*h.
            enc.setComputePipelineState(&p_resid_b);
            bind(&[&hidden, &mlp_down, &resid_dims]);
            let resid_n = b * h;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: resid_n.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ===== CPU oracle (per row): the EXACT same MoE FFN-block math =====
        let q4ks = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                    a: &[f32],
                    n: usize,
                    k: usize,
                    col_off: usize|
         -> Vec<f32> {
            // dequant-GEMV of cols [col_off..col_off+n) of the weight against activation a[k].
            let subs = k / 32;
            let supers = k / 256;
            let cd = w[0].contents().as_ptr() as *const u32;
            let sc = w[1].contents().as_ptr() as *const u8;
            let mn = w[2].contents().as_ptr() as *const i8;
            let dp = w[3].contents().as_ptr() as *const u32;
            (0..n)
                .map(|cc| {
                    let col = col_off + cc;
                    let mut acc = 0.0f32;
                    for bb in 0..subs {
                        let sblk = bb / 8;
                        let pair = unsafe { *dp.add(col * supers + sblk) };
                        let dv = f16_bits_to_f32(pair as u16);
                        let dmv = f16_bits_to_f32((pair >> 16) as u16);
                        let su8 = unsafe { *sc.add(col * subs + bb) } as f32;
                        let mi8 = unsafe { *mn.add(col * subs + bb) } as f32;
                        let (s, lo) = (dv * su8, dmv * mi8);
                        let ab = 32 * bb;
                        let (mut cs, mut asum) = (0.0f32, 0.0f32);
                        for wi in 0..4 {
                            let word = unsafe { *cd.add((col * subs + bb) * 4 + wi) };
                            for e in 0..8u32 {
                                let av = a[ab + wi * 8 + e as usize];
                                cs += av * ((word >> (4 * e)) & 0xF) as f32;
                                asum += av;
                            }
                        }
                        acc += s * cs + lo * asum;
                    }
                    acc
                })
                .collect()
        };
        let prew: Vec<f32> = (0..h)
            .map(|i| unsafe { *(pre_ffn_norm.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let mut want = vec![0.0f32; b * h];
        for r in 0..b {
            // add_norm: hidden += attn_res; normed = rmsnorm(hidden)·prew.
            let mut hid: Vec<f32> = (0..h)
                .map(|i| {
                    hid_before[r * h + i]
                        + unsafe { *(attn_res.contents().as_ptr() as *const f32).add(r * h + i) }
                })
                .collect();
            let mut ss = 0.0f32;
            for &v in &hid {
                ss += v * v;
            }
            let inv = 1.0 / (ss / h as f32 + eps).sqrt();
            let normed_r: Vec<f32> = (0..h).map(|i| hid[i] * inv * prew[i]).collect();
            // router GEMV → logits[num_experts].
            let logits_r = q4ks(&router_w, &normed_r, num_experts, h, 0);
            // route: softmax over logits, top_k argmax (ties → lowest), wts = softmax prob, renorm.
            let maxv = logits_r.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f32 = logits_r.iter().map(|&x| (x - maxv).exp()).sum();
            let mut taken = vec![false; num_experts];
            let mut ids_r = vec![0usize; top_k];
            let mut wts_r = vec![0.0f32; top_k];
            for slot in 0..top_k.min(num_experts) {
                let mut bv = f32::MIN;
                let mut bi = usize::MAX;
                for i in 0..num_experts {
                    if !taken[i] && logits_r[i] > bv {
                        bv = logits_r[i];
                        bi = i;
                    }
                }
                taken[bi] = true;
                ids_r[slot] = bi;
                wts_r[slot] = (logits_r[bi] - maxv).exp() / sum;
            }
            if norm_topk {
                let wsum: f32 = wts_r.iter().sum();
                if wsum > 0.0 {
                    for w in wts_r.iter_mut() {
                        *w /= wsum;
                    }
                }
            }
            // gate/up indirect per slot → swiglu → silu_all[slot*inter..].
            let mut mlp = vec![0.0f32; h];
            for slot in 0..top_k {
                let e = ids_r[slot];
                let gate = q4ks(&gate_w, &normed_r, inter, h, e * inter);
                let up = q4ks(&up_w, &normed_r, inter, h, e * inter);
                let silu: Vec<f32> = (0..inter)
                    .map(|i| {
                        let g = gate[i];
                        (g / (1.0 + (-g).exp())) * up[i]
                    })
                    .collect();
                // down: Wdown_e[h, inter] · silu → [h]; fold router weight; accumulate.
                let dn = q4ks(&down_w, &silu, h, inter, e * h);
                for j in 0..h {
                    mlp[j] += wts_r[slot] * dn[j];
                }
            }
            // add_residual: hidden += mlp.
            for j in 0..h {
                hid[j] += mlp[j];
                want[r * h + j] = hid[j];
            }
        }

        // compare GPU hidden vs oracle.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        unsafe {
            let hp = hidden.contents().as_ptr() as *const f32;
            for i in 0..b * h {
                let got = *hp.add(i);
                let w = want[i];
                let abs = (got - w).abs();
                max_abs = max_abs.max(abs);
                let rel = abs / w.abs().max(1e-3);
                if rel > max_rel {
                    max_rel = rel;
                    worst = (i, got, w);
                }
            }
        }
        if std::env::var("ARF_BATCH_MEGA_DUMP").is_ok() {
            unsafe {
                let dump = |name: &str,
                            buf: &ProtocolObject<dyn MTLBuffer>,
                            stride: usize,
                            count: usize| {
                    let p = buf.contents().as_ptr() as *const f32;
                    for r in 0..b.min(2) {
                        let g: Vec<f32> =
                            (0..count.min(4)).map(|i| *p.add(r * stride + i)).collect();
                        eprintln!("  [moe dbg] row{r} {name}={g:?}");
                    }
                };
                let dumpu = |name: &str,
                             buf: &ProtocolObject<dyn MTLBuffer>,
                             stride: usize,
                             count: usize| {
                    let p = buf.contents().as_ptr() as *const u32;
                    for r in 0..b.min(2) {
                        let g: Vec<u32> = (0..count).map(|i| *p.add(r * stride + i)).collect();
                        eprintln!("  [moe dbg] row{r} {name}={g:?}");
                    }
                };
                dump("normed", &normed, h, 4);
                dumpu("ids", &ids, top_k, top_k);
                dump("wts", &wts, top_k, top_k);
                dump("mlp_down", &mlp_down, h, 4);
                let hp = hidden.contents().as_ptr() as *const f32;
                for r in 0..b.min(2) {
                    let g: Vec<f32> = (0..4).map(|i| *hp.add(r * h + i)).collect();
                    let w: Vec<f32> = (0..4).map(|i| want[r * h + i]).collect();
                    eprintln!("  [moe dbg] row{r} hidden gpu={g:?} want={w:?}");
                }
                eprintln!(
                    "  [moe dbg] worst rel @ idx {} (row {}): gpu={} want={} (|want|={:.2e})",
                    worst.0,
                    worst.0 / h,
                    worst.1,
                    worst.2,
                    worst.2.abs()
                );
            }
        }
        Ok((max_abs, max_rel))
    }
    pub fn batched_full_layer_selftest(
        &mut self,
        ctx: &GpuContext,
        b: usize,
        h: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        ctx_len: usize,
        inter: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk: bool,
        eps: f32,
    ) -> Result<(f32, f32), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let rotary_dim = hd;
        let rot_half = rotary_dim / 2;
        let past_len: Vec<usize> = (0..b)
            .map(|r| 2 + r % (ctx_len.saturating_sub(2)).max(1))
            .collect();
        let positions: Vec<usize> = past_len.clone();

        // ---- B-row scratch (attention + MoE share `hidden`/`normed`) ----
        let hidden = al(b * h * 4)?;
        let normed = al(b * h * 4)?;
        let qb = al(b * q_dim * 4)?;
        let kb = al(b * kv_dim * 4)?;
        let vb = al(b * kv_dim * 4)?;
        let attn = al(b * q_dim * 4)?;
        let attn_out = al(b * h * 4)?; // attention output — the MoE add_norm residual
        let logits = al(b * num_experts * 4)?;
        let ids = al(b * top_k * 4)?;
        let wts = al(b * top_k * 4)?;
        let gate_all = al(b * top_k * inter * 4)?;
        let up_all = al(b * top_k * inter * 4)?;
        let silu_all = al(b * top_k * inter * 4)?;
        let mlp_down = al(b * h * 4)?;
        // per-seq KV pools + slots
        let mut kpool = Vec::with_capacity(b);
        let mut vpool = Vec::with_capacity(b);
        let mut slots = Vec::with_capacity(b);
        for _r in 0..b {
            kpool.push(al(ctx_len * kv_dim * 4)?);
            vpool.push(al(ctx_len * kv_dim * 4)?);
            let sl = al(ctx_len * 4)?;
            unsafe {
                let p = sl.contents().as_ptr() as *mut u32;
                for s in 0..ctx_len {
                    *p.add(s) = s as u32;
                }
            }
            slots.push(sl);
        }
        let qnw = al(hd * 4)?;
        let knw = al(hd * 4)?;
        let input_norm = al(h * 4)?;
        let pre_ffn_norm = al(h * 4)?; // add_norm weight (post-attn / pre-FFN)
        let max_pos = ctx_len + 1;
        let rope_cos = al(max_pos * rot_half * 4)?;
        let rope_sin = al(max_pos * rot_half * 4)?;
        let pos_buf = al(b * 4)?;

        // ---- deterministic Q4KS weight builder (shrink `extra` to keep residual O(1)) ----
        let subs_of = |k: usize| k / 32;
        let supers_of = |k: usize| k / 256;
        let mk_w = |n: usize,
                    k: usize,
                    seed: usize,
                    extra: f32|
         -> [Retained<ProtocolObject<dyn MTLBuffer>>; 5] {
            let subs = subs_of(k);
            let supers = supers_of(k);
            let codes = al(n * subs * 16).unwrap();
            let sc = al((n * subs).div_ceil(4) * 4).unwrap();
            let mn = al((n * subs).div_ceil(4) * 4).unwrap();
            let dd = al(n * supers * 4).unwrap(); // ddh: packed half2 {d,dmin}, one u32/super-block
            let dims = al(16).unwrap();
            unsafe {
                let cp = codes.contents().as_ptr() as *mut u32;
                for i in 0..(n * subs * 4) {
                    *cp.add(i) = (i
                        .wrapping_mul(2654435761)
                        .wrapping_add(seed.wrapping_mul(40503))
                        & 0xFFFF_FFFF) as u32;
                }
                let sp = sc.contents().as_ptr() as *mut u8;
                let mp = mn.contents().as_ptr() as *mut i8;
                for i in 0..(n * subs) {
                    *sp.add(i) = (((i + seed) % 31) + 1) as u8;
                    *mp.add(i) = ((((i + seed) % 17) as i32) - 8) as i8;
                }
                let scl = extra / (k as f32).sqrt();
                let dp = dd.contents().as_ptr() as *mut u32;
                for i in 0..(n * supers) {
                    let dv = (0.02 + (i % 5) as f32 * 0.003) * scl;
                    let dmv = (-0.01 - (i % 3) as f32 * 0.002) * scl;
                    *dp.add(i) =
                        (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
                }
                let d = dims.contents().as_ptr() as *mut u32;
                *d = 1;
                *d.add(1) = k as u32;
                *d.add(2) = n as u32;
                *d.add(3) = 0;
            }
            [codes, sc, mn, dd, dims]
        };
        // Shrink weights so the FULL-LAYER residual stays O(1): the attention output `aout` feeds
        // add_norm whose rmsnorm can amplify a near-zero `hidden` into huge values (saw ~2.7e4 on
        // one row), inflating the f32 abs floor and masking the strict bar. Smaller q/k/v/o keep
        // `aout` and thus `hidden` bounded; gate/up/down shrunk as in M2. Parity (max_rel) is
        // unaffected by magnitude — this only keeps abs honest.
        let qw = mk_w(q_dim, h, 1, 0.2);
        let kw = mk_w(kv_dim, h, 2, 0.2);
        let vw = mk_w(kv_dim, h, 3, 0.2);
        let ow = mk_w(h, q_dim, 4, 0.2);
        let router_w = mk_w(num_experts, h, 11, 0.3);
        let gate_w = mk_w(num_experts * inter, h, 21, 0.1);
        let up_w = mk_w(num_experts * inter, h, 31, 0.1);
        let down_w = mk_w(num_experts * h, inter, 41, 0.04);

        // ---- dims ----
        let norm_dims = al(16)?;
        let qk_dims = al(16)?;
        let rope_dims = al(32)?;
        let an_dims = al(16)?; // add_norm {B,h,eps,_}
        let router_gdims = al(16)?;
        let gu_dims = al(16)?;
        let down_dims = al(16)?;
        let route_dims = al(16)?;
        let swiglu_dims = al(16)?;
        let resid_dims = al(16)?;
        unsafe {
            let p = norm_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *(p.add(2) as *mut f32) = eps;
            *p.add(3) = 0;
            let p = qk_dims.contents().as_ptr() as *mut u32;
            *p = nh as u32;
            *p.add(1) = nkv as u32;
            *p.add(2) = hd as u32;
            *(p.add(3) as *mut f32) = eps;
            let p = rope_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = nh as u32;
            *p.add(2) = nkv as u32;
            *p.add(3) = hd as u32;
            *p.add(4) = rotary_dim as u32;
            *p.add(5) = 0;
            *p.add(6) = 0;
            *p.add(7) = 0;
            let p = an_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *(p.add(2) as *mut f32) = eps;
            *p.add(3) = 0;
            let p = router_gdims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = num_experts as u32;
            *p.add(3) = 0;
            let p = gu_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = inter as u32;
            *p.add(3) = top_k as u32;
            let p = down_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = inter as u32;
            *p.add(2) = h as u32;
            *p.add(3) = top_k as u32;
            let p = route_dims.contents().as_ptr() as *mut u32;
            *p = num_experts as u32;
            *p.add(1) = top_k as u32;
            *p.add(2) = norm_topk as u32;
            *p.add(3) = 0;
            let p = swiglu_dims.contents().as_ptr() as *mut u32;
            *p = (b * top_k * inter) as u32;
            *p.add(1) = 0;
            *p.add(2) = 0;
            *p.add(3) = 0;
            let p = resid_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let mk_gdims =
            |k: usize, n: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
                let bd = al(16)?;
                unsafe {
                    let p = bd.contents().as_ptr() as *mut u32;
                    *p = b as u32;
                    *p.add(1) = k as u32;
                    *p.add(2) = n as u32;
                    *p.add(3) = 0;
                }
                Ok(bd)
            };
        let gd_q = mk_gdims(h, q_dim)?;
        let gd_k = mk_gdims(h, kv_dim)?;
        let gd_v = mk_gdims(h, kv_dim)?;
        let gd_o = mk_gdims(q_dim, h)?;

        // ---- fill inputs ----
        unsafe {
            let hp = hidden.contents().as_ptr() as *mut f32;
            // Mild zero-mean input (matches M1/M2); weights are shrunk so the chained MoE down
            // (sum over inter × top_k slots) keeps the residual O(1) and the f32 abs floor honest.
            for r in 0..b {
                for i in 0..h {
                    *hp.add(r * h + i) = (((r * 13 + i) % 23) as f32 - 11.0) * 0.05;
                }
            }
            let qp = qnw.contents().as_ptr() as *mut f32;
            let kp = knw.contents().as_ptr() as *mut f32;
            for i in 0..hd {
                *qp.add(i) = 1.0 + ((i % 7) as f32 - 3.0) * 0.03;
                *kp.add(i) = 1.0 + ((i % 5) as f32 - 2.0) * 0.04;
            }
            let ip = input_norm.contents().as_ptr() as *mut f32;
            let fp = pre_ffn_norm.contents().as_ptr() as *mut f32;
            for i in 0..h {
                *ip.add(i) = 1.0 + ((i % 11) as f32 - 5.0) * 0.02;
                *fp.add(i) = 1.0 + ((i % 13) as f32 - 6.0) * 0.015;
            }
            let cp = rope_cos.contents().as_ptr() as *mut f32;
            let sp = rope_sin.contents().as_ptr() as *mut f32;
            for pos in 0..max_pos {
                for i in 0..rot_half {
                    let ang =
                        (pos as f32) * (1.0 / (10000f32).powf(2.0 * i as f32 / rotary_dim as f32));
                    *cp.add(pos * rot_half + i) = ang.cos();
                    *sp.add(pos * rot_half + i) = ang.sin();
                }
            }
            for r in 0..b {
                let k = kpool[r].contents().as_ptr() as *mut f32;
                let v = vpool[r].contents().as_ptr() as *mut f32;
                for row in 0..ctx_len {
                    for j in 0..kv_dim {
                        *k.add(row * kv_dim + j) = (((r * 5 + row + j) % 19) as f32 - 9.0) * 0.03;
                        *v.add(row * kv_dim + j) =
                            (((r * 7 + row * 3 + j) % 13) as f32 - 6.0) * 0.04;
                    }
                }
            }
            let pp = pos_buf.contents().as_ptr() as *mut u32;
            for r in 0..b {
                *pp.add(r) = positions[r] as u32;
            }
        }
        drop(guard);

        // per-seq attn/scatter/pos uniforms
        let mut attn_dims = Vec::with_capacity(b);
        let mut scatter_dims = Vec::with_capacity(b);
        let mut pos_one = Vec::with_capacity(b);
        {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            for r in 0..b {
                let ad = al2(48);
                unsafe {
                    let p = ad.contents().as_ptr() as *mut u32;
                    p.add(0).write(1);
                    p.add(1).write((past_len[r] + 1) as u32);
                    p.add(2).write(nh as u32);
                    p.add(3).write(nkv as u32);
                    p.add(4).write(hd as u32);
                    p.add(5).write(past_len[r] as u32);
                    (p.add(6) as *mut f32).write(scale);
                    p.add(7).write(group as u32);
                    p.add(8).write(0);
                    p.add(9).write(0);
                    p.add(10).write(0);
                    p.add(11).write(0);
                }
                attn_dims.push(ad);
                let sd = al2(16);
                unsafe {
                    let p = sd.contents().as_ptr() as *mut u32;
                    *p = 1;
                    *p.add(1) = kv_dim as u32;
                    *p.add(2) = 0;
                    *p.add(3) = 0;
                }
                scatter_dims.push(sd);
                let po = al2(16);
                unsafe {
                    *(po.contents().as_ptr() as *mut u32) = positions[r] as u32;
                }
                pos_one.push(po);
            }
            drop(guard);
        }
        let hid_before: Vec<f32> = unsafe {
            let p = hidden.contents().as_ptr() as *const f32;
            (0..b * h).map(|i| *p.add(i)).collect()
        };

        // ---- pipelines ----
        let p_gemv_b = self
            .pipelines
            .get("gemv_q4ks_batch")
            .ok_or("gemv_q4ks_batch")?
            .pso
            .clone();
        let p_rms_b = self
            .pipelines
            .get("rmsnorm_b")
            .ok_or("rmsnorm_b")?
            .pso
            .clone();
        let p_qk_b = self
            .pipelines
            .get("qk_norm_b")
            .ok_or("qk_norm_b")?
            .pso
            .clone();
        let p_rope_b = self
            .pipelines
            .get("rope_qk_b")
            .ok_or("rope_qk_b")?
            .pso
            .clone();
        let p_resid_b = self
            .pipelines
            .get("add_residual_b")
            .ok_or("add_residual_b")?
            .pso
            .clone();
        let p_scatter = self
            .pipelines
            .get("kv_scatter")
            .ok_or("kv_scatter")?
            .pso
            .clone();
        let p_attn = self
            .pipelines
            .get("attention_decode")
            .ok_or("attention_decode")?
            .pso
            .clone();
        let p_add_norm = self
            .pipelines
            .get("add_norm")
            .ok_or("add_norm")?
            .pso
            .clone();
        let p_route_b = self
            .pipelines
            .get("moe_route_b")
            .ok_or("moe_route_b")?
            .pso
            .clone();
        let p_id_b = self
            .pipelines
            .get("gemv_q4ks_id_b")
            .ok_or("gemv_q4ks_id_b")?
            .pso
            .clone();
        let p_swiglu_b = self
            .pipelines
            .get("moe_swiglu_b")
            .ok_or("moe_swiglu_b")?
            .pso
            .clone();
        let p_down_b = self
            .pipelines
            .get("gemv_q4ks_id_down_b")
            .ok_or("gemv_q4ks_id_down_b")?
            .pso
            .clone();

        // ===== record the FULL B-row layer in ONE concurrent command buffer =====
        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        let tg32 = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let tg64 = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let tg128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        unsafe {
            let bind = |bufs: &[&ProtocolObject<dyn MTLBuffer>]| {
                for (i, bf) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
            };
            let barrier = || enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            let gemv_b = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                          a: &ProtocolObject<dyn MTLBuffer>,
                          c: &ProtocolObject<dyn MTLBuffer>,
                          gd: &ProtocolObject<dyn MTLBuffer>,
                          n: usize| {
                enc.setComputePipelineState(&p_gemv_b);
                bind(&[a, &w[0], c, gd, &w[1], &w[2], &w[3]]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: 1,
                        depth: 1,
                    },
                    tg32,
                );
            };
            // ===== ATTENTION BLOCK (ends at attn_out — NO plain residual) =====
            enc.setComputePipelineState(&p_rms_b);
            bind(&[&hidden, &input_norm, &normed, &norm_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier();
            gemv_b(&qw, &normed, &qb, &gd_q, q_dim);
            gemv_b(&kw, &normed, &kb, &gd_k, kv_dim);
            gemv_b(&vw, &normed, &vb, &gd_v, kv_dim);
            barrier();
            enc.setComputePipelineState(&p_qk_b);
            bind(&[&qb, &kb, &qnw, &knw, &qk_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b * (nh + nkv),
                    height: 1,
                    depth: 1,
                },
                tg64,
            );
            barrier();
            enc.setComputePipelineState(&p_rope_b);
            bind(&[&qb, &kb, &rope_cos, &rope_sin, &pos_buf, &rope_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (b * (q_dim + kv_dim)).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier();
            for r in 0..b {
                enc.setComputePipelineState(&p_scatter);
                enc.useResource_usage(ProtocolObject::from_ref(&*kb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kb), (r * kv_dim * 4) as _, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*vb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vb), (r * kv_dim * 4) as _, 1);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*pos_one[r]),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&pos_one[r]), 0, 2);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*kpool[r]),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&kpool[r]), 0, 3);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*vpool[r]),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&vpool[r]), 0, 4);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*scatter_dims[r]),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&scatter_dims[r]), 0, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: kv_dim.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
            barrier();
            for r in 0..b {
                enc.setComputePipelineState(&p_attn);
                enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&qb), (r * q_dim * 4) as _, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*kpool[r]), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kpool[r]), 0, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&*vpool[r]), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vpool[r]), 0, 2);
                enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&attn), (r * q_dim * 4) as _, 3);
                enc.useResource_usage(ProtocolObject::from_ref(&*slots[r]), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&slots[r]), 0, 4);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*attn_dims[r]),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&attn_dims[r]), 0, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: nh,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
            barrier();
            gemv_b(&ow, &attn, &attn_out, &gd_o, h); // attn_out = o_proj(attn). NO residual add.
            barrier();
            // ===== MoE BLOCK (add_norm consumes attn_out as the residual) =====
            enc.setComputePipelineState(&p_add_norm);
            bind(&[&hidden, &attn_out, &pre_ffn_norm, &normed, &an_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier();
            gemv_b(&router_w, &normed, &logits, &router_gdims, num_experts);
            barrier();
            enc.setComputePipelineState(&p_route_b);
            bind(&[&logits, &ids, &wts, &route_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg128,
            );
            barrier();
            let gu_grid = |n: usize| MTLSize {
                width: n.div_ceil(8),
                height: b,
                depth: top_k,
            }; // COLS=8 in gemv_q4ks_id_b
            let tg_gu = tg32; // L29: the gate/up NSG=2 lever was deleted (measured -5% conc16)
            let (p_gu, gu_g, gu_tg) = (&p_id_b, gu_grid(inter), tg_gu);
            enc.setComputePipelineState(p_gu);
            bind(&[
                &normed, &gate_w[0], &gate_all, &gu_dims, &gate_w[1], &ids, &gate_w[2], &gate_w[3],
            ]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(gu_g, gu_tg);
            enc.setComputePipelineState(p_gu);
            bind(&[
                &normed, &up_w[0], &up_all, &gu_dims, &up_w[1], &ids, &up_w[2], &up_w[3],
            ]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(gu_g, gu_tg);
            barrier();
            enc.setComputePipelineState(&p_swiglu_b);
            bind(&[&gate_all, &up_all, &silu_all, &swiglu_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (b * top_k * inter).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier();
            enc.setComputePipelineState(&p_down_b);
            bind(&[
                &silu_all, &down_w[0], &mlp_down, &down_dims, &down_w[1], &ids, &wts, &down_w[2],
                &down_w[3],
            ]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: h.div_ceil(8),
                    height: b,
                    depth: 1,
                },
                tg32,
            ); // DCOLS=8 in gemv_q4ks_id_down_b
            barrier();
            enc.setComputePipelineState(&p_resid_b);
            bind(&[&hidden, &mlp_down, &resid_dims]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (b * h).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ===== CPU oracle (per row): the EXACT full-layer math =====
        let q4ks = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 5],
                    a: &[f32],
                    n: usize,
                    k: usize,
                    col_off: usize|
         -> Vec<f32> {
            let subs = k / 32;
            let supers = k / 256;
            let cd = w[0].contents().as_ptr() as *const u32;
            let sc = w[1].contents().as_ptr() as *const u8;
            let mn = w[2].contents().as_ptr() as *const i8;
            let dp = w[3].contents().as_ptr() as *const u32;
            (0..n)
                .map(|cc| {
                    let col = col_off + cc;
                    let mut acc = 0.0f32;
                    for bb in 0..subs {
                        let sblk = bb / 8;
                        let pair = unsafe { *dp.add(col * supers + sblk) };
                        let dv = f16_bits_to_f32(pair as u16);
                        let dmv = f16_bits_to_f32((pair >> 16) as u16);
                        let su8 = unsafe { *sc.add(col * subs + bb) } as f32;
                        let mi8 = unsafe { *mn.add(col * subs + bb) } as f32;
                        let (s, lo) = (dv * su8, dmv * mi8);
                        let ab = 32 * bb;
                        let (mut cs, mut asum) = (0.0f32, 0.0f32);
                        for wi in 0..4 {
                            let word = unsafe { *cd.add((col * subs + bb) * 4 + wi) };
                            for e in 0..8u32 {
                                let av = a[ab + wi * 8 + e as usize];
                                cs += av * ((word >> (4 * e)) & 0xF) as f32;
                                asum += av;
                            }
                        }
                        acc += s * cs + lo * asum;
                    }
                    acc
                })
                .collect()
        };
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let n = x.len();
            let mut ss = 0.0f32;
            for &v in x {
                ss += v * v;
            }
            let inv = 1.0 / (ss / n as f32 + eps).sqrt();
            (0..n).map(|i| x[i] * inv * w[i]).collect()
        };
        let qnw_v: Vec<f32> = (0..hd)
            .map(|i| unsafe { *(qnw.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let knw_v: Vec<f32> = (0..hd)
            .map(|i| unsafe { *(knw.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let innw_v: Vec<f32> = (0..h)
            .map(|i| unsafe { *(input_norm.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let prew_v: Vec<f32> = (0..h)
            .map(|i| unsafe { *(pre_ffn_norm.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let cos_v: Vec<f32> = (0..max_pos * rot_half)
            .map(|i| unsafe { *(rope_cos.contents().as_ptr() as *const f32).add(i) })
            .collect();
        let sin_v: Vec<f32> = (0..max_pos * rot_half)
            .map(|i| unsafe { *(rope_sin.contents().as_ptr() as *const f32).add(i) })
            .collect();

        let mut want = vec![0.0f32; b * h];
        for r in 0..b {
            let hin = &hid_before[r * h..r * h + h];
            // --- attention ---
            let normed_r = rms(hin, &innw_v);
            let mut q = q4ks(&qw, &normed_r, q_dim, h, 0);
            let mut k = q4ks(&kw, &normed_r, kv_dim, h, 0);
            let vv = q4ks(&vw, &normed_r, kv_dim, h, 0);
            for hh in 0..nh {
                let seg = rms(&q[hh * hd..hh * hd + hd], &qnw_v);
                q[hh * hd..hh * hd + hd].copy_from_slice(&seg);
            }
            for hh in 0..nkv {
                let seg = rms(&k[hh * hd..hh * hd + hd], &knw_v);
                k[hh * hd..hh * hd + hd].copy_from_slice(&seg);
            }
            let pos = positions[r];
            let rope = |buf: &mut [f32], heads: usize| {
                for hh in 0..heads {
                    let base = hh * hd;
                    for i in 0..rot_half {
                        let c = cos_v[pos * rot_half + i];
                        let s = sin_v[pos * rot_half + i];
                        let a = buf[base + i];
                        let bv = buf[base + i + rot_half];
                        buf[base + i] = a * c - bv * s;
                        buf[base + i + rot_half] = bv * c + a * s;
                    }
                }
            };
            rope(&mut q, nh);
            rope(&mut k, nkv);
            let kp_ptr = kpool[r].contents().as_ptr() as *const f32;
            let vp_ptr = vpool[r].contents().as_ptr() as *const f32;
            let pl = past_len[r];
            let mut attn_r = vec![0.0f32; q_dim];
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * hd;
                let mut scs = vec![0.0f32; pl + 1];
                let mut mx = f32::MIN;
                for key in 0..=pl {
                    let mut s = 0.0f32;
                    for i in 0..hd {
                        let kval = if key == pos {
                            k[kv_head * hd + i]
                        } else {
                            unsafe { *kp_ptr.add(key * kv_dim + hcol + i) }
                        };
                        s += q[hh * hd + i] * kval;
                    }
                    scs[key] = s * scale;
                    mx = mx.max(scs[key]);
                }
                let mut den = 0.0f32;
                for key in 0..=pl {
                    scs[key] = (scs[key] - mx).exp();
                    den += scs[key];
                }
                for dim in 0..hd {
                    let mut num = 0.0f32;
                    for key in 0..=pl {
                        let vval = if key == pos {
                            vv[kv_head * hd + dim]
                        } else {
                            unsafe { *vp_ptr.add(key * kv_dim + hcol + dim) }
                        };
                        num += scs[key] * vval;
                    }
                    attn_r[hh * hd + dim] = num / den;
                }
            }
            let aout = q4ks(&ow, &attn_r, h, q_dim, 0);
            // --- MoE (add_norm consumes aout) ---
            let mut hid: Vec<f32> = (0..h).map(|i| hin[i] + aout[i]).collect();
            let normed2 = rms(&hid, &prew_v);
            let logits_r = q4ks(&router_w, &normed2, num_experts, h, 0);
            let maxv = logits_r.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f32 = logits_r.iter().map(|&x| (x - maxv).exp()).sum();
            let mut taken = vec![false; num_experts];
            let mut ids_r = vec![0usize; top_k];
            let mut wts_r = vec![0.0f32; top_k];
            for slot in 0..top_k.min(num_experts) {
                let mut bv = f32::MIN;
                let mut bi = usize::MAX;
                for i in 0..num_experts {
                    if !taken[i] && logits_r[i] > bv {
                        bv = logits_r[i];
                        bi = i;
                    }
                }
                taken[bi] = true;
                ids_r[slot] = bi;
                wts_r[slot] = (logits_r[bi] - maxv).exp() / sum;
            }
            if norm_topk {
                let wsum: f32 = wts_r.iter().sum();
                if wsum > 0.0 {
                    for w in wts_r.iter_mut() {
                        *w /= wsum;
                    }
                }
            }
            let mut mlp = vec![0.0f32; h];
            for slot in 0..top_k {
                let e = ids_r[slot];
                let gate = q4ks(&gate_w, &normed2, inter, h, e * inter);
                let up = q4ks(&up_w, &normed2, inter, h, e * inter);
                let silu: Vec<f32> = (0..inter)
                    .map(|i| {
                        let g = gate[i];
                        (g / (1.0 + (-g).exp())) * up[i]
                    })
                    .collect();
                let dn = q4ks(&down_w, &silu, h, inter, e * h);
                for j in 0..h {
                    mlp[j] += wts_r[slot] * dn[j];
                }
            }
            for j in 0..h {
                hid[j] += mlp[j];
                want[r * h + j] = hid[j];
            }
        }

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        let mut worst_abs = (0usize, 0.0f32, 0.0f32);
        let mut n_big = 0usize;
        // BIT-PARITY relative metric: an element whose absolute error is below the bit-parity floor
        // (1e-4, the same bar M1/M2 gate on) is bit-identical — its relative ratio is a degenerate
        // near-zero-residual artifact (e.g. catastrophic cancellation (-0.4620)+(0.4629)=8.7e-4 where
        // a sub-floor 3.7e-5 reorder diff reads as 3.7%). Such elements MUST NOT gate parity. We
        // therefore report `max_rel` over ONLY the elements that exceed the abs floor; an element is
        // bit-parity iff abs < 1e-4 OR rel < 1e-3, so the returned (max_abs, max_rel) pair lets the
        // caller gate correctly with `max_abs < 1e-4 || max_rel < 1e-3`.
        unsafe {
            let hp = hidden.contents().as_ptr() as *const f32;
            for i in 0..b * h {
                let got = *hp.add(i);
                let w = want[i];
                let abs = (got - w).abs();
                if abs > max_abs {
                    max_abs = abs;
                    worst_abs = (i, got, w);
                }
                if abs > 1e-4 {
                    n_big += 1;
                }
                // skip sub-bit-parity-floor elements: they are bit-identical and their relative ratio
                // is a cancellation artifact, not a divergence.
                if abs < 1e-4 {
                    continue;
                }
                let rel = abs / w.abs().max(1e-3);
                if rel > max_rel {
                    max_rel = rel;
                    worst = (i, got, w);
                }
            }
        }
        if std::env::var("ARF_BATCH_MEGA_DUMP").is_ok() {
            eprintln!(
                "  [layer dbg] worst rel @ idx {} (row {}): gpu={} want={} (|want|={:.2e})",
                worst.0,
                worst.0 / h,
                worst.1,
                worst.2,
                worst.2.abs()
            );
            eprintln!("  [layer dbg] worst ABS @ idx {} (row {}): gpu={} want={} (abs={:.2e}); {} elems > 1e-4 of {}",
                worst_abs.0, worst_abs.0/h, worst_abs.1, worst_abs.2, (worst_abs.1-worst_abs.2).abs(), n_big, b*h);
        }
        Ok((max_abs, max_rel))
    }
    /// f16 KV cache parity gate (the 3rd leg: bandwidth). Runs the coalesced batched
    /// attention with the KV pool stored as `half` (f16 scatter + f16-load kernel) and compares its
    /// output against the SAME full-precision f32 CPU oracle that `batched_paged_attn_selftest`
    /// uses for the f32 coalesced kernel. f16 KV is LOSSY (f32->f16 round-trip), so the output is
    /// NOT bit-exact — the GATE is IDENTICAL ARGMAX vs the f32 oracle (greedy-robust; llama proves
    /// it): for every (row, head) the argmax DIMENSION of the hd-vector output must match the f32
    /// oracle's, a robust proxy for greedy-token stability. Also reports max_abs (expect ~1e-3, the
    /// f16 round-trip magnitude — NOT a fail). Returns (max_abs, argmax_identical, mismatch_count).
    ///
    /// This mirrors `batched_paged_attn_selftest` EXACTLY except: (a) the pool buffers are `half`
    /// (n*2 bytes), (b) cached slots are pre-populated as f16(value), (c) the f16 scatter + f16-load
    /// coalesced kernels are dispatched, (d) the oracle uses the FULL-precision f32 K/V/Q (the same
    /// oracle as the f32 gate) so the comparison isolates the f16-round-trip error, not a re-round.
    /// `mode` selects the kernel under test against the SAME oracle + gate: 0 = the coalesced
    /// f16 kernel; 1 = `attention_decode_kvhead_b_f16` (tg per (kv_head, seq), grid width nkv,
    /// requires group==8); 2 = `attention_decode_kvhead_splitk_b_f16` (the small-batch split-K
    /// form: pass1 grid (nkv, B, S=4) partials + the production `attention_splitk_combine_b`).
    #[cfg(target_os = "macos")]
    pub fn batched_paged_attn_selftest_f16(
        &mut self,
        ctx: &GpuContext,
        b: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pool_slots: usize,
        mode: u8,
    ) -> Result<(f32, bool, usize), String> {
        const SK_S: usize = 4; // split count exercised by mode 2 (chunk edges + empty chunks)
                               // IEEE-754 round-to-nearest-even f32 -> f16 (bits), matching Metal's half narrowing.
        fn f32_to_f16_bits(f: f32) -> u16 {
            let x = f.to_bits();
            let sign = ((x >> 16) & 0x8000) as u16;
            let mut mant = (x & 0x007f_ffff) as i32;
            let exp = ((x >> 23) & 0xff) as i32;
            if exp == 0xff {
                // Inf / NaN
                return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
            }
            let mut e = exp - 127 + 15;
            if e >= 0x1f {
                return sign | 0x7c00;
            } // overflow -> Inf
            if e <= 0 {
                if e < -10 {
                    return sign;
                } // underflow -> 0
                mant |= 0x0080_0000;
                let shift = 14 - e;
                let round = 1 << (shift - 1);
                let mut m = (mant + round) >> shift;
                if (mant & (round - 1)) == 0 && (m & 1) == 1 {
                    m -= (mant >> shift) & 1;
                } // ties-even guard
                return sign | m as u16;
            }
            // normalized: round mantissa 23 -> 10 bits, ties to even
            let round = 0x0000_1000;
            let mut m = mant + round;
            if (mant & 0x0000_1fff) == round {
                m &= !1;
            } // exactly halfway -> even
            if (m & 0x0080_0000) != 0 {
                e += 1;
                m = 0;
            } // mantissa carry
            if e >= 0x1f {
                return sign | 0x7c00;
            }
            sign | ((e as u16) << 10) | ((m >> 13) & 0x03ff) as u16
        }
        fn f16_bits_to_f32(h: u16) -> f32 {
            let sign = ((h >> 15) & 1) as u32;
            let exp = ((h >> 10) & 0x1f) as u32;
            let mant = (h & 0x03ff) as u32;
            let bits = if exp == 0 {
                if mant == 0 {
                    sign << 31
                } else {
                    let mut e = -1i32;
                    let mut m = mant;
                    while (m & 0x0400) == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    m &= 0x03ff;
                    (sign << 31) | (((e + 127 - 15) as u32) << 23) | (m << 13)
                }
            } else if exp == 0x1f {
                (sign << 31) | 0x7f80_0000 | (mant << 13)
            } else {
                (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
            };
            f32::from_bits(bits)
        }
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        // SAME synthetic geometry as batched_paged_attn_selftest (distinct past_len, strided slots).
        let past_len: Vec<usize> = (0..b).map(|r| 2 + r).collect();
        let seq_slots: Vec<Vec<u32>> = (0..b)
            .map(|r| (0..=past_len[r]).map(|i| (5 + r * 10 + i) as u32).collect())
            .collect();
        // f16 SHARED pool (half elements). Cached slots (0..past_len) pre-populated as f16(value);
        // the new-token slot (past_len) is left zero (the f16 scatter fills it on GPU). We also keep
        // the FULL-precision f32 K/V of the cached slots to feed the f32 oracle (the real gate).
        let kpool = al(pool_slots * kv_dim * 2)?; // half
        let vpool = al(pool_slots * kv_dim * 2)?; // half
        let mut kpool_f32 = vec![0.0f32; pool_slots * kv_dim]; // oracle: full-precision cached K
        let mut vpool_f32 = vec![0.0f32; pool_slots * kv_dim];
        unsafe {
            let kp = kpool.contents().as_ptr() as *mut u16;
            let vp = vpool.contents().as_ptr() as *mut u16;
            for r in 0..b {
                for (i, &slot) in seq_slots[r].iter().enumerate() {
                    if i == past_len[r] {
                        continue;
                    }
                    for j in 0..kv_dim {
                        let kv = (((r * 5 + i + j) % 19) as f32 - 9.0) * 0.03;
                        let vv = (((r * 7 + i * 3 + j) % 13) as f32 - 6.0) * 0.04;
                        *kp.add(slot as usize * kv_dim + j) = f32_to_f16_bits(kv);
                        *vp.add(slot as usize * kv_dim + j) = f32_to_f16_bits(vv);
                        kpool_f32[slot as usize * kv_dim + j] = kv; // FULL precision for the oracle
                        vpool_f32[slot as usize * kv_dim + j] = vv;
                    }
                }
            }
        }
        let qb = al(b * q_dim * 4)?; // Q stays f32
        let kb = al(b * kv_dim * 4)?; // new-token K/V fed as f32 (the scatter narrows to half)
        let vb = al(b * kv_dim * 4)?;
        let attn = al(b * q_dim * 4)?; // output f32
        let mut new_k_f32 = vec![0.0f32; b * kv_dim]; // FULL-precision new-token K/V for the oracle
        let mut new_v_f32 = vec![0.0f32; b * kv_dim];
        unsafe {
            let qp = qb.contents().as_ptr() as *mut f32;
            let kp = kb.contents().as_ptr() as *mut f32;
            let vp = vb.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..q_dim {
                    *qp.add(r * q_dim + i) = (((r * 3 + i) % 13) as f32 - 6.0) * 0.05;
                }
                for j in 0..kv_dim {
                    let kv = (((r * 11 + j) % 17) as f32 - 8.0) * 0.02;
                    let vv = (((r * 13 + j) % 11) as f32 - 5.0) * 0.03;
                    *kp.add(r * kv_dim + j) = kv;
                    *vp.add(r * kv_dim + j) = vv;
                    new_k_f32[r * kv_dim + j] = kv;
                    new_v_f32[r * kv_dim + j] = vv;
                }
            }
        }
        drop(guard);

        // Batched uniforms (identical layout to batched_paged_attn_selftest's use_attn_batched path).
        let max_slots = seq_slots.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let (sa, pla, wsa, adb, sdb, partials) = {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            let sa = al2(b * max_slots * 4);
            unsafe {
                let p = sa.contents().as_ptr() as *mut u32;
                for r in 0..b {
                    for (i, &s) in seq_slots[r].iter().enumerate() {
                        *p.add(r * max_slots + i) = s;
                    }
                    for i in seq_slots[r].len()..max_slots {
                        *p.add(r * max_slots + i) = 0;
                    }
                }
            }
            let pla = al2(b * 4);
            unsafe {
                let p = pla.contents().as_ptr() as *mut u32;
                for r in 0..b {
                    *p.add(r) = past_len[r] as u32;
                }
            }
            let wsa = al2(b * 4);
            unsafe {
                let p = wsa.contents().as_ptr() as *mut u32;
                for r in 0..b {
                    *p.add(r) = seq_slots[r]
                        .get(past_len[r])
                        .copied()
                        .unwrap_or(past_len[r] as u32);
                }
            }
            let ad = al2(48);
            unsafe {
                let p = ad.contents().as_ptr() as *mut u32;
                p.add(0).write(1);
                p.add(1).write(0);
                p.add(2).write(nh as u32);
                p.add(3).write(nkv as u32);
                p.add(4).write(hd as u32);
                p.add(5).write(0);
                (p.add(6) as *mut f32).write(scale);
                p.add(7).write(group as u32);
                p.add(8).write(max_slots as u32);
                p.add(9).write(0);
                p.add(10).write(if mode == 2 { SK_S as u32 } else { 0 });
                p.add(11).write(0);
            }
            let sdb = al2(16);
            unsafe {
                let p = sdb.contents().as_ptr() as *mut u32;
                *p = 1;
                *p.add(1) = kv_dim as u32;
                *p.add(2) = b as u32;
                *p.add(3) = 0;
            }
            let partials = al2(b * nh * SK_S * (hd + 2) * 4);
            drop(guard);
            (sa, pla, wsa, ad, sdb, partials)
        };

        // Require the f16 pipelines (compiled unconditionally by the example). Refuse a silent
        // fallback — a green report on the f32 kernels would be a LIE (the lm_head trap).
        let p_scatter_f16 = self
            .pipelines
            .get("kv_scatter_b_f16")
            .ok_or("kv_scatter_b_f16 pipeline absent (compile missing?)")?
            .pso
            .clone();
        let attn_kernel = match mode {
            2 => "attention_decode_kvhead_splitk_b_f16",
            1 => "attention_decode_kvhead_b_f16",
            _ => "attention_decode_coalesced_b_f16",
        };
        let p_attn_f16 = self
            .pipelines
            .get(attn_kernel)
            .ok_or_else(|| format!("{attn_kernel} pipeline absent (compile missing?)"))?
            .pso
            .clone();
        let p_combine = if mode == 2 {
            Some(
                self.pipelines
                    .get("attention_splitk_combine_b")
                    .ok_or("attention_splitk_combine_b pipeline absent (compile missing?)")?
                    .pso
                    .clone(),
            )
        } else {
            None
        };
        if !hd.is_multiple_of(128) {
            return Err(format!("f16 coalesced needs hd%128==0 (hd={hd})"));
        }
        if mode >= 1 && (nh / nkv != 8 || hd > 256) {
            return Err(format!(
                "kvhead needs group==8 && hd<=256 (nh={nh} nkv={nkv} hd={hd})"
            ));
        }
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        unsafe {
            let barrier = || enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            // f16 scatter: new-token f32 K/V -> half pool at write_slot[r]. Grid (kv_dim, B).
            enc.setComputePipelineState(&p_scatter_f16);
            enc.useResource_usage(ProtocolObject::from_ref(&*kb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&kb), 0, 0);
            enc.useResource_usage(ProtocolObject::from_ref(&*vb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&vb), 0, 1);
            enc.useResource_usage(ProtocolObject::from_ref(&*wsa), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&wsa), 0, 2);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*kpool),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&kpool), 0, 3);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*vpool),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&vpool), 0, 4);
            enc.useResource_usage(ProtocolObject::from_ref(&*sdb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&sdb), 0, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: kv_dim.div_ceil(256),
                    height: b,
                    depth: 1,
                },
                tg256,
            );
            barrier();
            // f16-load attention under test. Bindings 0-6 shared by all three modes; mode 2 adds
            // partials at 7 (pass1) then the production splitk combine (partials -> attn).
            enc.setComputePipelineState(&p_attn_f16);
            enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&qb), 0, 0);
            enc.useResource_usage(ProtocolObject::from_ref(&*kpool), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&kpool), 0, 1);
            enc.useResource_usage(ProtocolObject::from_ref(&*vpool), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&vpool), 0, 2);
            enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
            enc.setBuffer_offset_atIndex(Some(&attn), 0, 3);
            enc.useResource_usage(ProtocolObject::from_ref(&*sa), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&sa), 0, 4);
            enc.useResource_usage(ProtocolObject::from_ref(&*adb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&adb), 0, 5);
            enc.useResource_usage(ProtocolObject::from_ref(&*pla), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&pla), 0, 6);
            if mode == 2 {
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*partials),
                    MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&partials), 0, 7);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: nkv,
                        height: b,
                        depth: SK_S,
                    },
                    tg256,
                );
                barrier();
                enc.setComputePipelineState(p_combine.as_ref().unwrap());
                enc.useResource_usage(ProtocolObject::from_ref(&*partials), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&partials), 0, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&attn), 0, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&*adb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&adb), 0, 2);
                enc.useResource_usage(ProtocolObject::from_ref(&*pla), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&pla), 0, 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: nh,
                        height: b,
                        depth: 1,
                    },
                    tg256,
                );
            } else {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: if mode == 1 { nkv } else { nh },
                        height: b,
                        depth: 1,
                    },
                    tg256,
                );
            }
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ===== f32 ORACLE (the GATE) — full-precision K/V/Q, the SAME oracle the f32 coalesced
        // kernel matches. Scatter the new-token f32 K/V into the f32 pool copy, then run attention. =
        let row_floats = kv_dim;
        for r in 0..b {
            let ws = seq_slots[r]
                .get(past_len[r])
                .copied()
                .unwrap_or(past_len[r] as u32) as usize;
            for j in 0..kv_dim {
                kpool_f32[ws * row_floats + j] = new_k_f32[r * kv_dim + j];
                vpool_f32[ws * row_floats + j] = new_v_f32[r * kv_dim + j];
            }
        }
        let qb_v: Vec<f32> = unsafe {
            let p = qb.contents().as_ptr() as *const f32;
            (0..b * q_dim).map(|i| *p.add(i)).collect()
        };
        let mut want = vec![0.0f32; b * q_dim];
        for r in 0..b {
            let pl = past_len[r];
            let st = &seq_slots[r];
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * hd;
                let qbase = r * q_dim + hh * hd;
                let mut scs = vec![0.0f32; pl + 1];
                let mut mx = f32::MIN;
                for t in 0..=pl {
                    let kslot = st[t] as usize;
                    let mut s = 0.0f32;
                    for i in 0..hd {
                        s += qb_v[qbase + i] * kpool_f32[kslot * row_floats + hcol + i];
                    }
                    scs[t] = s * scale;
                    mx = mx.max(scs[t]);
                }
                let mut den = 0.0f32;
                for t in 0..=pl {
                    scs[t] = (scs[t] - mx).exp();
                    den += scs[t];
                }
                for dim in 0..hd {
                    let mut num = 0.0f32;
                    for t in 0..=pl {
                        let vslot = st[t] as usize;
                        num += scs[t] * vpool_f32[vslot * row_floats + hcol + dim];
                    }
                    want[r * q_dim + hh * hd + dim] = num / den;
                }
            }
        }

        // GATE: per (row, head), the argmax DIMENSION of the hd-vector must match the f32 oracle.
        // This is the greedy-stability proxy (a downstream lm_head over the o_proj'd output would
        // pick the same token iff the attention output's dominant structure is preserved). Also
        // report max_abs (expect ~1e-3, the f16 round-trip magnitude — NOT a fail).
        let _ = f16_bits_to_f32; // (available for debugging the pool if needed)
        let mut max_abs = 0.0f32;
        let mut argmax_ok = true;
        let mut mism = 0usize;
        unsafe {
            let ap = attn.contents().as_ptr() as *const f32;
            for i in 0..b * q_dim {
                max_abs = max_abs.max((*ap.add(i) - want[i]).abs());
            }
            for r in 0..b {
                for hh in 0..nh {
                    let base = r * q_dim + hh * hd;
                    let mut g_am = 0usize;
                    let mut g_best = f32::MIN;
                    let mut w_am = 0usize;
                    let mut w_best = f32::MIN;
                    for d in 0..hd {
                        let g = *ap.add(base + d);
                        if g > g_best {
                            g_best = g;
                            g_am = d;
                        }
                        let w = want[base + d];
                        if w > w_best {
                            w_best = w;
                            w_am = d;
                        }
                    }
                    if g_am != w_am {
                        argmax_ok = false;
                        mism += 1;
                    }
                }
            }
        }
        Ok((max_abs, argmax_ok, mism))
    }

    /// f16 KV PREFILL-MIRROR parity gate. Proves `kv_f32_to_f16_convert` produces a
    /// correct f16 pool from the f32 pool's PREFILL contents, so a real prompt (prefill → decode)
    /// is argmax-correct under ARF_KV_F16.
    ///
    /// The REAL-PROMPT SHAPE this models: prefill's WGSL `kv_scatter_batched` writes ALL positions
    /// [0, ctx_len) into the f32 pool (aliased keys_mtl); the f16 pool is a SEPARATE buffer never
    /// touched by that scatter, so its prefill slots are ZERO. Here we (1) write the FULL context
    /// [0, ctx_len) into the f32 pool ONLY (nothing pre-seeds the f16 pool — unlike the 9c test,
    /// which pre-seeded the f16 pool host-side); (2) run `kv_f32_to_f16_convert` to mirror the f32
    /// pool → the f16 pool over the valid range [0, pool_slots*kv_dim); (3) run the SAME f16-load
    /// coalesced attention the decode path uses; (4) assert the per-(row,head) argmax dimension
    /// matches the FULL-precision f32 oracle. WITHOUT the convert the f16 pool is all-zero → every
    /// attention output is ~0 → argmax diverges: this gate FAILS without the mirror, so it is real.
    ///
    /// Returns (max_abs_vs_oracle, argmax_ok, head_mismatches). Same contract as the f16 KV gate.
    pub fn prefill_mirror_f16_selftest(
        &mut self,
        ctx: &GpuContext,
        b: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pool_slots: usize,
    ) -> Result<(f32, bool, usize), String> {
        // IEEE-754 round-to-nearest-even f32 -> f16 (bits), matching Metal's half narrowing.
        fn f32_to_f16_bits(f: f32) -> u16 {
            let x = f.to_bits();
            let sign = ((x >> 16) & 0x8000) as u16;
            let mut mant = (x & 0x007f_ffff) as i32;
            let exp = ((x >> 23) & 0xff) as i32;
            if exp == 0xff {
                return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
            }
            let mut e = exp - 127 + 15;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
            if e <= 0 {
                if e < -10 {
                    return sign;
                }
                mant |= 0x0080_0000;
                let shift = 14 - e;
                let round = 1 << (shift - 1);
                let mut m = (mant + round) >> shift;
                if (mant & (round - 1)) == 0 && (m & 1) == 1 {
                    m -= (mant >> shift) & 1;
                }
                return sign | m as u16;
            }
            let round = 0x0000_1000;
            let mut m = mant + round;
            if (mant & 0x0000_1fff) == round {
                m &= !1;
            }
            if (m & 0x0080_0000) != 0 {
                e += 1;
                m = 0;
            }
            if e >= 0x1f {
                return sign | 0x7c00;
            }
            sign | ((e as u16) << 10) | ((m >> 13) & 0x03ff) as u16
        }
        let _ = f32_to_f16_bits; // oracle uses full-precision f32; helper kept for symmetry/debug
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        // SAME synthetic geometry as the 9c f16 gate (distinct past_len, strided slots). Here the
        // whole context [0..=past_len] is a "prefill" — ALL of it lands in the f32 pool.
        let past_len: Vec<usize> = (0..b).map(|r| 2 + r).collect();
        let seq_slots: Vec<Vec<u32>> = (0..b)
            .map(|r| (0..=past_len[r]).map(|i| (5 + r * 10 + i) as u32).collect())
            .collect();
        // The f32 KV pool (aliases keys_mtl in the real engine). We fill EVERY prefill slot here —
        // this is what the WGSL prefill scatter would have written. The f16 pool starts ZERO.
        let kpool_f32_buf = al(pool_slots * kv_dim * 4)?; // f32 source pool
        let vpool_f32_buf = al(pool_slots * kv_dim * 4)?;
        let kpool_f16 = al(pool_slots * kv_dim * 2)?; // half dest pool — ZERO until the convert
        let vpool_f16 = al(pool_slots * kv_dim * 2)?;
        let mut kpool_f32 = vec![0.0f32; pool_slots * kv_dim]; // oracle mirror (full precision)
        let mut vpool_f32 = vec![0.0f32; pool_slots * kv_dim];
        unsafe {
            let kp = kpool_f32_buf.contents().as_ptr() as *mut f32;
            let vp = vpool_f32_buf.contents().as_ptr() as *mut f32;
            // ALL positions incl. the last (the whole prompt is prefilled into the f32 pool).
            for r in 0..b {
                for (i, &slot) in seq_slots[r].iter().enumerate() {
                    for j in 0..kv_dim {
                        let kv = (((r * 5 + i + j) % 19) as f32 - 9.0) * 0.03;
                        let vv = (((r * 7 + i * 3 + j) % 13) as f32 - 6.0) * 0.04;
                        *kp.add(slot as usize * kv_dim + j) = kv;
                        *vp.add(slot as usize * kv_dim + j) = vv;
                        kpool_f32[slot as usize * kv_dim + j] = kv; // FULL precision for the oracle
                        vpool_f32[slot as usize * kv_dim + j] = vv;
                    }
                }
            }
        }
        let qb = al(b * q_dim * 4)?; // Q stays f32
        let attn = al(b * q_dim * 4)?; // output f32
        unsafe {
            let qp = qb.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..q_dim {
                    *qp.add(r * q_dim + i) = (((r * 3 + i) % 13) as f32 - 6.0) * 0.05;
                }
            }
        }
        drop(guard);

        // Attention uniforms (same layout as the 9c f16 gate). Note: NO new-token scatter here —
        // the whole context was prefilled into the f32 pool, and the convert mirrors it. Each seq's
        // causal frontier is its FULL past_len (row 0 attends [0..=past_len]).
        let max_slots = seq_slots.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let (sa, pla, adb, cdb) = {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            let sa = al2(b * max_slots * 4);
            unsafe {
                let p = sa.contents().as_ptr() as *mut u32;
                for r in 0..b {
                    for (i, &s) in seq_slots[r].iter().enumerate() {
                        *p.add(r * max_slots + i) = s;
                    }
                    for i in seq_slots[r].len()..max_slots {
                        *p.add(r * max_slots + i) = 0;
                    }
                }
            }
            let pla = al2(b * 4);
            unsafe {
                let p = pla.contents().as_ptr() as *mut u32;
                for r in 0..b {
                    *p.add(r) = past_len[r] as u32;
                }
            }
            let ad = al2(48);
            unsafe {
                let p = ad.contents().as_ptr() as *mut u32;
                p.add(0).write(1);
                p.add(1).write(0);
                p.add(2).write(nh as u32);
                p.add(3).write(nkv as u32);
                p.add(4).write(hd as u32);
                p.add(5).write(0);
                (p.add(6) as *mut f32).write(scale);
                p.add(7).write(group as u32);
                p.add(8).write(max_slots as u32);
                p.add(9).write(0);
                p.add(10).write(0);
                p.add(11).write(0);
            }
            // Convert dims: n_elems = pool_slots*kv_dim (whole valid range — the simplest-correct mirror).
            let cdb = al2(16);
            unsafe {
                let p = cdb.contents().as_ptr() as *mut u32;
                *p = (pool_slots * kv_dim) as u32;
                *p.add(1) = 0;
                *p.add(2) = 0;
                *p.add(3) = 0;
            }
            drop(guard);
            (sa, pla, ad, cdb)
        };

        // Require the convert + f16 attention pipelines (compiled by the example / weights.rs).
        let p_convert = self
            .pipelines
            .get("kv_f32_to_f16_convert")
            .ok_or("kv_f32_to_f16_convert pipeline absent (compile missing?)")?
            .pso
            .clone();
        let p_attn_f16 = self
            .pipelines
            .get("attention_decode_coalesced_b_f16")
            .ok_or("attention_decode_coalesced_b_f16 pipeline absent (compile missing?)")?
            .pso
            .clone();
        if !hd.is_multiple_of(128) {
            return Err(format!("f16 coalesced needs hd%128==0 (hd={hd})"));
        }
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let n_elems = pool_slots * kv_dim;

        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        unsafe {
            let barrier = || enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            // (1) CONVERT: f32 pool -> f16 pool over the whole valid range [0, n_elems). Grid 1D.
            enc.setComputePipelineState(&p_convert);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*kpool_f32_buf),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&kpool_f32_buf), 0, 0);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*vpool_f32_buf),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&vpool_f32_buf), 0, 1);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*kpool_f16),
                MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&kpool_f16), 0, 2);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*vpool_f16),
                MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&vpool_f16), 0, 3);
            enc.useResource_usage(ProtocolObject::from_ref(&*cdb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&cdb), 0, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_elems.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            barrier(); // convert (f16 pool write) must complete before attention reads it
                       // (2) f16-load coalesced attention over the mirrored f16 pool: grid (nh, B), 256 threads.
            enc.setComputePipelineState(&p_attn_f16);
            enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&qb), 0, 0);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*kpool_f16),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&kpool_f16), 0, 1);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*vpool_f16),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&vpool_f16), 0, 2);
            enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
            enc.setBuffer_offset_atIndex(Some(&attn), 0, 3);
            enc.useResource_usage(ProtocolObject::from_ref(&*sa), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&sa), 0, 4);
            enc.useResource_usage(ProtocolObject::from_ref(&*adb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&adb), 0, 5);
            enc.useResource_usage(ProtocolObject::from_ref(&*pla), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&pla), 0, 6);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: nh,
                    height: b,
                    depth: 1,
                },
                tg256,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ===== f32 ORACLE (the GATE) — full-precision K/V/Q over the SAME prefilled context. =====
        let row_floats = kv_dim;
        let qb_v: Vec<f32> = unsafe {
            let p = qb.contents().as_ptr() as *const f32;
            (0..b * q_dim).map(|i| *p.add(i)).collect()
        };
        let mut want = vec![0.0f32; b * q_dim];
        for r in 0..b {
            let pl = past_len[r];
            let st = &seq_slots[r];
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * hd;
                let qbase = r * q_dim + hh * hd;
                let mut scs = vec![0.0f32; pl + 1];
                let mut mx = f32::MIN;
                for t in 0..=pl {
                    let kslot = st[t] as usize;
                    let mut s = 0.0f32;
                    for i in 0..hd {
                        s += qb_v[qbase + i] * kpool_f32[kslot * row_floats + hcol + i];
                    }
                    scs[t] = s * scale;
                    mx = mx.max(scs[t]);
                }
                let mut den = 0.0f32;
                for t in 0..=pl {
                    scs[t] = (scs[t] - mx).exp();
                    den += scs[t];
                }
                for dim in 0..hd {
                    let mut num = 0.0f32;
                    for t in 0..=pl {
                        let vslot = st[t] as usize;
                        num += scs[t] * vpool_f32[vslot * row_floats + hcol + dim];
                    }
                    want[r * q_dim + hh * hd + dim] = num / den;
                }
            }
        }

        // GATE: per (row, head), the argmax DIMENSION must match the f32 oracle (greedy-stability
        // proxy). max_abs reported at the f16 round-trip magnitude (~1e-3, NOT a fail).
        let mut max_abs = 0.0f32;
        let mut argmax_ok = true;
        let mut mism = 0usize;
        unsafe {
            let ap = attn.contents().as_ptr() as *const f32;
            for i in 0..b * q_dim {
                max_abs = max_abs.max((*ap.add(i) - want[i]).abs());
            }
            for r in 0..b {
                for hh in 0..nh {
                    let base = r * q_dim + hh * hd;
                    let mut g_am = 0usize;
                    let mut g_best = f32::MIN;
                    let mut w_am = 0usize;
                    let mut w_best = f32::MIN;
                    for d in 0..hd {
                        let g = *ap.add(base + d);
                        if g > g_best {
                            g_best = g;
                            g_am = d;
                        }
                        let w = want[base + d];
                        if w > w_best {
                            w_best = w;
                            w_am = d;
                        }
                    }
                    if g_am != w_am {
                        argmax_ok = false;
                        mism += 1;
                    }
                }
            }
        }
        Ok((max_abs, argmax_ok, mism))
    }

    pub fn batched_paged_attn_selftest(
        &mut self,
        ctx: &GpuContext,
        b: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pool_slots: usize,
    ) -> Result<(f32, f32), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        // Per-seq NON-IDENTITY paged slot tables + distinct past_len. seq r occupies a disjoint,
        // non-contiguous run of physical slots in the SHARED pool. The new token (this decode step)
        // is at index past_len[r] → its physical slot = slots[r][past_len[r]].
        let past_len: Vec<usize> = (0..b).map(|r| 2 + r).collect(); // 2,3,4,... distinct
        let seq_slots: Vec<Vec<u32>> = (0..b)
            .map(|r| {
                // a non-contiguous, disjoint run: seq r starts at 5 + r*10, strides by 1, length past_len+1
                (0..=past_len[r]).map(|i| (5 + r * 10 + i) as u32).collect()
            })
            .collect();
        // SHARED pool: ONE keys + ONE values buffer, `pool_slots` rows of kv_dim. Pre-populated at
        // every seq's CACHED slots (0..past_len) with deterministic K/V; the slot for the NEW token
        // (past_len) is left ZERO (the scatter will fill it on GPU — and the oracle too).
        let kpool = al(pool_slots * kv_dim * 4)?;
        let vpool = al(pool_slots * kv_dim * 4)?;
        unsafe {
            let kp = kpool.contents().as_ptr() as *mut f32;
            let vp = vpool.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for (i, &slot) in seq_slots[r].iter().enumerate() {
                    if i == past_len[r] {
                        continue;
                    } // new-token slot: filled by scatter, not pre-pop
                    for j in 0..kv_dim {
                        *kp.add(slot as usize * kv_dim + j) =
                            (((r * 5 + i + j) % 19) as f32 - 9.0) * 0.03;
                        *vp.add(slot as usize * kv_dim + j) =
                            (((r * 7 + i * 3 + j) % 13) as f32 - 6.0) * 0.04;
                    }
                }
            }
        }
        // B-row scratch (the attention block).
        let hidden = al(b * 2048 * 4)?; // h not used directly; we feed q/k/v straight in instead
        let _ = hidden;
        let qb = al(b * q_dim * 4)?;
        let kb = al(b * kv_dim * 4)?;
        let vb = al(b * kv_dim * 4)?;
        let attn = al(b * q_dim * 4)?;
        // Fill q/k/v directly (skip the proj — we're isolating attention+scatter on paged KV).
        // The NEW token's k/v (per seq) goes into kb[r]/vb[r]; the scatter writes them to the pool.
        unsafe {
            let qp = qb.contents().as_ptr() as *mut f32;
            let kp = kb.contents().as_ptr() as *mut f32;
            let vp = vb.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..q_dim {
                    *qp.add(r * q_dim + i) = (((r * 3 + i) % 13) as f32 - 6.0) * 0.05;
                }
                for j in 0..kv_dim {
                    *kp.add(r * kv_dim + j) = (((r * 11 + j) % 17) as f32 - 8.0) * 0.02; // the NEW token's k
                    *vp.add(r * kv_dim + j) = (((r * 13 + j) % 11) as f32 - 5.0) * 0.03;
                    // the NEW token's v
                }
            }
        }
        drop(guard);

        // ===== per-seq uniforms — EXACTLY as try_batched_megakernel builds them =====
        let mut slots_b = Vec::with_capacity(b);
        let mut write_slot_b = Vec::with_capacity(b);
        let mut attn_dims_b = Vec::with_capacity(b);
        let mut scatter_dims_b = Vec::with_capacity(b);
        {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            for r in 0..b {
                let pl = past_len[r];
                let st = &seq_slots[r];
                let sb = al2(st.len().max(1) * 4);
                unsafe {
                    let p = sb.contents().as_ptr() as *mut u32;
                    for (i, &s) in st.iter().enumerate() {
                        *p.add(i) = s;
                    }
                }
                slots_b.push(sb);
                let ws = al2(16);
                unsafe {
                    *(ws.contents().as_ptr() as *mut u32) =
                        st.get(pl).copied().unwrap_or(pl as u32);
                }
                write_slot_b.push(ws);
                let ad = al2(48);
                unsafe {
                    let p = ad.contents().as_ptr() as *mut u32;
                    p.add(0).write(1);
                    p.add(1).write((pl + 1) as u32);
                    p.add(2).write(nh as u32);
                    p.add(3).write(nkv as u32);
                    p.add(4).write(hd as u32);
                    p.add(5).write(pl as u32);
                    (p.add(6) as *mut f32).write(scale);
                    p.add(7).write(group as u32);
                    p.add(8).write(0);
                    p.add(9).write(0);
                    p.add(10).write(0);
                    p.add(11).write(0);
                }
                attn_dims_b.push(ad);
                let sd = al2(16);
                unsafe {
                    let p = sd.contents().as_ptr() as *mut u32;
                    *p = 1;
                    *p.add(1) = kv_dim as u32;
                    *p.add(2) = 0;
                    *p.add(3) = 0;
                }
                scatter_dims_b.push(sd);
            }
            drop(guard);
        }

        let p_scatter = self
            .pipelines
            .get("kv_scatter")
            .ok_or("kv_scatter")?
            .pso
            .clone();
        let p_attn = self
            .pipelines
            .get("attention_decode")
            .ok_or("attention_decode")?
            .pso
            .clone();
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        // ===== ARF_ATTN_BATCHED: build the B-indexed buffers + use the B-ROW kernels (ONE
        // dispatch each) so this parity test validates the production batched path against the SAME
        // per-seq oracle. Falls back to the per-seq loop if the flag is off or the _b PSOs are
        // absent. =====
        let attn_batched = std::env::var_os("ARF_ATTN_BATCHED").is_some();
        let p_attn_b = self
            .pipelines
            .get("attention_decode_b")
            .map(|p| p.pso.clone());
        let p_scatter_b = self.pipelines.get("kv_scatter_b").map(|p| p.pso.clone());
        let use_attn_batched = attn_batched && p_attn_b.is_some() && p_scatter_b.is_some();
        // COALESCED simdgroup attention (ARF_ATTN_COALESCED, DEFAULT-OFF): A/B the 32-lane
        // coalesced kernel vs the SAME CPU oracle the (nh,B) _b kernel passes. Swaps PSO + tg size
        // (256→32) at the SAME (nh,B) grid when set + present + hd%128==0.
        let attn_coalesced = std::env::var_os("ARF_ATTN_COALESCED").is_some();
        let p_attn_coalesced_b = self
            .pipelines
            .get("attention_decode_coalesced_b")
            .map(|p| p.pso.clone());
        let max_slots = seq_slots.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let (slots_all, past_len_arr, write_slot_arr, attn_dims_bb, scatter_dims_bb) =
            if use_attn_batched {
                let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
                let dev = guard.raw_device();
                let al2 = |bytes: usize| {
                    dev.newBufferWithLength_options(
                        bytes.max(16),
                        objc2_metal::MTLResourceOptions::empty(),
                    )
                    .unwrap()
                };
                let sa = al2(b * max_slots * 4);
                unsafe {
                    let p = sa.contents().as_ptr() as *mut u32;
                    for r in 0..b {
                        for (i, &s) in seq_slots[r].iter().enumerate() {
                            *p.add(r * max_slots + i) = s;
                        }
                        for i in seq_slots[r].len()..max_slots {
                            *p.add(r * max_slots + i) = 0;
                        }
                    }
                }
                let pla = al2(b * 4);
                unsafe {
                    let p = pla.contents().as_ptr() as *mut u32;
                    for r in 0..b {
                        *p.add(r) = past_len[r] as u32;
                    }
                }
                let wsa = al2(b * 4);
                unsafe {
                    let p = wsa.contents().as_ptr() as *mut u32;
                    for r in 0..b {
                        *p.add(r) = seq_slots[r]
                            .get(past_len[r])
                            .copied()
                            .unwrap_or(past_len[r] as u32);
                    }
                }
                let ad = al2(48);
                unsafe {
                    let p = ad.contents().as_ptr() as *mut u32;
                    p.add(0).write(1);
                    p.add(1).write(0);
                    p.add(2).write(nh as u32);
                    p.add(3).write(nkv as u32);
                    p.add(4).write(hd as u32);
                    p.add(5).write(0);
                    (p.add(6) as *mut f32).write(scale);
                    p.add(7).write(group as u32);
                    p.add(8).write(max_slots as u32);
                    p.add(9).write(0);
                    p.add(10).write(0);
                    p.add(11).write(0);
                }
                let sdb = al2(16);
                unsafe {
                    let p = sdb.contents().as_ptr() as *mut u32;
                    *p = 1;
                    *p.add(1) = kv_dim as u32;
                    *p.add(2) = b as u32;
                    *p.add(3) = 0;
                }
                drop(guard);
                (Some(sa), Some(pla), Some(wsa), Some(ad), Some(sdb))
            } else {
                (None, None, None, None, None)
            };

        // ===== record the batched-mega scatter + attention (EXACT try_batched_megakernel binding) =====
        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        unsafe {
            let barrier = || enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            if use_attn_batched {
                let pa = p_attn_b.as_ref().unwrap();
                let ps = p_scatter_b.as_ref().unwrap();
                let (sa, pla, wsa, adb, sdb) = (
                    slots_all.as_ref().unwrap(),
                    past_len_arr.as_ref().unwrap(),
                    write_slot_arr.as_ref().unwrap(),
                    attn_dims_bb.as_ref().unwrap(),
                    scatter_dims_bb.as_ref().unwrap(),
                );
                // scatter_b: ONE dispatch, grid (kv_dim, B).
                enc.setComputePipelineState(ps);
                enc.useResource_usage(ProtocolObject::from_ref(&*kb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kb), 0, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*vb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vb), 0, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&**wsa), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&**wsa), 0, 2);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*kpool),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&kpool), 0, 3);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*vpool),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&vpool), 0, 4);
                enc.useResource_usage(ProtocolObject::from_ref(&**sdb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&**sdb), 0, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: kv_dim.div_ceil(256),
                        height: b,
                        depth: 1,
                    },
                    tg256,
                );
                barrier();
                // attention: ONE dispatch, grid (nh, B). COALESCED swaps the PSO at the same grid when
                // its flag is set + pipeline present + hd%128==0.
                let use_coalesced =
                    attn_coalesced && p_attn_coalesced_b.is_some() && hd.is_multiple_of(128);
                // Same integrity guard for COALESCED: refuse to green-report a silent fallback to _b.
                if attn_coalesced && !use_coalesced {
                    return Err(format!("ARF_ATTN_COALESCED set but coalesced pipeline absent or hd%128!=0 (hd={hd}) — refusing to report parity for the fallback path"));
                }
                let p_attn_sel = if use_coalesced {
                    p_attn_coalesced_b.as_ref().unwrap()
                } else {
                    pa
                };
                enc.setComputePipelineState(p_attn_sel);
                enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&qb), 0, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*kpool), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kpool), 0, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&*vpool), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vpool), 0, 2);
                enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&attn), 0, 3);
                enc.useResource_usage(ProtocolObject::from_ref(&**sa), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&**sa), 0, 4);
                enc.useResource_usage(ProtocolObject::from_ref(&**adb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&**adb), 0, 5);
                enc.useResource_usage(ProtocolObject::from_ref(&**pla), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&**pla), 0, 6);
                // COALESCED is now NSG=8 simdgroups/threadgroup (256 threads = oracle occupancy), NOT
                // the old tg32 single-simdgroup: coalesced KV access AT the oracle's occupancy.
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: nh,
                        height: b,
                        depth: 1,
                    },
                    tg256,
                );
            } else {
                // scatter PER-SEQ
                for r in 0..b {
                    enc.setComputePipelineState(&p_scatter);
                    enc.useResource_usage(ProtocolObject::from_ref(&*kb), MTLResourceUsage::Read);
                    enc.setBuffer_offset_atIndex(Some(&kb), (r * kv_dim * 4) as _, 0);
                    enc.useResource_usage(ProtocolObject::from_ref(&*vb), MTLResourceUsage::Read);
                    enc.setBuffer_offset_atIndex(Some(&vb), (r * kv_dim * 4) as _, 1);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*write_slot_b[r]),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&write_slot_b[r]), 0, 2);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*kpool),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&kpool), 0, 3);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*vpool),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&vpool), 0, 4);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*scatter_dims_b[r]),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&scatter_dims_b[r]), 0, 5);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: kv_dim.div_ceil(256),
                            height: 1,
                            depth: 1,
                        },
                        tg256,
                    );
                }
                barrier();
                // attention PER-SEQ
                for r in 0..b {
                    enc.setComputePipelineState(&p_attn);
                    enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
                    enc.setBuffer_offset_atIndex(Some(&qb), (r * q_dim * 4) as _, 0);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*kpool),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&kpool), 0, 1);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*vpool),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&vpool), 0, 2);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*attn),
                        MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&attn), (r * q_dim * 4) as _, 3);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*slots_b[r]),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&slots_b[r]), 0, 4);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*attn_dims_b[r]),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&attn_dims_b[r]), 0, 5);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: nh,
                            height: 1,
                            depth: 1,
                        },
                        tg256,
                    );
                }
            } // end else (per-seq; B-row is the if-branch above)
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ===== CPU oracle: m=1 attention math per-seq on the SAME pool/slots (after scatter) =====
        // Replicate the scatter on a CPU copy of the pool, then run the attention_decode math.
        let row_floats = kv_dim;
        let kpool_v: Vec<f32> = unsafe {
            let p = kpool.contents().as_ptr() as *const f32;
            (0..pool_slots * kv_dim).map(|i| *p.add(i)).collect()
        };
        let vpool_v: Vec<f32> = unsafe {
            let p = vpool.contents().as_ptr() as *const f32;
            (0..pool_slots * kv_dim).map(|i| *p.add(i)).collect()
        };
        let qb_v: Vec<f32> = unsafe {
            let p = qb.contents().as_ptr() as *const f32;
            (0..b * q_dim).map(|i| *p.add(i)).collect()
        };
        let mut want = vec![0.0f32; b * q_dim];
        for r in 0..b {
            let pl = past_len[r];
            let st = &seq_slots[r];
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * hd;
                let qbase = r * q_dim + hh * hd;
                let mut scs = vec![0.0f32; pl + 1];
                let mut mx = f32::MIN;
                for t in 0..=pl {
                    let kslot = st[t] as usize;
                    let mut s = 0.0f32;
                    for i in 0..hd {
                        s += qb_v[qbase + i] * kpool_v[kslot * row_floats + hcol + i];
                    }
                    scs[t] = s * scale;
                    mx = mx.max(scs[t]);
                }
                let mut den = 0.0f32;
                for t in 0..=pl {
                    scs[t] = (scs[t] - mx).exp();
                    den += scs[t];
                }
                for dim in 0..hd {
                    let mut num = 0.0f32;
                    for t in 0..=pl {
                        let vslot = st[t] as usize;
                        num += scs[t] * vpool_v[vslot * row_floats + hcol + dim];
                    }
                    want[r * q_dim + hh * hd + dim] = num / den;
                }
            }
        }

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        unsafe {
            let ap = attn.contents().as_ptr() as *const f32;
            for i in 0..b * q_dim {
                let got = *ap.add(i);
                let w = want[i];
                let abs = (got - w).abs();
                max_abs = max_abs.max(abs);
                let rel = abs / w.abs().max(1e-3);
                if rel > max_rel {
                    max_rel = rel;
                    worst = (i, got, w);
                }
            }
        }
        if std::env::var("ARF_BATCH_MEGA_DUMP").is_ok() {
            unsafe {
                let ap = attn.contents().as_ptr() as *const f32;
                for r in 0..b.min(2) {
                    let g: Vec<f32> = (0..4).map(|i| *ap.add(r * q_dim + i)).collect();
                    let w: Vec<f32> = (0..4).map(|i| want[r * q_dim + i]).collect();
                    eprintln!("  [paged dbg] row{r} past_len={} slots={:?} write_slot={} attn gpu={g:?} want={w:?}",
                        past_len[r], seq_slots[r], seq_slots[r].get(past_len[r]).copied().unwrap_or(0));
                }
                eprintln!(
                    "  [paged dbg] worst @ idx {} (row {}): gpu={} want={} (|want|={:.2e})",
                    worst.0,
                    worst.0 / q_dim,
                    worst.1,
                    worst.2,
                    worst.2.abs()
                );
            }
        }
        Ok((max_abs, max_rel))
    }
    pub fn perf_breakdown_microbench(
        &mut self,
        ctx: &GpuContext,
        b: usize,
        h: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        inter: usize,
        num_experts: usize,
        top_k: usize,
        vocab: usize,
        ctx_keys: usize,
    ) -> Result<Vec<(String, f64, usize)>, String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let eps = 1e-6f32;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // ---- Q4KS synthetic weight builder (same packing as the selftests) ----
        let subs_of = |k: usize| k / 32;
        let supers_of = |k: usize| k / 256;
        let mk_w =
            |n: usize, k: usize, seed: usize| -> [Retained<ProtocolObject<dyn MTLBuffer>>; 4] {
                let subs = subs_of(k);
                let supers = supers_of(k);
                let codes = al(n * subs * 16).unwrap();
                let sc = al((n * subs).div_ceil(4) * 4).unwrap();
                let mn = al((n * subs).div_ceil(4) * 4).unwrap();
                let dd = al(n * supers * 4).unwrap(); // ddh: packed half2 {d,dmin}, one u32/super-block
                unsafe {
                    let cp = codes.contents().as_ptr() as *mut u32;
                    for i in 0..(n * subs * 4) {
                        *cp.add(i) = (i
                            .wrapping_mul(2654435761)
                            .wrapping_add(seed.wrapping_mul(40503))
                            & 0xFFFF_FFFF) as u32;
                    }
                    let sp = sc.contents().as_ptr() as *mut u8;
                    let mp = mn.contents().as_ptr() as *mut i8;
                    for i in 0..(n * subs) {
                        *sp.add(i) = (((i + seed) % 31) + 1) as u8;
                        *mp.add(i) = ((((i + seed) % 17) as i32) - 8) as i8;
                    }
                    let scl = 1.0 / (k as f32).sqrt();
                    let dp = dd.contents().as_ptr() as *mut u32;
                    for i in 0..(n * supers) {
                        let dv = (0.02 + (i % 5) as f32 * 0.003) * scl;
                        let dmv = (-0.01 - (i % 3) as f32 * 0.002) * scl;
                        *dp.add(i) =
                            (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
                    }
                }
                [codes, sc, mn, dd]
            };
        // dense projections (q/k/v/o) + router (all dense Q4KS).
        let qw = mk_w(q_dim, h, 1);
        let kw = mk_w(kv_dim, h, 2);
        let vw = mk_w(kv_dim, h, 3);
        let ow = mk_w(h, q_dim, 4);
        let rw = mk_w(num_experts, h, 11);
        // all-experts MoE weights.
        let gate_w = mk_w(num_experts * inter, h, 21);
        let up_w = mk_w(num_experts * inter, h, 31);
        let down_w = mk_w(num_experts * h, inter, 41);
        // lm_head Q8: data packed 4 int8/u32 [vocab, h], one f32 scale/row.
        let lm_data = al(vocab * h)?; // 1 byte/elem (u8 view)
        let lm_scale = al(vocab * 4)?;
        unsafe {
            let p = lm_data.contents().as_ptr() as *mut u8;
            for i in 0..(vocab * h) {
                *p.add(i) = ((i % 17) as i32 - 8) as u8;
            }
            let s = lm_scale.contents().as_ptr() as *mut f32;
            for i in 0..vocab {
                *s.add(i) = 0.01;
            }
        }

        // ---- B-row scratch ----
        let hidden = al(b * h * 4)?;
        let normed = al(b * h * 4)?;
        let qb = al(b * q_dim * 4)?;
        let kb = al(b * kv_dim * 4)?;
        let vb = al(b * kv_dim * 4)?;
        let attn = al(b * q_dim * 4)?;
        let attn_out = al(b * h * 4)?;
        let mlogits = al(b * num_experts * 4)?;
        let ids = al(b * top_k * 4)?;
        let wts = al(b * top_k * 4)?;
        let gate_all = al(b * top_k * inter * 4)?;
        let up_all = al(b * top_k * inter * 4)?;
        let silu_all = al(b * top_k * inter * 4)?;
        let mlp_down = al(b * h * 4)?;
        let logits_b = al(b * vocab * 4)?;
        let out_tokens = al(b * 4)?;
        // per-head q/k norm weights + input/post norm.
        let qnw = al(hd * 4)?;
        let knw = al(hd * 4)?;
        let in_norm = al(h * 4)?;
        let post_norm = al(h * 4)?;
        let final_norm = al(h * 4)?;
        unsafe {
            for buf in [&qnw, &knw] {
                let p = buf.contents().as_ptr() as *mut f32;
                for i in 0..hd {
                    *p.add(i) = 1.0;
                }
            }
            for buf in [&in_norm, &post_norm, &final_norm] {
                let p = buf.contents().as_ptr() as *mut f32;
                for i in 0..h {
                    *p.add(i) = 1.0;
                }
            }
            // fill ids in-range + wts + activations so kernels do real work.
            let ip = ids.contents().as_ptr() as *mut u32;
            for i in 0..b * top_k {
                *ip.add(i) = ((i * 7) % num_experts) as u32;
            }
            let wp = wts.contents().as_ptr() as *mut f32;
            for i in 0..b * top_k {
                *wp.add(i) = 1.0 / top_k as f32;
            }
            for buf in [&hidden, &normed] {
                let p = buf.contents().as_ptr() as *mut f32;
                for i in 0..b * h {
                    *p.add(i) = ((i % 23) as f32 - 11.0) * 0.05;
                }
            }
            let sp = silu_all.contents().as_ptr() as *mut f32;
            for i in 0..b * top_k * inter {
                *sp.add(i) = ((i % 19) as f32 - 9.0) * 0.03;
            }
            let lp = mlogits.contents().as_ptr() as *mut f32;
            for i in 0..b * num_experts {
                *lp.add(i) = ((i % 29) as f32 - 14.0) * 0.1;
            }
        }
        // shared KV pool (paged-style) of ctx_keys rows + per-seq slot tables (identity 0..keys).
        let kpool = al(ctx_keys * kv_dim * 4)?;
        let vpool = al(ctx_keys * kv_dim * 4)?;
        unsafe {
            for buf in [&kpool, &vpool] {
                let p = buf.contents().as_ptr() as *mut f32;
                for i in 0..ctx_keys * kv_dim {
                    *p.add(i) = ((i % 13) as f32 - 6.0) * 0.03;
                }
            }
        }
        let mut slots_b = Vec::with_capacity(b);
        let mut attn_dims_b = Vec::with_capacity(b);
        let mut scatter_dims_b = Vec::with_capacity(b);
        let mut write_slot_b = Vec::with_capacity(b);
        for _r in 0..b {
            let sl = al(ctx_keys * 4)?;
            unsafe {
                let p = sl.contents().as_ptr() as *mut u32;
                for s in 0..ctx_keys {
                    *p.add(s) = s as u32;
                }
            }
            slots_b.push(sl);
            let ad = al(48)?;
            unsafe {
                let p = ad.contents().as_ptr() as *mut u32;
                p.add(0).write(1);
                p.add(1).write(ctx_keys as u32);
                p.add(2).write(nh as u32);
                p.add(3).write(nkv as u32);
                p.add(4).write(hd as u32);
                p.add(5).write((ctx_keys - 1) as u32);
                (p.add(6) as *mut f32).write(scale);
                p.add(7).write(group as u32);
                p.add(8).write(0);
                p.add(9).write(0);
                p.add(10).write(0);
                p.add(11).write(0);
            }
            attn_dims_b.push(ad);
            let sd = al(16)?;
            unsafe {
                let p = sd.contents().as_ptr() as *mut u32;
                *p = 1;
                *p.add(1) = kv_dim as u32;
                *p.add(2) = 0;
                *p.add(3) = 0;
            }
            scatter_dims_b.push(sd);
            let ws = al(16)?;
            unsafe {
                *(ws.contents().as_ptr() as *mut u32) = 0;
            }
            write_slot_b.push(ws);
        }
        // The batched attention timing row needs _b-style buffers: ONE slots_all [B, max_slots]
        // (identity 0..ctx_keys per row), past_len_b [B] (=ctx_keys-1), and a Dims with group@7 +
        // q_start=max_slots@8. max_slots = ctx_keys (the slot-table stride).
        let bt_max_slots = ctx_keys;
        let bt_slots_all = al(b * bt_max_slots * 4)?;
        unsafe {
            let p = bt_slots_all.contents().as_ptr() as *mut u32;
            for r in 0..b {
                for s in 0..ctx_keys {
                    *p.add(r * bt_max_slots + s) = s as u32;
                }
            }
        }
        let bt_past_len = al(b * 4)?;
        unsafe {
            let p = bt_past_len.contents().as_ptr() as *mut u32;
            for r in 0..b {
                *p.add(r) = (ctx_keys - 1) as u32;
            }
        }
        let bt_dims = al(48)?;
        unsafe {
            let p = bt_dims.contents().as_ptr() as *mut u32;
            p.add(0).write(1);
            p.add(1).write(0);
            p.add(2).write(nh as u32);
            p.add(3).write(nkv as u32);
            p.add(4).write(hd as u32);
            p.add(5).write(0);
            (p.add(6) as *mut f32).write(scale);
            p.add(7).write(group as u32);
            p.add(8).write(bt_max_slots as u32);
            p.add(9).write(0);
            p.add(10).write(0);
            p.add(11).write(0);
        }
        let pos_buf = al(b * 4)?;
        unsafe {
            let p = pos_buf.contents().as_ptr() as *mut u32;
            for r in 0..b {
                *p.add(r) = (ctx_keys - 1) as u32;
            }
        }
        let rope_cos = al((ctx_keys + 1) * (hd / 2) * 4)?;
        let rope_sin = al((ctx_keys + 1) * (hd / 2) * 4)?;
        unsafe {
            for buf in [&rope_cos, &rope_sin] {
                let p = buf.contents().as_ptr() as *mut f32;
                for i in 0..(ctx_keys + 1) * (hd / 2) {
                    *p.add(i) = 0.5;
                }
            }
        }

        // dims uniforms.
        let mk_gd = |k: usize, n: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let bf = al(16).unwrap();
            unsafe {
                let p = bf.contents().as_ptr() as *mut u32;
                *p = b as u32;
                *p.add(1) = k as u32;
                *p.add(2) = n as u32;
                *p.add(3) = 0;
            }
            bf
        };
        let gd_q = mk_gd(h, q_dim);
        let gd_k = mk_gd(h, kv_dim);
        let gd_v = mk_gd(h, kv_dim);
        let gd_o = mk_gd(q_dim, h);
        let gd_router = mk_gd(h, num_experts);
        let norm_dims = al(16)?;
        unsafe {
            let p = norm_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *(p.add(2) as *mut f32) = eps;
            *p.add(3) = 0;
        }
        let qk_dims = al(16)?;
        unsafe {
            let p = qk_dims.contents().as_ptr() as *mut u32;
            *p = nh as u32;
            *p.add(1) = nkv as u32;
            *p.add(2) = hd as u32;
            *(p.add(3) as *mut f32) = eps;
        }
        let rope_dims = al(32)?;
        unsafe {
            let p = rope_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = nh as u32;
            *p.add(2) = nkv as u32;
            *p.add(3) = hd as u32;
            *p.add(4) = hd as u32;
            *p.add(5) = 0;
            *p.add(6) = 0;
            *p.add(7) = 0;
        }
        let gu_dims = al(16)?;
        unsafe {
            let p = gu_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = inter as u32;
            *p.add(3) = top_k as u32;
        }
        let down_dims = al(16)?;
        unsafe {
            let p = down_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = inter as u32;
            *p.add(2) = h as u32;
            *p.add(3) = top_k as u32;
        }
        let route_dims = al(16)?;
        unsafe {
            let p = route_dims.contents().as_ptr() as *mut u32;
            *p = num_experts as u32;
            *p.add(1) = top_k as u32;
            *p.add(2) = 1;
            *p.add(3) = 0;
        }
        let swiglu_dims = al(16)?;
        unsafe {
            let p = swiglu_dims.contents().as_ptr() as *mut u32;
            *p = (b * top_k * inter) as u32;
            *p.add(1) = 0;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let resid_dims = al(16)?;
        unsafe {
            let p = resid_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let add_norm_dims = al(16)?;
        unsafe {
            let p = add_norm_dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *(p.add(2) as *mut f32) = eps;
            *p.add(3) = 0;
        }
        let q8_dims = al(16)?;
        unsafe {
            let p = q8_dims.contents().as_ptr() as *mut u32;
            *p = 1;
            *p.add(1) = h as u32;
            *p.add(2) = vocab as u32;
            *p.add(3) = 0;
        }
        let am_dims = al(16)?;
        unsafe {
            *(am_dims.contents().as_ptr() as *mut u32) = vocab as u32;
        }
        drop(guard);

        // ---- pipelines ----
        let pso = |k: &str| {
            self.pipelines
                .get(k)
                .map(|p| p.pso.clone())
                .ok_or_else(|| format!("pso {k}"))
        };
        let p_gemv_b = pso("gemv_q4ks_batch")?;
        let p_rms_b = pso("rmsnorm_b")?;
        let p_qk_b = pso("qk_norm_b")?;
        let p_rope_b = pso("rope_qk_b")?;
        let p_resid_b = pso("add_residual_b")?;
        let p_scatter = pso("kv_scatter")?;
        let p_attn = pso("attention_decode")?;
        let p_add_norm = pso("add_norm")?;
        let p_route_b = pso("moe_route_b")?;
        let p_id_b = pso("gemv_q4ks_id_b")?;
        let p_swiglu_b = pso("moe_swiglu_b")?;
        let p_down_b = pso("gemv_q4ks_id_down_b")?;
        let p_q8 = pso("gemv_q8")?;
        let p_argmax = pso("argmax")?;
        // GQA head-group attention (optional — only timed if the kernel was compiled into the island).
        let tg32 = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let tg64 = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let tg128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        // Per-kernel timer: N reps of ONE kernel-type in ONE concurrent encoder (no inter-rep
        // barrier — independent reps, measures steady-state per-CALL GPU throughput), warm up, then
        // GPUEndTime-GPUStartTime / N. The `record` closure issues ONE dispatch of the kernel.
        let reps = 200usize;
        let time_kernel = |record: &dyn Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>)| -> f64 {
            // warm-up
            for _ in 0..2 {
                let cmd = self.queue.commandBuffer().unwrap();
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .unwrap();
                for _ in 0..reps {
                    record(&enc);
                }
                enc.endEncoding();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let cmd = self.queue.commandBuffer().unwrap();
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .unwrap();
                for _ in 0..reps {
                    record(&enc);
                }
                enc.endEncoding();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                let ms = (unsafe { cmd.GPUEndTime() } - unsafe { cmd.GPUStartTime() }) * 1000.0;
                best = best.min(ms / reps as f64);
            }
            best
        };
        let bind_into = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
                         bufs: &[&ProtocolObject<dyn MTLBuffer>]| {
            for (i, bf) in bufs.iter().enumerate() {
                unsafe {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
            }
        };

        let mut out: Vec<(String, f64, usize)> = Vec::new();
        // (name, per-call ms, dispatches-per-layer). Once-per-token kernels use 0 in the
        // dispatches field as a flag (reported separately, NOT ×layers).
        macro_rules! push {
            ($name:expr, $ms:expr, $disp:expr) => {
                out.push(($name.to_string(), $ms, $disp));
            };
        }

        // rmsnorm_b (in_norm): grid.x=b, tg256.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_rms_b);
            }
            bind_into(enc, &[&hidden, &in_norm, &normed, &norm_dims]);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: b,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
        });
        push!("rmsnorm_b (in_norm)", ms, 1);
        // dense GEMV q/k/v/o: grid.x=n, tg32. Time each (k,n differs).
        let gemv = |w: &[Retained<ProtocolObject<dyn MTLBuffer>>; 4],
                    a: &ProtocolObject<dyn MTLBuffer>,
                    c: &ProtocolObject<dyn MTLBuffer>,
                    gd: &ProtocolObject<dyn MTLBuffer>,
                    n: usize|
         -> f64 {
            time_kernel(&|enc| {
                unsafe {
                    enc.setComputePipelineState(&p_gemv_b);
                }
                bind_into(enc, &[a, &w[0], c, gd, &w[1], &w[2], &w[3]]);
                unsafe {
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: n,
                            height: 1,
                            depth: 1,
                        },
                        tg32,
                    );
                }
            })
        };
        push!("gemv_b q_proj", gemv(&qw, &normed, &qb, &gd_q, q_dim), 1);
        push!("gemv_b k_proj", gemv(&kw, &normed, &kb, &gd_k, kv_dim), 1);
        push!("gemv_b v_proj", gemv(&vw, &normed, &vb, &gd_v, kv_dim), 1);
        push!("gemv_b o_proj", gemv(&ow, &attn, &attn_out, &gd_o, h), 1);
        // qk_norm_b: grid.x=b*(nh+nkv), tg64.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_qk_b);
            }
            bind_into(enc, &[&qb, &kb, &qnw, &knw, &qk_dims]);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: b * (nh + nkv),
                        height: 1,
                        depth: 1,
                    },
                    tg64,
                );
            }
        });
        push!("qk_norm_b", ms, 1);
        // rope_qk_b: grid covers b*(q_dim+kv_dim), tg256.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_rope_b);
            }
            bind_into(enc, &[&qb, &kb, &rope_cos, &rope_sin, &pos_buf, &rope_dims]);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (b * (q_dim + kv_dim)).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
        });
        push!("rope_qk_b", ms, 1);
        // kv_scatter PER-SEQ: B dispatches/layer, grid.x=kv_dim.div_ceil(256), tg256. Time ONE.
        let ms = time_kernel(&|enc| unsafe {
            enc.setComputePipelineState(&p_scatter);
            enc.useResource_usage(ProtocolObject::from_ref(&*kb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&kb), 0, 0);
            enc.useResource_usage(ProtocolObject::from_ref(&*vb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&vb), 0, 1);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*write_slot_b[0]),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&write_slot_b[0]), 0, 2);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*kpool),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&kpool), 0, 3);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*vpool),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&vpool), 0, 4);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*scatter_dims_b[0]),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&scatter_dims_b[0]), 0, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: kv_dim.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
        });
        push!("kv_scatter (per-seq ×B)", ms, b);
        // attention_decode PER-SEQ: B dispatches/layer, grid.x=nh, tg256. Time ONE.
        let ms = time_kernel(&|enc| unsafe {
            enc.setComputePipelineState(&p_attn);
            enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&qb), 0, 0);
            enc.useResource_usage(ProtocolObject::from_ref(&*kpool), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&kpool), 0, 1);
            enc.useResource_usage(ProtocolObject::from_ref(&*vpool), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&vpool), 0, 2);
            enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
            enc.setBuffer_offset_atIndex(Some(&attn), 0, 3);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*slots_b[0]),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&slots_b[0]), 0, 4);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*attn_dims_b[0]),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&attn_dims_b[0]), 0, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: nh,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
        });
        push!("attention_decode (per-seq ×B)", ms, b);
        // Production baseline attention at its REAL grid (nh, B) in ONE batched call (disp=0,
        // counted once/token, NOT xB — the earlier per-(grid,1) tagging double-counted the batch dim).
        let p_attn_b_mb = self
            .pipelines
            .get("attention_decode_b")
            .map(|p| p.pso.clone());
        if let Some(p_ab) = p_attn_b_mb.as_ref() {
            let ms_ab = time_kernel(&|enc| unsafe {
                enc.setComputePipelineState(p_ab);
                enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&qb), 0, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*kpool), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&kpool), 0, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&*vpool), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&vpool), 0, 2);
                enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&attn), 0, 3);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*bt_slots_all),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&bt_slots_all), 0, 4);
                enc.useResource_usage(ProtocolObject::from_ref(&*bt_dims), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&bt_dims), 0, 5);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*bt_past_len),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&bt_past_len), 0, 6);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: nh,
                        height: b,
                        depth: 1,
                    },
                    tg256,
                );
            });
            push!("attn_decode_b (nh,B) 1-call", ms_ab, 0); // disp=0 → once/token
        }
        // add_norm: grid.x=b, tg256.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_add_norm);
            }
            bind_into(
                enc,
                &[&hidden, &attn_out, &post_norm, &normed, &add_norm_dims],
            );
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: b,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
        });
        push!("add_norm", ms, 1);
        // router GEMV: dense, grid.x=num_experts, tg32.
        push!(
            "gemv_b router",
            gemv(&rw, &normed, &mlogits, &gd_router, num_experts),
            1
        );
        // moe_route_b: grid.x=b, tg128.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_route_b);
            }
            bind_into(enc, &[&mlogits, &ids, &wts, &route_dims]);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: b,
                        height: 1,
                        depth: 1,
                    },
                    tg128,
                );
            }
        });
        push!("moe_route_b", ms, 1);
        // gate GEMV (id_b, COLS=8): grid=(inter/8, b, top_k), tg32.
        let gu_grid = MTLSize {
            width: inter.div_ceil(8),
            height: b,
            depth: top_k,
        };
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_id_b);
            }
            bind_into(
                enc,
                &[
                    &normed, &gate_w[0], &gate_all, &gu_dims, &gate_w[1], &ids, &gate_w[2],
                    &gate_w[3],
                ],
            );
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(gu_grid, tg32);
            }
        });
        push!("gemv_id_b gate (MoE,COLS8)", ms, 1);
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_id_b);
            }
            bind_into(
                enc,
                &[
                    &normed, &up_w[0], &up_all, &gu_dims, &up_w[1], &ids, &up_w[2], &up_w[3],
                ],
            );
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(gu_grid, tg32);
            }
        });
        push!("gemv_id_b up (MoE,COLS8)", ms, 1);
        // moe_swiglu_b: grid covers b*top_k*inter, tg256.
        let sw_n = b * top_k * inter;
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_swiglu_b);
            }
            bind_into(enc, &[&gate_all, &up_all, &silu_all, &swiglu_dims]);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: sw_n.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
        });
        push!("moe_swiglu_b", ms, 1);
        // down GEMV (id_down_b, DCOLS=8): grid=(h/8, b, 1), tg32.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_down_b);
            }
            bind_into(
                enc,
                &[
                    &silu_all, &down_w[0], &mlp_down, &down_dims, &down_w[1], &ids, &wts,
                    &down_w[2], &down_w[3],
                ],
            );
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: h.div_ceil(8),
                        height: b,
                        depth: 1,
                    },
                    tg32,
                );
            }
        });
        push!("gemv_id_down_b (MoE,DCOLS8)", ms, 1);
        // add_residual_b: grid covers b*h, tg256.
        let ms = time_kernel(&|enc| {
            unsafe {
                enc.setComputePipelineState(&p_resid_b);
            }
            bind_into(enc, &[&hidden, &mlp_down, &resid_dims]);
            unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (b * h).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
        });
        push!("add_residual_b", ms, 1);
        // ---- once-per-token (NOT ×layers; disp flag = 0) ----
        // lm_head Q8 + argmax run ONCE per token, PER-ROW (B disjoint dispatches into DISTINCT
        // logits/out offsets). The real frame issues all B as one phase; to capture overlap, time
        // the WHOLE B-row phase per rep (B dispatches writing disjoint offsets) and divide by reps —
        // NOT one call ×B (which would miss concurrency). time_phase = reps of {B dispatches}.
        let time_phase = |record_b: &dyn Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>)| -> f64 {
            for _ in 0..2 {
                let cmd = self.queue.commandBuffer().unwrap();
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .unwrap();
                for _ in 0..reps {
                    record_b(&enc);
                }
                enc.endEncoding();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let cmd = self.queue.commandBuffer().unwrap();
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .unwrap();
                for _ in 0..reps {
                    record_b(&enc);
                }
                enc.endEncoding();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                let ms = (unsafe { cmd.GPUEndTime() } - unsafe { cmd.GPUStartTime() }) * 1000.0;
                best = best.min(ms / reps as f64);
            }
            best
        };
        // lm_head: B disjoint per-row Q8 GEMVs (read normed[r*h], write logits[r*vocab]).
        let lm_ms = time_phase(&|enc| {
            for r in 0..b {
                unsafe {
                    enc.setComputePipelineState(&p_q8);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*normed),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&normed), (r * h * 4) as _, 0);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*lm_data),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&lm_data), 0, 1);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*logits_b),
                        MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&logits_b), (r * vocab * 4) as _, 2);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*q8_dims),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&q8_dims), 0, 3);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*lm_scale),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&lm_scale), 0, 4);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: 1024,
                            height: 1,
                            depth: 1,
                        },
                        tg256,
                    );
                }
            }
        });
        push!("lm_head Q8 (1/tok, B-row phase)", lm_ms, 0);
        // argmax: B disjoint per-row reductions.
        let am_ms = time_phase(&|enc| {
            for r in 0..b {
                unsafe {
                    enc.setComputePipelineState(&p_argmax);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*logits_b),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&logits_b), (r * vocab * 4) as _, 0);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*out_tokens),
                        MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&out_tokens), (r * 4) as _, 1);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*am_dims),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&am_dims), 0, 2);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        tg256,
                    );
                }
            }
        });
        push!("argmax (1/tok, B-row phase)", am_ms, 0);
        Ok(out)
    }
    /// BATCHED Q8 lm_head + argmax parity (no model load). The HARD greedy gate for the
    /// conc64 lm_head lever: builds a deterministic Q8 lm_head weight `vocab,h` (4 int8/u32, one
    /// f32 scale/row) + deterministic normed`B,h`, then runs BOTH the shipped per-row path
    /// (gemv_q8 m=1 per row + per-row argmax = the oracle) AND the new ONE-dispatch batched path
    /// (gemv_q8_b m=B + argmax_b), and compares. The Q8 batched reduction REUSES the per-row
    /// 4-term dot / i8 unpack / scale-after-sum, so logits drift only by f32 reorder (≤1e-3,
    /// greedy-safe); the ARGMAX TOKEN must be byte-identical for every row. Returns
    /// (max_logit_drift, all_tokens_match, mismatch_count). vocab>8192 exercises the grid-stride.
    pub fn batched_lm_head_selftest(
        &mut self,
        ctx: &GpuContext,
        b: usize,
    ) -> Result<(f32, bool, usize), String> {
        let (drift, ok, mism, _gemm) = self.batched_lm_head_selftest_full(ctx, b)?;
        Ok((drift, ok, mism))
    }
    pub fn batched_lm_head_selftest_full(
        &mut self,
        ctx: &GpuContext,
        b: usize,
    ) -> Result<(f32, bool, usize, Option<(f32, bool, usize)>), String> {
        let h = 2048usize; // qwen3-coder hidden
        let vocab = 16384usize; // >8192 so gemv_q8_b grid-strides over multiple waves (1024tg×8sg)
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        // Q8 lm_head: data packed 4 int8/u32 [vocab,h] (u8 view), one f32 scale/row.
        let lm_data = al(vocab * h)?; // 1 byte/elem
        let lm_scale = al(vocab * 4)?;
        let normed = al(b * h * 4)?;
        let logits_ref = al(b * vocab * 4)?; // per-row path output
        let logits_b = al(b * vocab * 4)?; // batched path output
        let logits_mm = al(b * vocab * 4)?; // tiled GEMM path output
        let out_ref = al(b * 4)?; // per-row argmax tokens
        let out_b = al(b * 4)?; // batched argmax tokens
        let out_mm = al(b * 4)?; // tiled GEMM argmax tokens
        let q8_dims_one = al(16)?;
        unsafe {
            let p = q8_dims_one.contents().as_ptr() as *mut u32;
            *p = 1;
            *p.add(1) = h as u32;
            *p.add(2) = vocab as u32;
            *p.add(3) = 0;
        }
        let q8_dims_b = al(16)?;
        unsafe {
            let p = q8_dims_b.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = h as u32;
            *p.add(2) = vocab as u32;
            *p.add(3) = 0;
        }
        let am_dims = al(16)?;
        unsafe {
            *(am_dims.contents().as_ptr() as *mut u32) = vocab as u32;
        }
        // L143 — per-chunk dims for the row-chunked gemv_q8_b below (allocated here, before
        // `guard` is moved into the encoder scope).
        let q8_chunks: Vec<_> = {
            const MAXB_Q8: usize = 64;
            let mut v = Vec::new();
            let mut r0 = 0usize;
            while r0 < b {
                let rows = (b - r0).min(MAXB_Q8);
                let bf = al(16)?;
                unsafe {
                    let p = bf.contents().as_ptr() as *mut u32;
                    *p = rows as u32;
                    *p.add(1) = h as u32;
                    *p.add(2) = vocab as u32;
                    *p.add(3) = 0;
                }
                v.push((r0, bf));
                r0 += rows;
            }
            v
        };
        unsafe {
            // deterministic int8 weights (spread so columns differ → distinct argmaxes per row).
            let wp = lm_data.contents().as_ptr() as *mut u8;
            for i in 0..(vocab * h) {
                *wp.add(i) = ((i.wrapping_mul(2654435761) % 251) as i32 - 125) as u8;
            }
            let sp = lm_scale.contents().as_ptr() as *mut f32;
            for col in 0..vocab {
                *sp.add(col) = 0.005 + (col % 7) as f32 * 0.001;
            } // per-column dequant
              // deterministic activations, distinct per batch row (so rows argmax to different tokens).
            let np = normed.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..h {
                    *np.add(r * h + i) = (((r * 31 + i) % 53) as f32 - 26.0) * 0.04;
                }
            }
        }
        drop(guard);

        let pso = |k: &str| {
            self.pipelines
                .get(k)
                .map(|p| p.pso.clone())
                .ok_or_else(|| format!("pso {k}"))
        };
        let p_q8 = pso("gemv_q8")?;
        let p_argmax = pso("argmax")?;
        let p_q8_b = pso("gemv_q8_b")?;
        let p_argmax_b = pso("argmax_b")?;
        // Tiled Q8 GEMM (the DEFAULT lm_head path; only present when compiled). Drive it unless
        // ARF_NO_LM_HEAD_GEMM=1 — mirrors the production gate so parity is checked by default.
        let p_q8_mm = if std::env::var("ARF_NO_LM_HEAD_GEMM").ok().as_deref() != Some("1") {
            self.pipelines.get("matmul_mm_q8").map(|p| p.pso.clone())
        } else {
            None
        };
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let tg128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        unsafe {
            // ----- PER-ROW oracle: B disjoint gemv_q8 (m=1) + per-row argmax -----
            for r in 0..b {
                enc.setComputePipelineState(&p_q8);
                enc.useResource_usage(ProtocolObject::from_ref(&*normed), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&normed), (r * h * 4) as _, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*lm_data), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&lm_data), 0, 1);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*logits_ref),
                    MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&logits_ref), (r * vocab * 4) as _, 2);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*q8_dims_one),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&q8_dims_one), 0, 3);
                enc.useResource_usage(ProtocolObject::from_ref(&*lm_scale), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&lm_scale), 0, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                enc.setComputePipelineState(&p_argmax);
                enc.useResource_usage(
                    ProtocolObject::from_ref(&*logits_ref),
                    MTLResourceUsage::Read,
                );
                enc.setBuffer_offset_atIndex(Some(&logits_ref), (r * vocab * 4) as _, 0);
                enc.useResource_usage(ProtocolObject::from_ref(&*out_ref), MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&out_ref), (r * 4) as _, 1);
                enc.useResource_usage(ProtocolObject::from_ref(&*am_dims), MTLResourceUsage::Read);
                enc.setBuffer_offset_atIndex(Some(&am_dims), 0, 2);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
            // ----- BATCHED: ONE gemv_q8_b (m=B) + ONE argmax_b -----
            let bind = |bufs: &[(&ProtocolObject<dyn MTLBuffer>, MTLResourceUsage)]| {
                for (i, (bf, u)) in bufs.iter().enumerate() {
                    enc.useResource_usage(ProtocolObject::from_ref(*bf), *u);
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
            };
            // L143 — ROW-CHUNKED, mirroring the shipped path. `gemv_q8_b` clamps
            // `m = min(d.m, MAXB_Q8=64)`, so a single dispatch with b>64 leaves rows 64.. of
            // logits_b UNWRITTEN and argmax_b then argmaxes stale logits. Chunk by buffer offsets
            // so this oracle validates the pattern the record actually uses.
            for (row0, gd_c) in q8_chunks.iter() {
                let row0 = *row0;
                enc.setComputePipelineState(&p_q8_b);
                unsafe {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*normed),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&normed), (row0 * h * 4) as _, 0);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*lm_data),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&lm_data), 0, 1);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*logits_b),
                        MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(&logits_b), (row0 * vocab * 4) as _, 2);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&**gd_c),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&**gd_c), 0, 3);
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&*lm_scale),
                        MTLResourceUsage::Read,
                    );
                    enc.setBuffer_offset_atIndex(Some(&lm_scale), 0, 4);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&p_argmax_b);
            bind(&[
                (&logits_b, MTLResourceUsage::Read),
                (&out_b, MTLResourceUsage::Write),
                (&am_dims, MTLResourceUsage::Read),
            ]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: b,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            // ----- TILED Q8 GEMM (optional): ONE matmul_mm_q8 (m=B) + ONE argmax_b -----
            if let Some(p_q8_mm) = p_q8_mm.as_ref() {
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                enc.setComputePipelineState(p_q8_mm);
                enc.setThreadgroupMemoryLength_atIndex(12288, 0); // sW 8K + sA 4K
                bind(&[
                    (&normed, MTLResourceUsage::Read),
                    (&lm_data, MTLResourceUsage::Read),
                    (&logits_mm, MTLResourceUsage::Write),
                    (&q8_dims_b, MTLResourceUsage::Read),
                    (&lm_scale, MTLResourceUsage::Read),
                ]);
                let gx = (b as u32).div_ceil(32) as usize;
                let gy = (vocab as u32).div_ceil(64) as usize;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: gx,
                        height: gy,
                        depth: 1,
                    },
                    tg128,
                );
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                enc.setComputePipelineState(&p_argmax_b);
                bind(&[
                    (&logits_mm, MTLResourceUsage::Read),
                    (&out_mm, MTLResourceUsage::Write),
                    (&am_dims, MTLResourceUsage::Read),
                ]);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: b,
                        height: 1,
                        depth: 1,
                    },
                    tg256,
                );
            }
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // ----- compare: tokens MUST be identical; report max logit drift -----
        let (tokens_ref, tokens_b, lr, lb): (Vec<u32>, Vec<u32>, &[f32], &[f32]) = unsafe {
            let tr =
                std::slice::from_raw_parts(out_ref.contents().as_ptr() as *const u32, b).to_vec();
            let tb =
                std::slice::from_raw_parts(out_b.contents().as_ptr() as *const u32, b).to_vec();
            let lr =
                std::slice::from_raw_parts(logits_ref.contents().as_ptr() as *const f32, b * vocab);
            let lb =
                std::slice::from_raw_parts(logits_b.contents().as_ptr() as *const f32, b * vocab);
            (tr, tb, lr, lb)
        };
        let mut max_drift = 0.0f32;
        for i in 0..b * vocab {
            max_drift = max_drift.max((lr[i] - lb[i]).abs());
        }
        let mut mismatches = 0usize;
        for r in 0..b {
            if tokens_ref[r] != tokens_b[r] {
                mismatches += 1;
            }
        }
        // ----- TILED GEMM comparison (vs the SAME per-row oracle) -----
        let gemm = if p_q8_mm.is_some() {
            let (tokens_mm, lm): (Vec<u32>, &[f32]) = unsafe {
                let tm = std::slice::from_raw_parts(out_mm.contents().as_ptr() as *const u32, b)
                    .to_vec();
                let lm = std::slice::from_raw_parts(
                    logits_mm.contents().as_ptr() as *const f32,
                    b * vocab,
                );
                (tm, lm)
            };
            let mut max_drift_mm = 0.0f32;
            for i in 0..b * vocab {
                max_drift_mm = max_drift_mm.max((lr[i] - lm[i]).abs());
            }
            let mut mism_mm = 0usize;
            for r in 0..b {
                if tokens_ref[r] != tokens_mm[r] {
                    mism_mm += 1;
                }
            }
            Some((max_drift_mm, mism_mm == 0, mism_mm))
        } else {
            None
        };
        Ok((max_drift, mismatches == 0, mismatches, gemm))
    }
    /// Attention (decode, q_len=1) parity vs CPU oracle: scores=scale·Q·K, softmax, ·V.
    /// Sets up a small KV pool (nkeys keys, contiguous slots), runs the MSL kernel for ONE
    /// head, compares the `head_dim` output. Returns max_rel diff. nkv=1, group=num_heads
    /// (simplest GQA), window=0 (global). past_len = nkeys-1 (decode attends nkeys keys).
    /// S0 DE-RISK: full-GQA attention parity at the REAL gemma-4-12B per-layer geometries.
    /// Unlike `attention_check` (single head, no GQA), this tests multi-head with a GQA group
    /// (kv_head = head / group), a paged KV pool with the per-layer `kv_dim` row stride, and
    /// BOTH 12B profiles (sliding hd256/nkv8/group2, global hd512/nkv1/group16/DPL=2). scale=1.0
    /// (gemma query_pre_attn_scalar). Verifies EVERY head's `head_dim` output vs a CPU oracle —
    /// catches the DPL=2 strided-output bug + GQA mapping. `window`=0 for global causal.
    pub fn attention_geom_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        nh: usize,
        nkv: usize,
        head_dim: usize,
        nkeys: usize,
        window: usize,
    ) -> Result<f32, String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let group = nh / nkv;
        let row_floats = nkv * head_dim; // KV pool row stride (per-layer kv_dim)
        let scale = 1.0f32; // gemma query_pre_attn_scalar
        let q = al(nh * head_dim * 4);
        let kp = al(nkeys * row_floats * 4);
        let vp = al(nkeys * row_floats * 4);
        let ob = al(nh * head_dim * 4);
        let sl = al(nkeys * 4);
        let dims = al(48);
        unsafe {
            let qp = q.contents().as_ptr() as *mut f32;
            for hh in 0..nh {
                for i in 0..head_dim {
                    *qp.add(hh * head_dim + i) = (((hh * 7 + i) % 13) as f32 - 6.0) * 0.05;
                }
            }
            let k = kp.contents().as_ptr() as *mut f32;
            let v = vp.contents().as_ptr() as *mut f32;
            for key in 0..nkeys {
                for j in 0..row_floats {
                    *k.add(key * row_floats + j) = (((key + j) % 17) as f32 - 8.0) * 0.03;
                    *v.add(key * row_floats + j) = (((key * 3 + j) % 11) as f32 - 5.0) * 0.04;
                }
            }
            let s = sl.contents().as_ptr() as *mut u32;
            for key in 0..nkeys {
                *s.add(key) = key as u32;
            }
            let dp = dims.contents().as_ptr() as *mut u32;
            dp.add(0).write(1); // q_len
            dp.add(1).write(nkeys as u32); // ctx
            dp.add(2).write(nh as u32);
            dp.add(3).write(nkv as u32);
            dp.add(4).write(head_dim as u32);
            dp.add(5).write((nkeys - 1) as u32); // past_len
            (dp.add(6) as *mut f32).write(scale);
            dp.add(7).write(group as u32);
            dp.add(8).write(0); // q_start
            dp.add(9).write(window as u32);
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("attn pso")?.pso;
        let bufs = [&q, &kp, &vp, &ob, &sl, &dims];
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .unwrap();
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: nh,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        // CPU oracle — per head, with GQA mapping + window.
        let mut max_rel = 0.0f32;
        unsafe {
            let qp = q.contents().as_ptr() as *const f32;
            let k = kp.contents().as_ptr() as *const f32;
            let v = vp.contents().as_ptr() as *const f32;
            let o = ob.contents().as_ptr() as *const f32;
            let last = nkeys - 1;
            let lo = if window > 0 && last + 1 > window {
                last + 1 - window
            } else {
                0
            };
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * head_dim;
                let mut scs = vec![0.0f32; nkeys];
                let mut mx = f32::MIN;
                for keyi in lo..=last {
                    let mut s = 0.0f32;
                    for i in 0..head_dim {
                        s += *qp.add(hh * head_dim + i) * *k.add(keyi * row_floats + hcol + i);
                    }
                    scs[keyi] = s * scale;
                    mx = mx.max(scs[keyi]);
                }
                let mut den = 0.0f32;
                for keyi in lo..=last {
                    scs[keyi] = (scs[keyi] - mx).exp();
                    den += scs[keyi];
                }
                for dim in 0..head_dim {
                    let mut num = 0.0f32;
                    for keyi in lo..=last {
                        num += scs[keyi] * *v.add(keyi * row_floats + hcol + dim);
                    }
                    let want = num / den;
                    let got = *o.add(hh * head_dim + dim);
                    max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-4));
                }
            }
        }
        Ok(max_rel)
    }
    pub fn attention_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        head_dim: usize,
        nkeys: usize,
    ) -> Result<f32, String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let nh = 1usize;
        let kvh = 1usize;
        let row_floats = kvh * head_dim;
        let qb = al(head_dim * 4);
        let kp = al(nkeys * row_floats * 4);
        let vp = al(nkeys * row_floats * 4);
        let ob = al(head_dim * 4);
        let sl = al(nkeys * 4);
        let dims = al(48);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        unsafe {
            let q = qb.contents().as_ptr() as *mut f32;
            for i in 0..head_dim {
                *q.add(i) = ((i % 13) as f32 - 6.0) * 0.05;
            }
            let k = kp.contents().as_ptr() as *mut f32;
            let v = vp.contents().as_ptr() as *mut f32;
            for key in 0..nkeys {
                for i in 0..head_dim {
                    *k.add(key * row_floats + i) = (((key + i) % 17) as f32 - 8.0) * 0.03;
                    *v.add(key * row_floats + i) = (((key * 3 + i) % 11) as f32 - 5.0) * 0.04;
                }
            }
            let s = sl.contents().as_ptr() as *mut u32;
            for key in 0..nkeys {
                *s.add(key) = key as u32;
            } // contiguous slots
              // Dims: q_len, ctx, num_heads, kv_heads, head_dim, past_len, scale_bits, group,
              //       q_start, window, _, _
            let dp = dims.contents().as_ptr() as *mut u32;
            dp.add(0).write(1);
            dp.add(1).write(nkeys as u32);
            dp.add(2).write(nh as u32);
            dp.add(3).write(kvh as u32);
            dp.add(4).write(head_dim as u32);
            dp.add(5).write((nkeys - 1) as u32); // past_len
            (dp.add(6) as *mut f32).write(scale);
            dp.add(7).write((nh / kvh) as u32); // group
            dp.add(8).write(0); // q_start
            dp.add(9).write(0); // window (global)
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("attn pso")?.pso;
        let bufs = [&qb, &kp, &vp, &ob, &sl, &dims];
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .unwrap();
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: nh,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        // CPU oracle.
        let mut max_rel = 0.0f32;
        unsafe {
            let q = qb.contents().as_ptr() as *const f32;
            let k = kp.contents().as_ptr() as *const f32;
            let v = vp.contents().as_ptr() as *const f32;
            let o = ob.contents().as_ptr() as *const f32;
            let mut scs = vec![0.0f32; nkeys];
            let mut mx = f32::MIN;
            for key in 0..nkeys {
                let mut s = 0.0f32;
                for i in 0..head_dim {
                    s += *q.add(i) * *k.add(key * row_floats + i);
                }
                scs[key] = s * scale;
                mx = mx.max(scs[key]);
            }
            let mut den = 0.0f32;
            for key in 0..nkeys {
                scs[key] = (scs[key] - mx).exp();
                den += scs[key];
            }
            for dim in 0..head_dim {
                let mut num = 0.0f32;
                for key in 0..nkeys {
                    num += scs[key] * *v.add(key * row_floats + dim);
                }
                let want = num / den;
                let got = *o.add(dim);
                max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-4));
            }
        }
        Ok(max_rel)
    }
    /// rmsnorm_add parity: hidden += rmsnorm(src)·w vs CPU oracle. One row of `hidden`.
    /// Returns max_rel of the ADDED contribution (hidden_after - hidden_before).
    pub fn rmsnorm_add_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        hidden: usize,
        eps: f32,
    ) -> Result<f32, String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let (src, w, hid, dims) = (al(hidden * 4), al(hidden * 4), al(hidden * 4), al(16));
        unsafe {
            let sp = src.contents().as_ptr() as *mut f32;
            let wp = w.contents().as_ptr() as *mut f32;
            let hp = hid.contents().as_ptr() as *mut f32;
            for i in 0..hidden {
                *sp.add(i) = ((i % 17) as f32 - 8.0) * 0.04;
                *wp.add(i) = 1.0 + ((i % 11) as f32 - 5.0) * 0.03;
                *hp.add(i) = ((i % 7) as f32 - 3.0) * 0.5; // pre-existing residual
            }
            let dp = dims.contents().as_ptr() as *mut u32;
            *dp = 1;
            *dp.add(1) = hidden as u32;
            *(dp.add(2) as *mut f32) = eps;
            *dp.add(3) = 0;
        }
        // capture hidden-before for the delta oracle
        let before: Vec<f32> = unsafe {
            let p = hid.contents().as_ptr() as *const f32;
            (0..hidden).map(|i| *p.add(i)).collect()
        };
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("rmsnorm_add pso")?.pso;
        let bufs = [&src, &w, &hid, &dims];
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .unwrap();
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        let mut max_rel = 0.0f32;
        unsafe {
            let sp = src.contents().as_ptr() as *const f32;
            let wp = w.contents().as_ptr() as *const f32;
            let hp = hid.contents().as_ptr() as *const f32;
            let mut ss = 0.0f32;
            for i in 0..hidden {
                ss += *sp.add(i) * *sp.add(i);
            }
            let inv = 1.0 / (ss / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                let want = before[i] + *sp.add(i) * inv * *wp.add(i);
                let got = *hp.add(i);
                max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-3));
            }
        }
        Ok(max_rel)
    }
    /// GeGLU parity: out = gelu_tanh(gate)·up vs CPU oracle. Returns max_rel diff.
    pub fn geglu_check(&mut self, ctx: &GpuContext, key: &str, n: usize) -> Result<f32, String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let (g, u, o, dims) = (al(n * 4), al(n * 4), al(n * 4), al(16));
        unsafe {
            let gp = g.contents().as_ptr() as *mut f32;
            let up = u.contents().as_ptr() as *mut f32;
            for i in 0..n {
                *gp.add(i) = ((i % 29) as f32 - 14.0) * 0.3; // span incl. large (clamp path)
                *up.add(i) = ((i % 19) as f32 - 9.0) * 0.1;
            }
            let dp = dims.contents().as_ptr() as *mut u32;
            *dp = n as u32;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("geglu pso")?.pso;
        let bufs = [&g, &u, &o, &dims];
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .unwrap();
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        let mut max_rel = 0.0f32;
        unsafe {
            let gp = g.contents().as_ptr() as *const f32;
            let up = u.contents().as_ptr() as *const f32;
            let op = o.contents().as_ptr() as *const f32;
            const C: f32 = 0.797_884_6;
            for i in 0..n {
                let x = *gp.add(i);
                let inner = (C * (x + 0.044715 * x * x * x)).clamp(-15.0, 15.0);
                let want = 0.5 * x * (1.0 + inner.tanh()) * *up.add(i);
                let got = *op.add(i);
                max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-4));
            }
        }
        Ok(max_rel)
    }
    /// RMSNorm parity + timing: run the hand-MSL `rmsnorm` kernel (key compiled) on
    /// deterministic input vs a CPU oracle (y = x/sqrt(mean(x²)+eps)·w). Returns
    /// `(max_rel_diff, ref_norm, gpu_us)`. `tg` = threads/threadgroup (e.g. 256).
    pub fn rmsnorm_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        hidden: usize,
        eps: f32,
        tg: usize,
        iters: usize,
    ) -> Result<(f32, f32, f64), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let alloc = |b: usize| {
            dev.newBufferWithLength_options(b.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let x = alloc(hidden * 4);
        let w = alloc(hidden * 4);
        let yb = alloc(hidden * 4);
        let dims = alloc(16);
        unsafe {
            let xp = x.contents().as_ptr() as *mut f32;
            let wp = w.contents().as_ptr() as *mut f32;
            for i in 0..hidden {
                *xp.add(i) = ((i % 23) as f32 - 11.0) * 0.05;
                *wp.add(i) = 1.0 + ((i % 13) as f32 - 6.0) * 0.02; // gemma-ish (1+w)-folded
            }
            let dp = dims.contents().as_ptr() as *mut u32;
            *dp = 1;
            *dp.add(1) = hidden as u32;
            *(dp.add(2) as *mut f32) = eps;
            *dp.add(3) = 0;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("rmsnorm pso")?.pso;
        let bufs = [&x, &w, &yb, &dims];
        let run = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                    enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: tg,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6 // us
        };
        let _ = run();
        let mut us = f64::INFINITY;
        for _ in 0..iters {
            us = us.min(run());
        }
        // CPU oracle.
        let (mut max_rel, mut ref_norm) = (0.0f32, 0.0f32);
        unsafe {
            let xp = x.contents().as_ptr() as *const f32;
            let wp = w.contents().as_ptr() as *const f32;
            let yp = yb.contents().as_ptr() as *const f32;
            let mut ss = 0.0f32;
            for i in 0..hidden {
                ss += *xp.add(i) * *xp.add(i);
            }
            let inv = 1.0 / (ss / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                let want = *xp.add(i) * inv * *wp.add(i);
                let got = *yp.add(i);
                ref_norm += want * want;
                max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-4));
            }
        }
        Ok((max_rel, ref_norm.sqrt(), us))
    }
    /// Q3K GEMV parity: quantize a random weight with the canonical Q3KMatrix, run the MSL
    /// gemv_q3k, compare to the CPU dequant·activation oracle. Returns max_rel diff.
    pub fn gemv_q3k_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        k: usize,
        n: usize,
    ) -> Result<f32, String> {
        // canonical Q3K quantization of a deterministic weight [n, k].
        let mut wf = vec![0f32; n * k];
        for (i, x) in wf.iter_mut().enumerate() {
            *x = (((i * 2654435761usize) % 97) as f32 - 48.0) * 0.02;
        }
        let q = arf_core::tensor::Q3KMatrix::from_f32(&wf, n, k);
        let deq = q.to_f32(); // canonical dequant [n,k] — the oracle weight
        let af: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();
        let pack_u8 = |d: &[u8]| -> Vec<u32> {
            d.chunks_exact(4)
                .map(|q| {
                    (q[0] as u32)
                        | ((q[1] as u32) << 8)
                        | ((q[2] as u32) << 16)
                        | ((q[3] as u32) << 24)
                })
                .collect()
        };
        let pack_i8 = |d: &[i8]| -> Vec<u32> {
            d.chunks_exact(4)
                .map(|q| {
                    (q[0] as u8 as u32)
                        | ((q[1] as u8 as u32) << 8)
                        | ((q[2] as u8 as u32) << 16)
                        | ((q[3] as u8 as u32) << 24)
                })
                .collect()
        };
        let mut dd = Vec::with_capacity(q.d().len() * 2);
        for (a, b) in q.d().iter().zip(q.dmin()) {
            dd.push(*a);
            dd.push(*b);
        }
        let sc = pack_u8(q.scales());
        let mn = pack_i8(q.mins());
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let mk = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .unwrap()
        };
        let up = |buf: &Retained<ProtocolObject<dyn MTLBuffer>>, src: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                buf.contents().as_ptr() as *mut u8,
                src.len(),
            );
        };
        let a_b = mk(k * 4);
        up(&a_b, bytemuck::cast_slice(&af));
        let ql_b = mk(q.ql().len() * 4);
        up(&ql_b, bytemuck::cast_slice(q.ql()));
        let qh_b = mk(q.qh().len() * 4);
        up(&qh_b, bytemuck::cast_slice(q.qh()));
        let c_b = mk(n * 4);
        let dims_b = mk(16);
        unsafe {
            let p = dims_b.contents().as_ptr() as *mut u32;
            *p = 1;
            *p.add(1) = k as u32;
            *p.add(2) = n as u32;
        }
        let sc_b = mk(sc.len() * 4);
        up(&sc_b, bytemuck::cast_slice(&sc));
        let mn_b = mk(mn.len() * 4);
        up(&mn_b, bytemuck::cast_slice(&mn));
        let dd_b = mk(dd.len() * 4);
        up(&dd_b, bytemuck::cast_slice(&dd));
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("q3k pso")?.pso;
        let bufs = [&a_b, &ql_b, &qh_b, &c_b, &dims_b, &sc_b, &mn_b, &dd_b];
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .unwrap();
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(2),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        // Timing: dispatch 200× back-to-back, report kernel GB/s (Q3K bytes ≈ n*3k/8 codes + scales).
        {
            let q3k_bytes = (n * (k / 4 + k / 8 + k / 32 + k / 32)) as f64; // ql+qh+scales+mins
            let tcmd = self.queue.commandBuffer().unwrap();
            let tenc = tcmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                tenc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                    tenc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    tenc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                for _ in 0..200 {
                    tenc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: n.div_ceil(2),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                tenc.endEncoding();
            }
            tcmd.commit();
            tcmd.waitUntilCompleted();
            let ms = (tcmd.GPUEndTime() - tcmd.GPUStartTime()) * 1000.0;
            let gb_s = q3k_bytes * 200.0 / (ms / 1000.0) / 1e9;
            eprintln!(
                "  gemv_q3k k={k} n={n}: {:.2} ms/200, {:.0} GB/s (roofline ~410)",
                ms, gb_s
            );
        }
        // CPU oracle: c[col] = Σ_i deq[col*k+i] * a[i].
        let mut max_rel = 0.0f32;
        unsafe {
            let cp = c_b.contents().as_ptr() as *const f32;
            for col in 0..n {
                let mut want = 0.0f32;
                for i in 0..k {
                    want += deq[col * k + i] * af[i];
                }
                let got = *cp.add(col);
                max_rel = max_rel.max((want - got).abs() / want.abs().max(1e-2));
            }
        }
        Ok(max_rel)
    }
    pub fn gemv_q4ks_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        k: usize,
        n: usize,
        roofline_gb_s: f64,
        iters: usize,
    ) -> Result<(f32, f32, f64, f64), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32;
        let supers = k / 256;
        let alloc = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .expect("alloc")
        };
        let a_buf = alloc(k * 4);
        let codes_buf = alloc(n * subs * 16);
        let c_buf = alloc(n * 4);
        let dims_buf = alloc(16);
        let scales_buf = alloc((n * subs).div_ceil(4) * 4); // u8, 4/word
        let mins_buf = alloc((n * subs).div_ceil(4) * 4); // i8, 4/word
        let dd_buf = alloc(n * supers * 4); // ddh: packed half2 {d,dmin}/super
        unsafe {
            let ap = a_buf.contents().as_ptr() as *mut f32;
            for i in 0..k {
                *ap.add(i) = ((i % 13) as f32 - 6.0) * 0.04;
            }
            let cp = codes_buf.contents().as_ptr() as *mut u32;
            for i in 0..(n * subs * 4) {
                *cp.add(i) = (i.wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
            }
            // u8 scales 1..255 packed; i8 mins packed.
            let sp = scales_buf.contents().as_ptr() as *mut u8;
            let mp = mins_buf.contents().as_ptr() as *mut i8;
            for i in 0..(n * subs) {
                *sp.add(i) = ((i % 31) + 1) as u8;
                *mp.add(i) = (((i % 17) as i32) - 8) as i8;
            }
            let dp = dd_buf.contents().as_ptr() as *mut u32;
            for i in 0..(n * supers) {
                let dv = 0.02 + (i % 5) as f32 * 0.003; // d
                let dmv = -0.01 - (i % 3) as f32 * 0.002; // dmin
                *dp.add(i) = (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
            }
            let d = dims_buf.contents().as_ptr() as *mut u32;
            *d = 1;
            *d.add(1) = k as u32;
            *d.add(2) = n as u32;
            *d.add(3) = 0;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("q4ks pso")?.pso;
        let groups = MTLSize {
            width: n.div_ceil(2),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let bufs = [
            &a_buf,
            &codes_buf,
            &c_buf,
            &dims_buf,
            &scales_buf,
            &mins_buf,
            &dd_buf,
        ];
        let run = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                    enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0
        };
        let _ = run();
        let mut ms = f64::INFINITY;
        for _ in 0..iters {
            ms = ms.min(run());
        }
        // CPU oracle for max-rel + ref norm.
        let (mut max_rel, mut ref_norm) = (0.0f32, 0.0f32);
        unsafe {
            let a0 = a_buf.contents().as_ptr() as *const f32;
            let cd = codes_buf.contents().as_ptr() as *const u32;
            let sp = scales_buf.contents().as_ptr() as *const u8;
            let mp = mins_buf.contents().as_ptr() as *const i8;
            let dp = dd_buf.contents().as_ptr() as *const u32;
            let pb = c_buf.contents().as_ptr() as *const f32;
            for col in 0..n {
                let mut acc = 0.0f32;
                for b in 0..subs {
                    let sblk = b / 8;
                    let pair = *dp.add(col * supers + sblk);
                    let dv = f16_bits_to_f32(pair as u16);
                    let dmv = f16_bits_to_f32((pair >> 16) as u16);
                    let su8 = *sp.add(col * subs + b) as f32;
                    let mi8 = *mp.add(col * subs + b) as f32;
                    let s = dv * su8;
                    let lo = dmv * mi8;
                    let abase = 32 * b;
                    let (mut cs, mut asum) = (0.0f32, 0.0f32);
                    for w in 0..4 {
                        let word = *cd.add((col * subs + b) * 4 + w);
                        for e in 0..8u32 {
                            let av = *a0.add(abase + w * 8 + e as usize);
                            cs += av * ((word >> (4 * e)) & 0xF) as f32;
                            asum += av;
                        }
                    }
                    acc += s * cs + lo * asum;
                }
                let vb = *pb.add(col);
                ref_norm += acc * acc;
                max_rel = max_rel.max((acc - vb).abs() / acc.abs().max(1e-3));
            }
        }
        let weight_bytes = (n * k / 2) as f64 + (n * subs) as f64 * 2.0 + (n * supers * 8) as f64;
        let gb_s = weight_bytes / (ms / 1000.0) / 1e9;
        Ok((max_rel, ref_norm.sqrt(), gb_s, 100.0 * gb_s / roofline_gb_s))
    }
    pub fn gemv_q4ks_id_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        k: usize,
        n: usize,
        top_k: usize,
        num_experts: usize,
        roofline_gb_s: f64,
        iters: usize,
        tg_threads: usize, // 32 = 1 simdgroup (plain compile); 64 = the NSG=2 fc3 variant
    ) -> Result<(f32, f64, f64), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32;
        let supers = k / 256;
        let alloc = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .expect("alloc")
        };
        let ecols = num_experts * n; // total weight columns across all experts
        let a_buf = alloc(k * 4);
        let codes_buf = alloc(ecols * subs * 16);
        let c_buf = alloc(top_k * n * 4);
        let dims_buf = alloc(16);
        let scales_buf = alloc((ecols * subs).div_ceil(4) * 4);
        let mins_buf = alloc((ecols * subs).div_ceil(4) * 4);
        let dd_buf = alloc(ecols * supers * 4); // ddh: packed half2 {d,dmin}/super
        let ids_buf = alloc(top_k * 4);
        unsafe {
            let ap = a_buf.contents().as_ptr() as *mut f32;
            for i in 0..k {
                *ap.add(i) = ((i % 13) as f32 - 6.0) * 0.04;
            }
            let cp = codes_buf.contents().as_ptr() as *mut u32;
            for i in 0..(ecols * subs * 4) {
                *cp.add(i) = (i.wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
            }
            let sp = scales_buf.contents().as_ptr() as *mut u8;
            let mp = mins_buf.contents().as_ptr() as *mut i8;
            for i in 0..(ecols * subs) {
                *sp.add(i) = ((i % 31) + 1) as u8;
                *mp.add(i) = (((i % 17) as i32) - 8) as i8;
            }
            let dp = dd_buf.contents().as_ptr() as *mut u32;
            for i in 0..(ecols * supers) {
                let dv = 0.02 + (i % 5) as f32 * 0.003;
                let dmv = -0.01 - (i % 3) as f32 * 0.002;
                *dp.add(i) = (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
            }
            // ids: slot s routes to expert (s*37+1) % num_experts (deterministic spread).
            let ip = ids_buf.contents().as_ptr() as *mut u32;
            for s in 0..top_k {
                *ip.add(s) = ((s * 37 + 1) % num_experts) as u32;
            }
            let d = dims_buf.contents().as_ptr() as *mut u32;
            *d = 1; // m
            *d.add(1) = k as u32;
            *d.add(2) = n as u32;
            *d.add(3) = top_k as u32;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("id pso")?.pso;
        // grid = (ceil(n/2), 1, top_k); tg_threads-lane threadgroup. Z = routed slot.
        let groups = MTLSize {
            width: n.div_ceil(2),
            height: 1,
            depth: top_k,
        };
        let tg = MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        };
        let bufs = [
            &a_buf,
            &codes_buf,
            &c_buf,
            &dims_buf,
            &scales_buf,
            &ids_buf,
            &mins_buf,
            &dd_buf,
        ];
        let run = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                    enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0
        };
        let _ = run();
        let mut ms = f64::INFINITY;
        for _ in 0..iters {
            ms = ms.min(run());
        }
        // CPU oracle: for each slot, expert = ids[slot]; oracle c[slot*n+col] vs dense math at
        // weight column (expert*n + col).
        let mut max_rel = 0.0f32;
        unsafe {
            let a0 = a_buf.contents().as_ptr() as *const f32;
            let cd = codes_buf.contents().as_ptr() as *const u32;
            let sp = scales_buf.contents().as_ptr() as *const u8;
            let mp = mins_buf.contents().as_ptr() as *const i8;
            let dp = dd_buf.contents().as_ptr() as *const u32;
            let ip = ids_buf.contents().as_ptr() as *const u32;
            let pb = c_buf.contents().as_ptr() as *const f32;
            for slot in 0..top_k {
                let expert = *ip.add(slot) as usize;
                for col in 0..n {
                    let wcol = expert * n + col;
                    let mut acc = 0.0f32;
                    for b in 0..subs {
                        let sblk = b / 8;
                        let pair = *dp.add(wcol * supers + sblk);
                        let dv = f16_bits_to_f32(pair as u16);
                        let dmv = f16_bits_to_f32((pair >> 16) as u16);
                        let su8 = *sp.add(wcol * subs + b) as f32;
                        let mi8 = *mp.add(wcol * subs + b) as f32;
                        let s = dv * su8;
                        let lo = dmv * mi8;
                        let abase = 32 * b;
                        let (mut cs, mut asum) = (0.0f32, 0.0f32);
                        for w in 0..4 {
                            let word = *cd.add((wcol * subs + b) * 4 + w);
                            for e in 0..8u32 {
                                let av = *a0.add(abase + w * 8 + e as usize);
                                cs += av * ((word >> (4 * e)) & 0xF) as f32;
                                asum += av;
                            }
                        }
                        acc += s * cs + lo * asum;
                    }
                    let vb = *pb.add(slot * n + col);
                    max_rel = max_rel.max((acc - vb).abs() / acc.abs().max(1e-3));
                }
            }
        }
        // bandwidth = only the top_k ROUTED experts' weights (what decode actually streams).
        let weight_bytes =
            top_k as f64 * ((n * k / 2) as f64 + (n * subs) as f64 * 2.0 + (n * supers * 8) as f64);
        let gb_s = weight_bytes / (ms / 1000.0) / 1e9;
        Ok((max_rel, gb_s, 100.0 * gb_s / roofline_gb_s))
    }
    pub fn gemv_q4ks_id_down_check(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        k: usize, // mi (inter)
        n: usize, // h (hidden)
        top_k: usize,
        num_experts: usize,
        roofline_gb_s: f64,
        iters: usize,
        tg_threads: usize, // 32 = 1 simdgroup (plain compile); 64 = the NSG=2 fc3 variant
    ) -> Result<(f32, f64, f64), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32;
        let supers = k / 256;
        let alloc = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .expect("alloc")
        };
        let ecols = num_experts * n;
        let a_buf = alloc(top_k * k * 4); // silu_all [top_k*mi]
        let codes_buf = alloc(ecols * subs * 16);
        let c_buf = alloc(n * 4);
        let dims_buf = alloc(16);
        let scales_buf = alloc((ecols * subs).div_ceil(4) * 4);
        let mins_buf = alloc((ecols * subs).div_ceil(4) * 4);
        let dd_buf = alloc(ecols * supers * 4); // ddh: packed half2 {d,dmin}/super
        let ids_buf = alloc(top_k * 4);
        let wts_buf = alloc(top_k * 4);
        unsafe {
            let ap = a_buf.contents().as_ptr() as *mut f32;
            for i in 0..(top_k * k) {
                *ap.add(i) = ((i % 13) as f32 - 6.0) * 0.04;
            }
            let cp = codes_buf.contents().as_ptr() as *mut u32;
            for i in 0..(ecols * subs * 4) {
                *cp.add(i) = (i.wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
            }
            let sp = scales_buf.contents().as_ptr() as *mut u8;
            let mp = mins_buf.contents().as_ptr() as *mut i8;
            for i in 0..(ecols * subs) {
                *sp.add(i) = ((i % 31) + 1) as u8;
                *mp.add(i) = (((i % 17) as i32) - 8) as i8;
            }
            let dp = dd_buf.contents().as_ptr() as *mut u32;
            for i in 0..(ecols * supers) {
                let dv = 0.02 + (i % 5) as f32 * 0.003;
                let dmv = -0.01 - (i % 3) as f32 * 0.002;
                *dp.add(i) = (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
            }
            let ip = ids_buf.contents().as_ptr() as *mut u32;
            let wp = wts_buf.contents().as_ptr() as *mut f32;
            for s in 0..top_k {
                *ip.add(s) = ((s * 37 + 1) % num_experts) as u32;
                *wp.add(s) = 0.1 + s as f32 * 0.05;
            }
            let d = dims_buf.contents().as_ptr() as *mut u32;
            *d = 1;
            *d.add(1) = k as u32;
            *d.add(2) = n as u32;
            *d.add(3) = top_k as u32;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).ok_or("id_down pso")?.pso;
        let groups = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        };
        let bufs = [
            &a_buf,
            &codes_buf,
            &c_buf,
            &dims_buf,
            &scales_buf,
            &ids_buf,
            &wts_buf,
            &mins_buf,
            &dd_buf,
        ];
        let run = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                    enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0
        };
        let _ = run();
        let mut ms = f64::INFINITY;
        for _ in 0..iters {
            ms = ms.min(run());
        }
        let mut max_rel = 0.0f32;
        unsafe {
            let a0 = a_buf.contents().as_ptr() as *const f32;
            let cd = codes_buf.contents().as_ptr() as *const u32;
            let sp = scales_buf.contents().as_ptr() as *const u8;
            let mp = mins_buf.contents().as_ptr() as *const i8;
            let dp = dd_buf.contents().as_ptr() as *const u32;
            let ip = ids_buf.contents().as_ptr() as *const u32;
            let wp = wts_buf.contents().as_ptr() as *const f32;
            let pb = c_buf.contents().as_ptr() as *const f32;
            for h_j in 0..n {
                let mut acc = 0.0f32;
                for slot in 0..top_k {
                    let expert = *ip.add(slot) as usize;
                    let rw = *wp.add(slot);
                    let wcol = expert * n + h_j;
                    let sbase = slot * k; // silu segment start (in f32 elements)
                    let mut slot_acc = 0.0f32;
                    for b in 0..subs {
                        let sblk = b / 8;
                        let pair = *dp.add(wcol * supers + sblk);
                        let dv = f16_bits_to_f32(pair as u16);
                        let dmv = f16_bits_to_f32((pair >> 16) as u16);
                        let su8 = *sp.add(wcol * subs + b) as f32;
                        let mi8 = *mp.add(wcol * subs + b) as f32;
                        let s = dv * su8;
                        let lo = dmv * mi8;
                        let abase = sbase + 32 * b;
                        let (mut cs, mut asum) = (0.0f32, 0.0f32);
                        for w in 0..4 {
                            let word = *cd.add((wcol * subs + b) * 4 + w);
                            for e in 0..8u32 {
                                let av = *a0.add(abase + w * 8 + e as usize);
                                cs += av * ((word >> (4 * e)) & 0xF) as f32;
                                asum += av;
                            }
                        }
                        slot_acc += s * cs + lo * asum;
                    }
                    acc += rw * slot_acc;
                }
                let vb = *pb.add(h_j);
                max_rel = max_rel.max((acc - vb).abs() / acc.abs().max(1e-3));
            }
        }
        let weight_bytes =
            top_k as f64 * ((n * k / 2) as f64 + (n * subs) as f64 * 2.0 + (n * supers * 8) as f64);
        let gb_s = weight_bytes / (ms / 1000.0) / 1e9;
        Ok((max_rel, gb_s, 100.0 * gb_s / roofline_gb_s))
    }
    pub fn gemv_dispatch(
        &self,
        key: &str,
        a: &ProtocolObject<dyn MTLBuffer>,
        codes: &ProtocolObject<dyn MTLBuffer>,
        c: &ProtocolObject<dyn MTLBuffer>,
        dims: &ProtocolObject<dyn MTLBuffer>,
        scales: &ProtocolObject<dyn MTLBuffer>,
        mins: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
    ) -> Result<(), String> {
        let pso = &self.pipelines.get(key).ok_or("gemv pso missing")?.pso;
        let bufs: [&ProtocolObject<dyn MTLBuffer>; 6] = [a, codes, c, dims, scales, mins];
        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("enc")?;
        let groups = MTLSize {
            width: n.div_ceil(2),
            height: 1,
            depth: 1,
        }; // 2 cols/sg
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in bufs.iter().enumerate() {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(*b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(*b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }
    pub fn gemv_parity(
        &mut self,
        ctx: &GpuContext,
        key_a: &str,
        wg_a: usize,
        cols_a: usize,
        key_b: &str,
        wg_b: usize,
        cols_b: usize,
        k: usize,
        n: usize,
    ) -> Result<(f32, f32, f32), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32;
        let alloc = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .expect("alloc")
        };
        let a_buf = alloc(k * 4);
        let codes_buf = alloc(n * subs * 16);
        let ca_buf = alloc(n * 4);
        let cb_buf = alloc(n * 4);
        let dims_buf = alloc(16);
        let scales_buf = alloc(n * subs * 4);
        let mins_buf = alloc(n * subs * 4);
        // Deterministic fill: activations a smooth wave; codes pseudo-random nibbles;
        // scales/mins small bf16. Enough structure that wrong indexing diverges sharply.
        unsafe {
            let ap = a_buf.contents().as_ptr() as *mut f32;
            for i in 0..k {
                *ap.add(i) = ((i % 17) as f32 - 8.0) * 0.03;
            }
            let cp = codes_buf.contents().as_ptr() as *mut u32;
            for i in 0..(n * subs * 4) {
                *cp.add(i) = (i.wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
            }
            // bf16 of ~0.05 = 0x3D4C; of ~-0.02 = 0xBCA4 (low 16 used).
            let sp = scales_buf.contents().as_ptr() as *mut u32;
            let mp = mins_buf.contents().as_ptr() as *mut u32;
            for i in 0..(n * subs) {
                *sp.add(i) = 0x3D4C + ((i as u32 % 7) << 4);
                *mp.add(i) = 0xBCA4u32 ^ ((i as u32 % 5) << 3);
            }
            let dp = dims_buf.contents().as_ptr() as *mut u32;
            *dp = 1;
            *dp.add(1) = k as u32;
            *dp.add(2) = n as u32;
            *dp.add(3) = 0;
        }
        drop(guard);
        // Run a kernel into a given output buffer.
        let run =
            |key: &str, wg: usize, cols: usize, out: &Retained<ProtocolObject<dyn MTLBuffer>>| {
                let pso = &self.pipelines.get(key).expect("compiled").pso;
                let bufs = [&a_buf, &codes_buf, out, &dims_buf, &scales_buf, &mins_buf];
                let groups = MTLSize {
                    width: n.div_ceil(cols),
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: wg,
                    height: 1,
                    depth: 1,
                };
                let cmd = self.queue.commandBuffer().unwrap();
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                    .unwrap();
                unsafe {
                    enc.setComputePipelineState(pso);
                    for (i, b) in bufs.iter().enumerate() {
                        let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                        enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                        enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                    enc.endEncoding();
                }
                cmd.commit();
                cmd.waitUntilCompleted();
            };
        // Run only kernel B (the kernel under test, e.g. the hand-MSL); the ORACLE is a CPU
        // recompute from the same inputs (naga-WGSL through a raw encoder mis-binds buffers
        // — naga remaps @binding→[[buffer]] — so the GPU WGSL isn't a usable raw oracle).
        // CPU does the EXACT Q4_K-lite math the WGSL kernel does.
        let _ = (key_a, wg_a, cols_a, &ca_buf);
        run(key_b, wg_b, cols_b, &cb_buf);
        let (mut max_abs, mut max_rel, mut ref_norm) = (0.0f32, 0.0f32, 0.0f32);
        unsafe {
            let a0 = a_buf.contents().as_ptr() as *const f32;
            let cd = codes_buf.contents().as_ptr() as *const u32; // 4 u32 per sub-block
            let sc = scales_buf.contents().as_ptr() as *const u32;
            let mn = mins_buf.contents().as_ptr() as *const u32;
            let pb = cb_buf.contents().as_ptr() as *const f32;
            let bf16 = |w: u32| f32::from_bits((w & 0xFFFF) << 16);
            let nib = |word: u32, i: u32| ((word >> (4 * i)) & 0xF) as f32;
            if std::env::var("ARF_PARITY_DUMP").is_ok() {
                eprintln!(
                    "  [dump] a[0..4]={:?}",
                    (0..4).map(|i| *a0.add(i)).collect::<Vec<_>>()
                );
                eprintln!(
                    "  [dump] B(msl)[0..6]={:?}",
                    (0..6).map(|i| *pb.add(i)).collect::<Vec<_>>()
                );
            }
            for col in 0..n {
                // CPU oracle: same as matmul_vec_q4k.wgsl — sum over sub-blocks of
                // s·dot(a_sub, nibbles) + lo·sum(a_sub).
                let mut acc = 0.0f32;
                for b in 0..subs {
                    let idx = col * subs + b;
                    let s = bf16(*sc.add(idx));
                    let lo = bf16(*mn.add(idx));
                    let abase = 32 * b; // 32 activations per sub-block
                    let mut code_sum = 0.0f32;
                    let mut a_sum = 0.0f32;
                    // 4 u32 words × 8 nibbles = 32 codes, paired with 32 activations.
                    for w in 0..4 {
                        let word = *cd.add(idx * 4 + w);
                        for e in 0..8u32 {
                            let av = *a0.add(abase + (w * 8) + e as usize);
                            code_sum += av * nib(word, e);
                            a_sum += av;
                        }
                    }
                    acc += s * code_sum + lo * a_sum;
                }
                let vb = *pb.add(col);
                ref_norm += acc * acc;
                let abs = (acc - vb).abs();
                max_abs = max_abs.max(abs);
                max_rel = max_rel.max(abs / acc.abs().max(1e-3));
                if col < 4 && std::env::var("ARF_PARITY_DUMP").is_ok() {
                    eprintln!("  [dump] col{col}: cpu={acc:.4} msl={vb:.4}");
                }
            }
        }
        Ok((max_abs, max_rel, ref_norm.sqrt()))
    }
    /// Benchmark a Q4_K-lite decode GEMV kernel (given as WGSL, the SAME source the wgpu
    /// path uses) at a real gemma shape: out`n` = a`k` · W_q4kᵀ. Allocates the Q4_K weight
    /// layout (codes 4-bit, scales/mins bf16 per 32-element sub-block), runs `wg_per_col`
    /// workgroups grid, and returns (gpu_ms, achieved_gb_s, pct_of_roofline). The dominant
    /// DRAM traffic is the weight read: n·k·0.5 bytes (codes) + n·(k/32)·4 (scales+mins).
    /// `roofline_gb_s` is the device peak (~410 for M4 Max). `wg_per_col_div` divides the
    /// per-column grid (1 = one workgroup per output column, the matmul_vec_q4k_sg pattern).
    pub fn gemv_bench(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        wgsl: &str,
        k: usize,
        n: usize,
        wg_size: usize,
        cols_per_wg: usize,
        roofline_gb_s: f64,
        iters: usize,
    ) -> Result<(f64, f64, f64), String> {
        self.compile(ctx, key, wgsl)?;
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32; // sub-blocks per column
        let alloc = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .expect("alloc")
        };
        // bindings (matmul_vec_q4k_sg order): a[k] vec4, codes[n*subs] vec4<u32>, c[n],
        // dims uniform, scales[n*subs] u32, mins[n*subs] u32.
        let a_buf = alloc(k * 4);
        let codes_buf = alloc(n * subs * 16); // vec4<u32> = 16B per sub-block (32 nibbles)
        let c_buf = alloc(n * 4);
        let dims_buf = alloc(16);
        let scales_buf = alloc(n * subs * 4);
        let mins_buf = alloc(n * subs * 4);
        // dims = {m:1, k, n, _pad}
        unsafe {
            let p = dims_buf.contents().as_ptr() as *mut u32;
            *p = 1;
            *p.add(1) = k as u32;
            *p.add(2) = n as u32;
            *p.add(3) = 0;
        }
        drop(guard);
        let pso = &self.pipelines.get(key).unwrap().pso;
        let groups = MTLSize {
            width: n.div_ceil(cols_per_wg),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: wg_size,
            height: 1,
            depth: 1,
        };
        let bufs = [
            &a_buf,
            &codes_buf,
            &c_buf,
            &dims_buf,
            &scales_buf,
            &mins_buf,
        ];

        let run = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in bufs.iter().enumerate() {
                    let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***b);
                    enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                    enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0
        };
        let _ = run();
        let mut ms = f64::INFINITY;
        for _ in 0..iters {
            ms = ms.min(run());
        }
        // weight bytes read: codes (n·k/2) + scales (n·subs·2 used of 4) + mins (same).
        let weight_bytes = (n * k / 2) as f64 + (n * subs * 4) as f64; // codes + scales+mins words
        let gb_s = weight_bytes / (ms / 1000.0) / 1e9;
        Ok((ms, gb_s, 100.0 * gb_s / roofline_gb_s))
    }
    pub fn mm_q4ks_parity(
        &mut self,
        ctx: &GpuContext,
        b: usize,
        k: usize,
        n: usize,
        iters: usize,
    ) -> Result<(f32, f32, f64, f64), String> {
        assert!(k.is_multiple_of(256), "Q4_K_S k must be %256==0");
        // compile both kernels (idempotent). ARF_MM_V2=1 selects the v2 GEMM
        // (matmul_mm_q4ks_v2, KC=64 f32-tile variant) for the tiled side instead of v1 — A/B.
        self.compile_msl(
            ctx,
            "gemv_q4ks_batch",
            include_str!("../../shaders/metal/matmul_vec_q4ks_batch_msl.metal"),
            "gemv_q4ks_batch",
        )?;
        let use_v2 = std::env::var("ARF_MM_V2").ok().as_deref() == Some("1");
        let (mm_key, mm_smem) = if use_v2 {
            self.compile_msl(
                ctx,
                "matmul_mm_q4ks_v2",
                include_str!("../../shaders/metal/matmul_mm_q4ks_v2.metal"),
                "matmul_mm_q4ks_v2",
            )?;
            ("matmul_mm_q4ks_v2", 24576usize)
        } else {
            // ARF_DENSE_MM_HOIST: A/B the block-dequant-hoisted dense GEMM vs the per-element one.
            // The hoist adds scol[64]+lcol[64] = 512 B threadgroup mem (well under 12288). When set,
            // this selftest validates the HOISTED kernel against the GEMV oracle (the parity gate).
            let dense_base = include_str!("../../shaders/metal/matmul_mm_q4ks.metal");
            let dense_src: std::borrow::Cow<str> =
                if std::env::var_os("ARF_DENSE_MM_HOIST").is_some() {
                    std::borrow::Cow::Owned(format!("#define DENSE_MM_HOIST 1\n{dense_base}"))
                } else {
                    std::borrow::Cow::Borrowed(dense_base)
                };
            self.compile_msl(ctx, "matmul_mm_q4ks", &dense_src, "matmul_mm_q4ks")?;
            ("matmul_mm_q4ks", 12288usize)
        };

        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let subs = k / 32;
        let supers = k / 256;
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .expect("alloc")
        };

        // synthetic Q4_K_S weight [n,k] — SAME deterministic packing as ffn_block_selftest.
        let codes = al(n * subs * 16);
        let scales = al((n * subs).div_ceil(4) * 4);
        let mins = al((n * subs).div_ceil(4) * 4);
        let dd = al(n * supers * 4); // ddh: packed half2 {d,dmin}/super
        unsafe {
            let cp = codes.contents().as_ptr() as *mut u32;
            for i in 0..(n * subs * 4) {
                *cp.add(i) = (i.wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
            }
            let sp = scales.contents().as_ptr() as *mut u8;
            let mp = mins.contents().as_ptr() as *mut i8;
            for i in 0..(n * subs) {
                *sp.add(i) = ((i % 31) + 1) as u8;
                *mp.add(i) = (((i % 17) as i32) - 8) as i8;
            }
            let scl = 1.0 / (k as f32).sqrt();
            let dp = dd.contents().as_ptr() as *mut u32;
            for i in 0..(n * supers) {
                let dv = (0.02 + (i % 5) as f32 * 0.003) * scl;
                let dmv = (-0.01 - (i % 3) as f32 * 0.002) * scl;
                *dp.add(i) = (f32_to_f16_bits(dv) as u32) | ((f32_to_f16_bits(dmv) as u32) << 16);
            }
        }
        // B-row activation [B,k] row-major (k/4 float4 per row).
        let a = al(b * k * 4);
        unsafe {
            let ap = a.contents().as_ptr() as *mut f32;
            for r in 0..b {
                for i in 0..k {
                    *ap.add(r * k + i) = (((r * 7 + i) % 19) as f32 - 9.0) * 0.04;
                }
            }
        }
        // outputs [B,n].
        let c_gemv = al(b * n * 4);
        let c_mm = al(b * n * 4);
        // dims {m=B, k, n, _}.
        let dims = al(16);
        unsafe {
            let p = dims.contents().as_ptr() as *mut u32;
            *p = b as u32;
            *p.add(1) = k as u32;
            *p.add(2) = n as u32;
            *p.add(3) = 0;
        }
        drop(guard);

        let pso_gemv = self
            .pipelines
            .get("gemv_q4ks_batch")
            .ok_or("gemv pso")?
            .pso
            .clone();
        let pso_mm = self.pipelines.get(mm_key).ok_or("mm pso")?.pso.clone();

        // per-row GEMV: grid (n,1,1), tg 32; bindings [a, codes, c, dims, scales, mins, dd].
        let run_gemv = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .unwrap();
            let bufs: [&ProtocolObject<dyn MTLBuffer>; 7] =
                [&a, &codes, &c_gemv, &dims, &scales, &mins, &dd];
            unsafe {
                enc.setComputePipelineState(&pso_gemv);
                for (i, bf) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0
        };
        // tiled GEMM: grid (ceil(B/32), ceil(n/64), 1), tg 128, threadgroup mem 12288.
        // ARF_MM_CONC=1 → dispatch in a CONCURRENT encoder (matches the megakernel seam) to
        // reproduce any concurrent-encoder-specific fault without loading the 30B model.
        let conc = std::env::var_os("ARF_MM_CONC").is_some();
        let dtype = if conc {
            MTLDispatchType::Concurrent
        } else {
            MTLDispatchType::Serial
        };
        let run_mm = || -> f64 {
            let cmd = self.queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoderWithDispatchType(dtype).unwrap();
            let bufs: [&ProtocolObject<dyn MTLBuffer>; 7] =
                [&a, &codes, &c_mm, &dims, &scales, &mins, &dd];
            unsafe {
                enc.setComputePipelineState(&pso_mm);
                enc.setThreadgroupMemoryLength_atIndex(mm_smem, 0);
                for (i, bf) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    enc.setBuffer_offset_atIndex(Some(*bf), 0, i);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: b.div_ceil(32),
                        height: n.div_ceil(64),
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0
        };

        let _ = run_gemv();
        let _ = run_mm();
        let mut gemv_ms = f64::INFINITY;
        let mut mm_ms = f64::INFINITY;
        for _ in 0..iters {
            gemv_ms = gemv_ms.min(run_gemv());
            mm_ms = mm_ms.min(run_mm());
        }

        // compare outputs [B,n].
        let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let gp = unsafe { c_gemv.contents().as_ptr() as *const f32 };
        let mp = unsafe { c_mm.contents().as_ptr() as *const f32 };
        // GEMV reference caps at MAXB=32 rows; compare parity only over the rows it computes.
        let parity_rows = b.min(32);
        for i in 0..(parity_rows * n) {
            let (x, y) = unsafe { (*gp.add(i), *mp.add(i)) };
            let abs = (x - y).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / x.abs().max(1e-3));
        }
        drop(guard);
        Ok((max_abs, max_rel, gemv_ms, mm_ms))
    }
    /// PARITY GATE for the SHARED-PREFIX VERIFY attention kernel (attention_verify_shared_prefix).
    /// Builds the verify shape — ONE sequence, `k` query rows sharing a prefix [0, prefix_len), each
    /// row i's causal frontier = prefix_len+i (attends the prefix + drafted rows 0..i) — dispatches
    /// the shared-prefix kernel, and compares its attention output against a CPU oracle that mirrors
    /// EXACTLY the WGSL oracle (attention_batched.wgsl: row_meta`row`.last = prefix_len+i, ascending-
    /// key flash softmax). The drafted rows' K/V are scattered into the SHARED pool at the slot
    /// table's tail (slots[prefix_len+i]), so `all_accepted` forces the ADVERSARIAL case where row
    /// k-1 must attend row 0's freshly-scattered K/V (the causal-window path). Returns (max_abs,
    /// max_rel) of the attention output vs the oracle — bit-parity there ⇒ the downstream lm_head
    /// argmax is identical (the verify token contract).
    ///
    /// `prefix_len` = shared cached length; `k` = verify rows (<= KMAX=16); geometry nh/nkv/hd.
    /// pool_slots must be >= prefix_len + k + a scatter margin. window=0 (global) only.
    pub fn verify_shared_prefix_selftest(
        &mut self,
        ctx: &GpuContext,
        prefix_len: usize,
        k: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pool_slots: usize,
    ) -> Result<(f32, f32), String> {
        assert!((1..=16).contains(&k), "k must be 1..=16 (KMAX)");
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // ONE seq's slot table: prefix positions [0,prefix_len) at a NON-CONTIGUOUS disjoint run,
        // then the k drafted positions [prefix_len, prefix_len+k) at their own pool slots. This is
        // exactly `prefix_slots ++ write_slots` (progressive: slots[prefix_len+i] is row i's slot).
        let slots: Vec<u32> = (0..(prefix_len + k)).map(|p| (7 + p * 3) as u32).collect();
        // The guard must check the MAX SLOT, not the row count: the strided table's top slot is
        // 7+3*(prefix+k-1). The old `pool_slots >= prefix_len+k` let every sweep case scatter/read
        // past the pool allocation (true AGX address faults at k>1 — the 2026-07-02 wedge family).
        assert!(
            pool_slots as u32 > *slots.last().unwrap(),
            "pool too small: {} slots but slot table tops out at {}",
            pool_slots,
            slots.last().unwrap()
        );

        // SHARED pool: pre-populate the PREFIX slots [0,prefix_len) with deterministic K/V; the
        // drafted slots [prefix_len, prefix_len+k) are left ZERO (the scatter fills them on GPU +
        // the oracle replicates the scatter on a CPU copy).
        let kpool = al(pool_slots * kv_dim * 4)?;
        let vpool = al(pool_slots * kv_dim * 4)?;
        unsafe {
            let kp = kpool.contents().as_ptr() as *mut f32;
            let vp = vpool.contents().as_ptr() as *mut f32;
            for p in 0..prefix_len {
                let slot = slots[p] as usize;
                for j in 0..kv_dim {
                    *kp.add(slot * kv_dim + j) = (((p + j) % 19) as f32 - 9.0) * 0.03;
                    *vp.add(slot * kv_dim + j) = (((p * 3 + j) % 13) as f32 - 6.0) * 0.04;
                }
            }
        }

        // q for the k verify rows [k, q_dim]; new K/V for the k drafted rows [k, kv_dim].
        let qb = al(k * q_dim * 4)?;
        let newk = al(k * kv_dim * 4)?;
        let newv = al(k * kv_dim * 4)?;
        unsafe {
            let qp = qb.contents().as_ptr() as *mut f32;
            let kp = newk.contents().as_ptr() as *mut f32;
            let vp = newv.contents().as_ptr() as *mut f32;
            for i in 0..k {
                for x in 0..q_dim {
                    *qp.add(i * q_dim + x) = (((i * 5 + x) % 13) as f32 - 6.0) * 0.05;
                }
                for j in 0..kv_dim {
                    *kp.add(i * kv_dim + j) = (((i * 11 + j) % 17) as f32 - 8.0) * 0.02;
                    *vp.add(i * kv_dim + j) = (((i * 7 + j) % 11) as f32 - 5.0) * 0.03;
                }
            }
        }
        let attn = al(k * q_dim * 4)?;
        drop(guard);

        // ===== SCATTER the k drafted K/V into the pool at slots[prefix_len+i] (per-row kv_scatter).
        // grid (kv_dim, k), write_slot_b[i] = slots[prefix_len+i]. (This is the verify scatter — the
        // window keys row i attends live in the pool after this.)
        let (write_slot, scatter_dims) = {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            let ws = al2(k * 4);
            unsafe {
                let p = ws.contents().as_ptr() as *mut u32;
                for i in 0..k {
                    *p.add(i) = slots[prefix_len + i];
                }
            }
            let sd = al2(16);
            unsafe {
                let p = sd.contents().as_ptr() as *mut u32;
                *p = 1;
                *p.add(1) = kv_dim as u32;
                *p.add(2) = k as u32;
                *p.add(3) = 0;
            }
            drop(guard);
            (ws, sd)
        };
        // verify dims: q_start = k, past_len = prefix_len, window @9 = 0.
        let (slots_buf, vdims) = {
            let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
            let dev = guard.raw_device();
            let al2 = |bytes: usize| {
                dev.newBufferWithLength_options(
                    bytes.max(16),
                    objc2_metal::MTLResourceOptions::empty(),
                )
                .unwrap()
            };
            let sb = al2(slots.len() * 4);
            unsafe {
                let p = sb.contents().as_ptr() as *mut u32;
                for (i, &s) in slots.iter().enumerate() {
                    *p.add(i) = s;
                }
            }
            let ad = al2(48);
            unsafe {
                let p = ad.contents().as_ptr() as *mut u32;
                p.add(0).write(1);
                p.add(1).write(0);
                p.add(2).write(nh as u32);
                p.add(3).write(nkv as u32);
                p.add(4).write(hd as u32);
                p.add(5).write(prefix_len as u32);
                (p.add(6) as *mut f32).write(scale);
                p.add(7).write(group as u32);
                p.add(8).write(k as u32);
                p.add(9).write(0);
                p.add(10).write(0);
                p.add(11).write(0);
            }
            drop(guard);
            (sb, ad)
        };

        let p_scatter_b = self
            .pipelines
            .get("kv_scatter_b")
            .ok_or("kv_scatter_b")?
            .pso
            .clone();
        let p_verify = self
            .pipelines
            .get("attention_verify_shared_prefix")
            .ok_or("attention_verify_shared_prefix")?
            .pso
            .clone();
        let tg256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        unsafe {
            // scatter drafted K/V into the pool.
            enc.setComputePipelineState(&p_scatter_b);
            enc.useResource_usage(ProtocolObject::from_ref(&*newk), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&newk), 0, 0);
            enc.useResource_usage(ProtocolObject::from_ref(&*newv), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&newv), 0, 1);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*write_slot),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&write_slot), 0, 2);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*kpool),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&kpool), 0, 3);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*vpool),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            enc.setBuffer_offset_atIndex(Some(&vpool), 0, 4);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*scatter_dims),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&scatter_dims), 0, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: kv_dim.div_ceil(256),
                    height: k,
                    depth: 1,
                },
                tg256,
            );
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            // shared-prefix verify attention: grid (nh, 1), 256 threads.
            enc.setComputePipelineState(&p_verify);
            enc.useResource_usage(ProtocolObject::from_ref(&*qb), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&qb), 0, 0);
            enc.useResource_usage(ProtocolObject::from_ref(&*kpool), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&kpool), 0, 1);
            enc.useResource_usage(ProtocolObject::from_ref(&*vpool), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&vpool), 0, 2);
            enc.useResource_usage(ProtocolObject::from_ref(&*attn), MTLResourceUsage::Write);
            enc.setBuffer_offset_atIndex(Some(&attn), 0, 3);
            enc.useResource_usage(
                ProtocolObject::from_ref(&*slots_buf),
                MTLResourceUsage::Read,
            );
            enc.setBuffer_offset_atIndex(Some(&slots_buf), 0, 4);
            enc.useResource_usage(ProtocolObject::from_ref(&*vdims), MTLResourceUsage::Read);
            enc.setBuffer_offset_atIndex(Some(&vdims), 0, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: nh,
                    height: 1,
                    depth: 1,
                },
                tg256,
            );
            enc.endEncoding();
        }
        cmd.commit();
        // BOUNDED wait — never waitUntilCompleted here. A wedged command buffer must surface as a
        // reported failure, not park the process at 0% CPU (kill -9 of a process with a hung AGX
        // channel took the whole machine down twice on 2026-07-02: WindowServer watchdog → panic).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let st = unsafe { cmd.status() };
            if st == objc2_metal::MTLCommandBufferStatus::Completed {
                break;
            }
            if st == objc2_metal::MTLCommandBufferStatus::Error {
                return Err(format!("verify selftest GPU fault: {:?}", unsafe {
                    cmd.error()
                }));
            }
            if std::time::Instant::now() > deadline {
                return Err(format!(
                    "verify selftest TIMEOUT (30s, status {st:?}): command buffer never completed — \
                     GPU likely wedged; do NOT kill -9, let the process exit on its own"
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        // ===== CPU ORACLE: mirror attention_batched.wgsl. Replicate the scatter on a CPU copy, then
        // per row i (causal frontier = prefix_len+i) run plain (offline) softmax over keys [0..last].
        let kpool_v: Vec<f32> = unsafe {
            let p = kpool.contents().as_ptr() as *const f32;
            (0..pool_slots * kv_dim).map(|i| *p.add(i)).collect()
        };
        let vpool_v: Vec<f32> = unsafe {
            let p = vpool.contents().as_ptr() as *const f32;
            (0..pool_slots * kv_dim).map(|i| *p.add(i)).collect()
        };
        let qb_v: Vec<f32> = unsafe {
            let p = qb.contents().as_ptr() as *const f32;
            (0..k * q_dim).map(|i| *p.add(i)).collect()
        };
        let row_floats = kv_dim;
        let mut want = vec![0.0f32; k * q_dim];
        for i in 0..k {
            let last = prefix_len + i; // inclusive causal frontier for row i
            for hh in 0..nh {
                let kv_head = hh / group;
                let hcol = kv_head * hd;
                let qbase = i * q_dim + hh * hd;
                let mut scs = vec![0.0f32; last + 1];
                let mut mx = f32::MIN;
                for t in 0..=last {
                    let kslot = slots[t] as usize;
                    let mut s = 0.0f32;
                    for x in 0..hd {
                        s += qb_v[qbase + x] * kpool_v[kslot * row_floats + hcol + x];
                    }
                    scs[t] = s * scale;
                    mx = mx.max(scs[t]);
                }
                let mut den = 0.0f32;
                for t in 0..=last {
                    scs[t] = (scs[t] - mx).exp();
                    den += scs[t];
                }
                for dim in 0..hd {
                    let mut num = 0.0f32;
                    for t in 0..=last {
                        let vslot = slots[t] as usize;
                        num += scs[t] * vpool_v[vslot * row_floats + hcol + dim];
                    }
                    want[i * q_dim + hh * hd + dim] = num / den;
                }
            }
        }

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        unsafe {
            let ap = attn.contents().as_ptr() as *const f32;
            for i in 0..k * q_dim {
                let got = *ap.add(i);
                let w = want[i];
                let abs = (got - w).abs();
                max_abs = max_abs.max(abs);
                max_rel = max_rel.max(abs / w.abs().max(1e-3));
            }
        }
        Ok((max_abs, max_rel))
    }
    /// PARITY GATE for the FUSED gemma dense norm-pair (gemma_normpair). Dispatches the fused kernel
    /// over `tokens` synthetic rows and compares its outputs (updated `hidden` AND `normed`) against
    /// a CPU oracle that mirrors EXACTLY the 2-dispatch pair it replaces: rmsnorm_add
    /// (hidden += rmsnorm(attn_out)·post_norm) then rmsnorm (normed = rmsnorm(hidden)·pre_ffn).
    /// Because the fused kernel reuses the identical strided square-sum + 1/sqrt(ss/h+eps) idiom
    /// twice, parity here is bit-exact by construction — this gate proves it. Returns
    /// (max_abs_hidden, max_abs_normed). No model load; safe to run.
    pub fn gemma_normpair_selftest(
        &mut self,
        ctx: &GpuContext,
        tokens: usize,
        h: usize,
        eps: f32,
    ) -> Result<(f32, f32), String> {
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        // Deterministic synthetic inputs (spread across sign + magnitude so a wrong reduction diverges).
        let attn0: Vec<f32> = (0..tokens * h)
            .map(|i| (((i * 7) % 23) as f32 - 11.0) * 0.05)
            .collect();
        let hid0: Vec<f32> = (0..tokens * h)
            .map(|i| (((i * 5) % 19) as f32 - 9.0) * 0.04)
            .collect();
        let post_w: Vec<f32> = (0..h)
            .map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        let pre_w: Vec<f32> = (0..h)
            .map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.015)
            .collect();
        let attn = al(tokens * h * 4)?;
        let hid = al(tokens * h * 4)?;
        let postb = al(h * 4)?;
        let preb = al(h * 4)?;
        let normed = al(tokens * h * 4)?;
        let dims = al(16)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                attn0.as_ptr(),
                attn.contents().as_ptr() as *mut f32,
                tokens * h,
            );
            std::ptr::copy_nonoverlapping(
                hid0.as_ptr(),
                hid.contents().as_ptr() as *mut f32,
                tokens * h,
            );
            std::ptr::copy_nonoverlapping(
                post_w.as_ptr(),
                postb.contents().as_ptr() as *mut f32,
                h,
            );
            std::ptr::copy_nonoverlapping(pre_w.as_ptr(), preb.contents().as_ptr() as *mut f32, h);
            let p = dims.contents().as_ptr() as *mut u32;
            p.write(tokens as u32);
            p.add(1).write(h as u32);
            (p.add(2) as *mut f32).write(eps);
            p.add(3).write(0);
        }
        let pso = self
            .pipelines
            .get("gemma_normpair")
            .ok_or("gemma_normpair pipeline (compile it)")?
            .pso
            .clone();
        drop(guard);
        let cmd = self.queue.commandBuffer().ok_or("cmd")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("enc")?;
        unsafe {
            enc.setComputePipelineState(&pso);
            for (i, b) in [&attn, &postb, &hid, &preb, &normed, &dims]
                .iter()
                .enumerate()
            {
                enc.useResource_usage(
                    ProtocolObject::from_ref(&***b),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                enc.setBuffer_offset_atIndex(Some(&***b), 0, i);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: tokens,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cmd.commit();
        // BOUNDED wait — never bare waitUntilCompleted in a selftest (2026-07-02 wedge lesson).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let st = unsafe { cmd.status() };
            if st == objc2_metal::MTLCommandBufferStatus::Completed {
                break;
            }
            if st == objc2_metal::MTLCommandBufferStatus::Error {
                return Err(format!("gemma_normpair GPU fault: {:?}", unsafe {
                    cmd.error()
                }));
            }
            if std::time::Instant::now() > deadline {
                return Err("gemma_normpair TIMEOUT (GPU wedge?)".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // CPU oracle: EXACT rmsnorm_add then rmsnorm (same math as decode_ops:398-414 + rmsnorm:30-51).
        let mut want_hid = hid0.clone();
        let mut want_normed = vec![0.0f32; tokens * h];
        for t in 0..tokens {
            let b = t * h;
            let ss1: f32 = (0..h).map(|i| attn0[b + i] * attn0[b + i]).sum();
            let inv1 = 1.0 / (ss1 / h as f32 + eps).sqrt();
            for i in 0..h {
                want_hid[b + i] += attn0[b + i] * inv1 * post_w[i];
            }
            let ss2: f32 = (0..h).map(|i| want_hid[b + i] * want_hid[b + i]).sum();
            let inv2 = 1.0 / (ss2 / h as f32 + eps).sqrt();
            for i in 0..h {
                want_normed[b + i] = want_hid[b + i] * inv2 * pre_w[i];
            }
        }
        let (mut ma_h, mut ma_n) = (0.0f32, 0.0f32);
        unsafe {
            let hp = hid.contents().as_ptr() as *const f32;
            let np = normed.contents().as_ptr() as *const f32;
            for i in 0..tokens * h {
                ma_h = ma_h.max((*hp.add(i) - want_hid[i]).abs());
                ma_n = ma_n.max((*np.add(i) - want_normed[i]).abs());
            }
        }
        Ok((ma_h, ma_n))
    }

    /// SLOT-LIST pool-sync gate (the double-pool residency fix). Proves `kv_f32_to_f16_slots`
    /// (export) and `kv_f16_to_f32_slots` (import) convert EXACTLY the listed slots and NEVER
    /// touch unlisted ones — the anti-whole-pool property that protects decoded-token f16 slots
    /// from stale overwrite on mid-stream admissions. Drives the PRODUCTION host method
    /// (`convert_kv_slots`) end to end, over TWO synthetic layers with DIFFERENT kv_dim (the
    /// per-layer dims path). Exact-bits assertions: export must equal the IEEE round-to-nearest-
    /// even f32→f16 narrowing; import must equal the exact f16→f32 widening; untouched regions
    /// must be bit-identical to their sentinels.
    pub fn slot_convert_selftest(&self, ctx: &GpuContext, pool_slots: usize) -> Result<(), String> {
        fn f32_to_f16_bits(f: f32) -> u16 {
            let x = f.to_bits();
            let sign = ((x >> 16) & 0x8000) as u16;
            let mut mant = (x & 0x007f_ffff) as i32;
            let exp = ((x >> 23) & 0xff) as i32;
            if exp == 0xff {
                return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
            }
            let mut e = exp - 127 + 15;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
            if e <= 0 {
                if e < -10 {
                    return sign;
                }
                mant |= 0x0080_0000;
                let shift = 14 - e;
                let round = 1 << (shift - 1);
                let mut m = (mant + round) >> shift;
                if (mant & (round - 1)) == 0 && (m & 1) == 1 {
                    m -= (mant >> shift) & 1;
                }
                return sign | m as u16;
            }
            let round = 0x0000_1000;
            let mut m = mant + round;
            if (mant & 0x0000_1fff) == round {
                m &= !1;
            }
            if (m & 0x0080_0000) != 0 {
                e += 1;
                m = 0;
            }
            if e >= 0x1f {
                return sign | 0x7c00;
            }
            sign | ((e as u16) << 10) | ((m >> 13) & 0x03ff) as u16
        }
        fn f16_bits_to_f32(h: u16) -> f32 {
            let sign = ((h & 0x8000) as u32) << 16;
            let exp = ((h >> 10) & 0x1f) as u32;
            let mant = (h & 0x3ff) as u32;
            let bits = if exp == 0 {
                if mant == 0 {
                    sign
                } else {
                    let mut e: u32 = 113;
                    let mut m = mant;
                    while m & 0x400 == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    sign | (e << 23) | ((m & 0x3ff) << 13)
                }
            } else if exp == 0x1f {
                sign | 0x7f80_0000 | (mant << 13)
            } else {
                sign | ((exp + 112) << 23) | (mant << 13)
            };
            f32::from_bits(bits)
        }
        // Test values avoid the f16-subnormal band (|x| < 6.1e-5) except exact 0.0 — both
        // converters handle 0.0; subnormal round-trip is out of scope (real KV never lives there).
        let pat = |li: usize, i: usize| ((((li * 3 + i) % 19) as i32 - 9) as f32) * 0.03;
        let sentinel = |li: usize, i: usize| ((((li * 7 + i) % 13) as i32 - 6) as f32) * 0.04;
        let kv_dims = [512usize, 256];
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let al = |bytes: usize| {
            dev.newBufferWithLength_options(bytes.max(16), objc2_metal::MTLResourceOptions::empty())
                .ok_or("alloc")
        };
        let mut bufs = Vec::new(); // (kf32, vf32, kf16, vf16, kv_dim)
        for (li, &kv_dim) in kv_dims.iter().enumerate() {
            let n = pool_slots * kv_dim;
            let kf32 = al(n * 4)?;
            let vf32 = al(n * 4)?;
            let kf16 = al(n * 2)?;
            let vf16 = al(n * 2)?;
            unsafe {
                let kp = kf32.contents().as_ptr() as *mut f32;
                let vp = vf32.contents().as_ptr() as *mut f32;
                let kh = kf16.contents().as_ptr() as *mut u16;
                let vh = vf16.contents().as_ptr() as *mut u16;
                for i in 0..n {
                    *kp.add(i) = pat(li, i);
                    *vp.add(i) = pat(li, i + 5);
                    *kh.add(i) = f32_to_f16_bits(sentinel(li, i));
                    *vh.add(i) = f32_to_f16_bits(sentinel(li, i + 5));
                }
            }
            bufs.push((kf32, vf32, kf16, vf16, kv_dim));
        }
        drop(guard);
        // Non-contiguous, unsorted slot list — the paged-slot shape.
        let listed: Vec<u32> = (0..pool_slots as u32)
            .filter(|s| s % 3 == 1 || s % 7 == 0)
            .rev()
            .collect();
        if listed.is_empty() || listed.len() == pool_slots {
            return Err("slot list degenerate — pick a larger pool_slots".into());
        }
        let is_listed = |s: usize| {
            let s = s as u32;
            s % 3 == 1 || s.is_multiple_of(7)
        };
        let tuples: Vec<_> = bufs
            .iter()
            .map(|(k32, v32, k16, v16, kv_dim)| (&**k32, &**v32, &**k16, &**v16, *kv_dim))
            .collect();

        // (1) EXPORT the listed slots f32→f16; unlisted f16 must keep their sentinels.
        self.convert_kv_slots(ctx, &tuples, &listed, true)?;
        for (li, (_, _, kf16, vf16, kv_dim)) in bufs.iter().enumerate() {
            unsafe {
                let kh = kf16.contents().as_ptr() as *const u16;
                let vh = vf16.contents().as_ptr() as *const u16;
                for s in 0..pool_slots {
                    for j in 0..*kv_dim {
                        let i = s * kv_dim + j;
                        let (wk, wv) = if is_listed(s) {
                            (f32_to_f16_bits(pat(li, i)), f32_to_f16_bits(pat(li, i + 5)))
                        } else {
                            (
                                f32_to_f16_bits(sentinel(li, i)),
                                f32_to_f16_bits(sentinel(li, i + 5)),
                            )
                        };
                        if *kh.add(i) != wk || *vh.add(i) != wv {
                            return Err(format!(
                            "EXPORT mismatch layer{li} slot{s} j{j} (listed={}): k {:#06x} want {:#06x}, v {:#06x} want {:#06x}",
                            is_listed(s), *kh.add(i), wk, *vh.add(i), wv));
                        }
                    }
                }
            }
        }
        // (2) Scribble the listed slots' f32 rows, IMPORT them back f16→f32; unlisted f32 must
        // keep their original pattern (bit-identical — import never touches them).
        for (li, (kf32, vf32, _, _, kv_dim)) in bufs.iter().enumerate() {
            unsafe {
                let kp = kf32.contents().as_ptr() as *mut f32;
                let vp = vf32.contents().as_ptr() as *mut f32;
                for &s in &listed {
                    for j in 0..*kv_dim {
                        let i = s as usize * kv_dim + j;
                        *kp.add(i) = 777.0 + li as f32;
                        *vp.add(i) = -777.0 - li as f32;
                    }
                }
            }
        }
        self.convert_kv_slots(ctx, &tuples, &listed, false)?;
        for (li, (kf32, vf32, kf16, vf16, kv_dim)) in bufs.iter().enumerate() {
            unsafe {
                let kp = kf32.contents().as_ptr() as *const f32;
                let vp = vf32.contents().as_ptr() as *const f32;
                let kh = kf16.contents().as_ptr() as *const u16;
                let vh = vf16.contents().as_ptr() as *const u16;
                for s in 0..pool_slots {
                    for j in 0..*kv_dim {
                        let i = s * kv_dim + j;
                        let (wk, wv) = if is_listed(s) {
                            (f16_bits_to_f32(*kh.add(i)), f16_bits_to_f32(*vh.add(i)))
                        } else {
                            (pat(li, i), pat(li, i + 5))
                        };
                        if (*kp.add(i)).to_bits() != wk.to_bits()
                            || (*vp.add(i)).to_bits() != wv.to_bits()
                        {
                            return Err(format!(
                            "IMPORT mismatch layer{li} slot{s} j{j} (listed={}): k {} want {wk}, v {} want {wv}",
                            is_listed(s), *kp.add(i), *vp.add(i)));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl MetalIsland {
    /// Run the SHIPPED MPP path on one weight: `y = x · Wᵀ` as `[rows, n]` f32, the GPU seconds it
    /// took, and the dequantized weight the GPU actually multiplied by.
    ///
    /// The buffers are built exactly as `weights.rs` uploads a `Q4ksMtl` — u32 codes, u8/i8
    /// scales/mins, and `d`/`dmin` through the island's f16 `ddh` stream — and the matmul goes
    /// through `q4rm_encode`, the SAME routine the batched record calls, so the path tested is the
    /// path that ships. `x` is f32, as the engine's activations are: the narrowing passes are part
    /// of what is measured. The returned weight carries the f16 rounding of `d`/`dmin`, so a
    /// caller's reference multiplies by what the kernel saw, not by what `q` started as.
    ///
    /// `Err` on an OS whose Metal has no MPP tensor ops (below macOS 26) — "feature unavailable".
    pub fn mpp_q4_selftest(
        &mut self,
        ctx: &GpuContext,
        q: &arf_core::tensor::Q4KSMatrix,
        x: &[f32],
        rows: usize,
    ) -> Result<(Vec<f32>, f64, Vec<f32>), String> {
        // 8 or 16 = one unit; a multiple of 32 = the WIDE (prefill) form, tiles of 32.
        let wide = rows.is_multiple_of(Q4TILE_WIDE_UNIT);
        assert!(
            rows == Q4TILE_ROWS || rows == Q4TILE_MAX_ROWS || wide,
            "8, 16, or a multiple of 32"
        );
        let unit = if wide { Q4TILE_WIDE_UNIT } else { rows };
        let (n, k) = (q.rows(), q.cols());
        assert_eq!(x.len(), rows * k, "x must be [rows, cols]");
        let msl = include_str!("../../shaders/metal/matmul_mpp_q4ks_msl.metal");
        for name in [
            "q4tile_blockmax",
            "q4tile_scale",
            "q4tile_narrow",
            "q4rm_mm",
        ] {
            let key = format!("{name}_r{unit}");
            self.compile_msl(ctx, &key, msl, &key)?;
        }
        // The simdgroup_matrix decode matmul, when ARF_SG_Q4 selects it: this selftest must gate
        // whatever `q4rm_encode` will actually dispatch, or it gates nothing.
        if !wide && std::env::var_os("ARF_SG_Q4").is_some() {
            for name in ["q4sg_prepare", "q4sg_mm", "q4sg2_prepare", "q4sg2_mm"] {
                let key = format!("{name}_r{unit}");
                self.compile_msl(ctx, &key, msl, &key)?;
            }
        }
        if wide {
            // the prefill matmul `q4rm_encode_wide` dispatches, at both shapes
            for k in ["q4pf2_mm_r32_n64", "q4pf2_mm_r128_n64"] {
                self.compile_msl(ctx, k, msl, k)?;
            }
        } else {
            for name in ["q4rm_split", "q4rm_sum"] {
                let key = format!("{name}_r{unit}");
                self.compile_msl(ctx, &key, msl, &key)?;
            }
        }
        // The decode units are checked through the SPLIT matmul when it is on (the default), so
        // this example gates the kernel the engine actually dispatches; ARF_NO_MPP_SPLIT=1
        // checks the unsplit one.
        let parts = if !wide && self.q4rm_split_ready(unit) {
            self.q4rm_parts_for(ctx, q.rows())
        } else {
            None
        };
        let sc = self.q4tile_scratch_for(ctx, k).ok_or("q4 scratch alloc")?;
        let dims = self
            .q4rm_dims_for(ctx, n, k)
            .ok_or("shape is not whole tiles")?;

        // d/dmin as the island stores them: f16 pairs. Round-trip them for the reference too.
        let dd: Vec<f32> = q
            .d()
            .iter()
            .zip(q.dmin())
            .flat_map(|(a, b)| [*a, *b])
            .collect();
        let ddh = dd_to_ddh(&dd);
        let (subs, supers) = (k / 32, k / 256);
        let mut wd = vec![0.0f32; n * k];
        for r in 0..n {
            for sub in 0..subs {
                let h = ddh[r * supers + sub / 8];
                let s = f16_bits_to_f32(h as u16) * q.scales()[r * subs + sub] as f32;
                let lo = f16_bits_to_f32((h >> 16) as u16) * q.mins()[r * subs + sub] as f32;
                for i in 0..32 {
                    let pos = r * k + sub * 32 + i;
                    let code = (q.codes()[pos / 8] >> ((pos % 8) * 4)) & 0xF;
                    wd[pos] = code as f32 * s + lo;
                }
            }
        }

        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let opts = objc2_metal::MTLResourceOptions::StorageModeShared;
        let up = |bytes: &[u8]| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            let b = dev
                .newBufferWithLength_options(bytes.len().max(16), opts)
                .ok_or_else(|| "q4 buffer alloc".to_string())?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    b.contents().as_ptr() as *mut u8,
                    bytes.len(),
                );
            }
            Ok(b)
        };
        fn raw<T>(s: &[T]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        }
        let w = Q4ksMtl {
            codes: up(raw(q.codes()))?,
            scales: up(q.scales())?,
            mins: up(raw(q.mins()))?,
            dd: up(raw(&ddh))?,
            dims: up(raw(&[1u32, k as u32, n as u32, 0]))?,
            n,
            g64: None,
        };
        let b_x = up(raw(x))?;
        let b_y = up(&vec![0u8; rows * n * 4])?;
        drop(guard);

        let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("no encoder")?;
        unsafe {
            if wide {
                self.q4rm_encode_wide(
                    &enc,
                    rows,
                    &w,
                    &dims.0,
                    n,
                    k,
                    &sc,
                    &b_x,
                    &b_y,
                    &|_, _| {},
                    false,
                )?;
            } else {
                self.q4rm_encode(
                    &enc,
                    rows,
                    &w,
                    &dims.0,
                    n,
                    k,
                    &sc,
                    &b_x,
                    &b_y,
                    &|_, _| {},
                    false,
                    parts.as_ref().map(|b| &*b.0),
                )?;
            }
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(e) = cmd.error() {
            return Err(format!("q4 command buffer: {e:?}"));
        }
        let y: Vec<f32> = unsafe {
            let p = b_y.contents().as_ptr() as *const f32;
            (0..rows * n).map(|i| *p.add(i)).collect()
        };
        // TIMING: a single cold dispatch read 0.9-2.7 ms for ONE shape (2026-09-19) — useless.
        // 32 dispatches per command buffer, 9 buffers, median, per dispatch; the first buffer
        // above is the warm-up and is not counted.
        const K: usize = 32;
        let mut per = Vec::new();
        for _ in 0..9 {
            let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
            let enc = cmd
                // The wide form only pays off when its row-tiles OVERLAP, which needs the concurrent
                // encoder the engine's record uses; single units keep the serial one they were
                // first measured with.
                .computeCommandEncoderWithDispatchType(if wide {
                    MTLDispatchType::Concurrent
                } else {
                    MTLDispatchType::Serial
                })
                .ok_or("no encoder")?;
            unsafe {
                for _ in 0..K {
                    if wide {
                        self.q4rm_encode_wide(
                            &enc,
                            rows,
                            &w,
                            &dims.0,
                            n,
                            k,
                            &sc,
                            &b_x,
                            &b_y,
                            &|_, _| {},
                            false,
                        )?;
                    } else {
                        self.q4rm_encode(
                            &enc,
                            rows,
                            &w,
                            &dims.0,
                            n,
                            k,
                            &sc,
                            &b_x,
                            &b_y,
                            &|_, _| {},
                            false,
                            parts.as_ref().map(|b| &*b.0),
                        )?;
                    }
                }
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            per.push((cmd.GPUEndTime() - cmd.GPUStartTime()) / K as f64);
        }
        per.sort_by(|a, b| a.total_cmp(b));
        let gpu_s = per[per.len() / 2];
        // The narrowing ALONE, same method — printed, not returned, so the signature holds.
        let mut nar = Vec::new();
        for _ in 0..9 {
            let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
            let enc = cmd
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .ok_or("no encoder")?;
            unsafe {
                for _ in 0..K {
                    self.q4_narrow_encode(&enc, unit, &dims.0, k, &sc, &b_x, &|_, _| {})?;
                }
                enc.endEncoding();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            nar.push((cmd.GPUEndTime() - cmd.GPUStartTime()) / K as f64);
        }
        nar.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "    narrowing alone k={k} rows={rows}: {:.0} us of {:.0} us",
            nar[4] * 1e6,
            gpu_s * 1e6
        );
        Ok((y, gpu_s, wd))
    }
}

impl MetalIsland {
    /// GATE for `dflash_chain_sampled` (2026-09-26), no model: run the kernel ONCE on the given
    /// selector tables (`cands`/`unary` `[rows, 16]`, `edges` `[rows, 16, 16]`), the anchor, the
    /// host-keyed uniforms `[rows]` and the temperature — bound exactly as the GPU-select encode
    /// binds them (`dflash2_block.rs`) — and return the verify window `[rows]` and the q as
    /// `(ids, probabilities)`, both `[rows, 16]`. Compared against `dflash_chain_sampled_ref` by
    /// `examples/dflash_chain_sampled_probe.rs`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn dflash_chain_sampled_selftest(
        &mut self,
        ctx: &GpuContext,
        cands: &[u32],
        unary: &[f32],
        edges: &[f32],
        anchor: u32,
        unif: &[f32],
        temp: f32,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>), String> {
        const K: usize = 16;
        let rows = unif.len();
        assert_eq!(cands.len(), rows * K, "cands len");
        assert_eq!(unary.len(), rows * K, "unary len");
        assert_eq!(edges.len(), rows * K * K, "edges len");
        let msl = include_str!("../../shaders/metal/dflash2_msl.metal");
        self.compile_msl(ctx, "dflash_chain_sampled", msl, "dflash_chain_sampled")?;
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let opts = objc2_metal::MTLResourceOptions::StorageModeShared;
        let mk = |bytes: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            dev.newBufferWithLength_options(bytes.max(16), opts)
                .ok_or_else(|| "chain buffer alloc".to_string())
        };
        let up = |b: &Retained<ProtocolObject<dyn MTLBuffer>>, src: &[u32]| unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                b.contents().as_ptr() as *mut u32,
                src.len(),
            )
        };
        let b_c = mk(rows * K * 4)?;
        let b_u = mk(rows * K * 4)?;
        let b_e = mk(rows * K * K * 4)?;
        let b_in = mk(rows * 4)?;
        let b_win = mk(rows * 4)?;
        let b_qi = mk(rows * K * 4)?;
        let b_qp = mk(rows * K * 4)?;
        up(&b_c, cands);
        up(&b_u, &unary.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        up(&b_e, &edges.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let mut tok_in = vec![0u32; rows];
        tok_in[0] = anchor;
        up(&b_in, &tok_in);
        // sentinel fill: a kernel that never ran leaves these, not a plausible answer (rule 8)
        up(&b_win, &vec![0xDEAD_BEEFu32; rows]);
        up(&b_qi, &vec![0xDEAD_BEEFu32; rows * K]);
        drop(guard);
        let dims: [u32; 4] = [0, 0, rows as u32, 0];
        let pso = &self
            .pipelines
            .get("dflash_chain_sampled")
            .ok_or("pso dflash_chain_sampled")?
            .pso;
        let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("no encoder")?;
        let bufs: [(&ProtocolObject<dyn MTLBuffer>, usize); 7] = [
            (&b_c, 0),
            (&b_u, 1),
            (&b_e, 2),
            (&b_in, 3),
            (&b_win, 4),
            (&b_qi, 8),
            (&b_qp, 9),
        ];
        unsafe {
            enc.setComputePipelineState(pso);
            for (b, i) in bufs {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(b);
                enc.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(dims.as_ptr() as *mut _).unwrap(),
                16,
                5,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(unif.as_ptr() as *mut _).unwrap(),
                rows * 4,
                6,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&temp as *const f32 as *mut _).unwrap(),
                4,
                7,
            );
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        let rd_u = |b: &Retained<ProtocolObject<dyn MTLBuffer>>, n: usize| unsafe {
            std::slice::from_raw_parts(b.contents().as_ptr() as *const u32, n).to_vec()
        };
        let win = rd_u(&b_win, rows);
        let qi = rd_u(&b_qi, rows * K);
        let qp = rd_u(&b_qp, rows * K)
            .into_iter()
            .map(f32::from_bits)
            .collect();
        Ok((win, qi, qp))
    }
}

/// What `MetalIsland::g64_pfs_selftest` returns: every output of both prefill paths on the SAME
/// inputs, and their interleaved timings. Read by `examples/g64_pfs_probe.rs`.
pub struct G64PfsSelftest {
    /// `g64pf2_mm` (the default prefill path, through `g64pf2_encode`), f32 `[rows, n]`.
    pub pf2: Vec<f32>,
    /// `g64pfs_mm` (another engine's sg4 form, through `g64pfs_encode`), f32 `[rows, n]`.
    pub pfs: Vec<f32>,
    /// `g64pfs_mm_residual` with the given residual, f32 `[rows, n]`.
    pub pfs_residual: Vec<f32>,
    /// `g64pfs_mm_up_silu_sums` with the given gate: the bf16 bits `[rows, n]` …
    pub pfs_up_silu: Vec<u16>,
    /// … and its output sums, `[rows / 32][n / 64][32]` f32.
    pub pfs_up_silu_sums: Vec<f32>,
    /// Median GPU seconds per call (command-buffer GPUStartTime/GPUEndTime): operand pass
    /// included (`*_full`: the narrowing chain / `g64pfs_prepare`), and reused (`*_mm`).
    pub t_pf2_full: f64,
    pub t_pfs_full: f64,
    pub t_pf2_mm: f64,
    pub t_pfs_mm: f64,
    /// Timed samples per arm, calls per sample, and the weight copies rotated through.
    pub samples: usize,
    pub calls_per_sample: usize,
    pub copies: usize,
    /// Rule 7: (`g64pf2_mm`, `g64pfs_mm*`) dispatches the correctness run made.
    pub dispatched: (u64, u64),
}

/// The sentinel every f32 output is filled with before the correctness run: a NaN no kernel
/// writes, so an output the kernel never reached cannot pass (rule 8).
pub const G64PFS_SENTINEL_F32: u32 = 0xFFC0_DEAD;
/// The bf16 sentinel (also a NaN).
pub const G64PFS_SENTINEL_BF16: u16 = 0xFFDE;

impl MetalIsland {
    /// GATE + TIMING for `ARF_G64_PFS` (2026-09-27), NO MODEL: one group-64 weight (`w`/`s`/`b` in
    /// the StorageN=256 tiling, f16 scale/bias bits — as `G64Mtl` holds them), f32 activations
    /// `x` `[rows, k]`, `rows` a multiple of 32. Runs the SHIPPED encode routines on the same
    /// inputs:
    ///   * `g64pf2_encode` (the default prefill matmul, narrowing chain included), in windows of at
    ///     most `Q4TILE_WIDE_MAX_ROWS` rows — what the record does with a longer prompt;
    ///   * `g64pfs_encode` in one call of `rows` rows: plain, then residual (`residual`
    ///     `[rows, n]`) and up-silu-sums (`gate` `[rows, n]`) on the prepared operand.
    ///
    /// Every output is sentinel-filled first. Then `samples` command buffers per arm, the four arms
    /// interleaved (pf2 full, pfs full, pf2 operand reused, pfs operand reused), each buffer
    /// `CALLS` calls on ROTATED weight copies (>= 512 MiB of them, so the weight is not
    /// cache-resident), on a concurrent encoder with a barrier between every stage and call — the
    /// record's ordering. Times are medians of (GPU end - start) / calls.
    #[allow(clippy::too_many_arguments)]
    pub fn g64_pfs_selftest(
        &mut self,
        ctx: &GpuContext,
        n: usize,
        k: usize,
        w: &[u8],
        s: &[u16],
        b: &[u16],
        x: &[f32],
        rows: usize,
        residual: &[f32],
        gate: &[f32],
        samples: usize,
    ) -> Result<G64PfsSelftest, String> {
        const CALLS: usize = 4;
        const WINDOW: usize = Q4TILE_WIDE_MAX_ROWS;
        if rows == 0 || !rows.is_multiple_of(G64PFS_ROWS) || rows > G64PFS_MAX_ROWS {
            return Err(format!(
                "g64_pfs_selftest: {rows} rows (a multiple of 32, <= {G64PFS_MAX_ROWS})"
            ));
        }
        if x.len() != rows * k
            || residual.len() != rows * n
            || gate.len() != rows * n
            || w.len() != n * k / 2
            || s.len() != n * k / 64
            || b.len() != n * k / 64
            || samples == 0
        {
            return Err("g64_pfs_selftest: input lengths do not match n / k / rows".into());
        }
        let mpp = include_str!("../../shaders/metal/matmul_mpp_q4ks_msl.metal");
        for name in ["q4tile_blockmax", "q4tile_scale", "q4tile_narrow"] {
            let key = format!("{name}_r{Q4TILE_WIDE_UNIT}");
            self.compile_msl(ctx, &key, mpp, &key)?;
        }
        let pf = include_str!("../../shaders/metal/matmul_g64_pf_msl.metal");
        for key in ["g64pf2_mm_r32_n64", "g64pf2_mm_r128_n64"] {
            self.compile_msl(ctx, key, pf, key)?;
        }
        let pfs = include_str!("../../shaders/metal/matmul_g64_pfs_msl.metal");
        for key in [
            "g64pfs_prepare",
            "g64pfs_mm",
            "g64pfs_mm_residual",
            "g64pfs_mm_up_silu_sums",
        ] {
            self.compile_msl(ctx, key, pfs, key)?;
        }
        let sc = self.q4tile_scratch_for(ctx, k).ok_or("q4 scratch alloc")?;
        let dims = self
            .q4rm_dims_for(ctx, n, k)
            .ok_or("shape is not whole tiles")?;
        // rotated copies of the SAME weight: every call reads memory no recent call left cached
        let per_copy = w.len() + 4 * s.len();
        let copies = (512usize << 20).div_ceil(per_copy).max(2);
        let ws = (0..copies)
            .map(|_| Q4ksMtl::from_g64(ctx, n, k, w, s, b))
            .collect::<Result<Vec<_>, _>>()?;

        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let opts = objc2_metal::MTLResourceOptions::StorageModeShared;
        let mk = |bytes: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            dev.newBufferWithLength_options(bytes.max(16), opts)
                .ok_or_else(|| "g64pfs probe buffer alloc".to_string())
        };
        fn raw<T>(s: &[T]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        }
        let up = |bytes: &[u8]| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            let buf = mk(bytes.len())?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buf.contents().as_ptr() as *mut u8,
                    bytes.len(),
                )
            };
            Ok(buf)
        };
        let fill = |buf: &Retained<ProtocolObject<dyn MTLBuffer>>, words: usize, v: u32| unsafe {
            std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u32, words).fill(v)
        };
        // g64pf2: the record's windows of <= 256 rows, each its own a / c
        let windows: Vec<(usize, usize)> = (0..rows)
            .step_by(WINDOW)
            .map(|r0| (r0, (rows - r0).min(WINDOW)))
            .collect();
        let xs = windows
            .iter()
            .map(|&(r0, r)| up(raw(&x[r0 * k..(r0 + r) * k])))
            .collect::<Result<Vec<_>, _>>()?;
        let cs = windows
            .iter()
            .map(|&(_, r)| mk(r * n * 4))
            .collect::<Result<Vec<_>, _>>()?;
        let x_all = up(raw(x))?;
        let (c_pfs, c_res, c_up, c_sums) = (
            mk(rows * n * 4)?,
            mk(rows * n * 4)?,
            mk(rows * n * 2)?,
            mk(rows * (n / 64) * 4)?,
        );
        let (b_res, b_gate) = (up(raw(residual))?, up(raw(gate))?);
        for (c, &(_, r)) in cs.iter().zip(&windows) {
            fill(c, r * n, G64PFS_SENTINEL_F32);
        }
        fill(&c_pfs, rows * n, G64PFS_SENTINEL_F32);
        fill(&c_res, rows * n, G64PFS_SENTINEL_F32);
        let s2 = u32::from(G64PFS_SENTINEL_BF16);
        fill(&c_up, rows * n / 2, (s2 << 16) | s2);
        fill(&c_sums, rows * (n / 64), G64PFS_SENTINEL_F32);
        drop(guard);

        let w0 = &ws[0];
        let g0 = w0.g64.as_ref().ok_or("from_g64 built no group-64 weight")?;
        let before = (self.q4pf2_dispatches.get(), self.g64pfs_dispatches.get());
        // CORRECTNESS: a serial encoder (every dispatch sees the previous one's writes)
        let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("no encoder")?;
        let none = |_: &[&ProtocolObject<dyn MTLBuffer>], _: &[&ProtocolObject<dyn MTLBuffer>]| {};
        unsafe {
            for (i, &(_, r)) in windows.iter().enumerate() {
                self.g64pf2_encode(
                    &enc, r, w0, g0, &dims.0, n, k, &sc, &xs[i], &cs[i], &none, false,
                )?;
            }
            self.g64pfs_encode(
                &enc,
                rows,
                g0,
                n,
                &x_all,
                &c_pfs,
                &none,
                false,
                G64PfsEpilogue::Plain,
            )?;
            self.g64pfs_encode(
                &enc,
                rows,
                g0,
                n,
                &x_all,
                &c_res,
                &none,
                true,
                G64PfsEpilogue::Residual(&b_res),
            )?;
            self.g64pfs_encode(
                &enc,
                rows,
                g0,
                n,
                &x_all,
                &c_up,
                &none,
                true,
                G64PfsEpilogue::UpSiluSums {
                    gate: &b_gate,
                    out_sums: &c_sums,
                },
            )?;
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(e) = cmd.error() {
            return Err(format!("g64pfs probe command buffer: {e:?}"));
        }
        let dispatched = (
            self.q4pf2_dispatches.get() - before.0,
            self.g64pfs_dispatches.get() - before.1,
        );
        let rd32 = |b: &Retained<ProtocolObject<dyn MTLBuffer>>, words: usize| unsafe {
            std::slice::from_raw_parts(b.contents().as_ptr() as *const u32, words).to_vec()
        };
        let as_f32 = |v: Vec<u32>| v.into_iter().map(f32::from_bits).collect::<Vec<f32>>();
        let pf2: Vec<f32> = cs
            .iter()
            .zip(&windows)
            .flat_map(|(c, &(_, r))| as_f32(rd32(c, r * n)))
            .collect();
        let pfs_out = as_f32(rd32(&c_pfs, rows * n));
        let pfs_residual = as_f32(rd32(&c_res, rows * n));
        let pfs_up_silu = unsafe {
            std::slice::from_raw_parts(c_up.contents().as_ptr() as *const u16, rows * n).to_vec()
        };
        let pfs_up_silu_sums = as_f32(rd32(&c_sums, rows * (n / 64)));

        // TIMING: the four arms interleaved at the command-buffer grain (rule 3); round 0 warms up
        let mut next = 0usize;
        let mut times: [Vec<f64>; 4] = Default::default();
        for round in 0..=samples {
            for (arm, t) in times.iter_mut().enumerate() {
                let cmd = self.queue.commandBuffer().ok_or("no command buffer")?;
                let enc = cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .ok_or("no encoder")?;
                let bar = |_: &[&ProtocolObject<dyn MTLBuffer>],
                           _: &[&ProtocolObject<dyn MTLBuffer>]| unsafe {
                    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers)
                };
                let reuse = arm >= 2;
                unsafe {
                    for _ in 0..CALLS {
                        let wc = &ws[next % copies];
                        let gc = wc.g64.as_ref().ok_or("no group-64 weight")?;
                        next += 1;
                        if arm % 2 == 0 {
                            for (i, &(_, r)) in windows.iter().enumerate() {
                                self.g64pf2_encode(
                                    &enc, r, wc, gc, &dims.0, n, k, &sc, &xs[i], &cs[i], &bar,
                                    reuse,
                                )?;
                            }
                        } else {
                            self.g64pfs_encode(
                                &enc,
                                rows,
                                gc,
                                n,
                                &x_all,
                                &c_pfs,
                                &bar,
                                reuse,
                                G64PfsEpilogue::Plain,
                            )?;
                        }
                        bar(&[], &[]);
                    }
                    enc.endEncoding();
                }
                cmd.commit();
                cmd.waitUntilCompleted();
                if let Some(e) = cmd.error() {
                    return Err(format!("g64pfs probe timing buffer: {e:?}"));
                }
                if round > 0 {
                    t.push((cmd.GPUEndTime() - cmd.GPUStartTime()) / CALLS as f64);
                }
            }
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let [a0, a1, a2, a3] = times;
        Ok(G64PfsSelftest {
            pf2,
            pfs: pfs_out,
            pfs_residual,
            pfs_up_silu,
            pfs_up_silu_sums,
            t_pf2_full: med(a0),
            t_pfs_full: med(a1),
            t_pf2_mm: med(a2),
            t_pfs_mm: med(a3),
            samples,
            calls_per_sample: CALLS,
            copies,
            dispatched,
        })
    }
}
