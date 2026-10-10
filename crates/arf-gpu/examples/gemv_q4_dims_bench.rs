//! Isolate whether the Q4_0 m=1 decode GEMV (matmul_vec_q4.wgsl) is intrinsically slow at
//! gemma-12B's dimensions vs llama's. Times the kernel ALONE (no model loop, no per-dispatch
//! overhead, no profiling device-poll) across the real per-matmul (k=contract, n=out) shapes,
//! reporting median per-dispatch µs + achieved GB/s on the weight bytes (Q4_0 ≈ 0.5 B/weight +
//! block scale). If gemma's shapes are ~30× slower per byte → the kernel is the chat stall.
//! If they're fast in isolation → the 3.4s/step stall is inter-dispatch serialization (GPU
//! timeline needed), not the GEMV.
//!
//! Run: cargo run --release -p arf-gpu --example gemv_q4_dims_bench

use std::time::Instant;

const SRC: &str = include_str!("../src/shaders/wgsl/matmul_vec_q4.wgsl");

// (label, k=contract dim, n=output dim) — the real per-matmul shapes.
const SHAPES: &[(&str, u32, u32)] = &[
    // llama-1B (hidden 2048) — the FAST reference
    ("llama q_proj   k2048 n2048", 2048, 2048),
    ("llama down     k8192 n2048", 8192, 2048),
    // gemma-12B (hidden 3840, global q_dim 8192, FFN 15360) — the SLOW model
    ("g12 q_proj     k3840 n4096", 3840, 4096),
    ("g12 o_proj GLB k8192 n3840", 8192, 3840), // global-layer o_proj, the wide one
    ("g12 gate/up    k3840 n15360", 3840, 15360), // FFN up
    ("g12 down       k15360 n3840", 15360, 3840), // FFN down
    ("g12 lm_head    k3840 n262144", 3840, 262144),
];

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no adapter");
    println!("adapter: {:?}\n", adapter.get_info().name);
    let alim = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gemv-bench"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_buffer_size: alim.max_buffer_size,
            max_storage_buffer_binding_size: alim.max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        },
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("device");

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("q4-gemv"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("q4-gemv"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    use wgpu::util::DeviceExt;
    let st =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    println!(
        "{:32} {:>10} {:>10} {:>12}",
        "shape", "us/call", "GB/s", "weight_MB"
    );
    for &(label, k, n) in SHAPES {
        assert!(k % 32 == 0);
        let blocks = (k / 32) as usize; // blocks per row
                                        // codes: 4 u32 per 32-block per row; scales: 1 bf16/block, packed 2/word.
        let codes = vec![0x76543210u32; n as usize * blocks * 4];
        let scales = vec![0x3c003c00u32; (n as usize * blocks).div_ceil(2)]; // bf16 1.0 packed
        let a = vec![0.1f32; k as usize];
        let c = vec![0f32; n as usize];
        let dims = [1u32, k, n, 0u32];

        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&a),
            usage: st,
        });
        let buf_codes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&codes),
            usage: st,
        });
        let buf_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&c),
            usage: st,
        });
        let buf_d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buf_s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&scales),
            usage: st,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_codes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_d.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_s.as_entire_binding(),
                },
            ],
        });
        let groups = n.min(32768);

        // warmup
        for _ in 0..3 {
            dispatch_once(&device, &queue, &pipe, &bg, groups);
        }

        // time: batch many dispatches per submit so per-call cost dominates, median of runs.
        // DEPENDENT mode (DEP=1): every dispatch reads c as its `a` input (c→c chain) so wgpu
        // inserts a hazard barrier between each — the real decode's residual-chain pattern. This
        // discriminates "kernel is slow" (both modes slow) from "barriers serialize" (dep slow,
        // indep fast). Default INDEPENDENT (no deps): pure kernel throughput.
        let dep = std::env::var("DEP").is_ok();
        // For the dependent chain, bind c as both input(0) and output(2) across separate dispatches
        // by alternating two bind groups (c→tmp, tmp→c) so each reads the prior's write.
        let buf_c2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vec![0f32; n as usize]),
            usage: st,
        });
        let mk_bg = |a_in: &wgpu::Buffer, c_out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_codes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_d.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buf_s.as_entire_binding(),
                    },
                ],
            })
        };
        // dep chain needs a as [n]-sized (reads c). Only valid when n<=k (so c fits a's read);
        // skip dep timing where n>k by falling back to indep.
        let bg_a = mk_bg(&buf_c, &buf_c2);
        let bg_b = mk_bg(&buf_c2, &buf_c);
        // WB=1: interleave a queue.write_buffer BEFORE each dispatch (the real forward writes a
        // uniform per dispatch). This forces the encoder to close/reopen around the queued write
        // on Metal — the suspected real-path serializer the pure-dispatch loop doesn't replicate.
        let wb = std::env::var("WB").is_ok();
        const ITERS: u32 = 50;
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            if wb {
                // separate encoder per dispatch with a write_buffer between (mimics alloc_write+dispatch)
                for _ in 0..ITERS {
                    queue.write_buffer(&buf_d, 0, bytemuck::cast_slice(&dims));
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut p = enc.begin_compute_pass(&Default::default());
                        p.set_pipeline(&pipe);
                        p.set_bind_group(0, &bg, &[]);
                        p.dispatch_workgroups(groups, 1, 1);
                    }
                    queue.submit(Some(enc.finish()));
                }
            } else {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&pipe);
                    for i in 0..ITERS {
                        if dep && n <= k {
                            p.set_bind_group(0, if i % 2 == 0 { &bg_a } else { &bg_b }, &[]);
                        } else {
                            p.set_bind_group(0, &bg, &[]);
                        }
                        p.dispatch_workgroups(groups, 1, 1);
                    }
                }
                queue.submit(Some(enc.finish()));
            }
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            times.push(t.elapsed().as_secs_f64() / ITERS as f64);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[times.len() / 2];
        // weight bytes: Q4_0 = 0.5 B/weight (codes) + 2 B/block scale.
        let wbytes = (n as usize * k as usize) / 2 + n as usize * blocks * 2;
        let gbps = wbytes as f64 / med / 1e9;
        println!(
            "{:32} {:>10.1} {:>10.1} {:>12.1}",
            label,
            med * 1e6,
            gbps,
            wbytes as f64 / 1e6
        );
    }
    println!("\n(M4 Max bandwidth ceiling ~400 GB/s. A healthy decode GEMV should approach it.\n 30× below = the kernel is the chat stall; near-ceiling = the stall is inter-dispatch, not this kernel.)");
}

fn dispatch_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipe: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    groups: u32,
) {
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(pipe);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
}
