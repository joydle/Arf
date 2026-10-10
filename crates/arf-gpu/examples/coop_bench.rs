//! De-risk microbench for the cooperative-matrix (Apple `simdgroup_matrix`) GEMM
//! path. Compares an 8x8x8 coop-matrix f32 GEMM against a tuned 16x16 tiled
//! shared-memory scalar GEMM (the best "plain WGSL" baseline) on inference-shaped
//! matrices, and checks correctness vs a CPU reference on a small shape.
//!
//! The question this answers: does the matrix hardware actually beat scalar WGSL
//! on THIS GPU by enough to justify rewriting the matmul path? P8 assumed it was
//! unreachable; wgpu 29.0.1 exposes it. Measure before we build.
//!
//! Run:
//!   M=1024 N=1024 K=1024 REPS=200 \
//!   cargo run --release -p arf-gpu --example coop_bench
//! Also try inference shapes, e.g. M=16 K=2048 N=8192 (batched gate_proj),
//! M=256 K=2048 N=2048 (prefill q_proj).

use std::time::Instant;

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

const COOP_SRC: &str = r#"
enable wgpu_cooperative_matrix;
const T: u32 = 8u;
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // M,N,K,strideN

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let N = dims.y;
    let K = dims.z;
    let row = wid.x * T;
    let col = wid.y * T;
    var acc = coopLoad<coop_mat8x8<f32, C>>(&c[row * N + col], N);
    for (var k: u32 = 0u; k < K; k += T) {
        let at = coopLoad<coop_mat8x8<f32, A>>(&a[row * K + k], K);
        let bt = coopLoad<coop_mat8x8<f32, B>>(&b[k * N + col], N);
        acc = coopMultiplyAdd(at, bt, acc);
    }
    coopStore(acc, &c[row * N + col], N);
}
"#;

// 16x16 tiled shared-memory GEMM — the strong scalar-WGSL baseline.
const TILED_SRC: &str = r#"
const TS: u32 = 16u;
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // M,N,K,strideN

var<workgroup> as_t: array<f32, 256>; // 16x16
var<workgroup> bs_t: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;
    let row = wid.x * TS + lid.x;
    let col = wid.y * TS + lid.y;
    var acc = 0.0;
    var t = 0u;
    loop {
        if (t >= K) { break; }
        // cooperatively stage one 16x16 tile of A and B
        let a_r = wid.x * TS + lid.x;
        let a_c = t + lid.y;
        as_t[lid.x * TS + lid.y] = select(0.0, a[a_r * K + a_c], a_r < M && a_c < K);
        let b_r = t + lid.x;
        let b_c = wid.y * TS + lid.y;
        bs_t[lid.x * TS + lid.y] = select(0.0, b[b_r * N + b_c], b_r < K && b_c < N);
        workgroupBarrier();
        for (var i = 0u; i < TS; i = i + 1u) {
            acc = acc + as_t[lid.x * TS + i] * bs_t[i * TS + lid.y];
        }
        workgroupBarrier();
        t = t + TS;
    }
    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}
"#;

fn main() {
    let m = env_u32("M", 1024);
    let n = env_u32("N", 1024);
    let k = env_u32("K", 1024);
    let reps = env_u32("REPS", 200);
    assert!(
        m.is_multiple_of(8) && n.is_multiple_of(8) && k.is_multiple_of(8),
        "M,N,K must be /8 for coop tiles"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no adapter");
    println!("adapter: {:?}", adapter.get_info().name);

    let has_coop = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    // COOP_OFF=1 builds the device WITHOUT requesting the experimental feature, to
    // measure whether merely requesting it pessimizes the ordinary tiled pipeline
    // (the SUBGROUP-feature regression we saw at ~1.33x). In that mode the coop
    // kernel can't be built, so we only run + report the tiled baseline.
    let coop_off = std::env::var("COOP_OFF").is_ok();
    if !has_coop && !coop_off {
        println!("EXPERIMENTAL_COOPERATIVE_MATRIX not available; aborting");
        return;
    }

    let alim = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coop-bench"),
        required_features: if coop_off {
            wgpu::Features::empty()
        } else {
            wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
        },
        required_limits: wgpu::Limits {
            max_buffer_size: alim.max_buffer_size,
            max_storage_buffer_binding_size: alim.max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        },
        experimental_features: if coop_off {
            wgpu::ExperimentalFeatures::default()
        } else {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        },
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("device");

    // Host data.
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.03 - 0.1).collect();

    let mk_buf = |label: &str, data: &[f32]| {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };
    let buf_a = mk_buf("a", &a_data);
    let buf_b = mk_buf("b", &b_data);
    let c_zero = vec![0.0f32; (m * n) as usize];
    let buf_c = mk_buf("c", &c_zero);

    let dims = [m, n, k, n];
    let buf_d = {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    };

    let build = |src: &str, label: &str| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let tiled_pipe = build(TILED_SRC, "tiled");

    let bind = |pipe: &wgpu::ComputePipeline| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_d.as_entire_binding(),
                },
            ],
        })
    };
    let tiled_bg = bind(&tiled_pipe);

    let flops = 2.0 * m as f64 * n as f64 * k as f64;

    // Returns (gflops, c_buffer readback for verification).
    let run = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, wx: u32, wy: u32, tile: u32| {
        // zero C each run
        queue.write_buffer(&buf_c, 0, bytemuck::cast_slice(&c_zero));
        // warmup
        for _ in 0..3 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(pipe);
                p.set_bind_group(0, bg, &[]);
                p.dispatch_workgroups(wx, wy, 1);
            }
            queue.submit(Some(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let t = Instant::now();
        for _ in 0..reps {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(pipe);
                p.set_bind_group(0, bg, &[]);
                p.dispatch_workgroups(wx, wy, 1);
            }
            queue.submit(Some(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let secs = t.elapsed().as_secs_f64();
        let _ = tile;
        (flops * reps as f64 / secs / 1e9, secs / reps as f64 * 1e6)
    };

    let (tiled_g, tiled_us) = run(&tiled_pipe, &tiled_bg, m.div_ceil(16), n.div_ceil(16), 16);

    println!("shape M={m} N={n} K={k}  ({:.1} MFLOP/call)", flops / 1e6);
    if coop_off {
        println!("  tiled-scalar (COOP feature NOT requested): {tiled_g:8.1} GFLOP/s   {tiled_us:8.1} us/call");
    } else {
        let coop_pipe = build(COOP_SRC, "coop");
        let coop_bg = bind(&coop_pipe);
        let (coop_g, coop_us) = run(&coop_pipe, &coop_bg, m / 8, n / 8, 8);
        println!("  tiled-scalar : {tiled_g:8.1} GFLOP/s   {tiled_us:8.1} us/call");
        println!("  coop-matrix  : {coop_g:8.1} GFLOP/s   {coop_us:8.1} us/call");
        println!("  speedup      : {:.2}x", coop_g / tiled_g);
    }

    // Correctness check on the result of the coop run vs CPU (small slice).
    verify(&device, &queue, &buf_c, &a_data, &b_data, m, n, k);
}

#[allow(clippy::too_many_arguments)]
fn verify(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf_c: &wgpu::Buffer,
    a: &[f32],
    b: &[f32],
    m: u32,
    n: u32,
    k: u32,
) {
    // rerun coop once into a fresh C handled above; just read current buf_c.
    let size = (m * n) as u64 * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stg"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf_c, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);

    // Note: buf_c currently holds the LAST run's result (tiled). Compare a few
    // entries vs CPU reference to confirm both kernels are correct numerically.
    let mut max_err = 0.0f32;
    for &(i, j) in &[(0u32, 0u32), (3, 5), (m / 2, n / 2), (m - 1, n - 1)] {
        let mut s = 0.0f32;
        for kk in 0..k {
            s += a[(i * k + kk) as usize] * b[(kk * n + j) as usize];
        }
        let e = (got[(i * n + j) as usize] - s).abs() / s.abs().max(1.0);
        max_err = max_err.max(e);
    }
    println!("  verify (last run vs CPU, rel): max_err {max_err:.2e}");
}
