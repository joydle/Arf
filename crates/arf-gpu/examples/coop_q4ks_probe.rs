//! Parity-gate the new cooperative-matrix Q4_K_S GEMM kernel
//! (`shaders/wgsl/matmul_coop_q4ks.wgsl`) against the CPU `matmul_nt_q4ks` oracle BEFORE
//! it is wired into the FLUX DiT. The kernel computes C[m,n] = A[m,k] · dequant(Q4_K_S)ᵀ
//! with the two-level u8/i8 → f32 dequant folded into the staged weight element.
//!
//! It also cross-checks the existing CPU oracle is internally consistent. If this
//! passes (max rel err < 1e-3), the coop Q4_K_S GEMM is safe to route FLUX through.
//!
//! Run: `cargo run --release -p arf-gpu --example coop_q4ks_probe`
//!   env ARF_M / ARF_N / ARF_K override dims (m,n mult of 8; k mult of 256).

use arf_core::tensor::{matmul_nt_q4ks, Q4KSMatrix};

const SRC: &str = include_str!("../src/shaders/wgsl/matmul_coop_q4ks.wgsl");

fn main() {
    let envu = |name: &str, d: u32| {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let m = envu("ARF_M", 64) as usize; // joint-seq-like
    let n = envu("ARF_N", 256) as usize; // output dim
    let k = envu("ARF_K", 256) as usize; // must be mult of 256 for Q4_K_S
    assert_eq!(m % 8, 0, "m must be a multiple of 8 (8×8 coop tile)");
    assert_eq!(n % 8, 0, "n must be a multiple of 8");
    assert_eq!(
        k % 256,
        0,
        "k must be a multiple of 256 (Q4_K_S super-block)"
    );

    // A[m,k] and a weight W[n,k] quantized to Q4_K_S.
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i % 17) as f32) * 0.021 - 0.16)
        .collect();
    let q = Q4KSMatrix::from_f32(&w_f32, n, k);

    // CPU oracle: C[m,n] = A · dequant(W)ᵀ
    let want = matmul_nt_q4ks(
        &a_data,
        m,
        k,
        q.codes(),
        q.scales(),
        q.mins(),
        q.d(),
        q.dmin(),
        n,
    );

    // dd = interleaved [d, dmin] per super-block (the layout the kernel reads).
    let mut dd: Vec<f32> = Vec::with_capacity(q.d().len() * 2);
    for (dv, dmv) in q.d().iter().zip(q.dmin()) {
        dd.push(*dv);
        dd.push(*dmv);
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no adapter");
    println!("adapter: {:?}", adapter.get_info().name);
    if !adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        println!("coop matrix unavailable; aborting (kernel is gated on has_coop anyway)");
        return;
    }
    let alim = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coop-q4ks"),
        required_features: wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
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

    use wgpu::util::DeviceExt;
    let mk_f32 = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };
    let mk_u32 = |data: &[u32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };
    // scales (u8) / mins (i8) are read by the kernel as array<u32> (4 per word) — pack the raw
    // bytes into u32 words, exactly as storage_init_u8 / the i8 packer do on the LLM path.
    let pack_u8 = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .chunks(4)
            .map(|c| {
                let mut w = 0u32;
                for (b, &x) in c.iter().enumerate() {
                    w |= (x as u32) << (8 * b);
                }
                w
            })
            .collect()
    };
    let scales_u32 = pack_u8(q.scales());
    let mins_u32 = pack_u8(&q.mins().iter().map(|&x| x as u8).collect::<Vec<u8>>());

    let buf_a = mk_f32(&a_data);
    let buf_codes = mk_u32(q.codes());
    let buf_c = mk_f32(&vec![0.0f32; m * n]);
    let dims = [m as u32, k as u32, n as u32, 0u32];
    let buf_d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_scales = mk_u32(&scales_u32);
    let buf_mins = mk_u32(&mins_u32);
    let buf_dd = mk_f32(&dd);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coop-q4ks"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coop-q4ks"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
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
                resource: buf_scales.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_mins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf_dd.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipe);
        p.set_bind_group(0, &bg, &[]);
        // grid: one workgroup per 8×8 output tile (workgroup_id.x = row tiles, .y = col tiles).
        p.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
    }
    let size = (m * n * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);

    let mut max_err = 0.0f32;
    let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
    for i in 0..m {
        for j in 0..n {
            let g = got[i * n + j];
            let w = want[i * n + j];
            let e = (g - w).abs() / w.abs().max(1.0);
            if e > max_err {
                max_err = e;
                worst = (i, j, g, w);
            }
        }
    }
    println!("coop Q4_K_S GEMM  m={m} n={n} k={k}  max rel err {max_err:.2e}  (worst [{},{}] got={:.4} want={:.4})",
             worst.0, worst.1, worst.2, worst.3);
    if max_err < 1e-3 {
        println!("PASS — matmul_coop_q4ks matches the CPU matmul_nt_q4ks oracle. Safe to wire into FLUX.");
        std::process::exit(0);
    } else {
        println!("FAIL — mismatch; do NOT wire into the pipeline.");
        std::process::exit(1);
    }
}
