//! Probe the integration-critical questions for wiring coop-matrix into Arf's
//! matmul, which is C[M,N] = A[M,K] · Bᵀ with the weight stored [N,K] row-major
//! (NT layout). A direct storage `coopLoad` can't transpose via stride, so we
//! stage tiles through workgroup memory (transposing the weight in the scalar
//! copy) and `coopLoad` from there. This probe answers:
//!
//! 1. Does naga accept `coopLoad` from `var<workgroup>` memory?
//! 2. Does the shared-mem-staged NT GEMM match a CPU reference?
//!
//! If both pass, the C3 forward_batch wiring is unblocked.
//!
//! Run: `cargo run --release -p arf-gpu --example coop_nt_probe`

const NT_SRC: &str = r#"
enable wgpu_cooperative_matrix;
const T: u32 = 8u;
@group(0) @binding(0) var<storage, read>       a: array<f32>; // [M,K] row-major
@group(0) @binding(1) var<storage, read>       b: array<f32>; // [N,K] row-major (Bᵀ / weight)
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // [M,N] row-major
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // M,N,K,_

var<workgroup> as_t: array<f32, 64>; // 8x8 A tile [mm,kk] row-major
var<workgroup> bs_t: array<f32, 64>; // 8x8 logical-B tile [kk,nn] (transposed from Bᵀ)

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let N = dims.y;
    let K = dims.z;
    let row = wid.x * T; // output row base (m)
    let col = wid.y * T; // output col base (n)
    let lr = lid.x;      // 0..7
    let lc = lid.y;      // 0..7

    var acc = coopLoad<coop_mat8x8<f32, C>>(&c[row * N + col], N); // c pre-zeroed
    var k = 0u;
    loop {
        if (k >= K) { break; }
        // A[row+lr, k+lc] -> as_t[lr,lc]
        as_t[lr * T + lc] = a[(row + lr) * K + (k + lc)];
        // logical B[k+lr, col+lc] = Bᵀ[col+lc, k+lr] -> bs_t[lr,lc]  (transpose here)
        bs_t[lr * T + lc] = b[(col + lc) * K + (k + lr)];
        workgroupBarrier();
        let at = coopLoad<coop_mat8x8<f32, A>>(&as_t[0], T);
        let bt = coopLoad<coop_mat8x8<f32, B>>(&bs_t[0], T);
        acc = coopMultiplyAdd(at, bt, acc);
        workgroupBarrier();
        k = k + T;
    }
    coopStore(acc, &c[row * N + col], N);
}
"#;

fn main() {
    // ARF_M/N/K override the GEMM dims so the kill-check can SWEEP K to test
    // whether the coop NT GEMM diverges past K=256 (the dispatch.rs COOP_MAX_K
    // claim). m,n must be multiples of 8 (the 8×8 tile); k must be a multiple of 8.
    let envu = |name: &str, d: u32| {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let m = envu("ARF_M", 64);
    let n = envu("ARF_N", 64);
    let k = envu("ARF_K", 64);

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
        println!("coop matrix unavailable; aborting");
        return;
    }

    let alim = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coop-nt"),
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

    // A[M,K], B stored [N,K] (the weight). C = A·Bᵀ.
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
    let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.03 - 0.1).collect();
    let c_zero = vec![0.0f32; (m * n) as usize];

    use wgpu::util::DeviceExt;
    let mk = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };
    let buf_a = mk(&a_data);
    let buf_b = mk(&b_data);
    let buf_c = mk(&c_zero);
    let dims = [m, n, k, 0u32];
    let buf_d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coop-nt"),
        source: wgpu::ShaderSource::Wgsl(NT_SRC.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coop-nt"),
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
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipe);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(m / 8, n / 8, 1);
    }
    let size = (m * n) as u64 * 4;
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

    // CPU reference: C[m,n] = Σ_k A[m,k]·B[n,k]
    let mut max_err = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a_data[(i * k + kk) as usize] * b_data[(j * k + kk) as usize];
            }
            let e = (got[(i * n + j) as usize] - s).abs() / s.abs().max(1.0);
            max_err = max_err.max(e);
        }
    }
    println!(
        "NT coop GEMM (shared-mem staged, coopLoad from workgroup): max rel err {max_err:.2e}"
    );
    if max_err < 1e-4 {
        println!("PASS — coopLoad-from-workgroup + NT transpose works. C3 unblocked.");
    } else {
        println!("FAIL — result mismatch; need a different transpose strategy.");
    }
}
