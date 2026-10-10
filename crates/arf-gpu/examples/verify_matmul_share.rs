//! What share of one 8-row verify pass is Q4 matmul? (2026-09-23)
//!
//! The in-serving probe (`ARF_BMEGA_GPUTIME`) says a k=8 verify of the 27B is ~90 ms of GPU time
//! and ~97% GPU-busy, against a ~36-39 ms weight-read floor. Before building anything, size the
//! share (size the share of the whole first): time each projection shape of the verify census through
//! `mpp_q4_selftest` — which dispatches through `q4rm_encode`, the routine the verify record
//! calls — and multiply by how many times one pass runs it. `matmul total / 90 ms` decides
//! whether the next kernel work goes into the matmuls or into everything else (GDN scan,
//! attention, norms, conv).
//!
//! Random weights of the real shapes: a Q4 matmul's time is its bytes and its tiles, not the
//! values. Median of several runs per shape (the first touches cold buffers).
//!
//! Run: ARF_SG_Q4=1 cargo run --release -p arf-gpu --example verify_matmul_share
#[cfg(target_os = "macos")]
fn main() {
    use arf_core::tensor::Q4KSMatrix;
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;

    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island");
    let rows: usize = std::env::var("VMS_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reps: usize = std::env::var("VMS_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    // (label, out, in, count per verify pass) — the census `[gputime] matmuls this record` of a
    // k=8 verify of Qwen3.8-27B (64 layers: 48 GDN + 16 attention).
    let shapes = [
        ("ffn gate/up 5120->17408", 17408usize, 5120usize, 128usize),
        ("ffn down   17408->5120", 5120, 17408, 64),
        ("out proj    6144->5120", 5120, 6144, 64),
        ("gdn qkv    5120->10240", 10240, 5120, 48),
        ("gdn z      5120->6144", 6144, 5120, 48),
        ("attn k/v   5120->1024", 1024, 5120, 42),
        ("attn q     5120->12288", 12288, 5120, 16),
    ];

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };

    let (mut total_ms, mut total_bytes) = (0.0f64, 0.0f64);
    println!(
        "rows={rows} reps={reps} ARF_SG_Q4={}",
        std::env::var_os("ARF_SG_Q4").is_some()
    );
    for (label, n, k, count) in shapes {
        let wf: Vec<f32> = (0..n * k).map(|_| rnd() * 0.05).collect();
        let q = Q4KSMatrix::from_f32(&wf, n, k);
        let x: Vec<f32> = (0..rows * k).map(|_| rnd() * 4.0).collect();
        let mut t = Vec::with_capacity(reps);
        for _ in 0..reps {
            match island.mpp_q4_selftest(&ctx, &q, &x, rows) {
                Ok((_, gpu_s, _)) => t.push(gpu_s * 1e3),
                Err(e) => {
                    println!("{label:26} | not on the tiled path: {e}");
                    break;
                }
            }
        }
        if t.is_empty() {
            continue;
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = t[t.len() / 2];
        // Q4_K_S bytes the kernel must read: 4-bit codes + a u8 scale and u8 min per 32 + f16 d/dmin per 256.
        let bytes = (n * k) as f64 * 0.5 + (n * k / 32) as f64 * 2.0 + (n * k / 256) as f64 * 4.0;
        let gbs = bytes / (med * 1e-3) / 1e9;
        let share = med * count as f64;
        total_ms += share;
        total_bytes += bytes * count as f64;
        println!(
            "{label:26} | median {med:7.3} ms (min {:7.3}) | {gbs:6.1} GB/s | x{count:3} = {share:6.2} ms",
            t[0]
        );
    }
    println!(
        "Q4 projection matmuls per verify pass: {total_ms:.1} ms for {:.2} GB -> {:.0} GB/s effective",
        total_bytes / 1e9,
        total_bytes / (total_ms * 1e-3) / 1e9
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("verify_matmul_share: macOS only");
}
