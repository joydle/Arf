//! DFlash 2 port, STAGE 1 GATE: load the draft checkpoint, transcode
//! every projection bf16 -> Q4_K_S onto the GPU, and report what was loaded, what it costs, and
//! how much the 4-bit transcode moved the weights. Exits non-zero if any published tensor is
//! missing or mis-shaped, or if the transcode error is implausible.
//!
//! Run: cargo run --release -p arf-gpu --example dflash2_load -- models/qwen3.8-27b-dflash2
#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::metal::dflash2::Dflash2Draft;
    use arf_gpu::gpu::GpuContext;
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen3.8-27b-dflash2".into());
    let ctx = GpuContext::new().expect("gpu context");
    let t = std::time::Instant::now();
    let d = match Dflash2Draft::load(&ctx, std::path::Path::new(&dir)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    };
    let c = &d.cfg;
    println!("loaded {dir} in {:.1} s", t.elapsed().as_secs_f64());
    println!(
        "  {} layers, hidden {}, {} q / {} kv heads x {}, mlp {}, vocab {}",
        c.layers, c.hidden, c.heads, c.kv_heads, c.head_dim, c.intermediate, c.vocab
    );
    println!(
        "  taps target layers {:?} -> fc [{}, {}]; block of {} (mask token {}); conv groups of {} -> {} dynamic weights a row",
        c.target_layer_ids,
        c.hidden,
        c.target_layer_ids.len() * c.hidden,
        c.block_size,
        c.mask_token_id,
        c.conv_group_size,
        c.dynamic_size()
    );
    println!(
        "  selector rank {} over top-{}; window {}; rope theta {:.0}; eps {:e}",
        c.selector_rank, c.selector_top_k, c.sliding_window, c.rope_theta, c.rms_eps
    );
    println!(
        "  on the GPU: {:.2} GB (3.58 GB as published bf16)",
        d.gpu_bytes() as f64 / 2f64.powi(30)
    );
    println!(
        "  worst bf16 -> Q4_K_S relative RMS error: {:.4} ({})",
        d.worst_quant_error.0, d.worst_quant_error.1
    );
    // Q4_K on a trunk weight moves it ~5-8% RMS; a transcode that moved one 30% is a bug.
    if !(d.worst_quant_error.0 > 0.0 && d.worst_quant_error.0 < 0.2) {
        eprintln!("FAILED: implausible transcode error");
        std::process::exit(1);
    }
    println!("STAGE 1 GATE: PASS");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dflash2_load needs the Metal island (macOS).");
}
