//! TEMP MEASUREMENT: per-kernel-type GPU-time breakdown of the BATCHED native megakernel decode
//! frame at the REAL qwen3-coder-30B geometry, B=16 and B=32. No model load — synthetic Q4KS
//! weights (same packing as batched_mega_parity). Each kernel TYPE is timed in isolation via
//! N reps/command-buffer (GPUEndTime/N = steady-state per-call GPU ms). Reports a table of
//! per-call ms + per-LAYER ms (×dispatches-per-layer) + ×48-layer frame share, for B=16 and B=32,
//! plus the 16→32 scaling factor per kernel. Answers: which 2-3 kernels dominate, and which scale
//! ~per-row (1.9-2x = shareable-weight candidates).
//!
//! Run: ARF_MSL_GEMV=1 cargo run --release -p arf-gpu --example mega_perf_breakdown

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {}\n", ctx.info());
    let mut island = MetalIsland::new(&ctx).expect("metal island (needs Metal backend)");

    // Compile the FULL batched-megakernel island kernel set (mirror weights.rs compile order).
    let ops = include_str!("../src/shaders/metal/decode_ops_msl.metal");
    let attn = include_str!("../src/shaders/metal/attention_msl.metal");
    island
        .compile_msl(&ctx, "kv_scatter", ops, "kv_scatter")
        .expect("kv_scatter");
    island
        .compile_msl(&ctx, "attention_decode", attn, "attention_decode")
        .expect("attention_decode");
    island
        .compile_msl(&ctx, "attention_decode_b", attn, "attention_decode_b")
        .expect("attention_decode_b");
    island
        .compile_msl(&ctx, "gemv_q8", ops, "gemv_q8")
        .expect("gemv_q8");
    island
        .compile_msl(&ctx, "argmax", ops, "argmax")
        .expect("argmax");
    island
        .compile_msl(
            &ctx,
            "add_norm",
            include_str!("../src/shaders/metal/add_norm_msl.metal"),
            "add_norm",
        )
        .expect("add_norm");
    // B-row kernels.
    let ops_b = include_str!("../src/shaders/metal/decode_ops_batch_msl.metal");
    let moe_b = include_str!("../src/shaders/metal/moe_batch_msl.metal");
    island
        .compile_msl(
            &ctx,
            "gemv_q4ks_batch",
            include_str!("../src/shaders/metal/matmul_vec_q4ks_batch_msl.metal"),
            "gemv_q4ks_batch",
        )
        .expect("gemv_q4ks_batch");
    island
        .compile_msl(&ctx, "rmsnorm_b", ops_b, "rmsnorm_b")
        .expect("rmsnorm_b");
    island
        .compile_msl(&ctx, "qk_norm_b", ops_b, "qk_norm_b")
        .expect("qk_norm_b");
    island
        .compile_msl(&ctx, "rope_qk_b", ops_b, "rope_qk_b")
        .expect("rope_qk_b");
    island
        .compile_msl(&ctx, "add_residual_b", ops_b, "add_residual_b")
        .expect("add_residual_b");
    island
        .compile_msl(&ctx, "gemv_q4ks_id_b", moe_b, "gemv_q4ks_id_b")
        .expect("gemv_q4ks_id_b");
    island
        .compile_msl(&ctx, "gemv_q4ks_id_down_b", moe_b, "gemv_q4ks_id_down_b")
        .expect("gemv_q4ks_id_down_b");
    island
        .compile_msl(&ctx, "moe_route_b", moe_b, "moe_route_b")
        .expect("moe_route_b");
    island
        .compile_msl(&ctx, "moe_swiglu_b", moe_b, "moe_swiglu_b")
        .expect("moe_swiglu_b");

    // REAL qwen3-coder-30B-A3B geometry (config.rs::qwen3_coder_30b).
    let (h, nh, nkv, hd) = (2048usize, 32usize, 4usize, 128usize);
    let (inter, num_experts, top_k) = (768usize, 128usize, 8usize);
    let vocab = 151_936usize;
    let n_layers = 48usize;
    // steady-state decode context length proxy (keys per seq). Decode cost of attention scales with
    // this; pick a representative mid-context value.
    let ctx_keys = 512usize;

    let mut tables = Vec::new();
    for &b in &[16usize, 64] {
        let rows = island
            .perf_breakdown_microbench(
                &ctx,
                b,
                h,
                nh,
                nkv,
                hd,
                inter,
                num_experts,
                top_k,
                vocab,
                ctx_keys,
            )
            .expect("microbench");
        tables.push((b, rows));
    }

    // ---- print ----
    let (b16, t16) = (&tables[0].0, &tables[0].1);
    let (b32, t32) = (&tables[1].0, &tables[1].1);
    assert_eq!(*b16, 16);
    assert_eq!(*b32, 64);

    // per-layer ms = per-call ms × dispatches-per-layer (disp==0 => once-per-token, NOT ×layers).
    let per_layer = |ms: f64, disp: usize| if disp == 0 { 0.0 } else { ms * disp as f64 };
    let frame_layer = |rows: &Vec<(String, f64, usize)>| -> f64 {
        rows.iter()
            .map(|(_, ms, d)| per_layer(*ms, *d))
            .sum::<f64>()
            * n_layers as f64
    };
    let frame_once = |rows: &Vec<(String, f64, usize)>| -> f64 {
        rows.iter()
            .filter(|(_, _, d)| *d == 0)
            .map(|(_, ms, _)| *ms)
            .sum::<f64>()
    };

    let fl16 = frame_layer(t16);
    let fo16 = frame_once(t16);
    let tot16 = fl16 + fo16;
    let fl32 = frame_layer(t32);
    let fo32 = frame_once(t32);
    let tot32 = fl32 + fo32;

    println!("\n===== PER-KERNEL-TYPE GPU BREAKDOWN — qwen3-coder-30B Q4_K_S, {n_layers} layers, ctx_keys={ctx_keys} =====");
    println!("(per-call = one dispatch's GPU ms; per-layer = per-call × dispatches/layer; frame = per-layer × {n_layers} OR once/token)\n");
    println!(
        "{:<32} | {:>9} {:>9} {:>6} | {:>9} {:>9} {:>6} | {:>5}",
        "kernel", "call16ms", "frame16", "%16", "call64ms", "frame64", "%64", "16→64"
    );
    println!("{}", "-".repeat(108));
    for ((name, ms16, d16), (_, ms32, d32)) in t16.iter().zip(t32.iter()) {
        let f16 = if *d16 == 0 {
            *ms16
        } else {
            per_layer(*ms16, *d16) * n_layers as f64
        };
        let f32_ = if *d32 == 0 {
            *ms32
        } else {
            per_layer(*ms32, *d32) * n_layers as f64
        };
        let pct16 = 100.0 * f16 / tot16;
        let pct32 = 100.0 * f32_ / tot32;
        let scale = if *ms16 > 1e-9 { ms32 / ms16 } else { 0.0 };
        println!(
            "{:<32} | {:>9.4} {:>9.2} {:>5.1}% | {:>9.4} {:>9.2} {:>5.1}% | {:>5.2}x",
            name, ms16, f16, pct16, ms32, f32_, pct32, scale
        );
    }
    println!("{}", "-".repeat(108));
    println!(
        "{:<32} | {:>9} {:>9.2} {:>5} | {:>9} {:>9.2} {:>5} |",
        "PER-LAYER SUBTOTAL (×48)", "", fl16, "", "", fl32, ""
    );
    println!(
        "{:<32} | {:>9} {:>9.2} {:>5} | {:>9} {:>9.2} {:>5} |",
        "ONCE-PER-TOKEN SUBTOTAL", "", fo16, "", "", fo32, ""
    );
    println!(
        "{:<32} | {:>9} {:>9.2} {:>5} | {:>9} {:>9.2} {:>5} |",
        "FRAME TOTAL (modeled)", "", tot16, "", "", tot32, ""
    );
    println!(
        "\n16→64 frame scaling: {:.3}x (per-row would be 4.00x; real weight-reuse < 4.00)",
        tot32 / tot16
    );

    // ---- grouped: attention-block vs MoE-block vs dense-proj ----
    let group_share = |rows: &Vec<(String, f64, usize)>, total: f64, keys: &[&str]| -> f64 {
        rows.iter()
            .filter(|(n, _, _)| keys.iter().any(|k| n.contains(k)))
            .map(|(_, ms, d)| {
                if *d == 0 {
                    *ms
                } else {
                    per_layer(*ms, *d) * n_layers as f64
                }
            })
            .sum::<f64>()
            / total
            * 100.0
    };
    let dense_keys = ["q_proj", "k_proj", "v_proj", "o_proj"];
    let moe_keys = ["gate", "up (MoE", "down", "swiglu", "route"];
    let attn_keys = ["attention_decode", "kv_scatter", "qk_norm", "rope"];
    println!("\n===== BLOCK SHARES (% of modeled frame) =====");
    println!("{:<24} | {:>7} | {:>7}", "block", "B16 %", "B64 %");
    println!(
        "{:<24} | {:>6.1}% | {:>6.1}%",
        "dense q/k/v/o proj",
        group_share(t16, tot16, &dense_keys),
        group_share(t32, tot32, &dense_keys)
    );
    println!(
        "{:<24} | {:>6.1}% | {:>6.1}%",
        "MoE (gate/up/down/...)",
        group_share(t16, tot16, &moe_keys),
        group_share(t32, tot32, &moe_keys)
    );
    println!(
        "{:<24} | {:>6.1}% | {:>6.1}%",
        "attention block",
        group_share(t16, tot16, &attn_keys),
        group_share(t32, tot32, &attn_keys)
    );
    // GQA head-group attention A/B row (does NOT contain "attention_decode" so it's NOT in attn_keys
    // above — honest side-by-side vs the per-seq attention_decode row). Compare these two BEFORE
    // calling the GQA kernel a win at B16/B64.
    println!(
        "{:<24} | {:>6.1}% | {:>6.1}%",
        "gqa attn (A/B)",
        group_share(t16, tot16, &["gqa_attn"]),
        group_share(t32, tot32, &["gqa_attn"])
    );
    println!(
        "{:<24} | {:>6.1}% | {:>6.1}%",
        "router GEMV",
        group_share(t16, tot16, &["router"]),
        group_share(t32, tot32, &["router"])
    );
    println!(
        "{:<24} | {:>6.1}% | {:>6.1}%",
        "lm_head+argmax (1/tok)",
        group_share(t16, tot16, &["lm_head", "argmax"]),
        group_share(t32, tot32, &["lm_head", "argmax"])
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mega_perf_breakdown is macOS/Metal-only");
}
