//! M4 GATE: load the real Qwen3.6-27B (`qwen35`) hybrid model and run ONE forward over a
//! fixed prompt, then print the top-5 logits + the greedy token. Verifies SANE logits
//! (non-NaN, plausible distribution) — and, for "The capital of France is", ideally greedy
//! → "Paris".
//!
//! This drives the CPU REFERENCE hybrid forward (`forward_cpu_reference`): the faithful
//! bit-for-bit mirror of the M2/M3 GPU kernels' math (ssm_conv1d, gated-delta-net recurrence,
//! l2_norm/softplus/sigmoid prologue) plus the gated-GQA path + dense SwiGLU FFN. Weights are
//! read LAZILY from the GGUF mmap (one tensor dequantised at a time, then dropped), so the
//! peak host memory stays well under the model's 16 GB on-disk size.
//!
//! Run: cargo run --release -p arf-gpu --example qwen35_forward_probe -- [gguf] [prompt]

use arf_core::config::ModelConfig;
use arf_core::model::gguf::LazyGguf;
use arf_core::tokenizer::Tokenizer;
use arf_gpu::gpu::ssm_qwen35::{forward_cpu_reference, read_hparams};

/// top-5 helper: returns (idx, logit) for the 5 highest logits.
fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.iter().take(5).map(|&i| (i, logits[i])).collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_S.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());

    println!("opening GGUF: {path}");
    let g = LazyGguf::open_raw(std::path::Path::new(&path)).expect("open_raw");
    let cfg = ModelConfig::qwen35_27b();
    let hp = read_hparams(&g, &cfg).expect("read_hparams");
    println!(
        "hparams: n_embd={} n_head={} n_head_kv={} head_dim={} rope_dim={} theta={} layers={} (nextn {}) vocab={}",
        hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_embd_head_k, hp.rope_dim, hp.rope_theta,
        hp.n_layer, hp.n_layer_nextn, hp.n_vocab
    );
    let geom = hp.linear_geom();
    println!(
        "ssm geom: head_k={} head_v={} n_k_heads={} n_v_heads={} key_dim={} value_dim={} conv_dim={} d_conv={}",
        geom.head_k_dim, geom.head_v_dim, geom.num_k_heads, geom.num_v_heads,
        geom.key_dim, geom.value_dim, geom.conv_dim, geom.d_conv
    );

    // Tokenize (GGUF-embedded tokenizer). Add BOS so the prompt matches a normal chat prefill.
    let tok = Tokenizer::from_gguf_path(&path).expect("tokenizer");
    let mut ids = tok.encode(&prompt, false).expect("encode");
    let bos = g
        .get_metadata_u32("tokenizer.ggml.bos_token_id")
        .unwrap_or(248044);
    ids.insert(0, bos);
    println!("prompt: {prompt:?}\ntokens ({}): {:?}", ids.len(), ids);

    // Which path? GPU (default, the M4b gate) runs the island; --cpu forces the CPU oracle only.
    let gpu = std::env::args().all(|a| a != "--cpu");

    // ---- CPU ORACLE (the gate) ----
    let t0 = std::time::Instant::now();
    let logits = forward_cpu_reference(&g, &hp, &ids).expect("forward");
    let secs = t0.elapsed().as_secs_f64();
    println!("CPU forward done in {secs:.1}s, vocab={}", logits.len());

    let nan = logits.iter().filter(|v| !v.is_finite()).count();
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &logits {
        if v.is_finite() {
            mn = mn.min(v);
            mx = mx.max(v);
        }
    }
    println!("CPU non-finite logits: {nan} ; range [{mn:.3}, {mx:.3}]");
    let cpu_top = top5(&logits);
    println!("=== CPU ORACLE TOP-5 ===");
    for &(i, l) in &cpu_top {
        let dec = tok.decode(&[i as u32], false).unwrap_or_default();
        println!("  id={i:7}  logit={l:.4}  tok={dec:?}");
    }
    let cpu_greedy = cpu_top[0].0;
    let cpu_gdec = tok.decode(&[cpu_greedy as u32], false).unwrap_or_default();
    println!("CPU GREEDY: id={cpu_greedy} {cpu_gdec:?}");
    assert_eq!(nan, 0, "non-finite CPU logits — forward produced NaN/inf");

    if !gpu {
        println!("\nM4 GATE (CPU only): forward ran, logits finite. greedy={cpu_gdec:?}");
        return;
    }

    // ---- GPU FORWARD (M4b: the beast runs ON THE GPU island) ----
    #[cfg(target_os = "macos")]
    {
        use arf_gpu::gpu::concurrent_metal::MetalIsland;
        use arf_gpu::gpu::ssm_qwen35::forward_gpu_reference;
        use arf_gpu::gpu::GpuContext;

        let ctx = GpuContext::new().expect("gpu context");
        let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");

        let t1 = std::time::Instant::now();
        let glogits = forward_gpu_reference(&g, &hp, &ids, &mut island, &ctx).expect("gpu forward");
        let gsecs = t1.elapsed().as_secs_f64();
        println!("\nGPU forward done in {gsecs:.1}s, vocab={}", glogits.len());

        let gnan = glogits.iter().filter(|v| !v.is_finite()).count();
        println!("GPU non-finite logits: {gnan}");
        let gpu_top = top5(&glogits);

        println!("=== GPU vs CPU TOP-5 (side by side) ===");
        println!(
            "  {:>5}  {:>10}  {:>10}  {:>8}  tok",
            "rank", "GPU", "CPU", "diff"
        );
        for r in 0..5 {
            let (gi, gl) = gpu_top[r];
            let (ci, cl) = cpu_top[r];
            let dec = tok.decode(&[gi as u32], false).unwrap_or_default();
            // CPU logit for the GPU's r-th token (compare like-for-like by id).
            let cl_for_gi = glogits.get(gi).copied().unwrap_or(f32::NAN);
            let _ = cl_for_gi;
            println!(
                "  {r:>5}  id={gi:7} {gl:9.4}  id={ci:7} {cl:9.4}  {:>8.4}  {dec:?}",
                (gl - logits[gi])
            );
        }
        let gpu_greedy = gpu_top[0].0;
        let gpu_gdec = tok.decode(&[gpu_greedy as u32], false).unwrap_or_default();
        println!("GPU GREEDY: id={gpu_greedy} {gpu_gdec:?}");

        // cosine similarity of the full logit vectors (soft gate > 0.999).
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..glogits.len().min(logits.len()) {
            let a = glogits[i] as f64;
            let b = logits[i] as f64;
            if a.is_finite() && b.is_finite() {
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        let max_top1_diff = (gpu_top[0].1 - logits[gpu_top[0].0]).abs();
        println!("\nlogit cosine(GPU,CPU) = {cos:.6} ; |greedy logit diff| = {max_top1_diff:.4}");

        assert_eq!(
            gnan, 0,
            "non-finite GPU logits — GPU forward produced NaN/inf"
        );
        let paris = gpu_greedy == cpu_greedy;
        println!(
            "M4b GATE: greedy GPU=={cpu:?} CPU=={cpu:?} → {verdict} (cosine {cos:.5})",
            cpu = cpu_gdec,
            verdict = if paris {
                "PARIS MATCH ✓"
            } else {
                "DIVERGED ✗"
            }
        );
        assert!(paris, "GPU greedy {gpu_gdec:?} != CPU greedy {cpu_gdec:?}");
        assert!(cos > 0.999, "logit cosine {cos:.6} below 0.999 soft gate");
        println!(
            "\nM4b GATE PASS: the beast runs on the GPU. greedy={gpu_gdec:?}, cosine {cos:.6}"
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        println!("GPU forward is macOS-only (Metal island). CPU oracle ran above.");
    }
}
