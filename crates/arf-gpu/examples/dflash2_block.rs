//! DFlash 2 port, STAGE 4 — what the BLOCK FORWARD costs, by subtraction.
//!
//! Prefills a prompt on the 27B with the draft attached (so the context ring holds `CTX`
//! positions), then times the 8-row block forward whole and with one stage cut at a time,
//! interleaved round-robin in one process. "whole - cut" is that stage's cost; the proposals
//! under a cut are garbage and are never used. The correctness of the block is NOT gated here —
//! it is gated end to end: `scripts/spec_identity_check.py` (text identical to greedy) and the
//! acceptance the daemon logs under `ARF_SPEC_DEBUG`.
//!
//! `--ref`: instead of timing, run ONE block on the GPU and the same block through the plain CPU
//! reference (`dflash2_ref.rs`, written from z-lab/dflash's `model.py`) on the same embeddings,
//! the same ring and the same Q4_K_S weights, and compare the final normed hidden row by row.
//! This is the gate that separates "the block forward has a bug" from "the draft is only this
//! good at 4 bits".
//!
//! Run: cargo run --release -p arf-gpu --example dflash2_block -- [CTX=128] [draft dir] [gguf] [--ref]

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::backend::BatchedBackend;
    use arf_core::config::{EngineConfig, KvQuant, ModelConfig, Quant};
    use arf_core::engine::build_forward;
    use arf_core::sampling::{SamplingParams, SeqSampling};
    use arf_core::scheduler::{Request, Scheduler};
    use arf_gpu::gpu::metal::dflash2_block::{
        DFLASH_CUT_ATTN_BLOCK, DFLASH_CUT_ATTN_KERNEL, DFLASH_CUT_ATTN_MATMULS, DFLASH_CUT_LM_HEAD,
        DFLASH_CUT_MLP, DFLASH_CUT_MLP_MATMULS,
    };
    use arf_gpu::gpu::GpuContext;
    use arf_gpu::WgpuBatched;
    use std::sync::Arc;

    let arg = |i: usize, d: &str| std::env::args().nth(i).unwrap_or_else(|| d.into());
    let n_ctx: usize = arg(1, "128").parse().expect("CTX");
    let dir = std::path::PathBuf::from(arg(2, "models/qwen3.8-27b-dflash2"));
    let gguf = arg(3, "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf");
    arf_gpu::apply_fast_path_defaults("dflash2_block");
    std::env::remove_var("ARF_KV_F16");
    std::env::set_var("ARF_NO_KV_F16", "1");

    let cfg = ModelConfig::qwen35_27b();
    let (block_size, num_blocks) = (16usize, (n_ctx + 64).div_ceil(16));
    let ctx = Arc::new(GpuContext::new().expect("gpu context"));
    eprintln!("loading {gguf} ...");
    let model = arf_gpu::weights::load_gguf_gpu_kv_quant_pooled(
        &cfg,
        std::path::Path::new(&gguf),
        block_size * num_blocks,
        &ctx,
        Quant::Q4KS,
        KvQuant::None,
        Some(num_blocks),
    )
    .expect("load gguf");
    let backend = WgpuBatched(model);
    backend.0.dflash_attach(&dir).expect("attach draft");

    let prompt: Vec<u32> = (0..n_ctx)
        .map(|i| ((i * 7919 + 104_729) % 50_000 + 1_000) as u32)
        .collect();
    let mut sched = Scheduler::new(EngineConfig {
        block_size,
        num_blocks,
        max_batch_size: 1,
        max_prefill_tokens: 512,
        enable_prefix_cache: false,
        ..Default::default()
    });
    sched.add(Request::new(1, prompt, SamplingParams::greedy(1)));
    let g = SamplingParams::greedy(1);
    let mut first = 0u32;
    while sched.has_unfinished() {
        let Some(plan) = sched.schedule().unwrap() else {
            break;
        };
        let (ids, batch) = build_forward(&plan, sched.block_size());
        let samp: Vec<SeqSampling> = plan
            .seqs
            .iter()
            .map(|sp| SeqSampling {
                params: &g,
                position: sp.past_len + sp.q_len,
                generated: &[],
            })
            .collect();
        let toks = backend.step(&ids, &batch, &samp).unwrap();
        for o in sched.commit_tokens(&toks).unwrap() {
            first = o.token;
        }
    }

    if std::env::args().any(|a| a == "--ref") {
        use arf_gpu::gpu::metal::dflash2_ref::Dflash2BlockReference;
        let draft = backend.0.dflash_draft(first, n_ctx, 1);
        let isl = backend.0.island.as_ref().unwrap().lock().unwrap();
        let (_, gpu) = isl.dflash_block_debug().expect("block debug");
        let dc = arf_gpu::gpu::metal::dflash2::Dflash2Config::from_json(
            &std::fs::read_to_string(dir.join("config.json")).unwrap(),
        )
        .unwrap();
        let (mut rk, mut rv) = (vec![], vec![]);
        for l in 0..dc.layers {
            let (k, v) = isl.dflash_read_ring(l).expect("ring");
            rk.push(k);
            rv.push(v);
        }
        drop(isl);
        let mut toks = vec![dc.mask_token_id; dc.block_size];
        toks[0] = first;
        let embeds = backend.0.dflash_embed_rows(&toks).expect("embed rows");
        eprintln!("loading the CPU reference weights ...");
        let refm = Dflash2BlockReference::load(&dir, &dc, false).expect("reference");
        let t = std::time::Instant::now();
        let cpu = refm.forward(&embeds, &rk, &rv, n_ctx);
        eprintln!("cpu block forward: {:.1} s", t.elapsed().as_secs_f64());
        let h = dc.hidden;
        let mut worst = 0.0f64;
        for r in 0..dc.block_size {
            let (a, b) = (&cpu[r * h..][..h], &gpu[r * h..][..h]);
            let (mut e, mut n, mut dot, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                e += (*x as f64 - *y as f64).powi(2);
                n += (*x as f64).powi(2);
                nb += (*y as f64).powi(2);
                dot += *x as f64 * *y as f64;
            }
            let rel = (e / n).sqrt();
            worst = worst.max(rel);
            println!(
                "row {r}: final hidden GPU vs CPU reference: relative RMS error {rel:.3e}, cosine {:.6}",
                dot / (n.sqrt() * nb.sqrt())
            );
        }
        if std::env::args().any(|a| a == "--published") {
            // What 4 bits cost the draft's OUTPUT, and WHICH tensors: the same block with one
            // group of projections at a time kept at the published bf16, against the block with
            // ALL of them published. (The ring was still built by the 4-bit k/v projections.)
            drop(refm);
            eprintln!("loading the PUBLISHED bf16 weights (~6 GB of f32) ...");
            let full = Dflash2BlockReference::load(&dir, &dc, true)
                .expect("published")
                .forward(&embeds, &rk, &rv, n_ctx);
            let err = |x: &[f32]| {
                let (mut e, mut n) = (0.0f64, 0.0f64);
                for (a, b) in full.iter().zip(x) {
                    e += (*a as f64 - *b as f64).powi(2);
                    n += (*a as f64).powi(2);
                }
                (e / n).sqrt()
            };
            println!("final hidden vs the all-bf16 block, relative RMS over the 8 rows:");
            println!("  all Q4_K_S (what ships)              {:.3}", err(&cpu));
            let groups: [(&str, &[&str]); 5] = [
                ("conv kernel_projection at bf16", &["kernel_projection"]),
                ("attention q/k/v/o at bf16", &["self_attn"]),
                (
                    "mlp gate/up/down at bf16",
                    &["mlp.gate", "mlp.up", "mlp.down"],
                ),
                ("mlp down only at bf16", &["mlp.down"]),
                (
                    "kernel_projection + attention",
                    &["kernel_projection", "self_attn"],
                ),
            ];
            for (label, pats) in groups {
                let m = Dflash2BlockReference::load_mixed(&dir, &dc, &|n| {
                    pats.iter().any(|p| n.contains(p))
                })
                .expect("mixed");
                println!(
                    "  {label:36} {:.3}",
                    err(&m.forward(&embeds, &rk, &rv, n_ctx))
                );
            }
        }
        println!("GPU draft (selector): {draft:?}");
        // Five layers of MPP matmuls with half-narrowed activations: a correct block sits at a
        // few 1e-3; a wrong conv tap, position, head order or residual sits near 1.
        println!("worst relative RMS error: {worst:.3e}  (gate: < 3e-2)");
        if worst < 3e-2 {
            println!("BLOCK REFERENCE GATE: PASS");
        } else {
            println!("BLOCK REFERENCE GATE: FAIL");
            std::process::exit(1);
        }
        return;
    }

    let arms: [(&str, u32); 8] = [
        ("whole block", 0),
        ("- lm_head", DFLASH_CUT_LM_HEAD),
        ("- attention kernel", DFLASH_CUT_ATTN_KERNEL),
        ("- attention sub-block", DFLASH_CUT_ATTN_BLOCK),
        ("- MLP sub-block", DFLASH_CUT_MLP),
        ("- MLP matmuls only", DFLASH_CUT_MLP_MATMULS),
        ("- attention matmuls only", DFLASH_CUT_ATTN_MATMULS),
        (
            "- all of the above",
            DFLASH_CUT_LM_HEAD | DFLASH_CUT_ATTN_BLOCK | DFLASH_CUT_MLP,
        ),
    ];
    let mut ms: Vec<Vec<f64>> = vec![vec![]; arms.len()];
    for round in 0..13 {
        for (i, (_, cut)) in arms.iter().enumerate() {
            let t = backend
                .0
                .dflash_draft_timed(first, n_ctx, 1, *cut)
                .expect("the draft declined — the ring is not at CTX");
            if round >= 3 {
                ms[i].push(t);
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let whole = med(&mut ms[0]);
    println!("block forward at {n_ctx} context positions, 10 interleaved rounds after 3 warm:");
    for (i, (name, _)) in arms.iter().enumerate() {
        let m = med(&mut ms[i]);
        println!(
            "  {name:24} {m:7.2} ms   (that stage = {:6.2} ms)   [min {:.2}, max {:.2}]",
            whole - m,
            ms[i][0],
            ms[i][ms[i].len() - 1]
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dflash2_block drives the native-Metal island; macOS only.");
}
