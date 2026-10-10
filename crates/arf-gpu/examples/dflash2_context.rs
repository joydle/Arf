//! DFlash 2 port, STAGE 3 GATE: the context ring.
//!
//! Prefills a prompt on the 27B with the draft attached, then recomputes the draft's context
//! K/V on the CPU from exactly the tap rows the GPU consumed — `fc` -> `hidden_norm` -> each
//! layer's `k_proj`/`v_proj` -> per-head k-norm -> half-split RoPE at the logical position — with
//! the SAME Q4_K_S weights (the CPU Q4 matmul, f64 norms and rotation), and compares every
//! checked ring slot of all five layers.
//!
//! `short` (default): 40 tokens = two records (39 rows + the split-off last row), every position
//! checked, and every slot past 40 must still be zero.
//! `long`: 2,100 tokens = 17+ windows and a WRAPPED ring (window 2048): checks the 52 wrapped
//! positions, the oldest survivors right after them, and a window boundary.
//!
//! The GPU path narrows activations to scaled half inside the MPP matmul, so this is a
//! tolerance gate, not a bit gate: error is reported relative to the RMS of the reference.
//!
//! `time`: a 128-token prompt in ONE record, then the wall time of a context commit at 8, 16 and
//! 128 rows (encode + GPU, fenced), so the draft's per-cycle context cost is a measurement.
//!
//! `ring-ab`: the same 128-row record, then the ring commit encoded SERIAL
//! (`ARF_DFLASH_COMMIT_SERIAL`) and CONCURRENT from the same taps into NaN-poisoned rings, at 8,
//! 16 and 128 rows (the split-K 8/16-row units and the wide path), as one run, as two streams'
//! runs, and as two runs of one stream. The rings must be BIT-identical, every committed slot
//! written, and each arm must have run the encode it names (the commit counters).
//!
//! Run: cargo run --release -p arf-gpu --example dflash2_context -- [short|long|time|ring-ab] [draft dir] [gguf]

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::backend::BatchedBackend;
    use arf_core::config::{EngineConfig, KvQuant, ModelConfig, Quant};
    use arf_core::engine::build_forward;
    use arf_core::sampling::{SamplingParams, SeqSampling};
    use arf_core::scheduler::{Request, Scheduler};
    use arf_core::tensor::{matmul_nt_q4ks, Q4KSMatrix};
    use arf_gpu::gpu::metal::dflash2::{Dflash2ContextReference, Dflash2Draft};
    use arf_gpu::gpu::GpuContext;
    use arf_gpu::WgpuBatched;
    use std::sync::Arc;

    let arg = |i: usize, d: &str| std::env::args().nth(i).unwrap_or_else(|| d.into());
    let long = arg(1, "short") == "long";
    // `ring-ab` shares the timing setup: one 128-row record, no tap log
    let ring_ab = arg(1, "short") == "ring-ab";
    let timing = arg(1, "short") == "time" || ring_ab;
    if timing {
        // One 128-row record, so the taps hold 128 rows and any commit width can be timed.
        std::env::set_var("ARF_NO_PREFILL_LAST_ROW", "1");
    }
    let dir = std::path::PathBuf::from(arg(2, "models/qwen3.8-27b-dflash2"));
    let gguf = arg(3, "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf");
    arf_gpu::apply_fast_path_defaults("dflash2_context");
    std::env::remove_var("ARF_KV_F16");
    std::env::set_var("ARF_NO_KV_F16", "1");

    let n_prompt = if long {
        2100
    } else if timing {
        128
    } else {
        40
    };
    let cfg = ModelConfig::qwen35_27b();
    let (block_size, num_blocks) = (16usize, 160usize);
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
    let draft = Dflash2Draft::load(&ctx, &dir).expect("load draft");
    let dc = draft.cfg.clone();
    let prompt: Vec<u32> = (0..n_prompt)
        .map(|i| ((i * 7919 + 104_729) % 50_000 + 1_000) as u32)
        .collect();
    let draft = std::cell::Cell::new(Some(draft));
    let attach = || {
        backend
            .0
            .island
            .as_ref()
            .expect("the Metal island")
            .lock()
            .unwrap()
            .dflash_attach(&ctx, draft.take().expect("attach once"), !timing)
            .expect("attach draft")
    };
    let next_id = std::cell::Cell::new(1u64);
    // Fresh sequence id per pass: the recurrent bank pins state to the id (see dflash2_taps).
    let run = |gen: usize| -> Vec<u32> {
        let id = next_id.get();
        next_id.set(id + 1);
        let mut sched = Scheduler::new(EngineConfig {
            block_size,
            num_blocks,
            max_batch_size: 1,
            max_prefill_tokens: 512,
            enable_prefix_cache: false,
            ..Default::default()
        });
        sched.add(Request::new(
            id,
            prompt.clone(),
            SamplingParams::greedy(gen),
        ));
        let g = SamplingParams::greedy(gen);
        let mut out = vec![];
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
            out.extend(
                sched
                    .commit_tokens(&toks)
                    .unwrap()
                    .into_iter()
                    .map(|o| o.token),
            );
        }
        out
    };
    // The target's text BEFORE a draft exists — compared below with the text after attach: the
    // context pass shares the target's MPP scratch (regrown at attach) and its queue.
    let before = if !long && !timing { run(16) } else { vec![] };
    attach();
    let t = std::time::Instant::now();
    let after = run(if !long && !timing { 16 } else { 1 });
    eprintln!("prefill + decode: {:.1} s", t.elapsed().as_secs_f64());

    let isl = backend.0.island.as_ref().unwrap().lock().unwrap();
    let (len, valid) = isl.dflash_context_len().expect("attached");
    println!("context ring: {len} positions, valid = {valid}");
    if ring_ab {
        use arf_gpu::gpu::metal::dflash2_ctx::CommitRun;
        let run_of = |stream: u64, row0: usize, rows: usize, start: usize| CommitRun {
            stream: Some(stream),
            row0,
            rows,
            start,
        };
        const POISON: u32 = 0x7fc0_dead; // dflash_poison_rings' NaN
        let per_pos = 2 * dc.layers * dc.kv_heads * dc.head_dim;
        let mut all_ok = true;
        for rows in [8usize, 16, 128] {
            let h1 = rows / 2;
            let cases: [(&str, Vec<CommitRun>, Vec<(u64, usize)>); 3] = [
                ("one run", vec![run_of(1, 0, rows, 0)], vec![(1, rows)]),
                (
                    "two streams",
                    vec![run_of(1, 0, h1, 0), run_of(2, h1, rows - h1, 0)],
                    vec![(1, h1), (2, rows - h1)],
                ),
                (
                    "one stream, two runs",
                    vec![run_of(1, 0, h1, 0), run_of(1, h1, rows - h1, h1)],
                    vec![(1, rows)],
                ),
            ];
            for (name, runs, streams) in cases {
                let mut arms: Vec<Vec<Vec<u32>>> = vec![];
                for serial in [true, false] {
                    let arm = if serial { "serial" } else { "concurrent" };
                    isl.dflash_set_commit_serial(serial).expect("attached");
                    let before = isl.dflash_commit_counts().unwrap();
                    isl.dflash_poison_rings();
                    isl.dflash_commit_runs(&runs).expect("commit");
                    let after = isl.dflash_commit_counts().unwrap();
                    let ran = if serial {
                        after.0 == before.0 + 1 && after.1 == before.1
                    } else {
                        after.1 == before.1 + 1 && after.0 == before.0
                    };
                    if !ran {
                        println!(
                            "{rows:3} rows, {name}: the {arm} encode did NOT run ({before:?} -> {after:?})"
                        );
                        all_ok = false;
                    }
                    let mut bits = vec![];
                    for &(s, n) in &streams {
                        let b = isl.dflash_ring_bits(Some(s)).expect("ring");
                        let written = b.iter().filter(|&&x| x != POISON).count();
                        let want = n.min(dc.sliding_window) * per_pos;
                        if written != want {
                            println!(
                                "{rows:3} rows, {name}, {arm}: stream {s} has {written} words written, want {want}"
                            );
                            all_ok = false;
                        }
                        bits.push(b);
                    }
                    arms.push(bits);
                }
                let ndiff: usize = arms[0]
                    .iter()
                    .zip(&arms[1])
                    .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
                    .sum();
                let same = arms[0] == arms[1];
                println!(
                    "{rows:3} rows, {name}: serial vs concurrent ring {}",
                    if same {
                        "BIT-IDENTICAL".to_string()
                    } else {
                        format!("DIFFER in {ndiff} words")
                    }
                );
                all_ok &= same;
            }
        }
        println!(
            "RING COMMIT A/B GATE: {}",
            if all_ok { "PASS" } else { "FAIL" }
        );
        if !all_ok {
            std::process::exit(1);
        }
        return;
    }
    if timing {
        // The taps are stale after the first commit; the work per commit is identical.
        let mut at = len;
        for rows in [8usize, 16, 128] {
            let mut ms = vec![];
            for i in 0..13 {
                isl.dflash_fence();
                let t = std::time::Instant::now();
                isl.dflash_commit_context(Some(1), rows, at)
                    .expect("commit");
                isl.dflash_fence();
                if i >= 3 {
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
                at += rows;
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "context commit, {rows:3} rows: median {:.2} ms (min {:.2}, max {:.2}) over 10",
                ms[5], ms[0], ms[9]
            );
        }
        return;
    }
    let taps = isl.dflash_tap_log().expect("tap log");
    let (h, hd, heads, win) = (dc.hidden, dc.head_dim, dc.kv_heads, dc.sliding_window);
    let fc_in = dc.target_layer_ids.len() * h;
    let mut ok = valid && len == n_prompt && taps.len() == n_prompt * fc_in;
    if !long {
        let same = before == after && before.len() == 16;
        println!(
            "target text, no draft vs draft attached (16 greedy tokens): {}",
            if same { "IDENTICAL" } else { "DIFFER" }
        );
        if !same {
            println!("  before {before:?}\n  after  {after:?}");
            std::process::exit(1);
        }
    }
    if !ok {
        println!(
            "FAILED before any comparison: expected {n_prompt} positions and {} tap floats, got {}",
            n_prompt * fc_in,
            taps.len()
        );
        std::process::exit(1);
    }

    // Positions to check. Rows are independent, so a subset is a full check of those rows.
    let check: Vec<usize> = if long {
        (win..n_prompt) // wrapped: these overwrote slots 0..52
            .chain(n_prompt - win..n_prompt - win + 12) // the oldest survivors, just past the wrap
            .chain(126..130) // a prefill-window boundary
            .chain(1022..1026)
            .collect()
    } else {
        (0..n_prompt).collect()
    };
    let refw = Dflash2ContextReference::load(&dir, &dc).expect("reference weights");
    let mean = refw.hidden_norm.iter().sum::<f32>() / h as f32;
    println!(
        "hidden_norm.weight mean {mean:.3} (a plain-weight norm sits near 1, a (1+w) norm near 0)"
    );
    let mm = |q: &Q4KSMatrix, a: &[f32], m: usize| {
        matmul_nt_q4ks(
            a,
            m,
            q.cols(),
            q.codes(),
            q.scales(),
            q.mins(),
            q.d(),
            q.dmin(),
            q.rows(),
        )
    };
    let m = check.len();
    let mut rows_in = Vec::with_capacity(m * fc_in);
    for &p in &check {
        rows_in.extend_from_slice(&taps[p * fc_in..][..fc_in]);
    }
    let t = std::time::Instant::now();
    let mut ctxn = mm(&refw.fc, &rows_in, m);
    for r in ctxn.chunks_mut(h) {
        let ss: f64 = r.iter().map(|x| *x as f64 * *x as f64).sum();
        let inv = 1.0 / (ss / h as f64 + dc.rms_eps as f64).sqrt();
        for (x, w) in r.iter_mut().zip(&refw.hidden_norm) {
            *x = (*x as f64 * inv * *w as f64) as f32;
        }
    }
    if std::env::args().any(|a| a == "--fc-error") {
        // What does 4-bit `fc` cost on the inputs it really sees? Same tap rows through the
        // PUBLISHED bf16 weights (f64 accumulate) vs the Q4_K_S ones, before and after the norm.
        let wf = Dflash2ContextReference::fc_published(&dir, &dc).expect("fc bf16");
        let q4 = mm(&refw.fc, &rows_in, m);
        let (mut e_raw, mut n_raw, mut e_n, mut n_n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut amax, mut arms) = (0.0f64, 0.0f64);
        for x in &rows_in {
            amax = amax.max(x.abs() as f64);
            arms += *x as f64 * *x as f64;
        }
        for r in 0..m {
            let x = &rows_in[r * fc_in..][..fc_in];
            let exact: Vec<f64> = (0..h)
                .map(|o| {
                    let w = &wf[o * fc_in..][..fc_in];
                    x.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum()
                })
                .collect();
            let got = &q4[r * h..][..h];
            let norm = |v: &[f64]| {
                let ss: f64 = v.iter().map(|a| a * a).sum();
                let inv = 1.0 / (ss / h as f64 + dc.rms_eps as f64).sqrt();
                v.iter().map(|a| a * inv).collect::<Vec<f64>>()
            };
            let gotf: Vec<f64> = got.iter().map(|a| *a as f64).collect();
            for (a, b) in exact.iter().zip(&gotf) {
                e_raw += (a - b) * (a - b);
                n_raw += a * a;
            }
            for (a, b) in norm(&exact).iter().zip(&norm(&gotf)) {
                e_n += (a - b) * (a - b);
                n_n += a * a;
            }
        }
        println!(
            "fc on real taps ({m} rows): input max |x| {amax:.1} vs rms {:.3}; 4-bit vs published \
             bf16 output error: {:.1}% relative RMS raw, {:.1}% after hidden_norm",
            (arms / rows_in.len() as f64).sqrt(),
            100.0 * (e_raw / n_raw).sqrt(),
            100.0 * (e_n / n_n).sqrt()
        );
    }
    let (mut worst_k, mut worst_v) = (0.0f64, 0.0f64);
    for l in 0..dc.layers {
        let (rk, rv) = isl.dflash_read_ring(l).expect("ring");
        let (kraw, vraw) = (mm(&refw.k_proj[l], &ctxn, m), mm(&refw.v_proj[l], &ctxn, m));
        let (mut ek, mut nk, mut ev, mut nv, mut mk, mut mv) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, &pos) in check.iter().enumerate() {
            let slot = pos % win;
            for hh in 0..heads {
                let ks = &kraw[(i * heads + hh) * hd..][..hd];
                let vs = &vraw[(i * heads + hh) * hd..][..hd];
                let ss: f64 = ks.iter().map(|x| *x as f64 * *x as f64).sum();
                let inv = 1.0 / (ss / hd as f64 + dc.rms_eps as f64).sqrt();
                let gk = &rk[(hh * win + slot) * hd..][..hd];
                let gv = &rv[(hh * win + slot) * hd..][..hd];
                for d in 0..hd / 2 {
                    let a = ks[d] as f64 * inv * refw.k_norm[l][d] as f64;
                    let b = ks[d + hd / 2] as f64 * inv * refw.k_norm[l][d + hd / 2] as f64;
                    let ang = pos as f64 * (dc.rope_theta as f64).powf(-2.0 * d as f64 / hd as f64);
                    let (s, c) = ang.sin_cos();
                    for (want, got) in [(a * c - b * s, gk[d]), (b * c + a * s, gk[d + hd / 2])] {
                        let e = (want - got as f64).abs();
                        ek += e * e;
                        nk += want * want;
                        mk = mk.max(e);
                    }
                }
                for d in 0..hd {
                    let e = (vs[d] as f64 - gv[d] as f64).abs();
                    ev += e * e;
                    nv += vs[d] as f64 * vs[d] as f64;
                    mv = mv.max(e);
                }
            }
        }
        let cnt = (m * heads * hd) as f64;
        let (rms_k, rms_v) = ((nk / cnt).sqrt(), (nv / cnt).sqrt());
        println!(
            "layer {l}: K rel-rms {:.2e} max/rms {:.2e} | V rel-rms {:.2e} max/rms {:.2e}  ({m} positions x {heads} heads)",
            (ek / nk).sqrt(), mk / rms_k, (ev / nv).sqrt(), mv / rms_v
        );
        worst_k = worst_k.max((ek / nk).sqrt());
        worst_v = worst_v.max((ev / nv).sqrt());
        if !rk.iter().chain(&rv).all(|x| x.is_finite()) {
            println!("layer {l}: NON-FINITE ring values");
            ok = false;
        }
        if !long {
            // Nothing past the prompt may have been written (padded matmul rows are garbage).
            let dirty = (0..heads)
                .flat_map(|hh| (n_prompt..win).map(move |s| (hh * win + s) * hd))
                .filter(|&o| {
                    rk[o..o + hd]
                        .iter()
                        .chain(&rv[o..o + hd])
                        .any(|x| *x != 0.0)
                })
                .count();
            if dirty > 0 {
                println!(
                    "layer {l}: {dirty} (head, slot) pairs past position {n_prompt} were written"
                );
                ok = false;
            }
        }
    }
    eprintln!("cpu reference: {:.1} s", t.elapsed().as_secs_f64());
    // Two chained MPP matmuls (each gated at 5e-3 max/rms on its own, mpp_q4_parity) and two
    // normalisations: a correct ring sits at ~1e-3 relative RMS; a wrong weight, position, head
    // order or RoPE convention sits at ~1.
    println!("worst relative RMS error: K {worst_k:.2e}, V {worst_v:.2e}  (gate: < 1e-2)");
    if ok && worst_k < 1e-2 && worst_v < 1e-2 {
        println!(
            "STAGE 3 GATE ({}): PASS",
            if long { "long" } else { "short" }
        );
    } else {
        println!("STAGE 3 GATE: FAIL");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dflash2_context drives the native-Metal island; macOS only.");
}
