//! DFlash 2 port, STAGE 2 GATE: the target's taps.
//!
//! `check` (default): runs the 27B through the serving loop with the taps on AND the debug
//! capture (`ARF_GDN_CAPTURE`, the instrument that localised the hybrid layers against
//! llama.cpp) on the same five layers, and after every step that ran a batched record compares
//! the two BIT FOR BIT — every row, every slot. They are written by different kernels into
//! different layouts ([row][slot][h] vs [slot][row][h]), so agreement checks the tap's index
//! math, its row count and its ordering against the layer's last write. Also asserts the token
//! streams with taps on are identical to the streams with taps off.
//!
//! `time`: no capture; interleaves taps off/on over a 120-token prefill and a 4-wide decode and
//! prints the wall time of each, so the overhead is a measurement rather than an estimate.
//!
//! Run: cargo run --release -p arf-gpu --example dflash2_taps -- [check|time] [gguf]

#[cfg(target_os = "macos")]
const TAPS: [usize; 5] = [5, 19, 33, 47, 61];

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::backend::BatchedBackend;
    use arf_core::config::{EngineConfig, KvQuant, ModelConfig, Quant};
    use arf_core::engine::build_forward;
    use arf_core::sampling::{SamplingParams, SeqSampling};
    use arf_core::scheduler::{Request, Scheduler};
    use arf_gpu::gpu::GpuContext;
    use arf_gpu::WgpuBatched;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    let mode = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    let gguf = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf".into());
    let check = mode == "check";
    // The levers the daemon ships with — without them the model loads with no island at all and
    // there is no batched record to tap. Then the daemon's own rule for a hybrid model serving
    // more than one sequence: f32 KV (arf-serve main.rs, the f16-KV refusal).
    arf_gpu::apply_fast_path_defaults("dflash2_taps");
    std::env::remove_var("ARF_KV_F16");
    std::env::set_var("ARF_NO_KV_F16", "1");
    if check {
        let list = TAPS.map(|l| l.to_string()).join(",");
        std::env::set_var("ARF_GDN_CAPTURE", list);
    }

    let cfg = ModelConfig::qwen35_27b();
    let h = cfg.hidden_size;
    let (block_size, num_blocks) = (16usize, 64usize);
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
    let island = || {
        backend
            .0
            .island
            .as_ref()
            .expect("the Metal island")
            .lock()
            .unwrap()
    };
    let prompt = |seed: usize, n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104_729) % 50_000 + 1_000) as u32)
            .collect()
    };

    // One pass of the serving loop. `after` runs after every step and gets the step's wall time.
    //
    // Sequence ids are NEVER reused across passes. The recurrent bank pins a row to a sequence id
    // for its lifetime (batch.rs, L162), so a second pass that reuses id 0 CONTINUES the first
    // pass's conv ring and delta-net state instead of starting clean — the first version of this
    // gate did that and its off-vs-off control came back DIFFER. The daemon's ids are unique.
    let next_id = std::cell::Cell::new(0u64);
    let run = |prompts: &[Vec<u32>], gen: usize, after: &mut dyn FnMut(usize, f64)| {
        let base = next_id.get();
        next_id.set(base + prompts.len() as u64);
        let mut sched = Scheduler::new(EngineConfig {
            block_size,
            num_blocks,
            max_batch_size: prompts.len(),
            max_prefill_tokens: 512,
            enable_prefix_cache: false,
            ..Default::default()
        });
        for (i, p) in prompts.iter().enumerate() {
            sched.add(Request::new(
                base + i as u64,
                p.clone(),
                SamplingParams::greedy(gen),
            ));
        }
        let g = SamplingParams::greedy(gen);
        let mut streams: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        let mut step = 0usize;
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
            let t = std::time::Instant::now();
            let toks = backend.step(&ids, &batch, &samp).unwrap();
            after(step, t.elapsed().as_secs_f64() * 1e3);
            for out in sched.commit_tokens(&toks).unwrap() {
                streams.entry(out.id - base).or_default().push(out.token);
            }
            step += 1;
        }
        streams
    };
    let enable = || {
        island()
            .dflash_enable_taps(&ctx, &TAPS, h, 128)
            .expect("enable taps")
    };

    if check {
        // 20 tokens: one prefill record of 20 rows (a final window is split only past 24).
        // Two sequences: the decode steps are 2-row batched records.
        let prompts = [prompt(1, 20), prompt(2, 20)];
        let off = run(&prompts, 12, &mut |_, _| {});
        // CONTROL: the same pass again, taps still off. If a second pass over one loaded model
        // does not reproduce the first, "off vs on" below would blame the taps for it.
        let off2 = run(&prompts, 12, &mut |_, _| {});
        println!(
            "control, taps off vs off again: {}",
            if off == off2 { "IDENTICAL" } else { "DIFFER" }
        );
        for (id, t) in &off {
            println!("  off  seq {id}: {t:?}");
        }
        for (id, t) in &off2 {
            println!("  off2 seq {id}: {t:?}");
        }
        enable();
        let (mut seen, mut compared, mut rows_checked, mut widths) = (0u64, 0usize, 0usize, vec![]);
        let mut bad = 0usize;
        let on = run(&prompts, 12, &mut |step, _| {
            let isl = island();
            let n = isl.dflash_tap_records();
            if n == seen {
                return; // this step ran no batched record (the m=1 path) — nothing new to compare
            }
            seen = n;
            let (rows, taps) = isl.dflash_read_taps().expect("taps");
            for (slot, layer) in TAPS.iter().enumerate() {
                let Some(cap) = isl.gdn_capture_hidden(slot, rows, h) else {
                    eprintln!("step {step}: no capture for slot {slot} at {rows} rows");
                    bad += 1;
                    continue;
                };
                for r in 0..rows {
                    let t = &taps[(r * TAPS.len() + slot) * h..][..h];
                    let c = &cap[r * h..][..h];
                    let same = t.iter().zip(c).all(|(a, b)| a.to_bits() == b.to_bits());
                    let live = t.iter().all(|x| x.is_finite()) && t.iter().any(|x| *x != 0.0);
                    if !same || !live {
                        bad += 1;
                        eprintln!("step {step} layer {layer} row {r}: same={same} live={live}");
                    }
                }
            }
            // Slots must hold DIFFERENT layers: a tap that wrote one layer five times passes the
            // comparison above only if the capture did the same, but make it explicit.
            for s in 1..TAPS.len() {
                if taps[s * h..][..h] == taps[..h] {
                    bad += 1;
                    eprintln!("step {step}: slot {s} equals slot 0");
                }
            }
            compared += 1;
            rows_checked += rows * TAPS.len();
            if !widths.contains(&rows) {
                widths.push(rows);
            }
        });
        println!(
            "taps vs debug capture: {compared} records, {rows_checked} row-slots of {h} floats, \
             record widths {widths:?}, {bad} mismatches"
        );
        for (id, t) in &on {
            println!("  on   seq {id}: {t:?}");
        }
        let same_text = off2 == on && off == off2;
        println!(
            "token streams, taps off vs on: {}",
            if same_text { "IDENTICAL" } else { "DIFFER" }
        );
        let wide = widths.iter().any(|&w| w >= 20);
        let narrow = widths.iter().any(|&w| (2..20).contains(&w));
        if bad == 0 && same_text && wide && narrow {
            println!("STAGE 2 GATE: PASS");
        } else {
            println!("STAGE 2 GATE: FAIL (needs 0 mismatches, identical text, a prefill record and a decode record)");
            std::process::exit(1);
        }
        return;
    }

    // ---- time: interleaved off/on. Prefill = one 120-token prompt; decode = 4 sequences. ----
    let arm = |taps_on: bool| -> (f64, f64) {
        if taps_on {
            enable();
        } else {
            island().dflash_disable_taps();
        }
        let mut prefill = 0.0;
        run(&[prompt(3, 120)], 1, &mut |step, ms| {
            if step == 0 {
                prefill = ms;
            }
        });
        let mut dec: Vec<f64> = vec![];
        let four = [prompt(4, 8), prompt(5, 8), prompt(6, 8), prompt(7, 8)];
        run(&four, 40, &mut |step, ms| {
            if step >= 8 {
                dec.push(ms);
            }
        });
        dec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (prefill, dec[dec.len() / 2])
    };
    arm(false); // discarded: warms every shape
    arm(true);
    for round in 1..=3 {
        for on in [false, true] {
            let (p, d) = arm(on);
            println!(
                "round {round} taps {:3}: 120-token prefill {p:8.1} ms, 4-wide decode step (median) {d:7.2} ms",
                if on { "ON" } else { "off" }
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dflash2_taps drives the native-Metal island; macOS only.");
}
