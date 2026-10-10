//! Speculative-decode economics: greedy vs the n-gram speculative-verify loop on
//! the resident GPU model — wall-clock ratio, tokens/forward, draft acceptance,
//! and exact-output parity. The n-gram acceptance is a FLOOR (a trained MTP head
//! drafts far better); the decision number is the verify-pass economics, i.e. the
//! tokens/forward the verify loop sustains — that times a better drafter's
//! acceptance is the MTP ceiling.
//!
//! Generalized over models via env (was Llama-1B-only):
//!   ARCH      llama-3.2-1b (default) | qwen3-coder-30b | gemma3-4b
//!   MODEL     path to a safetensors dir OR a single .gguf file
//!             (default: models/<arch>/{model.safetensors|*.gguf})
//!   QUANT     int8 (default) | bf16/none | q4   — the VERIFY pass runs m>1
//!             batched matmuls, which only int8/bf16 support (Q4 is m==1 GEMV
//!             only), so spec-decode must run int8/bf16 here.
//!   TOKENIZER optional tokenizer.json; without it the harness seeds a few token
//!             ids and lets the model extend them into a realistic stream (timing
//!             + acceptance need the realized distribution, not decoded text).
//!   K         draft window (default 4)     ORDER  n-gram order (default 3)
//!   N         tokens to generate (default 200)
//!
//! Run: ARCH=qwen3-coder-30b QUANT=int8 MODEL=models/qwen3-coder-30b-a3b/<file>.gguf \
//!        cargo run --release -p arf-gpu --example gpu_spec_timing

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arf_core::config::{ModelConfig, Quant};
use arf_core::Tokenizer;
use arf_gpu::gpu::{GpuContext, GpuModel};

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

fn env_usize(key: &str, default: usize) -> usize {
    env(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn is_gguf(path: &Path) -> bool {
    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        return true;
    }
    // Fall back to the magic bytes so an extensionless blob (ollama) is detected.
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut m = [0u8; 4];
            f.read_exact(&mut m)?;
            Ok(&m == b"GGUF")
        })
        .unwrap_or(false)
}

fn return_seed(ids: Vec<u32>) -> Vec<(String, Vec<u32>)> {
    vec![("seeded".to_string(), ids)]
}

fn main() {
    let arch = env("ARCH").unwrap_or_else(|| "llama-3.2-1b".into());
    let (mut cfg, default_dir): (ModelConfig, &str) = match arch.as_str() {
        "llama-3.2-1b" | "llama3.2-1b" => (ModelConfig::llama_3_2_1b(), "models/llama-3.2-1b"),
        "qwen3-coder-30b" | "qwen3moe" => {
            (ModelConfig::qwen3_coder_30b(), "models/qwen3-coder-30b-a3b")
        }
        "gemma3-4b" | "gemma3" => (ModelConfig::gemma3_4b(), "models/gemma3-4b"),
        // L222 — the hybrid (48 gated-delta-net + 16 attention) 27B this engine now targets.
        "qwen3.8" | "qwen3.8-27b" | "qwen35" => (ModelConfig::qwen35_27b(), "models/qwen3.8-27b"),
        other => {
            panic!("unknown ARCH {other:?} (llama-3.2-1b | qwen3-coder-30b | gemma3-4b | qwen3.8)")
        }
    };
    // A GGUF blob may carry a slightly different vocab than the built-in config
    // (gemma variants differ by a handful of tokens); override without a config.json.
    if let Some(v) = env("VOCAB").and_then(|s| s.parse().ok()) {
        cfg.vocab_size = v;
    }

    // Q4/Q4KS now run the verify pass too (the shared-prefix verify megakernel handles
    // m=k on the Q4KS island; the WGSL fallback still needs int8/bf16). Unknown QUANT
    // values are an ERROR — the old catch-all silently upgraded q4ks→Int8, which on a
    // 30B model wired ~34 GB and froze the machine (jetsam 2026-07-02).
    let quant = match env("QUANT").as_deref() {
        Some("q4") => Quant::Q4,
        Some("q4k") => Quant::Q4K,
        Some("q4ks") => Quant::Q4KS,
        Some("bf16") | Some("none") => Quant::None,
        Some("int8") | None => Quant::Int8,
        Some(other) => panic!("unknown QUANT {other:?} (int8 | bf16/none | q4 | q4k | q4ks)"),
    };
    let k = env_usize("K", 4);
    let order = env_usize("ORDER", 3);
    let n = env_usize("N", 200);
    let max_ctx = env_usize("MAX_CTX", 2048);

    // Resolve the model path: explicit MODEL, else the arch's default dir (a
    // safetensors `model.safetensors` or the first `*.gguf` in it).
    let model_path = env("MODEL").map(PathBuf::from).unwrap_or_else(|| {
        let dir = PathBuf::from(default_dir);
        let st = dir.join("model.safetensors");
        if st.exists() {
            return st;
        }
        std::fs::read_dir(&dir)
            .ok()
            .and_then(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .find(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
            })
            .unwrap_or(st)
    });

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    let gguf = is_gguf(&model_path);
    eprint!(
        "loading {arch} ({quant:?}, {}) onto {}... ",
        if gguf { "gguf" } else { "safetensors" },
        ctx.info()
    );
    std::io::stderr().flush().ok();
    let model: GpuModel = if gguf {
        arf_gpu::weights::load_gguf_gpu_quant(&cfg, &model_path, max_ctx, &ctx, quant)
            .expect("load gguf")
    } else {
        let paths: [&Path; 1] = [model_path.as_path()];
        arf_gpu::weights::load_safetensors_gpu_quant(&cfg, &paths, max_ctx, &ctx, quant)
            .expect("load safetensors")
    };
    eprintln!("done");

    // Prompts: real text when a tokenizer is available, else token-id seeds the
    // model extends into a realistic stream (the n-gram drafter only needs the
    // realized distribution, not decoded text).
    let tok = env("TOKENIZER")
        .map(PathBuf::from)
        .or_else(|| {
            let t = model_path
                .parent()
                .map(|d| d.join("tokenizer.json"))
                .filter(|p| p.exists());
            t
        })
        .and_then(|p| Tokenizer::from_file(p).ok());

    let prompts: Vec<(String, Vec<u32>)> = if let Some(t) = &tok {
        [
            ("natural", "The capital of France is"),
            ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci"),
            ("repetitive", "the quick brown fox jumps. the quick brown fox jumps. the quick brown fox jumps."),
        ]
        .iter()
        .map(|(l, p)| ((*l).to_string(), t.encode(p, true).expect("encode")))
        .collect()
    } else {
        eprintln!("note: no tokenizer — seeding token ids. 'natural-seed' = the model's own continuation (acceptance floor); 'cycle-seed' = a repetitive seed that drives the greedy stream into a loop so the drafter proposes the FULL k every round — that row measures the verify-pass economics at full width (the MTP ceiling), independent of any real drafter's acceptance.");
        if let Some(csv) = env("SEED_IDS") {
            let ids: Vec<u32> = csv
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            eprintln!("[seed] SEED_IDS override: {} tokens", ids.len());
            return_seed(ids)
        } else {
            vec![
                (
                    "natural-seed".to_string(),
                    vec![1u32, 785, 311, 5651, 264, 2875, 3364],
                ),
                // A short cycle the model will tend to continue → ~100% n-gram hits →
                // every verify forward runs full width (1+k). Its ratio IS the ceiling.
                (
                    "cycle-seed".to_string(),
                    vec![1u32, 220, 16, 220, 17, 220, 18, 220, 16, 220, 17, 220, 18],
                ),
            ]
        }
    };

    println!(
        "ARCH={arch} QUANT={quant:?} k={k} order={order} N={n} — greedy vs n-gram speculative\n"
    );
    // The MTP ceiling is the spec ratio when verify runs at FULL width (drafter
    // proposes k every round) — a property of the model's batched-verify cost, NOT
    // of the n-gram's acceptance. We capture it empirically as the ratio of the row
    // with the highest tokens/forward (widest realized verify). Extrapolating from a
    // low-acceptance row is WRONG: the drafter is adaptive (no match ⇒ width-1
    // verify ≈ a decode), so reaching k+1 tok/fwd costs the full k× verify.
    let mut ceiling: (f64, f64, f64) = (0.0, 0.0, 0.0); // (tok_per_fwd, ratio, accept%)
    for (label, ids) in &prompts {
        // Warm up BOTH paths. generate() alone leaves the SPECULATIVE verify megakernel
        // (attention_verify_shared_prefix + verify_dims) cold — its first-use Metal pipeline JIT
        // (MTLCompilerService) front-loads ~7 frames of 16-28s into the FIRST timed spec call,
        // which tanked the measured ratio to ~0.04x while the true warm verify cost is ~44ms
        // (~2.3x a decode). Warm the spec path too so we time the steady state, not the JIT.
        let _ = model.generate(ids, 16);
        let _ = model.generate_speculative(ids, 16, k, order);

        let t = Instant::now();
        let greedy = model.generate(ids, n);
        let greedy_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let (spec, fwd, drafted, accepted) = model.generate_speculative(ids, n, k, order);
        let spec_s = t.elapsed().as_secs_f64();

        // L228 — DETERMINISM PROBE: run greedy TWICE before comparing. If two greedy
        // runs of the same prompt differ, `spec == greedy` is not a parity test of the
        // speculative path at all — it is a coin flip, and every MISMATCH reported here
        // is uninterpretable.
        if std::env::var_os("SPEC_DET_PROBE").is_some() {
            let g2 = model.generate(ids, n);
            eprintln!("[det-probe] {label}: greedy==greedy2 ? {}", g2 == greedy);
            if g2 != greedy {
                let d = greedy.iter().zip(g2.iter()).position(|(a, b)| a != b);
                eprintln!(
                    "[det-probe]   first divergence at index {d:?} of {} tokens",
                    greedy.len()
                );
                eprintln!(
                    "[det-probe]   run1[..12] = {:?}",
                    &greedy[..greedy.len().min(12)]
                );
                eprintln!("[det-probe]   run2[..12] = {:?}", &g2[..g2.len().min(12)]);
            }
        }
        let ok = spec == greedy;
        let ratio = greedy_s / spec_s;
        let tpf = spec.len() as f64 / fwd as f64;
        let accept = if drafted > 0 {
            accepted as f64 / drafted as f64 * 100.0
        } else {
            0.0
        };
        if tpf > ceiling.0 {
            ceiling = (tpf, ratio, accept);
        }
        println!(
            "[{label:>12}] greedy {:6.1} tok/s | spec {:6.1} tok/s | {ratio:.2}x | {tpf:.2} tok/fwd | accept {accept:3.0}% | parity {}",
            greedy.len() as f64 / greedy_s,
            spec.len() as f64 / spec_s,
            if ok { "OK" } else { "MISMATCH!" },
        );
        std::io::stdout().flush().ok();
    }

    // The decision number. At full-width verify (highest tok/fwd row) the ratio is
    // what a perfect drafter would sustain — the MTP ceiling. >1.0x ⇒ the verify
    // wall is low enough that better drafts convert to a win (GO); <=1.0x ⇒ the
    // batched-verify fixed cost dominates and no drafter quality saves it (NO-GO).
    println!(
        "\nMTP ceiling (full-width verify, {:.2} tok/fwd @ {:.0}% accept): ~{:.2}x greedy",
        ceiling.0, ceiling.2, ceiling.1
    );
    if ceiling.1 > 1.3 {
        println!("=> GO-ish: verify economics leave real slack for a trained MTP drafter.");
    } else if ceiling.1 > 1.0 {
        println!("=> MARGINAL: thin slack; MTP must draft very well to net a win.");
    } else {
        println!("=> NO-GO on this model: the batched-verify cost wall caps even a perfect drafter at <=1x.");
    }
}
