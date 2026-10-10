//! FAST-PATH ENV DEFAULTING — the measured-winner levers, defaulted ON for every process
//! that loads a GPU model.
//!
//! This block lived in `arf-serve`'s `main()` (and only there), which meant `arf run` /
//! `arf generate` — the CLI's in-process chat/generation — silently took the slow portable
//! path: the fast path is gated on env (`ARF_MSL_GEMV`, `ARF_MEGAKERNEL`, …) that only
//! the daemon defaulted. Extracted here VERBATIM so every entry point applies the SAME
//! conditions in the SAME order with the SAME messages (prefixed with its role).
//!
//! No behavior change for the daemon: `arf-serve` calls this with role `"serve"` exactly
//! where the block used to run (before `Args::parse()`); the CLI calls it with role `"cli"`
//! at the top of its chat/generation entry points, before the GPU model load (where
//! `ARF_MSL_GEMV` is read).

/// Default the fast-path env levers ON (macOS only — the island is Metal; no-op elsewhere).
/// `role` prefixes the startup messages (`[serve]`, `[cli]`). Every default respects an
/// explicit setting and its `ARF_NO_*` opt-out. Must run BEFORE the GPU model load,
/// where `ARF_MSL_GEMV` is read.
pub fn apply_fast_path_defaults(role: &str) {
    apply_fast_path_defaults_for(role, arch_from_argv());
}

/// The same head_dim rule as the `--arch` guard in [`apply_fast_path_defaults`], decided from the
/// model's RESOLVED config. Call it BEFORE [`apply_fast_path_defaults`], once the config is known.
///
/// WHY: the argv guard only sees `--arch`. A model resolved from its own `config.json` (`arf pull
/// llama3.2:1b && arf run llama3.2:1b`, the README quickstart) passes no `--arch`, so the guard never
/// fired and the island ran a head_dim-64 model: `!!!!!!!!` on the GPU, at defaults. Deciding it here,
/// before any lever is defaulted, lands on the configuration `make quickstart` has always run and
/// that is measured correct — `ARF_NO_MSL_GEMV` set before anything else.
pub fn decline_fast_path_for_model(role: &str, cfg: &arf_core::config::ModelConfig) {
    #[cfg(not(target_os = "macos"))]
    let _ = (role, cfg);
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_NO_MSL_GEMV").is_none() {
        if let Some(hd) = cfg.island_unsupported_head_dim() {
            decline_island(role, hd);
        }
    }
}

/// Turn the island off for a model it cannot address, saying why. Removes the positive lever too:
/// every downstream gate reads `ARF_MSL_GEMV.is_some()` alone.
#[cfg(target_os = "macos")]
fn decline_island(role: &str, hd: usize) {
    let overrode = std::env::var_os("ARF_MSL_GEMV").is_some();
    std::env::set_var("ARF_NO_MSL_GEMV", "1");
    std::env::remove_var("ARF_MSL_GEMV");
    eprintln!(
        "[{role}] fast path: DISABLED for this model — head_dim {hd} is not a multiple \
         of 128 (<= 512) and the island decode kernels require it; running the portable \
         wgpu path, which is correct and slower{}",
        if overrode {
            " (ARF_MSL_GEMV was set on the launch line; overriding it, because the \
             island cannot address this model's KV rows and would emit garbage)"
        } else {
            ""
        }
    );
}

/// The `--arch` this process was launched with, in either spelling. Separate from the decision
/// below so a test can supply an arch: this runs before `Args::parse()`, so the real source is
/// argv, and a test binary's argv is the harness's — which would otherwise leave the head_dim
/// guard unreachable from a test and the invariant unpinned.
fn arch_from_argv() -> Option<String> {
    let argv: Vec<String> = std::env::args().collect();
    argv.iter()
        .position(|a| a == "--arch")
        .and_then(|i| argv.get(i + 1).cloned())
        .or_else(|| {
            argv.iter()
                .find_map(|a| a.strip_prefix("--arch=").map(str::to_string))
        })
}

/// The body of [`apply_fast_path_defaults`], with the arch passed in rather than read from argv.
fn apply_fast_path_defaults_for(role: &str, arch: Option<String>) {
    #[cfg(not(target_os = "macos"))]
    let _ = (role, arch);
    // 🔴 head_dim NOT A MULTIPLE OF 128 => DO NOT DEFAULT THE ISLAND ON (llama-3.2-1B, hd 64).
    //
    // The island's decode attention kernels load KV as 32 lanes × float4/half4 and say so in their
    // own headers ("Requires hd % 128 == 0" — attention_msl.metal,
    // `attention_decode_coalesced_b_f16`). The host guards for that rule live in `island.rs` and
    // `selftest.rs` and none runs early enough to stop the island being ENABLED, so hd=64
    // dispatches them anyway. MEASURED, `arf generate --arch llama-3.2-1b --device gpu`, greedy,
    // "The capital of France is":
    //     ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 -> "!!!!!!!!!!!!!!"
    //     ARF_NO_MSL_GEMV=1                  -> " Paris. The capital of France is Paris."
    // Every other supported arch is hd 128/256/512, so the 1B is the only violator — and nothing in
    // CI runs it on the GPU, which is why it sat here.
    //
    // WHY IT IS DECIDED **HERE**, BEFORE THE LEVERS, AND NOT AT MODEL BUILD: refusing later means
    // the island is still BUILT, and merely having it built changes what the portable path reads.
    // The configuration that is measured correct is `ARF_NO_MSL_GEMV` set before any of this
    // runs — one with real mileage, rather than a new one.
    //
    // The arch comes from argv because this must run before `Args::parse()` (same reason the
    // KV_F16 block below reads `--max-batch-size` that way). A model loaded from safetensors with a
    // sibling `config.json` passes no `--arch` and keeps the old behaviour; no such model is in
    // `ARCHITECTURES` today, so nothing supported is left on the bad path.
    //
    // 🔴 THIS RUNS EVEN WHEN THE LAUNCHER SET `ARF_MSL_GEMV` EXPLICITLY. Gating it on
    // `ARF_MSL_GEMV.is_none()` — "only decide this if nobody else has" — reads like ordinary
    // respect for an explicit setting but is an OFF SWITCH ON THE SAFETY RULE: `make dev`/`make
    // serve` pass `ARF_MSL_GEMV=1` on the launch line, so the guard would skip itself on exactly
    // the path everybody uses. An explicit `ARF_MSL_GEMV=1` is a request for the fast path, not
    // a waiver of the kernels' addressing limits, and there is no configuration in which honouring
    // it helps: the island cannot read a head_dim-64 KV row, so the only outcomes are "portable and
    // right" or "island and `!!!!`". `ARF_NO_MSL_GEMV` remains a true override because it asks
    // for the SAFE direction. Removing `ARF_MSL_GEMV` as well as setting `ARF_NO_MSL_GEMV` is
    // required, not belt-and-braces: every downstream gate tests `ARF_MSL_GEMV.is_some()`, so
    // leaving it set would build island views for a model that must not dispatch them.
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_NO_MSL_GEMV").is_none() {
        if let Some(cfg) = arch
            .as_deref()
            .and_then(|a| arf_core::config::config_for_arch(a).ok())
        {
            if let Some(hd) = cfg.island_unsupported_head_dim() {
                decline_island(role, hd);
            }
        }
    }
    // FAST PATH BY DEFAULT: the megakernel (batched MoE + B-row attention + mm_id GEMM, the whole
    // conc scoreboard) needs the native-Metal island, which is gated on ARF_MSL_GEMV. Historically
    // this had to be set on the launch line (make dev/dev-conc prepend it), so `make serve`, a raw
    // `arf-serve`, or a systemd unit that forgot it silently ran the SLOW path and chat felt
    // nothing like the bench numbers. Default it ON here so EVERY launch method gets the fast path;
    // opt out with ARF_NO_MSL_GEMV=1 (the portable/debug path). macOS-only (the island is Metal).
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_MSL_GEMV").is_none() && std::env::var_os("ARF_NO_MSL_GEMV").is_none() {
        std::env::set_var("ARF_MSL_GEMV", "1");
        eprintln!("[{role}] fast path: ARF_MSL_GEMV defaulted ON (native-Metal island + megakernel); ARF_NO_MSL_GEMV=1 to opt out");
    } else if std::env::var_os("ARF_NO_MSL_GEMV").is_some() {
        // 🔴 THE OPT-OUT HAS TO REMOVE THE POSITIVE LEVER, not merely decline to set it. Nothing
        // downstream reads `ARF_NO_MSL_GEMV`: the island is built by `weights.rs` under a bare
        // `ARF_MSL_GEMV.is_some()`, and so is every one of the gates behind it. So
        // `ARF_NO_MSL_GEMV=1 ARF_MSL_GEMV=1` printed "DISABLED — running the portable/serial
        // path" and then built the island anyway: the message was false, and it was false in the
        // opt-out people reach for when they already suspect the island.
        //
        // The head_dim guard above does this correctly and says why in its own comment ("leaving it
        // set would build island views for a model that must not dispatch them"); this branch is
        // the same rule for the case the USER asks for.
        //
        // Reachable by accident, not just by typing both: `ARF_MSL_GEMV=1` is on every `make
        // dev`/`make serve` line, so anyone appending `ARF_NO_MSL_GEMV=1` to that command to
        // test the portable path lands here.
        let contradicted = std::env::var_os("ARF_MSL_GEMV").is_some();
        std::env::remove_var("ARF_MSL_GEMV");
        eprintln!(
            "[{role}] fast path: DISABLED (ARF_NO_MSL_GEMV set) — running the portable/serial path{}",
            if contradicted {
                " (ARF_MSL_GEMV was also set; NO_MSL_GEMV wins, clearing it — it is the broader statement)"
            } else {
                ""
            }
        );
    } else {
        eprintln!("[{role}] fast path: ARF_MSL_GEMV already set by launcher");
    }
    // 🔴 CONTRADICTORY PAIR: the island off, the megakernel explicitly on.
    //
    // The megakernel IS the island — `try_m1_megakernel` dispatches island kernels against weights
    // whose island views `ARF_NO_MSL_GEMV` told the loader not to build. The DEFAULTING block
    // just below already knows this and declines to turn the megakernel on when
    // `ARF_NO_MSL_GEMV` is set. But a default only guards the value nobody typed; an explicit
    // `ARF_MEGAKERNEL=1` walks straight past it and the invariant is enforced nowhere.
    //
    // MEASURED on llama-3.2-1b, greedy, "The capital of France is":
    //     ARF_NO_MSL_GEMV=1                     -> " Paris. The capital of France is Paris."
    //     ARF_NO_MSL_GEMV=1 ARF_MEGAKERNEL=1 -> "!!!!!!!!!!!!!!"
    // Confident garbage at full speed with no diagnostic, from the very path people fall back TO
    // when they suspect the island. A debug lever that silently corrupts output is worse than no
    // debug lever.
    //
    // This is also REACHABLE WITHOUT ANYONE ASKING FOR IT, which is why it sits here rather than in
    // a doc note: the head_dim guard above sets `ARF_NO_MSL_GEMV` itself, so a `make dev` on
    // llama-3.2-1b — whose launch line carries `ARF_MEGAKERNEL=1` — lands in this pair by
    // construction. The guard's fix does not hold without this block.
    //
    // NO_MSL_GEMV WINS. It is the broader statement ("do not run the island"), and the megakernel
    // is a part of the island rather than an alternative to it, so clearing the narrower lever
    // lands on the configuration that is measured correct. Refusing to start was the other option
    // and is worse: this pair appears in old A/B scripts and bisection notes, and a hard failure
    // would turn each of them into a bug report for a run that now only needs to be slower.
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_NO_MSL_GEMV").is_some() && std::env::var_os("ARF_MEGAKERNEL").is_some()
    {
        std::env::remove_var("ARF_MEGAKERNEL");
        eprintln!(
            "[{role}] fast path: IGNORING ARF_MEGAKERNEL — it needs the native-Metal island \
             that ARF_NO_MSL_GEMV disabled, and that pair emits fluent GARBAGE rather than \
             slow-but-correct tokens; running the portable wgpu path"
        );
    }
    // SINGLE-CHAT FAST PATH: a lone chat is B=1; it routes through the m=1 megakernel
    // (try_m1_megakernel), which is gated on ARF_MEGAKERNEL. Default it ON for the same
    // reason as ARF_MSL_GEMV above — a raw launch must not silently run a slow chat.
    // Opt out with ARF_NO_MEGAKERNEL=1.
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_MEGAKERNEL").is_none()
        && std::env::var_os("ARF_NO_MEGAKERNEL").is_none()
        && std::env::var_os("ARF_NO_MSL_GEMV").is_none()
    {
        std::env::set_var("ARF_MEGAKERNEL", "1");
        eprintln!("[{role}] single-chat fast path: ARF_MEGAKERNEL defaulted ON (B=1 → m=1 megakernel); ARF_NO_MEGAKERNEL=1 to opt out");
    }
    // SINGLEQ + the island fence: together they give the B=1 chat the island-embed + 4-byte
    // token read (~78-80 tok/s, reference-exact) instead of the 600KB logits readback (~67).
    // The fence (BLOCKING) is a CORRECTNESS requirement for that branch — without it the island
    // buffers pile up unwaited (nondeterministic reads + multi-second p99 stalls, measured).
    // ARF_NO_MEGA_SINGLEQ=1 opts out of both.
    // L1 (2026-08-04): the m=1 bridge now overrides this env for ITS OWN record only (NOWAIT ring
    // + per-step slot_done fence, see try_m1_megakernel); the batched fallback still honors it.
    // ARF_M1_BLOCKING=1 restores the fully-blocking m=1 behavior for A/B.
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_MEGAKERNEL").is_some()
        && std::env::var_os("ARF_NO_MEGA_SINGLEQ").is_none()
    {
        if std::env::var_os("ARF_MEGA_SINGLEQ").is_none() {
            std::env::set_var("ARF_MEGA_SINGLEQ", "1");
        }
        if std::env::var_os("ARF_MEGA_BLOCKING").is_none() {
            std::env::set_var("ARF_MEGA_BLOCKING", "1");
        }
        eprintln!("[{role}] single-chat fast path: SINGLEQ + island fence defaulted ON (island embed + 4-byte token read); ARF_NO_MEGA_SINGLEQ=1 to opt out");
    }
    // ── CONCURRENT-SERVING WIN CONFIG (2026-07-28/29) — default ON, opt out ARF_NO_WIN_CONFIG=1.
    // These are the measured H2H-winning set on the batched (multi-request) path, and the
    // reason the daemon needed them spelled out on the launch line until now:
    //   ATTN_COALESCED   32-lane coalesced KV co-load + NSG=8 flash attention.
    //   KV_F16           half-precision KV pool (HALF the KV DRAM traffic) + slot-list pool sync.
    //   ATTN_KVHEAD      one threadgroup per (kv_head, seq): the GQA group's 8 q-heads share ONE
    //                    staged KV read (traffic /8), with a measured occupancy auto-gate that
    //                    routes small batches to the split-K form.
    //   PREFILL_CHUNK_ROWS=256  (L107, 2026-08-08) fewer/larger prefill packs — kills the
    //                    expensive ~19-row tail pack (105ms fixed weight-stream). Verified real
    //                    HTTP g2_race GEN=128: conc8 180.6 vs llama 166.9 = **1.082× WIN**, user
    //                    text green. Code also unwrap_or(256); pinning here makes the value
    //                    visible in the env and in the startup line. Opt out: set the env to 128
    //                    (or any N) before launch; ARF_NO_WIN_CONFIG does not clear an explicit
    //                    PREFILL_CHUNK_ROWS (size knobs stay under user control).
    // MEASURED with the attn/kv set (a head-to-head run, PREFIX=512, alternating pairs vs
    // llama.cpp b9660 -fa on): conc64 1.10x (best 1.12x), conc32 1.04x, conc8 1.004x, conc16
    // 0.957x; and conc64 SUSTAINS 441 tok/s over 36 min (2026-07-29).
    // Correctness: full parity harness (119 blocks) + the live
    // ARF_KV_F16_SELFCHECK (token streams identical to the f32 oracle) are green on this set.
    // ARF_KV_F16 — CONCURRENCY-CONDITIONAL as of L135 (2026-08-12). See the block below the
    // loop; it is no longer a member of this unconditional set.
    #[cfg(target_os = "macos")]
    if std::env::var_os("ARF_MSL_GEMV").is_some() && std::env::var_os("ARF_NO_WIN_CONFIG").is_none()
    {
        for (k, v) in [
            ("ARF_ATTN_COALESCED", "1"),
            ("ARF_ATTN_KVHEAD", "1"),
            // L107 — shipping default for packed prefill chunk width (also unwrap_or(256) in
            // batch.rs). Only set when unset so an explicit ARF_PREFILL_CHUNK_ROWS=N wins.
            ("ARF_PREFILL_CHUNK_ROWS", "256"),
        ] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
        // ── ARF_KV_F16: CONCURRENCY-CONDITIONAL (L135, 2026-08-12) ───────────────────────
        // The f16 KV pool is a SECOND pool allocated ALONGSIDE the f32 staging pool
        // (weights.rs:1700) — +3.0 GB at --num-blocks 2048. The f32 pool cannot be dropped
        // instead: prefill scatters into AND attends it (batch.rs:1485). So it is a pure
        // memory ADD, and 9.0 GB of KV on a 36 GB Mac forced 4.3 GB of swap and OOM-rebooted
        // this box twice (L133).
        //
        // Whether that 3 GB buys anything depends ENTIRELY on concurrency. Measured ABA per
        // tier on a rebooted box, GEN 128, toggle verified AT THE ENGINE each arm
        // ([m1-kv-f16] engaged=true vs no f16 line at all — the env alone is not proof, which
        // is how a bad control arm slipped through earlier the same day):
        //
        //     conc1   OFF 59.8/59.9/59.9   ON 81.9/82.0/81.8   -> +37%  ⭐ BIG WIN for ON
        //     conc8   OFF 186.1/185.8      ON 184.5/184.3      -> -0.9% (OFF wins)
        //     conc16  OFF 198.4            ON 198.1            -> -0.15% (tie)
        //     conc32  OFF 281.9  ON 282.0  OFF 282.1           -> tie (L136, ABABA)
        //     conc64  🔴 UNMEASURABLE on a 36 GB box (L136)
        //
        // conc64 failed twice. The mechanism is known: the ON arms grew swap by 1,935 MB and
        // 757 MB, so the treatment degrades the machine it is being measured on, and the OFF
        // controls decay monotonically (384.8 -> 384.7 -> 368.8). Bounded at ~0.3% by the largest
        // gap any arm showed. Do NOT spend more time on it without a bigger box.
        //
        // The single-stream win is real and LARGE (+37%, arms replicate to 0.1%), and it
        // independently confirms nightly_gate's banked "+31.6%, off reads ~62.4". It makes
        // mechanical sense: at batch 1 the decode is KV-bandwidth-bound, so halving KV bytes
        // per token pays directly. Under concurrency the batch amortizes weight streaming and
        // KV bandwidth stops being the binding constraint — the win goes to zero.
        //
        // L5b's ON-by-default was therefore RIGHT for conc1 and WRONG about the concurrent
        // tiers it cited (+7.0%/+1.4%/+4.2% do not reproduce; conc8 flips sign). Neither a
        // blanket ON nor a blanket OFF is correct: ON costs 3 GB for nothing at high
        // concurrency, OFF costs 37% on the most common interactive case (one agent typing).
        //
        // Default: ON at or below ARF_KV_F16_MAX_BATCH (8), OFF above it. Explicit
        // ARF_KV_F16=1 / ARF_NO_KV_F16=1 always win.
        // ⚠️ Polarity trap: downstream gates are is_some()-based, so ARF_KV_F16=0 would
        // still ENABLE it — the opt-out must be ARF_NO_KV_F16=1.
        if std::env::var_os("ARF_KV_F16").is_none() && std::env::var_os("ARF_NO_KV_F16").is_none() {
            // Runs before Args::parse(), so read --max-batch-size from argv directly.
            let argv: Vec<String> = std::env::args().collect();
            let batch = argv
                .iter()
                .position(|a| a == "--max-batch-size")
                .and_then(|i| argv.get(i + 1))
                .and_then(|v| v.parse::<usize>().ok())
                .or_else(|| {
                    argv.iter()
                        .find_map(|a| a.strip_prefix("--max-batch-size=")?.parse::<usize>().ok())
                })
                .unwrap_or(1);
            let cutoff = std::env::var("ARF_KV_F16_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8);
            if batch <= cutoff {
                std::env::set_var("ARF_KV_F16", "1");
            }
        }
        // L134: report the RESOLVED state, not the intent. This banner used to claim "f16 KV
        // [L5b]" even under ARF_NO_KV_F16=1, where the f16 pool is never allocated. A log used
        // to audit which path ran must not say the opposite of what ran.
        eprintln!(
            "[{role}] concurrent win config defaulted ON (coalesced attn + {} + \
kv-head attn + prefill_chunk_rows={} [L107]) — ARF_NO_WIN_CONFIG=1 to opt out",
            if std::env::var_os("ARF_KV_F16").is_some() {
                "f16 KV ON [L135: +37% at conc1, costs ~3.0GB @2048 blocks]"
            } else {
                "f16 KV OFF [L135: a tie above batch 8 — saves ~3.0GB]"
            },
            std::env::var("ARF_PREFILL_CHUNK_ROWS").unwrap_or_else(|_| "256".into())
        );
    }
}

/// Both invariants here are ones an EXPLICIT env var used to be able to switch off, and both
/// failure modes are silent wrong output rather than an error — so a test is the only thing that
/// keeps them. Each case below was reproduced on hardware before being written down; the expected
/// strings in the comments are what the engine actually printed.
///
/// These call the real defaulting rather than an extracted pure copy, because the bug was never in
/// the decision — it was in the CONDITION GUARDING the decision, which a reimplementation would
/// just reproduce. That means mutating process env, so they take a lock and restore every variable
/// they touch.
#[cfg(all(test, target_os = "macos"))]
mod lever_contradiction_tests {
    use super::apply_fast_path_defaults_for;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const TOUCHED: [&str; 7] = [
        "ARF_MSL_GEMV",
        "ARF_NO_MSL_GEMV",
        "ARF_MEGAKERNEL",
        "ARF_NO_MEGAKERNEL",
        "ARF_MEGA_SINGLEQ",
        "ARF_MEGA_BLOCKING",
        "ARF_KV_F16",
    ];

    /// Run the defaulting with exactly `set` present and launched as `arch`, and report the
    /// resolved values of `ARF_MSL_GEMV` / `ARF_MEGAKERNEL`. Restores the ambient env on the
    /// way out so the rest of the suite is unaffected.
    fn resolve_as(arch: Option<&str>, set: &[(&str, &str)]) -> (bool, bool) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&str, Option<std::ffi::OsString>)> =
            TOUCHED.iter().map(|k| (*k, std::env::var_os(k))).collect();
        for k in TOUCHED {
            std::env::remove_var(k);
        }
        for (k, v) in set {
            std::env::set_var(k, v);
        }
        apply_fast_path_defaults_for("test", arch.map(str::to_string));
        let out = (
            std::env::var_os("ARF_MSL_GEMV").is_some(),
            std::env::var_os("ARF_MEGAKERNEL").is_some(),
        );
        for (k, v) in saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        out
    }

    /// The common case: no `--arch` on the launch line, so the head_dim guard has nothing to
    /// look up and only the lever logic runs.
    fn resolve(set: &[(&str, &str)]) -> (bool, bool) {
        resolve_as(None, set)
    }

    /// `ARF_NO_MSL_GEMV=1 ARF_MEGAKERNEL=1` — the megakernel dispatches island kernels for
    /// weights whose island views were never built. MEASURED on llama-3.2-1b, greedy, "The capital
    /// of France is": " Paris. The capital of France is Paris." without the pair,
    /// "!!!!!!!!!!!!!!" with it.
    #[test]
    fn megakernel_cannot_outlive_the_island_it_runs_on() {
        let (msl, mega) = resolve(&[("ARF_NO_MSL_GEMV", "1"), ("ARF_MEGAKERNEL", "1")]);
        assert!(!msl, "ARF_NO_MSL_GEMV must leave the island off");
        assert!(
            !mega,
            "an explicit ARF_MEGAKERNEL must be cleared when the island is off — the pair \
             emits fluent garbage, and the defaulting block alone never sees an explicit value"
        );
    }

    /// The head_dim guard must fire even when the launcher asked for the island — `make dev` and
    /// `make serve` both pass `ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1` on the launch line, which is
    /// precisely the case a `ARF_MSL_GEMV.is_none()` condition would skip. MEASURED on
    /// llama-3.2-1b (head_dim 64), greedy, "The capital of France is": " Paris. The capital of
    /// France is Paris." on the portable path, "!!!!!!!!!!!!!!" under that launch line.
    #[test]
    fn explicit_msl_gemv_cannot_waive_the_head_dim_rule() {
        // Only meaningful if llama-3.2-1b is still the ineligible arch this pins.
        let cfg = arf_core::config::config_for_arch("llama-3.2-1b").expect("arch");
        assert!(
            cfg.island_unsupported_head_dim().is_some(),
            "llama-3.2-1b is the ineligible-head_dim fixture; if it became eligible, repoint this \
             test at whatever `island_unsupported_head_dim` now rejects rather than deleting it"
        );

        // The launch line everybody actually uses: both levers asked for by name.
        let (msl, mega) = resolve_as(
            Some("llama-3.2-1b"),
            &[("ARF_MSL_GEMV", "1"), ("ARF_MEGAKERNEL", "1")],
        );
        assert!(
            !msl,
            "an explicit ARF_MSL_GEMV must NOT waive the head_dim rule — the island cannot \
             address a head_dim-64 KV row, and the measured output under this exact launch line \
             is `!!!!!!!!!!!!!!`"
        );
        assert!(
            !mega,
            "clearing the island must take the megakernel with it: the head_dim guard sets \
             ARF_NO_MSL_GEMV, and NO_MSL_GEMV + explicit MEGAKERNEL is the fluent-garbage pair"
        );

        // The same arch with nothing on the launch line must land identically — the guard is a
        // property of the model, not of who asked.
        let (msl, mega) = resolve_as(Some("llama-3.2-1b"), &[]);
        assert!(!msl && !mega, "the guard must fire on a bare launch too");
    }

    /// `ARF_NO_MSL_GEMV=1 ARF_MSL_GEMV=1` — the opt-out must WIN, not just print that it
    /// did. Nothing downstream reads `ARF_NO_MSL_GEMV`; `weights.rs` builds the island under a
    /// bare `ARF_MSL_GEMV.is_some()`, so leaving the positive lever set means the daemon
    /// announces the portable path and runs the island. Reachable from `make dev` by adding the
    /// opt-out to a line whose launch env already carries `ARF_MSL_GEMV=1`.
    #[test]
    fn the_opt_out_clears_the_lever_it_overrides() {
        let (msl, mega) = resolve_as(None, &[("ARF_NO_MSL_GEMV", "1"), ("ARF_MSL_GEMV", "1")]);
        assert!(
            !msl,
            "ARF_NO_MSL_GEMV must REMOVE ARF_MSL_GEMV — every downstream island gate reads \
             the positive lever alone, so leaving it set builds the island the user opted out of"
        );
        assert!(
            !mega,
            "and the megakernel must not survive the island either"
        );
    }

    /// The guard is scoped to the ineligible arch and nothing else: an eligible model launched
    /// the same way must keep both levers, or the fix above would be a global off-switch.
    #[test]
    fn the_head_dim_guard_does_not_touch_an_eligible_arch() {
        let cfg = arf_core::config::config_for_arch("gemma-3-4b").expect("arch");
        assert!(
            cfg.island_unsupported_head_dim().is_none(),
            "gemma-3-4b is the eligible fixture here; repoint if its head_dim ever changes"
        );
        let (msl, mega) = resolve_as(Some("gemma-3-4b"), &[]);
        assert!(
            msl && mega,
            "an island-eligible arch must still get the fast path by default"
        );
    }

    /// The README quickstart: a model resolved from its own `config.json`, so no `--arch` on the
    /// launch line. MEASURED before this test existed: `arf run llama3.2:1b` at defaults printed
    /// `!!!!!!!!` on the GPU, because only the argv guard existed. The config entry point must land
    /// on the same levers as the `--arch` guard.
    #[test]
    fn a_config_without_arch_still_declines_the_island() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&str, Option<std::ffi::OsString>)> =
            TOUCHED.iter().map(|k| (*k, std::env::var_os(k))).collect();
        for k in TOUCHED {
            std::env::remove_var(k);
        }
        let cfg = arf_core::config::config_for_arch("llama-3.2-1b").expect("arch");
        super::decline_fast_path_for_model("test", &cfg);
        apply_fast_path_defaults_for("test", None);
        let (msl, mega) = (
            std::env::var_os("ARF_MSL_GEMV").is_some(),
            std::env::var_os("ARF_MEGAKERNEL").is_some(),
        );
        // and an eligible model is untouched by the config entry point
        for k in TOUCHED {
            std::env::remove_var(k);
        }
        let ok = arf_core::config::config_for_arch("gemma-3-4b").expect("arch");
        super::decline_fast_path_for_model("test", &ok);
        apply_fast_path_defaults_for("test", None);
        let (msl_ok, mega_ok) = (
            std::env::var_os("ARF_MSL_GEMV").is_some(),
            std::env::var_os("ARF_MEGAKERNEL").is_some(),
        );
        for (k, v) in saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        assert!(
            !msl && !mega,
            "a head_dim-64 config must leave the island and the megakernel off"
        );
        assert!(
            msl_ok && mega_ok,
            "an eligible config must still get the fast path"
        );
    }

    /// The levers the fast path is FOR must survive all of this: nothing above should touch a
    /// launch on a model the island can actually run.
    #[test]
    fn a_normal_launch_still_gets_the_fast_path() {
        let (msl, mega) = resolve(&[]);
        assert!(
            msl,
            "the island must still default ON for an ordinary launch"
        );
        assert!(mega, "the m=1 megakernel must still default ON with it");

        let (msl, mega) = resolve(&[("ARF_MSL_GEMV", "1"), ("ARF_MEGAKERNEL", "1")]);
        assert!(
            msl && mega,
            "an explicit fast-path request must pass through"
        );
    }
}
