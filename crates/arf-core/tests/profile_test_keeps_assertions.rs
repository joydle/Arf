//! L358d — proof that making tests FAST did not make them WEAKER.
//!
//! `[profile.test] opt-level = 2` was added because the workspace suite was effectively
//! unrunnable: `gpu_vision_encoder_matches_cpu_real_gguf` burned 32 minutes of CPU in 10 minutes
//! of wall clock at 98.9%, all inside the unoptimised CPU reference SigLIP tower, and CI runs
//! `cargo test --workspace`. A gate people skip is not a gate.
//!
//! But optimising a test profile is exactly the kind of change that can quietly delete the
//! checks it was meant to preserve. `debug_assert!` compiles to nothing without
//! `debug-assertions`, and `ssm_qwen35.rs` alone has NINE of them guarding the gated-delta-net
//! recurrence — the destructive per-row state that a wrong verify window corrupts into *fluent*
//! garbage rather than a crash. Silently losing those is a worse outcome than a slow suite.
//!
//! `profile.test` inherits `dev`, where both settings are `true`, and setting `opt-level` alone
//! does not change either. That is the theory. These tests are the measurement — the whole point
//! of L358 was that a check which cannot fail is not evidence, so this one is written to fail if
//! anyone ever adds `debug-assertions = false` to chase compile time.

/// `debug_assert!` must still be compiled IN.
#[test]
#[allow(clippy::assertions_on_constants)]
fn debug_assertions_are_live_under_profile_test() {
    // L364 — clippy sees a constant here (on a given build `cfg!(debug_assertions)` IS constant)
    // and that is exactly the point: this test asserts the BUILD CONFIGURATION, not a runtime
    // value. It exists because [profile.test] once compiled `debug_assert!` out and silently
    // stopped checking the 9 assertions guarding the GDN recurrence. Denied warnings on Linux
    // surfaced it; allowing it here keeps the test that catches a real regression.
    //
    // The allow is UNCONDITIONAL rather than `not(target_os = "macos")`. It was scoped to Linux
    // because that is where it first fired, but the lint is a property of the assertion, not of
    // the platform — a new enough clippy raises it on macOS too, which broke
    // `cargo clippy --workspace --all-targets` on the machine most of this is developed on. A
    // platform-scoped allow for a platform-independent lint is a gate that fails somewhere else
    // later.
    assert!(
        cfg!(debug_assertions),
        "debug_assert! is compiled OUT under [profile.test] — the 9 debug_assert!s guarding the \
         GDN recurrence in ssm_qwen35.rs are no longer checked by any test run"
    );
}

/// Arithmetic overflow must still panic rather than wrap. Wrapping silently is how a KV slot
/// index or a past-length becomes valid-looking nonsense.
#[test]
fn overflow_checks_are_live_under_profile_test() {
    let r = std::panic::catch_unwind(|| {
        let x: u8 = 255;
        std::hint::black_box(x) + std::hint::black_box(1)
    });
    assert!(
        r.is_err(),
        "u8 overflow did not panic — overflow-checks are off under [profile.test], so wrapping \
         indices would go undetected in every test in this workspace"
    );
}
