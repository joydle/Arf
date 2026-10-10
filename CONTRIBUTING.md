# Contributing to Arf

Thanks for your interest. Arf is a from-scratch LLM inference engine in Rust
(native-Metal + wgpu). Bug reports, model requests and performance reports are welcome as
[issues](https://github.com/joydle/Arf/issues/new/choose).

This repository is the published release: its `main` branch is locked and does not take pull
requests. Changes ship in new releases, made by the maintainers. If you want to propose a change,
open an issue that describes it, with the measurements the rules below ask for; a fork is the place
to try it.

## Read this first

The ground rules below are the working contract for this codebase; [CLAUDE.md](CLAUDE.md) is the
short form that the maintainers' tools read, and it applies to human contributors too.

Before proposing a performance change, read the comments around the code you want to change: where an
idea was tried, measured and lost, the code says so, so nobody re-chases it.

## Ground rules

**Correctness gates speed.** Every kernel is verified bit-close against a CPU reference. A
faster-but-wrong kernel is worthless — it makes the model output garbage. So:

- Any change to a compute path must **pass the parity suite** (see below) and produce correct
  text at batch size 1 **and** at batch size 2 or more. Verify the text before measuring speed.
- New kernels ship with a parity gate (a CPU oracle + a selftest). Look at the
  existing `*_check` / `*_selftest` methods in
  `crates/arf-gpu/src/gpu/metal/selftest.rs` for the pattern.
- `crates/arf-gpu/src/shaders/metal/matmul_vec_q4ks_batch_msl.metal` is on the path every token
  takes. Its md5 stays unchanged unless your PR says why it changed.

**Numbers are measured, never projected.**

- **No number without its method.** Every tok/s figure comes with the machine, the date and the
  command that produced it ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) shows the shape). A figure
  without them is unverified: say so, or leave it out.
- **Interleave A/B arms in one session.** Machines drift several percent between sessions; a
  number from twenty minutes ago is not a baseline. Check load average and swap first.
- **Prove the instrument.** Confirm from the log that the feature you are measuring actually
  ran. A green arm in which the feature did not dispatch is not evidence.
- **Retract in public.** If a published number turns out wrong, correct it where it was
  published and say what replaced it.
- **Keep the comments that record failures** (`MEASURED-OUT`, `DO NOT RE-CHASE`, retractions).
  They are the most valuable text in the tree; do not delete them to tidy up.

## Build & test

```sh
cargo check --workspace          # first move — must be clean
cargo build --release            # release build (debug is far too slow for GPU work)
cargo test --workspace           # CPU unit + integration tests
cargo test -p arf-gpu --test gpu   # GPU parity vs CPU (skips cleanly with no GPU adapter)
```

Toolchain: pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (rustup picks it up
automatically). The minimum supported Rust version is 1.89 (`rust-version` in `Cargo.toml`,
enforced by the `msrv` job in CI). Metal features are `#[cfg(target_os = "macos")]`; the wgpu
path is portable.

## The gate: CI must pass

`cargo build`, `cargo test` and `cargo clippy --all-targets` each reach a different set of
targets, so all three are needed. Always pass `--workspace`: `cargo build -p <crate>` can report
"Finished" without compiling the crate you meant.

```sh
make check    # every CI gate, in CI's order: fmt, clippy (with and without `profiling`),
              # `cargo test --workspace --locked`, `cargo doc` with -D warnings
```

or by hand:

```sh
cargo fmt --all
cargo build --workspace                  # 0 warnings
cargo clippy --workspace --all-targets   # 0 warnings; CI denies them — fix, don't #[allow] blindly
cargo test --workspace
```

## Running the engine

```sh
# single generation
cargo run --release -p arf-cli -- generate --model <path> --prompt "..."
# the serve daemon (OpenAI-compatible /v1/)
make dev-conc        # coder-30B concurrency demo (macOS, native-Metal fast path)
```

See the [README](README.md) and [docs/](docs/) for the full tour. Kernel layout
and what each shader is for: [docs/KERNELS.md](docs/KERNELS.md). Runtime flags:
[docs/ENV_SUPPORTED.md](docs/ENV_SUPPORTED.md) is the supported contract;
[docs/ENV_INVENTORY.md](docs/ENV_INVENTORY.md) is the generated census of every variable the
code reads.

## Proposing a change

- One focused change per issue.
- Explain **what** and **why**; if it's a perf change, include before/after
  numbers from interleaved arms, the exact command, the machine and its load and swap.
- If something you tried did not work, say so — what was measured is as useful as what won.
- Match the surrounding code style (the codebase favors clear names, dense but
  purposeful comments that state constraints, and parity gates over trust).
- New env flags are discouraged — prefer a measured default. If you must add one: explain
  what it does and why in a comment at the `env::var` call site, then run
  `python3 scripts/gen_env_inventory.py` so [docs/ENV_INVENTORY.md](docs/ENV_INVENTORY.md)
  picks it up (that file is generated — do not hand-edit it). Add it to
  [docs/ENV_SUPPORTED.md](docs/ENV_SUPPORTED.md) only if it is user-facing and you intend to
  keep it stable.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you agree
to uphold it.

## Security

Report vulnerabilities privately — see [SECURITY.md](SECURITY.md), which also states the threat
model: model files are parsed input, and the server is a local development daemon with no auth.

## License

By contributing, you agree your contributions are licensed under the project's
[Apache-2.0](LICENSE) license.
