# Working in this repository

Guidance for agents and contributors. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the full contributor guide.

## The rules, short

1. **Correct text before speed.** Verify the output first. A speed number from a path that produces wrong
   text is worse than no number.
2. **Measure on one machine, in one session, interleaved.** A Mac drifts several percent between sessions and
   more once it swaps; a number from twenty minutes ago is not a baseline. Check `vm.swapusage` and the load
   average before trusting anything. [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) says how the published
   numbers were taken.
3. **Verify the thing, not the proxy.** A flag that changes nothing looks exactly like a flag that works:
   confirm in the log that the feature actually ran before quoting what it did.
4. **Keep the comments that record what was tried and measured out.** They stop the next person from
   rebuilding something that already lost.
5. **Size the share of the whole before building.** A large speed-up of a small stage is worth little; compute
   `stage / total` first.

## Before you claim it builds

`cargo build` is not `cargo test` is not `clippy --all-targets`. Each reaches a different set of targets.

```sh
cargo build --workspace                  # must be 0 warnings
cargo clippy --workspace --all-targets   # must be 0 — this is what CI runs
cargo test --workspace
```

`cargo build -p arf-gpu` reports "Finished" **without compiling the crate**. It is not a verification. Always
use `--workspace`.

`make check` runs every CI gate in CI's order (fmt, clippy with and without the `profiling` feature,
`cargo test --workspace --locked`, `cargo doc` with `-D warnings`). The toolchain is pinned in
`rust-toolchain.toml`; `make disk` / `make disk-gc` show and free regenerable space.

## Touching the decode path

`gpu/batch.rs`, `gpu/metal/island.rs`, `gpu/decode.rs` and the shaders are the path every token takes. Any
change there needs, at minimum:

- `md5 crates/arf-gpu/src/shaders/metal/matmul_vec_q4ks_batch_msl.metal` unchanged
  (`f2416500fd51658ff620c62b77e16539` since 2026-10-05, when one comment pointing at a removed document was reworded and
  nothing else changed), or a stated reason it changed
- correct text out at b=1 and b≥2
- an interleaved A/B if a speed claim is being made at all

## Refactoring

Prefer the contiguous move over the rename. Measure a function's shape before splitting it. A change to a
public signature is an API change, not a refactor, and examples are callers too.

## This repository is public

Everything committed here is published. Write each line as if a stranger reads it tomorrow.

**Never commit:** traces of how the work was done by tools (agent, session, workflow or memory ids; scratch or
temp-directory paths; transcripts); home-directory paths, hostnames, IP addresses, emails (except the project
contact), tokens; private names or plans; a citation of a script that is not in the repository.

**Before every push:** `make check`, then `make leak-scan`. Run `make hooks` once per clone so the pre-push
hook runs the scan for you; it also reads an optional local denylist (`~/.config/arf/denylist`). A hit is fixed,
not bypassed.

**Changes land by pull request** against `main`, with the PR template's gates filled in.
