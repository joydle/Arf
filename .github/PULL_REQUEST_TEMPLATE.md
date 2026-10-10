## What and why

<!-- What changes, and what problem it solves. If it fixes a bug, say what the bug did. -->

## The gate

<!-- CLAUDE.md / CONTRIBUTING.md. Each command reaches a different set of targets; always
     --workspace (`cargo build -p <crate>` can say "Finished" without compiling it).
     `make check` runs every CI gate in CI's order. -->

- [ ] `cargo build --workspace` — 0 warnings
- [ ] `cargo clippy --workspace --all-targets` — 0 warnings
- [ ] `cargo test --workspace` — passes
- [ ] `make check` — passes

## If this touches the decode path

<!-- gpu/batch.rs, gpu/metal/island.rs, gpu/decode.rs, or any shader. Delete if it doesn't. -->

- [ ] Output text is correct at **b=1** and at **b≥2** (checked before measuring speed)
- [ ] `md5 crates/arf-gpu/src/shaders/metal/matmul_vec_q4ks_batch_msl.metal` is unchanged,
      or the change is explained here:

## If this claims a performance change

<!-- Delete if it doesn't. docs/PERFORMANCE.md says how numbers are taken here. -->

- [ ] A/B arms **interleaved in one session** — this box drifts more between sessions than most wins
- [ ] `vm.swapusage` and load average checked before trusting the numbers
- [ ] The log shows the feature was actually dispatched in the B arm
- [ ] Every figure quoted here comes with its machine, date and command

Before/after, with the exact command:

```
```

## Notes

<!-- Anything that did NOT work is worth keeping. If you tried an approach and it lost, say so
     here and leave a comment in the code saying what was measured, so nobody re-chases it. -->
