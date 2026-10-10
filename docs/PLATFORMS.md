# Platforms — what runs where, and what has actually been verified

This file exists because "supports Linux" is the kind of claim that is easy to imply and hard to
retract. Every row below says which of three things is true, and they are not the same thing:

| tier | means |
|---|---|
| **MEASURED** | run on real hardware, with recorded numbers |
| **CHECKED** | compiles, lints and passes CPU tests in CI — no GPU execution proven |
| **UNVERIFIED** | code exists, nobody has run it. Not a promise. |

---

## macOS (Apple Silicon) — MEASURED

The shipping target. The native-Metal island (`arf-gpu/src/gpu/metal/island.rs`) carries
every fast path: the batched megakernel, B-row attention, the sort-by-expert MoE GEMM, speculative
decode through the MTP head.

Seven models raced against llama.cpp on the same GGUFs, same box, interleaved arms. CI runs on `macos-14`.

## Linux (CPU + build) — CHECKED, as of L364

The workspace **compiles on Linux across every target** — library, tests, examples and benches
(`cargo check --workspace --all-targets`: 0 errors). That was not true before 2026-09-08:

| stage | errors before |
|---|---:|
| `cargo check --workspace` | **32** |
| `--all-targets` (after the lib was fixed) | **9 more**, all in one island benchmark |

Every one had the same root cause: code outside `#[cfg(target_os = "macos")]` naming island-only
struct fields and methods. Nothing caught it because CI was macOS-only — and the `--all-targets`
nine are the reason this repo's own rule exists that `cargo build` is not `cargo test` is not
`clippy --all-targets`.

### Dead code off macOS — why it is a lint policy and not 20 more `cfg`s

With the errors gone, Linux still reported ~20 dead-code warnings: the speculation drafter, the
depth-2 bank ring, the GDN helpers all exist solely to serve the island, so off macOS they are
genuinely unreachable.

**Per-item `#[cfg(target_os = "macos")]` was tried first and abandoned.** It cascades — gating a
type dead-ends its `impl`, then the struct field holding it, then that field's initializer, then
the helpers that touched it. Eight rounds in it was still not closed, and every island helper
added later would join the queue.

The fix is one crate-level attribute in `arf-gpu/src/lib.rs`, applying **only off macOS**:

```rust
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports, unused_variables))]
```

On macOS these lints stay denied by CI, which is where they can catch something real — the island
*is* compiled there, so genuinely unused code still surfaces on the platform that ships.

Verify it yourself in a real Linux container, no cross-toolchain needed:

```sh
make linux-check     # cargo check --workspace
make linux-lint      # clippy -D warnings, what CI runs
make linux-test      # the CPU suite
make linux-shell     # poke around with the same cached volumes
```

**Test suite on Linux: every suite passes, 0 failures** (run in the container above, 2026-09-08).
CI runs the same commands on `ubuntu-latest` on every push and PR, so it stays fixed.

### What is NOT proven

**The portable wgpu path has never decoded on Linux.** The wgpu/WGSL path (121 WGSL shaders) is
compiled and type-checked there and should reach Vulkan, but "should" is doing real work in that
sentence — nobody has loaded a model through it on a Linux GPU and read the output. Until someone
does, this is CHECKED, not MEASURED, and it should not be described as working.

Two paths are macOS-only by construction and will refuse rather than silently misbehave:

- **Hybrid (GDN) CLI generation** — every token must advance the recurrent state through the
  island's megakernel. Off macOS `generate_hybrid` panics with a message saying so.
- **Speculative decode via the MTP head** — reads raw-Metal weight views and the depth-2 bank
  ring. Off macOS `mtp_draft_chain` returns an empty draft, which the caller already treats as
  "no speculation this step" (the same thing that happens on Mac when the head declines).

## NVIDIA

An NVIDIA backend is developed separately and is not part of this repository.

---

## Why there is no `arf-metal` crate

Reasonable question — `arf-gpu` is 47k lines of which 33k touch Metal, so the name is a lie.
It was measured before being rejected:

| | |
|---|---|
| public items in the island module | 274 |
| `cfg(target_os = "macos")` gates to re-home | 241 |
| `as_hal::<Metal>()` calls — the island borrowing wgpu's device | 48 |
| crates outside `arf-gpu` that name `MetalIsland` | **0** |

The island does not sit *on top of* wgpu, it reaches *through* it: it borrows wgpu's device and
reads the same allocations through raw Metal handles. A `arf-metal` crate would therefore
still depend on `arf-gpu`, and since nothing outside the crate references `MetalIsland`, the
split would move 33k lines to fix a name and change no structure.

The house rule covers it — prefer the contiguous move over the rename, and measure a function's
shape before splitting it (`CLAUDE.md`).

### What did move (L364b), and what deliberately did not

| change | why |
|---|---|
| `arf-wgpu` → `arf-gpu` | 637 references, purely mechanical; the name now describes both backends |
| `shaders/` → `shaders/{metal,wgsl}/` | 145 files in one directory; a pure file move, `include_str!` paths updated |
| the 3 Metal-pure files → `gpu/metal/` | 21.4k lines no portable code touches: `island.rs`, `selftest.rs`, `shim.rs`. Existing `concurrent_metal::` paths still resolve via a re-export, so nothing outside the folder changed |

**`types.rs`, `batch.rs`, `decode.rs` and `ssm_qwen35.rs` stayed put.** They interleave portable
code with macOS branches — 27, 37, 19 and 32 `cfg` gates respectively — so filing them under
`metal/` would misdescribe them and separating them is the 33k-line split rejected above. A folder
structure should tell the truth about its contents or stay out of the way.
