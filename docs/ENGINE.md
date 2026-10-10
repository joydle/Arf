# The Arf engine

The complete technical account: the constraint, the mathematics, the mechanisms, the models, and
the measured results. Every claim below carries a citation to the file that implements it or the
row that measured it.

**Verification status.** Every formula was checked against the implementing source; every number was
recomputed from its inputs. Where a figure could not be traced to a measurement row it is marked
**projected** or omitted. Discrepancies found during verification are recorded in §12.

Companion documents: [KERNELS.md](KERNELS.md) for per-kernel detail, [PERFORMANCE.md](PERFORMANCE.md) for the
measured results and how they are taken.

---

## 1. The constraint

Generating one token requires reading **every weight in the model**. A single token is one row of
activations against the whole weight set — there is no reuse to exploit.

```
18.6 GB streamed per token  ÷  410 GB/s (M4 Max)  =  22.04 tok/s
measured 20.0                                     =  90.7% of that
```

*Measured. The 18.6 GB is the transcoded resident sidecar, not the 17.11 GB
GGUF on disk — see §12.1.*

Three consequences shape every decision that follows:

| consequence | why | evidence |
|---|---|---|
| Arithmetic is nearly free | the ALU idles waiting on memory | five separate ALU-for-bytes trades all lost — §5.2 |
| Batching is nearly free | a second sequence reads the same weights | marginal row cost 0.29 of a step |
| Only more tokens per pass beats the wall | the ceiling is per weight-pass | §8 |

---

## 2. The stack

Six crates, acyclic. Each depends only on layers beneath it, and the compiler enforces that.

```
  arf-serve   arf-cli    arf-router      binaries
       │              │         (no engine deps)
       ├──────────────┘
  arf-gpu                                  backend
       │
         arf-core                              engine
              │
          arf-ml                               numerics
```

An NVIDIA backend is developed separately and is not part of this repository. A backend plugs in behind `BatchedBackend`  and depends
on `arf-core` alone.

| crate | lines | holds |
|---|---:|---|
| `arf-ml` | 4,073 | f32 tensor, block-quant formats, TurboQuant, norms, CPU matmul |
| `arf-core` | 17,737 | model logic, GGUF/safetensors, paged KV, scheduler, sampling, tokenizer |
| `arf-gpu` | 63,733 | Metal island + portable wgpu, 147 kernels, FLUX, vision |
| `arf-serve` | 7,561 | OpenAI-compatible HTTP server, actor loop |
| `arf-cli` | 4,931 | `pull` / `run` / `serve` |
| `arf-router` | 2,098 | optional fleet front, zero engine dependencies |

**Why the numerics are their own crate.** They had no dependency on the rest of the engine — the
separation existed in fact, but nothing enforced it. Splitting `arf-ml` out makes the compiler
check it, and gives the CPU kernels a second job: they are the **correctness oracle** every GPU
kernel is tested against.

The boundary is a rule, not a habit: **`arf-ml` may not know what a transformer is.** RMSNorm and
QK-norm live there because they are pure numerics. RoPE does not, because it needs `ModelConfig` for
its scaling variants; nor do MLP/MoE/attention, which need `model::Linear` and the cache.

**Why two GPU backends in one crate.** `arf-gpu` contains a portable WebGPU path and a native
Metal path implementing the same operations. The portable one runs anywhere and is easier to reason
about; the native one is what ships on macOS. Keeping both means every Metal kernel has a reference
to be checked against.


## 3. Attention

### 3.1 The form

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
```

The `√d` divisor: if `q` and `k` have unit-variance components, `q·k` over `d` dimensions has
variance `d`. At `d = 128` that is a standard deviation of **11.3**, wide enough to saturate the
softmax toward one-hot. Dividing by `√d` restores unit variance.

*Verified numerically: `sqrt(128) = 11.31`.*

**The cost.** `Q Kᵀ` is `n × n` — doubling context quadruples the work. Caching K and V makes
generation linear per token instead of quadratic, but the cache then grows with the sequence and so
does the per-token read.

### 3.2 Grouped-query attention

Several query heads share one key/value head. Qwen3.8: **24 query heads, 4 KV heads = 6× reduction**
in both KV cache size and bytes read per token.

*Source: `arf_core::config::ModelConfig::qwen35_27b` — `num_attention_heads: 24`,
`num_kv_heads: 4`.*

### 3.3 Rotary position embeddings

Attention's dot product is permutation-invariant. RoPE encodes position by rotating each dimension
pair:

```
θᵢ = p · base^(−2i/d)

[a']   [cos θᵢ  −sin θᵢ] [a]
[b'] = [sin θᵢ   cos θᵢ] [b]
```

The identity that makes it work:

```
⟨R(m) q, R(n) k⟩ = ⟨q, R(n − m) k⟩
```

Absolute positions go in; **relative** position falls out of the arithmetic. Nothing is learned.

*Source: `shaders/wgsl/rope.wgsl`. Identity verified numerically to machine precision (`|lhs − rhs| <
1e-12`).*

⚠️ **Pairing is a convention, not a law.** Half-split pairs `(xᵢ, x₍ᵢ₊d/2₎)`; interleaved pairs
`(x₂ᵢ, x₂ᵢ₊₁)`. Same angles, different memory stride. Applying the wrong one silently scrambles
every query and key — this cost the project a real bug (L266).

---

## 4. The delta rule

A gated delta net replaces the growing KV cache with a fixed `d × d` state matrix per head, edited
as tokens arrive. Cost per token does **not** depend on sequence length.

```
1. decay     S ← exp(g) · S              g = learned per-head gate
2. recall    ŝ = Sᵀ k                    what S currently returns for this key
3. error     d = β · (v − ŝ)             how wrong that is
4. correct   S ← S + k dᵀ                rank-1 write fixing exactly that error
   read out  o = Sᵀ q / √dₖ
```

*Source: `shaders/metal/gated_delta_net_msl.metal`, a bit-exact port of llama.cpp
`src/models/delta-net-base.cpp::build_delta_net_autoregressive`, whose per-element derivation the
kernel header reproduces line by line.*

**Step 3 is the idea.** The update is proportional to the *delta* between what memory already
returns and what should be stored. If `S` already maps `k → v`, then `v − ŝ ≈ 0` and almost nothing
is written — capacity is spent only on what is new. Plain linear attention (`S ← S + k vᵀ`) writes
the full value every time and saturates; repeated keys blur together.

### 4.1 The hybrid trade

| | state | read cost | recall |
|---|---|---|---|
| Full attention | O(n) | O(n) | exact |
| Gated delta net | O(d²) | O(d²) | lossy, gated |

Qwen3.8 interleaves them on a period of 4: **48 recurrent + 16 attention layers**. Most of the depth
stays constant-cost at long context; a quarter keeps exact recall.

*Source: `qwen35_27b()` — `num_layers: 64`, `full_attention_interval: 4`. Verified: layers 3, 7,
11, … are attention, giving 16 of 64.*

⚠️ **The recurrence is serial** — token *t+1* depends on the state left by *t*. State must be banked
per **sequence**, not per batch row. Keyed by row, row 0 is sequence A one step and B the next, each
inheriting the other's memory: four concurrent sequences produced garbage while the same four run
serially were perfect.

---

## 5. The mechanisms

### 5.1 The Metal island

The portable path dispatches through wgpu — a bind group, pipeline set and dispatch per kernel per
layer, hundreds of encoder operations per token across 64 layers.

The **island** is a second native-Metal implementation that records the whole decode step into one
command buffer, binding raw `MTLBuffer` handles. Same math, same kernels, no abstraction. Default on
macOS; the reason two backends exist.

Command buffers are chunked at 8 layers — not for speed, but because recording every layer into one
buffer trips Metal's watchdog and hangs the device.

*Source: `gpu/metal/island.rs`, `flush_k` default 8.*

### 5.2 GEMV loop order — the largest single win

Decode is one activation row against a huge quantized matrix. Column-outer re-reads the activation
row once per output column. **Row-outer / column-inner** loads each row's eight `float4` into named
registers once and dots them against all columns in the tile.

Identical arithmetic, identical weight bytes. Only activation traffic changes.

| | marginal cost of a second batch row |
|---|---:|
| column-outer | 0.96 of a full step |
| row-outer, `COLS=4` | **0.29** |

*3.31× improvement. Source: `shaders/metal/matmul_vec_q4ks_batch_msl.metal`; measured.*

**What lost**, each measured:

| attempt | result | why |
|---|---:|---|
| Two output columns per threadgroup, b=1 | −25% | traded spare parallelism for unneeded reuse |
| Hoist column-independent sum to a prepass | **−31%** | removed 99.994% of a computation, got slower |
| Q3_K instead of Q4_K | −4% | more unpack work than bytes saved |
| Wider tiles, tighter accumulator stride | flat | lever already exhausted |

Four losses were needed before the diagnosis stuck: **the kernel is latency-bound on outstanding
loads, not throughput-bound on arithmetic.** The hoisted sum's 24 adds were free — they ran in the
shadow of memory the loop already awaited. Adding ALU to save bytes loses; adding *requests in
flight* wins (`NSGB=4`, +11.6% — `KERNELS.md`).

### 5.3 Paged KV with cross-request prefix sharing

Attention history lives in a block pool with per-sequence block tables — the OS page-table trick
applied to the KV cache. Sequences grow into free blocks rather than reserving a contiguous span.

The consequence that matters: **two requests sharing a prompt prefix share the physical blocks
holding it.** A coding agent sending the same system prompt to eight requests prefills it once.

*Source: `arf_core::scheduler::block_manager`.*

Architectural rather than tuning — llama.cpp's KV is per-slot physically, so sharing a prefix across
requests would mean copying where Arf aliases. The effect is on **time to first token** under
concurrency, and it decays toward 1× as the shared prefix shrinks.

### 5.4 Lossless native-Q4_K transcode

A GGUF stores Q4_K blocks. The naive load dequantizes to f32 and re-quantizes into the engine's
format — a double quantization that adds sampling-visible logit noise.

Arf transcodes the ggml block layout directly, bit-exactly, and caches the result. Weight buffers
are then wrapped **no-copy** over the mmap, which is why the cache pads every region to a 16 KB page:
`newBufferWithBytesNoCopy` only accepts page-aligned memory.

*Source: `weight_cache.rs` (`PAGE = 16 * 1024`). Bit-exactness is pinned by two
ggml-parity tests in `arf-core/src/model/gguf.rs`.*

### 5.5 On-GPU sampling

Sampling on the host reads back a full logit vector per token — **600 KB** at this vocabulary.
Sampling on-device reads back **4 bytes**, and writes the chosen id straight into the buffer the next
step reads from.

A per-token readback once turned a GPU-bound loop into a CPU-bound one; removing it was worth more
than any kernel change that followed.

### 5.6 Grouped MoE expert GEMM

Top-*k* routing done naively is *k* small matrix multiplies per token — *k* launches, each too small
to fill the GPU. Arf sorts tokens by expert and issues one grouped GEMM per expert over its
contiguous run.

---

## 6. Quantization

### 6.1 Weights: Q4_K

```
super-block: 256 weights = 8 sub-blocks × 32
w ≈ scale · code + min          code ∈ [0,15], 4 bits
each SUB-BLOCK carries its own (scale, min) as bf16

codes         256 × 4 bits = 128 bytes
scales+mins   8 × 2 × 2    =  32 bytes
                             ─────────
                              160 bytes / 256 weights = 5.0 bits each
```

*Verified: `(128 + 32) × 8 / 256 = 5.00`. Source: `arf-ml/src/weight.rs`.*

Two ideas do the work. **Scale locally** — 32 adjacent weights span a far narrower range than the
whole tensor, so 16 levels across that range are much finer. **Do not assume symmetry** — Q4_0 uses
`[−8, 7]` centred on zero; Q4_K stores an explicit `min` and fits the actual range, which matters for
the skewed distributions real weights have.

> **Fewer bits is not fewer bytes.** Q3_K cut bits per weight and ran 4% *slower*: the extra unpack
> arithmetic cost more than the bytes saved, because the kernel was never ALU-limited. The metric is
> bytes streamed per token, not bits per weight.

### 6.2 KV cache: TurboQuant

Quantizing a vector directly is hard — a few outlier coordinates dominate the range. TurboQuant
removes outliers by *spreading* them:

1. normalize to unit length (store `‖v‖` separately)
2. apply a fixed random orthogonal rotation `R`
3. scalar-quantize each rotated coordinate

After rotation the coordinates are i.i.d. from a **known closed-form density** — the marginal of a
uniform point on the sphere `S^(d−1)`:

```
f(x) ∝ (1 − x²)^((d−3)/2)   on [−1, 1]
```

so the optimal levels follow from `d` alone. No calibration set, no training pass, no per-model
tuning: the rotation makes every vector look statistically identical.

*Source: `arf-ml/src/turboquant.rs`, implementing [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).*

---

## 7. Diffusion transformers

FLUX runs on the same stack. A language model predicts the next token; a diffusion transformer
predicts the **direction from noise toward an image**, then steps that way repeatedly.

```
x_{i+1} = x_i + (σ_{i+1} − σ_i) · v(x_i, σ_i, text)

σ = 1 is pure noise, σ = 0 is the image
σ schedule: linspace(1, 1/N, N) with a resolution-dependent shift
```

That is Euler integration of an ODE. The denoiser is a transformer: images are cut into patches,
each patch is a token, attention runs over them as it does over words, and the text prompt enters
through the same attention as a second stream. Position is 3-axis RoPE rather than 1-axis, because a
patch has coordinates.

*Source: `gpu/flux/scheduler.rs`, mirroring diffusers `FlowMatchEulerDiscreteScheduler`.*

**Flow matching is the newer idea:** rather than learning to reverse a noising process step by step,
learn the straight-line velocity field between the two distributions. Straighter paths mean fewer
steps — FLUX-schnell generates in four.

This is why one engine runs both: a DiT needs the same primitives — GEMV, quantized weights, norms,
attention — in a different order.

---

## 8. Speculative decoding

Every mechanism above reduces the cost of one pass. This one changes how many tokens a pass
produces, and it is the only lever left.

```
speedup = tokens emitted per pass / cost of that pass
```

Guess several tokens cheaply, verify them all in one batched forward. The verifier accepts a draft
only if it matches what the full model would have produced, so **a bad draft costs a wasted
verification, never a wrong output** — a performance technique with no quality term.

**Measured:** at k=2 a verify costs **1.59 decode-steps** — which is what the 0.96 → 0.29 marginal
row cost bought. **Projected:** 1.64× at 80% acceptance.

⚠️ **Correct, at parity — not yet a win.** The model ships a trained draft head (`blk.64`); as of
L308 it drafts the right token — **83.9% top-1** in probe mode, a hit rate, not a tok/s number
(measured 2026-08-30) — and as of L309 speculative output is **byte-identical to
greedy** at 128/256/512 tokens on both KV pools, with 79–80% of k=1 drafts accepted. Wall time is
1.03× — parity, not interleaved, not a quotable row. What blocks the speedup is overhead, not
correctness: the draft record costs ~25 ms against a ~3 ms roofline, and the head has no prompt
history or catch-up. *Earlier revisions of this paragraph said the head
"currently predicts the wrong tokens" and named the `eh_proj` GEMV orientation as the leading
suspect; L308 found four other causes, each sufficient alone.*

---

## 9. Vision transformers

Gemma 3 sees. The mechanism is the same transformer, with one interesting twist in how images are
allowed to attend.

An image is cut into fixed patches. Each patch is flattened, projected to the model's width, and
from then on it **is** a token. The encoder (SigLIP) is 27 transformer layers doing what the text
layers do: normalise, attend, feed-forward, add back. The output is a short sequence of **soft
tokens** — vectors in the same space words live in — spliced into the text prompt.

**The twist is the mask.** Text is causal: a token cannot see the future. An image has no
left-to-right order, so patches within one image attend **bidirectionally**, while everything
crossing the span boundary stays causal.

The engine implements this as a per-sequence list of image spans: a query inside a span extends its
causal frontier to that span's end, and is otherwise ordinary.

*Source: `arf_core::model::batch::SeqAttn::image_spans` — `(local_start, len)` pairs;
`gpu/vision.rs` for the GPU encoder, gated by a bit-parity test against the CPU one.*

One design note that follows from the arithmetic: **patch embedding runs on the CPU.** It is a tiny
gather plus one small matmul — negligible — while the 27 transformer layers with their O(n²)
attention over thousands of patches are where the GPU time actually is.

---

## 10. Where it stands

| Single stream · Qwen3.8-27B Q4_K_M · M4 Max · greedy · 32K ctx | tok/s | ratio |
|---|---:|---:|
| **Arf** — HTTP daemon, rebooted quiet box | **20.0** | 1.20× vs the row below (different harness, different session — indicative only) |
| llama.cpp — `llama-bench -n 64 -r 2`, same box and file, separate session | 16.60 | 1.0× |
| **Arf vs llama-server, HTTP both sides, 4 arms interleaved** (L224 — transcribed from the commit log, not re-run) | 16.6–17.1 vs 14.8–15.0 | **1.12×** |

**Wins:** shared-prefix workloads (architectural), saturated concurrency, long context where the
hybrid's recurrent layers stop scaling with history.

**Does not win:** short prompts with no shared prefix — nothing to exploit, and both engines read the
same bytes at the same bandwidth.

**Left:** at ~91% of a hard ceiling, kernel work is finished as a lever.

### Supported architectures

| model | shape | the hard part |
|---|---|---|
| Qwen3.8-27B | 64 layers, 5120 hidden, 24q/4kv, head_dim 256 | two layer kinds in one pass, per-sequence recurrent state |
| Qwen3-Coder-30B | MoE, top-k router | expert routing without *k* tiny launches |
| muse-glimmer-30B | 52 layers, 6656 hidden, 32q/2kv, head_dim 128, dense | attention output gate + four-norm block + pre-trunk embed norm; 3:1 sliding/global, window 2048 |
| Gemma 4 (12B/31B) | four-norm blocks, QK-norm | per-layer geometry — global layers have own head_dim, partial rotary |
| Gemma 3 4B | dense + SigLIP vision | image tokens attend bidirectionally in-span, causally outside |
| Llama 3.2 1B | dense, small | the correctness model |

*Source: `arf_core::config::ARCHITECTURES` — one row per architecture.*

---

## 11. Reading the repository

| path | what is there |
|---|---|
| `crates/arf-ml` | the numerics floor — no dependency on anything else in the tree |
| `crates/arf-core` | the engine proper. **Start here** for logic rather than pointers |
| `crates/arf-gpu` | GPU backend, both paths, 147 kernels — where the `unsafe` lives |
| `crates/arf-serve` | OpenAI-compatible server and the batched-decode actor loop |
| `crates/arf-cli` | `pull` / `run` / `serve` |
| `crates/arf-router` | optional fleet front, zero engine dependencies |
| `crates/arf-gpu/src/shaders` | 147 kernels, WGSL and MSL — indexed in [KERNEL_MAP.md](KERNEL_MAP.md) |
| [PERFORMANCE.md](PERFORMANCE.md) | measured results, with dates and method |

### What to run first

```sh
cargo build --workspace                   # must be 0 warnings
cargo clippy --workspace --all-targets    # must be 0 — this is what CI runs
cargo test --workspace
```

These three reach **different sets of targets**. The same broken initializer once hid in three
places at once — integration tests, benches and doctests — each reachable only by a different
invocation.

---

## 12. Corrections found during verification

Recorded because the discipline that produced these numbers requires it.

**12.1 The roofline was once computed from the wrong file.** An early analysis used the 17.11 GB GGUF
rather than the 18.6 GB transcoded sidecar that is actually streamed, concluding 69% of peak and
inventing ~30% of "diffuse kernel loss" that did not exist. Corrected to 81%, and now ~91% after
L220. *Always parse what is resident, not what is on disk.*

**12.2 The speculation formula did not reproduce its own number.** Writing it as
`(1 + accepted) / verify_cost` and computing from k=2, 80% acceptance, 1.59-step verify gives
**1.53×**, not the documented 1.64×. The documented figure is retained as a projection with its
source cited; the derivation is not reconstructed to fit.

**12.3 An MLX concurrency comparison was removed.** Figures quoted in a previous README had no recorded
measurement behind them. By the project's own rule that makes them unverified, so they are omitted
rather than repeated.

**12.4 A llama.cpp source citation by line number was removed.** The mechanism claim (per-slot
physical KV) stands; a line reference into a version not checked out here does not.

**12.5 A code comment describes a "48-layer decode".** That is from the Coder-30B era and is wrong for
Qwen3.8's 64 layers. The chunking mechanism is layer-count-agnostic; this document describes what it
does rather than quoting the stale number.
**12.6 The llama.cpp baseline was quoted as 17.27, which has no measurement row.** Written into
ENGINE.md and the README in L295/L296, carried forward from an unrecorded figure rather than from the log.
The traceable figure is **16.60 tok/s** (`llama-bench -n 64 -r 2`, same box, same GGUF), which
makes the ratio **1.20×**, not 1.16×. Corrected everywhere — and note the error ran AGAINST us:
this project's rule caught a figure that was understating its own result.

**12.7 The 1.20× that §12.6 re-instated had itself been retracted.** L224 (2026-08-23) redid the Qwen3.8 comparison the fair way — HTTP server on both sides, same p50
inter-token method, four arms interleaved — and measured **1.12×** (Arf 16.6–17.1 vs llama-server
14.8–15.0), after L223 found that `llama-bench` understates llama.cpp's server by ~20%. The commit
said the 1.20× "is not the number to quote". §12.6 restored it anyway, from the L217/L220 row, because
that was the only llama.cpp figure with a row. Both are now rows: 20.0 vs 16.60 is a pair of
standalone absolutes from different harnesses and sessions, and 1.12× is the like-for-like ratio
(transcribed from the commit log, not re-run). Corrected in README, §10, HOW_IT_WORKS §5 and the
per-model table.
