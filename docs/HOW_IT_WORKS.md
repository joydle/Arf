# How Arf works

The engine's design, the models it runs, and the specific mechanisms behind its speed on Apple
silicon — where it leads llama.cpp in some regimes and trails it in others; [PERFORMANCE.md](PERFORMANCE.md) has the
measured results, wins and losses alike.

Every number here was measured on the machine and in the way [PERFORMANCE.md](PERFORMANCE.md) describes.
Kernel-level detail lives in [KERNELS.md](KERNELS.md).

---

## 1. The constraint everything is designed around

Generating one token requires reading **every weight in the model**. Not a subset — all of them.
There is no reuse to exploit, because a single token is a single row of activations against the
whole weight set.

That makes decode **bandwidth-bound**, and it sets a hard ceiling:

```
18.6 GB streamed per token  ÷  410 GB/s (M4 Max)  =  22.0 tok/s
```

Arf measures 20.0 — about 91% of that. This one fact determines the whole design:

- **Arithmetic is nearly free.** The ALU idles waiting on memory. Adding math to save bytes is a
  losing trade, and this project proved it five separate times before believing it.
- **The win is never "compute faster."** It is: stream fewer bytes, keep more requests in flight,
  or emit more than one token per pass.
- **Batching is close to free.** A second sequence reads the same weights. After L220 the marginal
  cost of an extra batch row is **0.29** of a full step, down from 0.96.

> Compute the roofline from the bytes you actually **stream**, not the file on disk. An early
> version of this analysis used the 17.11 GB GGUF instead of the 18.6 GB transcoded sidecar and
> invented a 30% "diffuse kernel loss" that did not exist.

---

## 2. What one token does

```
token id
   │
   ├─ embed              gather one row from the [vocab, hidden] table
   │
   ├─ 64 layers ─────────────────────────────────────────────────
   │    normalise → mix across the sequence → add back
   │                 └─ full attention (every 4th layer), or
   │                 └─ gated delta net (the other three)
   │    normalise → feed-forward (expand, gate, contract) → add back
   │
   ├─ final norm
   ├─ lm_head            [hidden] → [vocab] scores
   └─ sample             argmax or a sampling kernel — ON the GPU
```

**The whole sequence runs on-device.** The sampled id is written straight into the buffer the next
step reads from; the host never sees an intermediate value. This matters more than it sounds — a
per-token readback once turned a GPU-bound loop into a CPU-bound one, and removing it was worth
more than every kernel change that followed.

Command buffers are chunked at 8 layers each — not because that is faster, but because recording
every layer into one buffer trips Metal's watchdog and hangs the device. The island queue serialises
commits in order, and each chunk's completion publishes its hidden-state write before the next chunk
reads it.

---

## 3. Supported models, and what each one demands

| model | shape | the hard part |
|---|---|---|
| **Qwen3.8-27B** | 64 layers, 5120 hidden, 24 q-heads / 4 kv-heads, head_dim 256 | Hybrid: 48 recurrent + 16 attention. Two entirely different layer kinds in one forward pass. |
| **muse-glimmer-30B** | 52 layers, 6656 hidden, 32 q-heads / 2 kv-heads, head_dim 128, dense SwiGLU | 3:1 sliding-window (2048) / global attention, four-norm blocks with QK-norm, a sigmoid attention-output gate (`attn_gate`) no other supported arch has, a weightless pre-trunk embedding norm, interleaved RoPE, logit softcap 20 |
| **Qwen3-Coder-30B** | MoE, top-k router | Expert routing without launching *k* tiny kernels |
| **Gemma 4** (12B / 31B) | four-norm blocks, QK-norm | Per-layer geometry: global layers have their own head_dim and partial rotary |
| **Gemma 3 4B** | dense + SigLIP vision | Image soft-tokens attend bidirectionally inside their span, causally outside |
| **Llama 3.2 1B** | dense, small | The correctness model — fast enough to iterate on |

The registry is one table (`arf_core::config::ARCHITECTURES`); adding an architecture is one row.
FLUX text→image runs on the same GPU stack through `/v1/images/generations`.

### Why the hybrid is interesting

Qwen3.8 alternates two layer types on a period of four:

- **Gated delta net** (48 layers) keeps a fixed-size recurrent state — a conv ring and a delta-net
  matrix — advanced one token at a time. Its cost does **not** grow with context length.
- **Full attention** (16 layers) reads the whole KV history. Its cost does.

Three-to-one is the architectural bet: most of the depth stays cheap at long context. The
engineering cost is that recurrent state is **serial** — token *t+1* depends on *t* — so it must be
banked per sequence, not per batch row.

> That distinction is the subtlest bug this engine has had. Keying the recurrent state by batch row
> means row 0 can be sequence A one step and sequence B the next, each inheriting the other's
> memory. Four concurrent sequences produced garbage while the same four run serially were perfect.
> The state now follows a stable per-sequence bank row, pinned for the sequence's lifetime.

---

## 4. The mechanisms — what actually makes it fast

### 4.1 The Metal island: one command buffer, not two hundred

The portable path dispatches each operation through wgpu — a bind group, a pipeline set, a dispatch,
per kernel per layer. At 64 layers that is hundreds of encoder operations and their driver overhead
per token.

The **island** is a second, native-Metal implementation of the same decode step that records the
whole thing into one command buffer, binding raw `MTLBuffer` handles directly. Same math, same
kernels, none of the abstraction. This is the default on macOS and the reason the engine ships two
backends rather than one.

### 4.2 Loop order in the GEMV — the single biggest win

The decode GEMV is one activation row against a huge quantised weight matrix. The obvious loop is
column-outer: for each output column, walk the activations. That re-reads the activation row once
per column.

**Row-outer / column-inner** loads each row's eight `float4` into named registers *once* and dots
them against all columns in the tile. Identical arithmetic, identical weight bytes — the activation
traffic collapses.

| | marginal cost of a second batch row |
|---|---:|
| before (column-outer) | 0.96 of a full step |
| after (row-outer, `COLS=4`) | **0.29** |

That 3.3× is what makes speculative verification viable, because verifying a draft *is* a batched
step.

**What did not work**, each measured:

| attempt | result |
|---|---:|
| two output columns per threadgroup at b=1 | −25% |
| hoist the column-independent activation sum into a prepass | **−31%** |
| Q3_K instead of Q4_K (fewer bits) | −4% |
| wider tiles (`COLS=8`), tighter accumulator stride | flat |

The pattern took four measurements to accept: **the kernel is latency-bound on outstanding loads,
not throughput-bound on arithmetic.** The hoisted sum's 24 adds were free — they ran in the shadow
of memory the loop was already waiting on. Adding ALU to save bytes loses. Adding *requests in
flight* wins (`NSGB=4`, +11.6%).

### 4.3 Paged KV with cross-request prefix sharing

The KV cache is a block pool with per-sequence block tables — the operating-system trick, applied to
attention history. Two consequences:

1. **No per-sequence contiguous reservation.** Sequences grow into free blocks; memory is not
   fragmented by early over-allocation.
2. **Sequences sharing a prompt prefix share the physical blocks holding it.** A coding agent
   sending the same system prompt and repo context to eight requests prefills it once.

This is architectural, not tuning: llama.cpp's KV is per-slot physically, so sharing a prefix
across requests would mean copying where Arf aliases. The measured effect is on **time to first
token** under concurrency, not on steady-state decode.

> Quote this win with its shape. The advantage grows with shared-prefix length and decays toward 1×
> as the prefix shrinks — at a ~260-token prefix it is parity or behind.

### 4.4 Lossless native-Q4_K transcode

A GGUF stores Q4_K blocks. The naive load is dequantise to f32, then re-quantise into the engine's
own format — a double-quantisation that adds sampling-visible logit noise.

Arf transcodes the ggml block layout directly into its Q4_K_S representation, bit-exactly, and
caches the result in a sidecar. Weight buffers are then wrapped **no-copy** over the mmap with
`newBufferWithBytesNoCopy` — page-aligned so Metal accepts them, which is why the cache pads every
region to 16 KB.

### 4.5 On-GPU sampling and the batched lm_head

Sampling on the host means reading back a `[vocab]` logit vector per token per sequence — 600 KB for
one token at this vocabulary. Sampling on-device reads back 4 bytes.

For batched decode the lm_head is a real GEMM: `[B, hidden] × [hidden, vocab]`, one weight stream for
all B rows, rather than B separate GEMVs.

### 4.6 Grouped MoE expert GEMM

Top-*k* routing naively means *k* small matrix multiplies per token — *k* kernel launches, each too
small to fill the GPU. Arf sorts tokens by expert and issues one grouped GEMM per expert over its
contiguous run.

---

## 5. Where Arf stands, honestly

**Single stream**, Qwen3.8-27B Q4_K_M, M4 Max, greedy, 32K context:

| | tok/s | |
|---|---:|---|
| Arf | **20.0** | 91% of the 22.0 roofline — HTTP daemon, rebooted quiet box |
| llama.cpp | 16.60 | same box, same file — but `llama-bench -n 64 -r 2`, a separate session; the 1.20× between these two is indicative only |
| Arf vs llama-server, same HTTP harness, 4 arms interleaved | 16.6–17.1 vs 14.8–15.0 | **1.12×** (L224, transcribed from the commit log, not re-run) |

**Where it wins:** shared-prefix workloads (architectural), saturated concurrency, and the
long-context regime where the hybrid's recurrent layers stop scaling with history.

**Where it does not:** short prompts with no shared prefix, where there is nothing to exploit and
both engines read the same bytes at the same bandwidth. A ~260-token prefix measures at parity or
behind.

**What is left.** Kernel micro-optimisation is finished as a lever: at 91% of a hard ceiling there is
nothing underneath. The only remaining move is **speculative decoding** — emitting several tokens
per weight pass, which beats a per-pass limit by changing the arithmetic rather than the constant.
The groundwork is done (b=1..4 correct, marginal row 0.29). The model ships a trained draft head
(`blk.64`); it is wired, runs end to end, and as of L308 predicts the right token — 83.9% top-1 in
probe mode (measured 2026-08-30) — with speculative output byte-identical to greedy since
L309. Wall time is at parity: the draft record's ~25 ms overhead and the head's missing prompt
history are what stand between 79–80% acceptance and a speedup.

---

## 6. The rules that produced these numbers

The short form:

1. **Interleave A/B arms in one session.** This box drifts 4–8% between sessions — more than most
   wins are worth. A number from twenty minutes ago is not a baseline.
2. **Correctness gates speed.** A benchmark once ran for five commits against a path producing
   corrupt output; the "batching penalty" it measured was mostly a count of surviving streams.
3. **Verify the feature is dispatched, not just configured.** A flag that changes nothing looks
   exactly like a flag that works.
4. **Every GPU kernel has a CPU oracle.** And when a parity test fails, measure *both* sides against
   a third thing before deciding which one is wrong — the oracle has been the broken one.
5. **Keep the failures.** The comments recording what was tried and lost are the most valuable text
   in the repository.
