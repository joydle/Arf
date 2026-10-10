# Performance

Measured results, each with its date and method. All on one Apple M4 Max (36 GB, 546 GB/s), Qwen3.8-27B at
~4 bits, macOS on AC power, unless a row says otherwise. To measure your own Mac: `make bench-me`.

## How these are measured

- **Same machine, same client, same prompts** for every engine, one engine resident at a time.
- **Rounds interleaved** (A, B, B, A …) in one session: this machine drifts several percent between sessions,
  and more once it swaps, so a number from another session is not a baseline. A round in which swap grew is
  thrown away.
- **Correct text first.** A speed number counts only from a run whose output is checked (greedy text against
  a reference, or tests passing for the agent runs).
- **Each engine as its users get it:** its own default ~4-bit build and its own default speculation.

## Coding agents

Wall time from the prompt to passing tests, three tasks, median of three rounds (2026-09-27):

| agent | time to passing tests |
|---|---:|
| Claude Code | 256 s |
| Pi | 85 s |
| OpenCode | 155 s |

## SPEED-Bench, side by side (2026-10-05)

SPEED-Bench's coding prompts (`scripts/speed_bench.py`: 8 single-turn coding prompts at 1,024 output tokens, one
35K-token code-completion prompt cold and then again, 4 prompts at once), greedy, reasoning on. Each engine on its
own default ~4-bit build and speculation, one resident at a time, 3 rounds with the order rotating; the medians of
the rounds in which swap did not grow (Arf 2 of 3, Splash 2 of 3, MLX 3 of 3).

| | Arf | Splash 1.1.0 | MLX (mlx-lm 0.31.3) |
|---|---:|---:|---:|
| decode, one stream (tok/s) | **90.5** | 88.2 | 18.5 |
| 4 requests at once (tok/s) | 99.3 | **99.6** | 61.2 |
| prefill, the 35K-token prompt (tok/s) | 174 | **182** | 180 |
| the same prompt again, time to first token | **0.18 s** | 0.33 s | 0.68 s |

llama.cpp (Q4_K_M, no speculation) ran in the same rounds; every one of its rounds grew swap, so none is shown.

## Five engines, one client (2026-09-27)

| | Arf | Splash 1.1.0 | MLX 0.31.3 | llama.cpp | Ollama 0.34.4 |
|---|---:|---:|---:|---:|---:|
| 4 requests at once (tok/s) | **63.3** | 60.4 | 45.4 | 18.2 | 16.7 |
| decode, one stream (tok/s) | 61.3 | **67.3** | 20.1 | 17.3 | 16.8 |
| time to first token, 2.45K-token prompt | 13.09 s | 12.98 s | **12.62 s** | 15.47 s | 16.11 s |

A re-run on a heavily loaded machine (2026-09-29) read ~20% lower for every engine, with Arf behind in every
column; numbers from a loaded machine are not comparable to these.

## Prefix reuse for agents

| what | before | after | measured |
|---|---:|---:|---|
| a new Claude Code session's first turn (the shared system prompt and tools are reused) | 88 s | 14.4 s | 2026-09-26 |
| a new agent session right after a server restart (anchor replay, once it has run) | 60 s | 1.9 s | 2026-10-05 |
| the same 34,972-token prompt again (SPEED-Bench's 32K coding prompt), time to first token | 1.67 s | 0.18 s | 2026-10-05 |

Text is identical with and without the reuse in the first two rows. In the third, the repeat's first token
equals the cold request's, and the prefix-cache gate passes: with the cache on and off, answers are identical
or part only at a near-tie (the other token within 0.1 nats of the top choice on both servers), the kind of
parting speculative decoding's batch shapes already cause. The gain comes from resuming closer to the
prompt's end: the cached state is now kept within a few tokens of it (the chat template's reply header,
measured per template) on a 16-token boundary, where it was 32 tokens rounded down to 128. Snapshots inside
the prompt (the shared-prefix anchors) stay on 128-token boundaries: there, finer ones made cold requests slower.

## Memory

Over 24 agent-style sessions the server's footprint stays flat at 8.49 GB from the tenth session on, with
the soft cap on cached KV (`ARF_KV_CACHE_CAP_TOKENS`, default 49,152 tokens); without it, it had reached
9.54 GB and was still rising (2026-09-27).

## Images

Qwen3.8's vision tower on the GPU: a 512×384 image in ~0.3 s (2026-09-27).
