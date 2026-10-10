# Release notes — the Mac release (Apple silicon, Metal)

Every number here is dated, with its machine and method, in [`PERFORMANCE.md`](PERFORMANCE.md) (one M4 Max, 36 GB,
unless an entry says otherwise).

**Version:** 1.0.0 · **Tag:** `v1.0.0` · **Date:** 2026-10-10 ([RELEASING.md](RELEASING.md))

**Install:** `brew install joydle/tap/arf` (Apple silicon), or build from source — see the README.

## Arf 1.0.0

- **Qwen3.8-27B in one command.** `arf` alone shows what this Mac can run, what is downloaded and what is
  running. `arf pull qwen3.8:27b` fetches a bundle — our calibrated group-64 weights, the DFlash-2 speculative
  draft, the vision projector — and `arf run qwen3.8:27b` chats with it through a background server
  (`arf status`, `arf stop`). The weights are public at `joydle/Qwen3.8-27B-Arf-GGUF`, sha256-exact against the
  file the numbers were measured on; `scripts/get_qwen38.sh` rebuilds them from public files instead. Any GGUF
  pulled from a repository runs by name, and `arf pull` refuses a model this Mac cannot load before downloading it.
- **Built for coding agents.** `arf launch claude` / `arf launch opencode` start the agent on the local model.
  OpenAI `/v1/chat/completions` and `/v1/responses`, Anthropic `/v1/messages`, tool calls through the model's own
  template. In the 2026-10-09 release test all six agent-bench tasks pass (Claude Code 407 s, OpenCode 111 s).
- **Long agent prompts are read once.** A hybrid prefix cache reuses the tool definitions across sessions in
  different directories and the project context (AGENTS.md / CLAUDE.md) across sessions in one project, even
  after a file changed. The prompt states are saved to disk and loaded after a restart, the ones in use are the
  ones kept, and `arf launch claude` reads Claude Code's ~30K-token safety check ahead in the background
  (`~/.arf/logs/warm-PORT.log`). A long prompt shows its progress (`arf status`, the `[start]` / `[done]` log
  lines) instead of looking hung.
- **Images and video.** Qwen3.8's vision tower on the GPU (a 512×384 image in ~0.3 s), `image_url` and Anthropic
  `image` blocks; `video_url` and `video` parts, frames decoded by `ffmpeg` from PATH and encoded on the GPU.
- **Throughput under load.** Multi-stream speculative decoding, greedy and sampled: 4 concurrent requests
  63.3 tok/s on a quiet box (2026-09-27). Each request's `[done]` line reports the tokens a verify pass yielded.
- **Prefill.** A reworked GDN scan layout and fused operand passes: time to first token at 2,450 tokens is
  13.09 s, within 4% of the fastest engine measured (12.62 s, 2026-09-27). Prefill attention has since taken
  32 keys an iteration by default, for long prompts.
- **Memory that stays put.** A soft cap on cached KV (`ARF_KV_CACHE_CAP_TOKENS`, default 131,072 tokens; `0` =
  off) keeps the footprint flat over many sessions. The GPU is idle whenever a memory mapping changes, and after a
  GPU driver panic naming `arf-serve` the next start uses safe memory (a fixed working set mapped at load). A
  stopped server releases its GPU memory before it exits.

## Honest limits

- One stream, plain chat: ~10% behind the fastest engine measured (61.3 vs 67.3 tok/s greedy).
- Image and video requests are greedy and do not speculate. Remote media URLs are off unless
  `ARF_MEDIA_URLS=1`. Video needs `ffmpeg` on PATH; long videos
  (past the 2,048-token default cap, `ARF_QWEN_VIDEO_MAX_TOKENS`) are not measured.
- `arf serve` is a local daemon with no auth (see SECURITY.md).
- Linux GPU is unverified. An NVIDIA backend is developed separately and is not part of this repository.

## Measured and not shipped

GSQ-RCO 3-bit models (slower than Q4_K_M on Metal, worse perplexity); wider prefill windows; a full-vocabulary
draft (net zero); a group-64 draft (a wash). Each has its ledger entry and, where code exists, a DO-NOT-RE-CHASE
note.
