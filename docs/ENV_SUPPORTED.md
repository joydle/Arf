# Supported environment variables

**This is the contract. Everything else is internal.**

[`ENV_INVENTORY.md`](ENV_INVENTORY.md) is the generated census of every `ARF_*` variable the code
reads (its header carries the current count; non-`ARF_` names are listed only in this file). That
file is a census, not a promise — most of its entries are GPU tuning levers inside `arf-gpu`
that exist so a kernel experiment can be A/B'd without a rebuild. They may be renamed or removed
without notice, and you should not need any of them: **the defaults are the measured winners.**

The variables below are the ones a user or operator is expected to set. They will not change
without a note in the release notes and a deprecation period.

## Model and runtime

The model path, tokenizer and quantization are command-line arguments, not environment variables —
see `arf serve --help`. Only one runtime path is set through the environment:

| variable | what it does |
|---|---|
| `ARF_FLUX_DIR` | Where the FLUX image-generation weights live. Defaults to `models/flux-schnell`. |

## Serving

`arf-serve` reads `./.env` at startup (see `env.template`); a variable already in the real
environment always wins. Any `ARF_*` lever placed there becomes live, including internal ones
from ENV_INVENTORY.md. `arf` (the CLI) does not read `.env`.

| variable | what it does |
|---|---|
| `DATABASE_URL` | Postgres connection string for chat persistence (needs `--features postgres`). Without it the server persists nothing. |
| `HF_TOKEN` | HuggingFace token. Server: attached as a Bearer header on `/hf/search` upstream calls only, never echoed to clients; its presence (not value) is printed at startup. Also read by the CLI, below. |
| `ARF_PREFILL_CHUNK_ROWS` | Rows per prefill chunk. Raising it trades latency for prefill throughput. |
| `ARF_KV_F16_MAX_BATCH` | Cutoff (default `8`) for the concurrency-conditional f16 KV pool on the macOS fast path: `serve` enables it when `--max-batch-size` is omitted or ≤ this value, and leaves it off above. Force on with `ARF_KV_F16=1`, off with `ARF_NO_KV_F16=1` — `ARF_KV_F16=0` does NOT disable it. |
| `ARF_NO_KV_F16` | The f16 KV pool opt-out (the only operator-facing one; see the polarity note above). |

## Qwen3.8: agents, images, video

| variable | what it does |
|---|---|
| `ARF_KV_CACHE_CAP_TOKENS` | Soft cap on cached KV, in tokens (default `131072`; `0` = no cap). Past it, a new block reuses the oldest cached one instead of mapping more memory, so the server's footprint stays flat over many sessions; live sequences still grow past it. |
| `ARF_IMAGE_FILES` | `1` lets `image_url` / `video_url` / `video` parts name a local file (`file:///abs/path` or `/abs/path`). Off by default: the server would read files on behalf of any HTTP client. |
| `ARF_MEDIA_URLS` | `1` lets image, audio and video parts name an `http(s)://` URL, which the server fetches: public addresses only (every DNS answer checked, the connection pinned to them), no redirects, 64 MB and 20 s per reference, no proxy. Off by default: `arf serve` has no authentication, so the server would fetch on behalf of any HTTP client. |
| `ARF_ANCHOR_REPLAY` | **Off by default.** `1` records each prefix anchor's token prefix to `~/.arf/anchors/` and re-takes it in the background after a restart, so the first agent session after a restart can resume past the shared system + tools prefix. Off by default because the replay currently competes with the first requests after a start instead of yielding to them (issue #17). |
| `ARF_EARLY_ANCHOR` | **Experimental, off by default.** `1` publishes a prefix anchor as soon as a request's prefill passes it (not when the request finishes), and makes a request that shares an anchor still being computed wait for it instead of computing it again — so agents started together read their shared system prompt once. Not yet verified on the GPU. |
| `ARF_PORT` | The port `arf run`, `arf launch` and `arf stop` use for the background server (default 8080). |
| `ARF_ANCHOR_DIR` | Directory for the recorded prefix anchors (default `~/.arf/anchors`). |
| `ARF_QWEN_VIDEO_MAX_TOKENS` | Language-model tokens one whole video may take (default `2048`; the model allows ~12K). Frames are sized to fit it. Budgets above the default are not measured. |
| `ARF_QWEN38_REPO` | CLI: the Hugging Face repo `arf pull qwen3.8:27b` fetches the weights from (default `joydle/Qwen3.8-27B-Arf-GGUF`). |

## Set by the engine

| variable | what it does |
|---|---|
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` | On macOS, any Metal-backed binary (serve and CLI) sets this to `1` before creating the GPU device unless it is already set, to relax the AGX interactivity watchdog that preempts long command buffers (port of llama.cpp's fix). Set it to `0` explicitly to restore the watchdog for an A/B. |

## Speculative decoding

These tune the in-model MTP / prompt-lookup speculation, which is off by default — it has not yet earned
its keep on general text. The trained block draft is separate: it runs whenever a draft is attached (the
`qwen3.8:27b` bundle attaches its DFlash-2 draft automatically; `--no-draft` turns it off), and none of the
variables below is needed for it.

| variable | what it does |
|---|---|
| `ARF_SPEC_K` | Draft length. Default `0`, which disables speculation. `2` is the measured sweet spot (`arf-serve/src/actor.rs`). |
| `ARF_SPEC_MIN_MATCH` | Shortest suffix the lookup drafter will match. Default `4`; values below `1` fall back to the default. |
| `ARF_SPEC_MIN_OCC` | Independent prior occurrences a match at exactly `ARF_SPEC_MIN_MATCH` length needs before it is drafted from; longer matches are trusted after one. Default `2`; set `1` to trust a minimum-length match off a single occurrence. Values below `1` fall back to the default. |

## CLI

| variable | what it does |
|---|---|
| `HF_TOKEN` | HuggingFace token, for `arf pull` and gated-repo search (`--token` overrides). Shared with the server, above. |
| `ARF_MODELS_DIR` | Where `arf pull` downloads and `ls` / `run` / `serve` look. Unset: `./models` if it exists, else `~/.arf/models`. `--dir` overrides it. |
| `ARF_PULL_MAX_MBPS` | Download rate cap for `arf pull`. |
| `ARF_SHOW_IDS` | Print token ids alongside decoded text. |
| `NO_COLOR`, `TERM` | Standard terminal conventions; honoured, not invented here. |

## Testing

Used by the test suite and probes so no developer's home directory ends up in the repo
(see `scripts/check_no_local_paths.sh`).

| variable | what it does |
|---|---|
| `ARF_MODEL_PATH` | Directory holding the Llama-3.2-1B correctness weights (`model.safetensors` plus `tokenizer.json`, as laid out by `arf pull`) — not the file itself. Read by the `#[ignore]`d arf-core tests (`real_model`, `quant`) and by the arf-gpu benches indexed in [EXAMPLES.md](EXAMPLES.md). Use an absolute path: cargo runs tests with CWD = the crate directory. |
| `ARF_GGUF_BLOB` | GGUF blob for the tokenizer round-trip test. |
| `ARF_TOKENIZER` | `tokenizer.json` for the decode probe. |
| `ARF_TEST_GGUF_BLOB` | GGUF fixture for loader tests; the test skips if unset. |
| `ARF_TEST_SAFETENSORS` | safetensors fixture, same. |
| `ARF_VISION_SUM` | Instrumentation, not a path: prints per-stage element sums from the vision encoder (library code, `arf-core/src/model/vision.rs`) to diff against llama.cpp. |

Apart from `ARF_VISION_SUM`, none of these are read by the engine at runtime — they exist only so
tests and probes can find a multi-gigabyte model without a hardcoded home directory.

---

## Why the rest still exist

They are not dead code, and deleting them is not free:

- **Escape hatches** (`ARF_NO_*`, `ARF_SKIP_*`, `ARF_FORCE_*`) turn a default back off.
  They exist so that a regression on someone's machine has a one-line workaround instead of a
  rebuild. Keep them.
- **Instrumentation** (`*_DUMP`, `*_TRACE`, `*_STATS`, `*_TIME`, `*_SUM`) prints or dumps and changes no
  behaviour.
- **Kernel levers** are how this engine was measured. `ARF_GEMV_B_NSG` and `ARF_GEMV_B_COLS`
  are why the marginal batch-row cost went from 0.96 to 0.29 — the A/B needed both arms available
  in one binary, in one session ([PERFORMANCE.md](PERFORMANCE.md)).

Only three flags are provably settled — their comments record the losing arm — and even those are
kept as escape hatches rather than removed. A flag with a measured default and a documented reason
to exist is not bloat. An *undocumented* flag surface is, which is what this file fixes.
