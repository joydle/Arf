# arf — developer tasks.
#
# Run `make` or `make help` for the list. The wgpu/Metal GPU backend lives in
# the `arf-gpu` crate (a default dependency, not a Cargo feature); the GPU
# targets below build and test it. The only optional feature is `profiling`.

CARGO       ?= cargo
GPU_PKG     := arf-gpu
CLI_PKG     := arf-cli
SERVE_PKG   := arf-serve
NATIVE      := RUSTFLAGS="-C target-cpu=native"

# --- HUD / serve defaults (override on the command line, e.g. `make dev MODEL=...`) ---
# `make dev`        = gemma-4-12B (the image/vision demo). `make dev-conc` = qwen3-coder-30B (the
# concurrency fast path: native-Metal megakernel + batched lm_head + mm_id MoE, all DEFAULT-ON).
MODEL     ?= models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
ARCH      ?= gemma-4-12b
TOKENIZER ?= models/gemma-4-31b-it/tokenizer.json
QUANT     ?= q4
# qwen3-coder-30B concurrency-demo defaults (the fast path; `make dev-conc`):
QWEN_GGUF ?= $(HOME)/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a
# ^ q4 = native Q4_0 transcode (this GGUF is Q4_0). NOT q4k: that path double-quantizes a
# Q4_0 blob (dequant→re-quant), adding logit noise that surfaces junk tokens when sampling.
PORT      ?= 8080
SERVE_BIN := ./target/release/arf-serve

# --- FAST PATH (macOS): high-perf env vars prepended to the daemon launch. ---
# DEFAULT-ON: the native-Metal island + batched megakernel is the fast path, carrying every conc
# win (B-row attention, mm_id sort-by-expert MoE GEMM, tiled dense GEMM) — cold conc16/32/64 =
# 160/198/259 agg vs llama 196/234/402. It auto-engages only when ELIGIBLE (macOS + qwen3moe Q4KS
# + pure greedy decode) and bails CLEANLY to the portable wgpu path otherwise (e.g. gemma, prefill,
# quantized-KV), so it's safe to default. The serve loop itself defaults the megakernel on
# (ARF_NO_BATCH_MEGA=1 forces serial); ARF_MSL_GEMV builds the island + the .mtl weight views
# it needs. To run the COLD-MEASURED fast path: `make dev` already does it for the qwen target.
#   ARF_MSL_GEMV            : build the native-Metal island (REQUIRED for the megakernel; the keystone).
#   ARF_MEGAKERNEL          : the m=1 megakernel — a SINGLE chat (B=1) routes through it (the
#                                try_m1_megakernel bridge; chat p50 ~67 tok/s vs ~11 on the fallback).
#   ARF_PREFILL_CHUNK_ROWS  : packed prefill width (L107). Default 256 — fewer/larger packs;
#                                verified conc8 1.082× llama (real HTTP). Code + WIN_CONFIG also
#                                default this; setting it here keeps `make dev` / `make serve`
#                                explicit and launcher-proof. A/B: FAST_ENV='... PREFILL_CHUNK_ROWS=128'.
# (The serve binary also defaults these internally, so a raw launch is covered; setting them here
# keeps the make targets explicit and launcher-proof.)
# Override to the portable wgpu path with `make dev FAST_ENV='ARF_NO_MSL_GEMV=1'` — FAST_ENV=''
# alone is NOT enough: arf-serve defaults ARF_MSL_GEMV on when it is unset (main.rs).
# ARF_NO_BATCH_MEGA=1 is a different lever: island still up, batched megakernel forced serial.
#
# DON'T hunt the old DECISIONS_LOG (gone since 2026-08-28) for "why is it fast right now" — the daemon prints it itself at
# load, one line: `[msl] fast-path levers: doty=... nsg2=... vec4=... megakernel=... singleq=...
# batch_mega=... spec=...`. That line is generated from the SAME booleans the kernel compile
# uses (crates/arf-gpu/src/weights.rs, near "island up"), so it can't drift from what's
# actually running the way this comment block can. `make dev`/`make dev-conc` tail it below;
# to see it again later: tail -f /tmp/arf-serve.log | grep 'fast-path levers'.
# Also look for: `[serve] concurrent win config ... prefill_chunk_rows=256 [L107]`.
FAST_ENV  ?= ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARF_PREFILL_CHUNK_ROWS=256

.DEFAULT_GOAL := help

# Optional: a checkout may add gates of its own to the end-of-`check` hook by setting EXTRA_CHECK
# in `local.mk`. Absent (the usual case), nothing extra runs and nothing here depends on it.
EXTRA_CHECK ?=
-include local.mk

.PHONY: help
help: ## Show this help.
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- linux (cross-platform verification) --------------------------------------
#
# L364 — Arf's fast path is the native-Metal island, which is macOS-only, but the WORKSPACE
# must still compile and pass its CPU tests on Linux: the portable wgpu/WGSL path targets
# Vulkan, and it cannot be developed if `cargo check` is red there.
#
# These targets run a REAL Linux toolchain in Docker rather than cross-compiling from macOS
# (cross-compiling needs an x86_64-linux-gnu C toolchain that a stock Mac does not have, and it
# would still not be Linux). Named volumes cache the target dir and the cargo registry, so the
# first run is slow and every run after it is fast. (`rust:1-bookworm` ships no clippy component,
# so `linux-lint` adds it inside the container; the volume keeps it for later runs.)
#
# ⚠️ WHAT THESE DO AND DO NOT PROVE. They prove the workspace COMPILES and its CPU tests pass on
# Linux. They do NOT prove any GPU path works there: nobody has run a Vulkan decode. Do not
# read a green `make linux-test` as "Arf serves on Linux" — see docs/PLATFORMS.md.
LINUX_IMAGE ?= rust:1-bookworm
LINUX_DOCKER = docker run --rm -v "$(PWD)":/w -w /w \
	-v arf-linux-target:/w/target-linux \
	-v arf-linux-cargo:/usr/local/cargo/registry \
	-e CARGO_TARGET_DIR=/w/target-linux $(LINUX_IMAGE)

.PHONY: linux-check
linux-check: ## Type-check the whole workspace on real Linux (Docker).
	$(LINUX_DOCKER) cargo check --workspace

.PHONY: linux-lint
linux-lint: ## Clippy the workspace on Linux, warnings denied (what CI runs).
	$(LINUX_DOCKER) bash -c 'rustup component add clippy >/dev/null 2>&1; \
		cargo clippy --workspace --all-targets -- -D warnings'

.PHONY: linux-test
linux-test: ## Run the CPU test suite on Linux (GPU tests skip themselves there).
	$(LINUX_DOCKER) cargo test --workspace

.PHONY: linux-shell
linux-shell: ## Interactive Linux shell with the same cached volumes (debugging).
	docker run --rm -it -v "$(PWD)":/w -w /w \
		-v arf-linux-target:/w/target-linux \
		-v arf-linux-cargo:/usr/local/cargo/registry \
		-e CARGO_TARGET_DIR=/w/target-linux $(LINUX_IMAGE) bash

.PHONY: linux-clean
linux-clean: ## Drop the cached Linux target dir + registry volumes.
	-docker volume rm arf-linux-target arf-linux-cargo

## --- build -------------------------------------------------------------------

.PHONY: build
build: ## Build the workspace (CPU).
	$(CARGO) build --workspace

.PHONY: release
release: ## Build optimized, with native CPU features (SIMD).
	$(NATIVE) $(CARGO) build --release --workspace

.PHONY: build-gpu
build-gpu: ## Build the wgpu GPU backend + the CLI that drives it.
	$(CARGO) build -p $(GPU_PKG) -p $(CLI_PKG)

## --- test --------------------------------------------------------------------

.PHONY: test
test: ## Run all tests (CPU).
	$(CARGO) test --workspace

.PHONY: test-gpu
test-gpu: ## Run the GPU parity test (skips if no adapter is present).
	$(CARGO) test -p $(GPU_PKG) --test gpu -- --nocapture

## --- quality -----------------------------------------------------------------

.PHONY: fmt
fmt: ## Format the code.
	$(CARGO) fmt

.PHONY: fmt-check
fmt-check: ## Check formatting without writing.
	$(CARGO) fmt --check

.PHONY: lint
lint: ## Clippy with warnings denied (CPU, all targets).
	$(CARGO) clippy --workspace --all-targets -- -D warnings

.PHONY: lint-gpu
lint-gpu: ## Clippy with warnings denied (GPU backend crate).
	$(CARGO) clippy -p $(GPU_PKG) --all-targets -- -D warnings

.PHONY: doc
doc: ## Build the API docs.
	$(CARGO) doc --no-deps --workspace

.PHONY: check
check: ## Every macOS CI job, locally (check, gpu, lean-build, deps-pin-guard) — run before claiming it builds.
	$(CARGO) fmt --all --check
	$(CARGO) clippy --workspace --all-targets --locked -- -D warnings
	$(CARGO) clippy -p $(GPU_PKG) --features profiling --all-targets --locked -- -D warnings
	$(CARGO) build -p arf-cli --features profiling --locked
	$(CARGO) test --workspace --locked
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --workspace --locked
	@# gates a checkout adds in `local.mk`, if any
	$(EXTRA_CHECK)
	@# lean-build: the crates compile without default features
	$(CARGO) check -p arf-ml --no-default-features --locked
	$(CARGO) check -p arf-core --no-default-features --locked
	$(CARGO) check -p $(GPU_PKG) --no-default-features --locked
	$(CARGO) check -p arf-serve --no-default-features --locked
	@# deps-pin-guard: the island must resolve the same single objc2-metal/objc2 as wgpu-hal (see ci.yml)
	@for pv in 'objc2-metal 0.3.2' 'objc2 0.6.4' 'wgpu-hal 29.0.3'; do \
	  set -- $$pv; got=$$(grep -A1 "^name = \"$$1\"$$" Cargo.lock | grep '^version' | tr -d '\n'); \
	  test "$$got" = "version = \"$$2\"" || { echo "deps pin guard: $$1 is '$$got', want single $$2"; exit 1; }; \
	done

.PHONY: nextest
hooks: ## Install the repository's git hooks (pre-push leak scan). Once per clone.
	git config core.hooksPath .githooks

leak-scan: ## Scan the tracked tree for leaks (generic patterns + an optional local denylist, ~/.config/arf/denylist).
	bash scripts/leak_scan.sh

nextest: ## Run the tests with cargo-nextest (one process per test; see .config/nextest.toml).
	@command -v cargo-nextest >/dev/null 2>&1 || { echo "cargo-nextest not installed: brew install cargo-nextest"; exit 1; }
	$(CARGO) nextest run --workspace

## --- local disk ---------------------------------------------------------------

.PHONY: disk
disk: ## Show what the build and the trace tools hold on disk.
	@df -h . | tail -1
	@du -sh target target/*/ 2>/dev/null || true
	@du -sch "$${TMPDIR:-/tmp}"/instruments*.ktrace 2>/dev/null | tail -1 || true

.PHONY: disk-gc
disk-gc: ## Free regenerable space: incremental caches + Instruments temp traces (2026-09-26: 12.5 GB of .ktrace).
	rm -rf target/debug/incremental target/release/incremental
	rm -f "$${TMPDIR:-/tmp}"/instruments*.ktrace
	@df -h . | tail -1

## --- security ----------------------------------------------------------------

.PHONY: audit
audit: ## Scan dependencies for known vulnerabilities (needs cargo-audit).
	@command -v cargo-audit >/dev/null 2>&1 || { \
		echo "cargo-audit not found; install with: cargo install cargo-audit"; exit 1; }
	$(CARGO) audit

.PHONY: deny
deny: ## Check licenses, bans and advisories (needs cargo-deny).
	@command -v cargo-deny >/dev/null 2>&1 || { \
		echo "cargo-deny not found; install with: cargo install cargo-deny"; exit 1; }
	$(CARGO) deny check

## --- benchmarks --------------------------------------------------------------

.PHONY: bench
bench: ## Run criterion benchmarks (native CPU features).
	$(NATIVE) $(CARGO) bench --workspace

.PHONY: repro-conc
repro-conc: ## "Are we still fast?" — coder-30B conc sweep vs docs/PERFORMANCE.md (footgun-proof: warms cache, sets levers, 300s timeouts). CONCS="64" to narrow.
	./scripts/repro_conc.sh

.PHONY: parity-conc
parity-conc: ## Prove the batched kernels are correct (~16s, must be all PASS). Run this before suspecting a perf regression.
	@echo "== parity-conc [1/2]: default MoE path (b<16 GEMV) =="
	ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARCH=qwen3-coder GGUF="$(QWEN_GGUF)" QUANT=q4ks \
	  $(CARGO) run --release -q -p $(GPU_PKG) --example batched_mega_parity
	@echo "== parity-conc [2/2]: SORTED tiled mm_id path (what conc>=16 ACTUALLY runs) =="
	ARF_MOE_SORTED=1 ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARCH=qwen3-coder GGUF="$(QWEN_GGUF)" QUANT=q4ks \
	  $(CARGO) run --release -q -p $(GPU_PKG) --example batched_mega_parity

.PHONY: gate gate-quick
gate: ## THE ANTI-ROT GATE: parity + conc1 floor + H2H ratios + sustained soak (~70min).
	./scripts/nightly_gate.sh
gate-quick: ## Same gate without the 36-min soak — use before merging a perf change.
	./scripts/nightly_gate.sh --quick

## --- profiling ---------------------------------------------------------------

.PHONY: profile-gpu
profile-gpu: ## Per-kernel GPU timing (where each token's GPU time goes). BATCH/QUANT/N env.
	./scripts/profile.sh gpu

.PHONY: profile-cpu
profile-cpu: ## CPU flamegraph via samply (host-side wall-clock: recording, readback, argmax).
	./scripts/profile.sh cpu

.PHONY: profile-pgo
profile-pgo: ## Profile-guided optimization: instrument, run, rebuild, A/B tok/s.
	./scripts/profile.sh pgo

## --- aggregate ---------------------------------------------------------------

.PHONY: ci
ci: fmt-check lint lint-gpu test build-gpu ## What CI runs: format, lint (CPU+GPU), test, GPU build.
	@echo "ci: ok"

## --- run: the serve daemon ----------------------------------------------------

## --- quickstart: zero to a running model ------------------------------------
# `make dev` needs a 7-30 GB GGUF you may not have. These targets get a NEW CONTRIBUTOR
# from clone to a talking model with no local files and no flags to read.

# ⚠️ ARF_NO_MSL_GEMV=1 is deliberate and load-bearing here. The `llama3.2:1b` alias pulls a
# SAFETENSORS repo, and the native-Metal island emits garbage ("!!!!!") on a bf16 safetensors
# source — an OPEN BUG, bisected (L349). Disabling the island uses the
# portable wgpu GPU path, which is correct AND fast for a 1B model (~90 tok/s measured).
# Correctness outranks speed (rule 4), and this is still a real GPU run.
# CORRECTED 2026-09-04 (L360). This used to read "the island bug does NOT affect the GGUF
# models this engine is measured on". That scope was ASSUMED from the L349 safetensors
# bisect, never tested — and it was wrong: gemma-4-12b is a GGUF and emitted 'Count' then
# empty deltas forever on the batched island path, because the batched record binds ONE
# attention geometry and Gemma 4 has two (global head_dim 512 / kv_heads 1 vs sliding
# 256 / 8). A guard now falls back to the serial path for dual-geometry models, so gemma-4
# is correct at defaults. The other GGUF models are unaffected and still take the full-speed
# path: `make dev MODEL=... ARCH=... QUANT=q4ks` is the 21.1 tok/s route.
QS_ENV := ARF_NO_MSL_GEMV=1
.PHONY: quickstart
quickstart: ## ⭐ START HERE: build, pull a 1B model, run it on the GPU. No local files needed.
	@echo "→ [1/3] building the CLI (the first build takes a few minutes)"
	@$(NATIVE) $(CARGO) build --release -p $(CLI_PKG) -q
	@echo "→ [2/3] pulling llama3.2:1b (~2.3 GB) if it is not already here"
	@./target/release/arf pull llama3.2:1b
	@echo "→ [3/3] generating on the GPU (portable path — see the note above)"
	@$(QS_ENV) ./target/release/arf generate --model models/llama-3.2-1b \
		--device gpu --max-tokens 40 --prompt "Explain what an inference engine does, in two sentences."
	@echo ""
	@echo "→ that was the portable GPU path. The main model, on the full-speed native-Metal engine"
	@echo "  (group-64 weights + speculative draft + images, ~22 GB; needs ~27 GB of free memory):"
	@echo "     target/release/arf pull qwen3.8:27b && target/release/arf serve qwen3.8:27b"
	@echo "   then point a coding agent at it (README: Coding agents). 'make bench-me' shows where"
	@echo "   your box lands against its memory roofline."

.PHONY: quickstart-serve
quickstart-serve: ## Same 1B model as an OpenAI-compatible server on :$(PORT).
	@$(NATIVE) $(CARGO) build --release -p $(CLI_PKG) -p $(SERVE_PKG) -q
	@./target/release/arf pull llama3.2:1b
	@echo "→ serving on http://127.0.0.1:$(PORT)/v1 — try:"
	@echo "   curl -s localhost:$(PORT)/v1/chat/completions -H 'Content-Type: application/json' \\"
	@echo "     -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}' | jq -r .choices[0].message.content"
	@$(QS_ENV) ./target/release/arf serve models/llama-3.2-1b --port $(PORT)

.PHONY: bench-me
bench-me: ## "How fast is MY box?" — decode-only tok/s vs this machine's memory roofline.
	@./scripts/bench_me.sh

.PHONY: setup
setup: ## One-time setup: .env from template (idempotent).
	@if [ ! -f .env ]; then cp env.template .env && echo "✓ created .env from env.template — add your HF_TOKEN for HF search"; else echo "✓ .env exists"; fi
	@echo "✓ setup done — run 'make dev'"

.PHONY: doctor
doctor: ## Diagnose the dev environment (tools, model files, .env, running daemon).
	@echo "arf doctor"
	@printf "  %-22s" "cargo:";    command -v cargo >/dev/null 2>&1 && cargo --version || echo "✗ missing"
	@printf "  %-22s" ".env:";     [ -f .env ] && echo "✓ present" || echo "· run 'make setup'"
	@printf "  %-22s" "HF_TOKEN:"; ([ -f .env ] && grep -q '^HF_TOKEN=.\+' .env) && echo "✓ set (HF search enabled)" || echo "· unset (public HF search only)"
	@printf "  %-22s" "serve bin:"; [ -x $(SERVE_BIN) ] && echo "✓ built" || echo "· not built (make serve-build)"
	@printf "  %-22s" "model file:"; [ -f "$(MODEL)" ] && echo "✓ $(MODEL)" || echo "✗ missing: $(MODEL)"
	@printf "  %-22s" "tokenizer:"; [ -f "$(TOKENIZER)" ] && echo "✓ present" || echo "✗ missing: $(TOKENIZER)"
	@printf "  %-22s" "daemon :$(PORT):"; curl -fsS http://127.0.0.1:$(PORT)/healthz >/dev/null 2>&1 && echo "✓ up" || echo "· down"

.PHONY: serve-build
serve-build: ## Build the optimized serve daemon.
	$(CARGO) build --release -p $(SERVE_PKG)

.PHONY: serve
serve: serve-build ## Run the serve daemon (resident model) in the foreground.
	$(FAST_ENV) $(SERVE_BIN) --model "$(MODEL)" --arch "$(ARCH)" $(if $(strip $(TOKENIZER)),--tokenizer "$(TOKENIZER)",) --quant "$(QUANT)" --no-tty --port $(PORT)

.PHONY: dev
dev: serve-build ## Build and start the daemon in the background, waiting until it answers /healthz.
	@echo "→ starting arf-serve on :$(PORT) — loading $(MODEL)"
	@echo "  (a ~7GB GGUF onto Metal takes ~20-40s on first load; logs: tail -f /tmp/arf-serve.log)"
	@pkill -f "$(SERVE_BIN)" 2>/dev/null || true; sleep 1
	@echo "  fast-path env: $(FAST_ENV)"
	@$(FAST_ENV) $(SERVE_BIN) --model "$(MODEL)" --arch "$(ARCH)" $(if $(strip $(TOKENIZER)),--tokenizer "$(TOKENIZER)",) --quant "$(QUANT)" --no-tty --port $(PORT) > /tmp/arf-serve.log 2>&1 &
	@printf "  loading model"; \
		for i in $$(seq 1 120); do \
			curl -fsS http://127.0.0.1:$(PORT)/healthz >/dev/null 2>&1 && { echo " ✓ ready (loaded in ~$$((i*2))s)"; break; }; \
			pgrep -f "$(SERVE_BIN)" >/dev/null 2>&1 || { echo ""; echo "  ✗ serve exited during load — last log lines:"; tail -8 /tmp/arf-serve.log; exit 1; }; \
			printf "."; sleep 2; \
		done; \
		curl -fsS http://127.0.0.1:$(PORT)/healthz >/dev/null 2>&1 || { echo ""; echo "  ✗ not ready after 240s — check: tail -f /tmp/arf-serve.log"; exit 1; }
	@grep -m1 'fast-path levers' /tmp/arf-serve.log 2>/dev/null | sed 's/^/  /' || echo "  (no [msl] lever line — not on the native-Metal fast path; see FAST_ENV in Makefile)"
	@echo "→ API ready → http://127.0.0.1:$(PORT)/v1  (daemon log: tail -f /tmp/arf-serve.log)"

.PHONY: dev-conc
dev-conc: ## CONCURRENCY DEMO: qwen3-coder-30B on the native-Metal fast path (megakernel +
dev-conc: ## batched lm_head + mm_id MoE, all default-on).
	@$(MAKE) dev MODEL="$(QWEN_GGUF)" ARCH=qwen3-coder QUANT=q4ks TOKENIZER=""

.PHONY: dev-serve
dev-serve: ## Foreground the qwen3-coder-30B daemon on the fast path.
	@$(MAKE) serve MODEL="$(QWEN_GGUF)" ARCH=qwen3-coder QUANT=q4ks TOKENIZER=""

.PHONY: stop dev-stop
stop dev-stop: ## Stop the serve daemon.
	@pkill -f "$(SERVE_BIN)" 2>/dev/null && echo "✓ daemon stopped" || echo "· no daemon running"

## --- aggregate (cont.) --------------------------------------------------------

.PHONY: clean
clean: ## Remove build artifacts.
	$(CARGO) clean
