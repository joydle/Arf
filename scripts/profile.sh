#!/usr/bin/env bash
# End-to-end profiling for arf's GPU decode.
#
#   scripts/profile.sh gpu      # per-kernel GPU timing (timestamp queries) — where the GPU spends each token
#   scripts/profile.sh cpu      # CPU flamegraph via samply — where wall-clock goes on the host side
#   scripts/profile.sh pgo      # profile-guided optimization: instrument → run → rebuild → A/B
#   scripts/profile.sh all      # gpu + cpu (pgo is opt-in; it rebuilds the binary)
#
# Env:
#   MODEL    path to the model (default: ./models/llama-3.2-1b/model.safetensors)
#   TOK      path to tokenizer.json (default: sibling of MODEL)
#   PROMPT   prompt text (default: a long story prompt that decodes EOS-free)
#   N        tokens to generate (default 128)
#   QUANT    none|int8|q4k|q4ks (default q4k — the fast path for the 1B)
#   ARCH     arch for a GGUF blob with no config.json (e.g. qwen3-coder-30b)
#   BATCH    GPU batch size for batched-decode profiling (default 1)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-$ROOT/models/llama-3.2-1b/model.safetensors}"
PROMPT="${PROMPT:-Write a long story about a robot who learns to paint. Once upon a time,}"
N="${N:-128}"
QUANT="${QUANT:-q4k}"
BATCH="${BATCH:-1}"
# ARCH is REQUIRED for a single-file GGUF blob (no sibling config.json), e.g.
# ARCH=qwen3-coder-30b. Leave empty for a safetensors dir with a config.json.
ARCH="${ARCH:-}"
# Tokenizer defaults to the sibling tokenizer.json next to the model file (or in the
# model dir). Override TOK for a GGUF blob (which has no sibling) — or omit entirely,
# since the CLI can read a GGUF's embedded tokenizer when --tokenizer is absent.
TOK="${TOK:-$(dirname "$MODEL")/tokenizer.json}"
OUT="$ROOT/target/profile"
mkdir -p "$OUT"

quant_flag() { [ "$QUANT" != "none" ] && echo "--quant $QUANT" || true; }
batch_flag() { [ "$BATCH" -gt 1 ] && echo "--batch $BATCH" || true; }
tok_flag()   { [ -f "$TOK" ] && echo "--tokenizer $TOK" || true; }
arch_flag()  { [ -n "$ARCH" ] && echo "--arch $ARCH" || true; }

build_release() { # $1 = extra features (e.g. "profiling"); GPU is always compiled in
  # arf-gpu is an unconditional dependency — there is no `gpu` feature to request.
  local feat=""
  [ -n "${1:-}" ] && feat="--features $1"
  RUSTFLAGS="-C target-cpu=native" cargo build --release -p arf-cli $feat >&2
}

# --- GPU: per-kernel timestamp timing ------------------------------------------
# The `profiling` feature wraps each dispatch in GPU timestamp queries and emits
# per-kernel and per-token totals on tracing target `arf::gpu`. NOTE: the
# profiling build adds a device poll per submit, so its tok/s is NOT the real
# throughput — read the per-kernel *shares*, not the wall time, from this run.
profile_gpu() {
  echo "==> GPU per-kernel timing (model=$MODEL quant=$QUANT batch=$BATCH N=$N)"
  build_release profiling
  local log="$OUT/gpu-kernels.log"
  RUST_LOG=arf::gpu=trace ./target/release/arf generate \
    --device gpu $(quant_flag) $(batch_flag) $(tok_flag) $(arch_flag) \
    --model "$MODEL" --prompt "$PROMPT" --max-tokens "$N" --profile \
    2>"$log" 1>/dev/null || true
  echo "    raw trace: $log"
  echo "==> per-kernel GPU time (summed over the run, % of total):"
  sed -E 's/\x1b\[[0-9;]*m//g; s/.*kernel="?([a-z_]+)"? gpu_us=([0-9.]+).*/\1 \2/' "$log" \
    | awk 'NF==2{s[$1]+=$2; tot+=$2} END{
        for (k in s) printf "%-14s %12.1f us  %5.1f%%\n", k, s[k], 100*s[k]/tot;
        printf "%-14s %12.1f us\n", "TOTAL", tot
      }' | sort -k2 -rn
}

# --- CPU: flamegraph via samply ------------------------------------------------
# samply samples the host process and opens the Firefox-profiler UI. Shows where
# WALL-CLOCK goes on the CPU side: command recording, bind-group creation, the
# readback poll, CPU argmax, tokenizer — the non-GPU half of decode.
profile_cpu() {
  echo "==> CPU flamegraph via samply (model=$MODEL quant=$QUANT batch=$BATCH N=$N)"
  build_release ""
  echo "    samply will open the profiler in your browser when the run finishes."
  samply record -o "$OUT/cpu.json.gz" -- \
    ./target/release/arf generate \
    --device gpu $(quant_flag) $(batch_flag) $(tok_flag) $(arch_flag) \
    --model "$MODEL" --prompt "$PROMPT" --max-tokens "$N"
  echo "    profile saved: $OUT/cpu.json.gz  (re-open with: samply load $OUT/cpu.json.gz)"
}

# --- PGO: profile-guided optimization ------------------------------------------
# Build an instrumented binary, run the representative workload to collect a
# branch/value profile, then rebuild with that profile so the optimizer lays out
# the hot paths (GEMV reductions, the matmul, sampling) for this exact workload.
# A/B reports before/after tok/s.
profile_pgo() {
  command -v cargo-pgo >/dev/null || { echo "cargo-pgo not installed: cargo install cargo-pgo"; exit 1; }
  echo "==> PGO: instrument → run → optimize (model=$MODEL quant=$QUANT N=$N)"

  # cargo-pgo merges .profraw with `llvm-profdata`. It ships in the rustup toolchain's
  # llvm-tools (`rustup component add llvm-tools-preview`) but is NOT on PATH — and on
  # this box the active `rustc` is Homebrew's (different sysroot), so search the rustup
  # toolchains dir directly and prepend it.
  if ! command -v llvm-profdata >/dev/null; then
    local profdata
    profdata="$(find "${RUSTUP_HOME:-$HOME/.rustup}/toolchains" -name llvm-profdata 2>/dev/null | head -1)"
    [ -x "$profdata" ] && export PATH="$(dirname "$profdata"):$PATH"
  fi
  command -v llvm-profdata >/dev/null || {
    echo "llvm-profdata not found — run: rustup component add llvm-tools-preview"; exit 1; }

  # Args as an ARRAY so the multi-word --prompt isn't word-split on expansion.
  local args=(generate --device gpu)
  [ "$QUANT" != "none" ] && args+=(--quant "$QUANT")
  [ "$BATCH" -gt 1 ] && args+=(--batch "$BATCH")
  [ -f "$TOK" ] && args+=(--tokenizer "$TOK")
  [ -n "$ARCH" ] && args+=(--arch "$ARCH")
  args+=(--model "$MODEL" --prompt "$PROMPT" --max-tokens "$N")

  echo "--> baseline (no PGO):"
  build_release ""
  ./target/release/arf "${args[@]}" 2>&1 1>/dev/null | grep -E "tok/s" || true

  local triple; triple="$(rustc -vV | sed -n 's/host: //p')"
  local bin="./target/$triple/release/arf"

  echo "--> instrumented build + profiling run (collects .profraw under target/pgo-profiles):"
  RUSTFLAGS="-C target-cpu=native" cargo pgo instrument build -- -p arf-cli >&2
  "$bin" "${args[@]}" >/dev/null 2>&1 || true   # emits the profile cargo-pgo will merge

  echo "--> optimized build with the collected profile:"
  RUSTFLAGS="-C target-cpu=native" cargo pgo optimize build -- -p arf-cli >&2

  echo "--> PGO-optimized result:"
  "$bin" "${args[@]}" 2>&1 1>/dev/null | grep -E "tok/s" || true
}

case "${1:-all}" in
  gpu) profile_gpu ;;
  cpu) profile_cpu ;;
  pgo) profile_pgo ;;
  all) profile_gpu; echo; profile_cpu ;;
  *) echo "usage: $0 {gpu|cpu|pgo|all}"; exit 2 ;;
esac
