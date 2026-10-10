#!/usr/bin/env bash
# two_model_gate.sh — CORRECTNESS gate for BOTH shipped models, on their DEFAULT config.
#
# WHY THIS EXISTS. Muse Glimmer work now touches shared island code — the megakernel's dense
# FFN activation, the MegaLayer struct, the embed path. Every one of those is code Qwen also
# runs, and Qwen is where the win lives (2.51x @conc32 — measured
# 2026-08-11; the 7.28x TTFT figure that used to sit here has no recorded measurement and is not
# quotable). A silent Qwen regression
# while chasing a second model would be the worst possible trade, and "I checked by hand once"
# is not protection.
#
# So: both models, zero env overrides (the config a user actually gets), a real answer each.
# Text gates, not throughput — this catches WRONG, which is the failure mode island edits cause.
# Speed is nightly_gate.sh's job.
#
# CONTRACT: prints PASS only if both models answer correctly. Any failure => non-zero exit and
# a named reason. A refused run is the gate WORKING.
#
# USAGE:  scripts/two_model_gate.sh
#   env:  QWEN_GGUF / MUSE_GGUF override the models (see scripts/lib/arf_env.sh)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=lib/arf_env.sh
source scripts/lib/arf_env.sh

PORT="${PORT:-8790}"
FAILED=0

# ⚠️ MEASUREMENT HYGIENE (learned the hard way, 2026-08-14). A benchmark on this box measured
# 11.27 tok/s where the same commit had measured 16.90 — a fake 33% "regression" that cost a
# real debugging detour. Cause: a Chrome window with GPU compositing plus a vite dev server
# were running. Arf decode is GPU-bandwidth-bound, so ANY other GPU client silently taxes it.
# After killing them the same binary measured 17.99. Warn loudly; do not silently benchmark.
if pgrep -f "Google Chrome" >/dev/null 2>&1 || pgrep -f "[v]ite" >/dev/null 2>&1; then
  echo "WARNING: Chrome and/or vite are running — they contend for the GPU and DEPRESS every"
  echo "         number below. Kill them before trusting any throughput measurement."
fi

cleanup() { pkill -f "arf-serve.*--port $PORT" 2>/dev/null; }
trap cleanup EXIT

# ask <name> <gguf> <arch> <prompt> <must-contain...>
ask() {
  local name="$1" gguf="$2" arch="$3" prompt="$4"; shift 4
  if [ ! -f "$gguf" ]; then
    echo "SKIP $name — model not present ($gguf)"; return 0
  fi
  cleanup; sleep 2
  # DEFAULT config on purpose: no ARF_* overrides. This is what a user gets.
  nohup ./target/release/arf-serve --model "$gguf" --arch "$arch" \
    --port "$PORT" --num-blocks 512 --max-batch-size 1 > "/tmp/tmg_$name.log" 2>&1 &
  local waited=0
  until curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null 2>&1; do
    sleep 4; waited=$((waited+4))
    if [ $waited -gt 600 ]; then echo "FAIL $name — daemon never came up"; FAILED=1; return 1; fi
  done
  local body reply
  body=$(python3 -c 'import json,sys; print(json.dumps({"model":"m","messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":120}))' "$prompt")
  reply=$(curl -s -m 400 "localhost:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$body" \
          | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
  if [ -z "$reply" ]; then echo "FAIL $name — empty reply"; FAILED=1; return 1; fi
  local missing=""
  for need in "$@"; do
    case "$reply" in *"$need"*) ;; *) missing="$missing '$need'";; esac
  done
  if [ -n "$missing" ]; then
    echo "FAIL $name — reply missing:$missing"
    echo "     got: $(printf '%s' "$reply" | head -c 200)"
    FAILED=1; return 1
  fi
  echo "PASS $name"
}

echo "=== two-model correctness gate (default config, no env overrides) ==="
# Qwen: the model the WIN is measured on. Any island change must leave this correct.
ask qwen "$QWEN_GGUF" qwen3-coder "Write a Python function to reverse a string." "def" "[::-1]"
# Muse Glimmer: exercises the four-norm block, the attention gate, the pre-trunk embedding norm
# and the interleaved RoPE. "2 + 2 = 4" is llama's exact answer to this prompt.
ask muse "$MUSE_GGUF" muse-glimmer "What is 2+2?" "2 + 2 = 4"

if [ "$FAILED" -ne 0 ]; then echo "GATE: FAILED"; exit 1; fi
echo "GATE: PASS — both models correct on default config"
