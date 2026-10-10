#!/usr/bin/env bash
# REPRODUCE THE CANONICAL coder-30B CONCURRENCY NUMBERS — one command, footgun-proof.
#
# WHY THIS EXISTS (2026-07-07): a "we lost our speed, can't reproduce!" panic burned an
# evening. The speed was NEVER lost. The cause was harness friction that masqueraded as a
# regression:
#   1. COLD weight-cache — first run transcodes the 18 GB GGUF (one-time). A short timeout
#      dies mid-transcode with zero output → looks like a hang.
#   2. ~10 s FIRST-TOKEN stall — Metal pipeline JIT on the first decode step (one-time per
#      process). p99 shows it; steady-state is unaffected.
#   3. Timeouts too short (120–240 s) → the harness never survived to print steady numbers.
#   4. ARF_BATCH_MEGA / MSL_GEMV / MEGAKERNEL not set → silent slow path (per-row fallback).
#      (The serve binary defaults these on now, but a raw serve_loop_bench invocation did not.)
#
# This script removes all four: warms the cache first (discarded), sets every fast-path lever
# explicitly, and gives each level a timeout that clears the first-token stall. Steady-state
# DECODE tok/s + p50/p90/p95/p99 are what you compare to docs/PERFORMANCE.md.
#
# Usage:  scripts/repro_conc.sh            # full sweep 8/16/32/64
#         CONCS="64" scripts/repro_conc.sh # just conc64
#         GGUF=/path scripts/repro_conc.sh # override model
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

GGUF="${GGUF:-${QWEN_GGUF:-models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf}}"
CONCS="${CONCS:-8 16 32 64}"
BIN=./target/release/examples/serve_loop_bench
LEVERS=(ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1)   # batch_mega is default-on in weights.rs; kept minimal + explicit

[ -f "$GGUF" ] || { echo "ABORT: GGUF not found: $GGUF"; exit 1; }

# --- machine discipline: absolute tok/s needs AC + no LPM ---
pmset -g batt 2>/dev/null | grep -q "AC Power" || echo "⚠️  NOT on AC power — absolute tok/s will be throttled (ratios still valid)."
pmset -g 2>/dev/null | grep -qi "lowpowermode.*1" && echo "⚠️  Low Power Mode is ON — turn it off for real numbers."

# --- one GPU load at a time (a 0%-CPU stuck proc = wedge; never kill -9, per gpu-wedge post-mortem) ---
if pgrep -f 'serve_loop_bench|arf-serve|batched_mega_parity' >/dev/null; then
  echo "ABORT: another GPU proc is live — let it exit first (one 30B load at a time)."; exit 1
fi

echo "== building serve_loop_bench =="
cargo build --release -q -p arf-gpu --example serve_loop_bench 2>&1 | tail -3

# --- warm the weight-cache ONCE (discarded): transcode + pipeline JIT happen here, not in a timed run ---
if ! ls "$GGUF".arf-q4ks-cache >/dev/null 2>&1; then
  echo "== cold cache: warming (transcode 18 GB, one-time; ~2-3 min) — discarded =="
  timeout 400 env "${LEVERS[@]}" ARCH=qwen3-coder GGUF="$GGUF" QUANT=q4ks CONC=8 REQS=8 GEN=4 "$BIN" >/dev/null 2>&1
fi

echo "== coder-30B concurrency sweep (steady-state DECODE tok/s + latency percentiles) =="
echo "   GGUF: $GGUF"
for c in $CONCS; do
  reqs=$(( c * 2 )); [ "$reqs" -lt 16 ] && reqs=16
  # Timeout budget: load (~15 s warm) + first-token stall (~10 s) + reqs/c batches * ~0.2 s * gen.
  # 300 s clears every level comfortably; short timeouts were the whole false-alarm.
  line=$(timeout 300 env "${LEVERS[@]}" ARCH=qwen3-coder GGUF="$GGUF" QUANT=q4ks \
           CONC="$c" REQS="$reqs" GEN=64 "$BIN" 2>/dev/null | grep -iE "^agg|LATENCY")
  agg=$(echo "$line"    | grep -oE 'DECODE [0-9]+\.[0-9]+' | grep -oE '[0-9.]+' | head -1)
  p50=$(echo "$line"    | grep -oE 'p50 [0-9.]+ms->[0-9]+' | grep -oE '>[0-9]+' | tr -d '>' | head -1)
  p95=$(echo "$line"    | grep -oE 'p95 [0-9.]+ms->[0-9]+' | grep -oE '>[0-9]+' | tr -d '>' | head -1)
  printf "  conc%-3s  DECODE %-7s tok/s   p50=%-5s p95=%-5s tok/s\n" "$c" "${agg:-FAIL}" "${p50:-?}" "${p95:-?}"
  [ -z "${agg:-}" ] && echo "     ^ FAIL: check GPU health (ioreg recoveryCount) — if a proc is stuck at 0% CPU, LET IT EXIT."
  sleep 3
done
echo ""
echo "Compare against docs/PERFORMANCE.md. If these are within ~10% and no level FAILed,"
echo "the fast path is intact. A FAIL is almost always cold-cache or a busy machine, NOT lost perf —"
echo "run 'cargo run --release -p arf-gpu --example batched_mega_parity' to prove the kernels."
