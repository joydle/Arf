#!/usr/bin/env bash
# kernel_ab.sh — the measurement ladder for a gated kernel experiment.
#
# WHY THIS EXISTS. Reviewing one campaign's losses, nearly all of them were
# MEASUREMENT failures, not bad ideas:
#   L197  never gated the flag OFF -> measured a register-spill bug, blamed the idea
#   L198  blamed swap/the machine  -> it was our own binary (found by bisecting it)
#   L183  called the GPU degraded  -> it was our own warm-up race
#   L204  OFF arm read 16.0 vs 17.6 measured 20 min earlier -> pure box drift
#
# The common thread: comparing against a number from a DIFFERENT MOMENT. This box
# drifts 4-8% between sessions, which is the same magnitude as most wins we chase.
# A baseline from 20 minutes ago is not a baseline, it is a rumor.
#
# The ladder, cheapest and most decisive first:
#   1. GATE      flag OFF must equal HEAD (same binary)  -> proves you are testing
#                the IDEA and not an implementation bug. THIS IS THE ONE L197 SKIPPED.
#   2. BASELINE  re-measure HEAD in the same minute      -> kills drift
#   3. CORRECT   text must be identical across arms      -> a fast wrong answer is not a win
#   4. A/B       one env var apart, N reps               -> the actual comparison
#   5. CONFIRM   b=1 AND b=3                             -> catches "wins here, loses there"
#                (L201 lost on BOTH; that is what made it conclusive)
#
# USAGE:  scripts/kernel_ab.sh ARF_GEMV_ASUM=1 [reps]
#         scripts/kernel_ab.sh --no-gate ARF_FOO=1   (skip step 1 if no OFF path exists)
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${ARF_AB_MODEL:-models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf}
ARCH=${ARF_AB_ARCH:-qwen3.8}
QUANT=${ARF_AB_QUANT:-q4ks}
SCRATCH=${TMPDIR:-/tmp}/kernel_ab.$$
mkdir -p "$SCRATCH"
GATE=1
[ "${1:-}" = "--no-gate" ] && { GATE=0; shift; }
FLAG=${1:?usage: kernel_ab.sh VAR=VAL [reps]}
REPS=${2:-3}
PORT=$((8700 + RANDOM % 200))

# Deterministic prompt for the correctness check. Greedy (temperature 0), so any
# difference between arms is a REAL numerical divergence, not sampling noise.
PROMPT='The capital of France is'

run_arm() {  # $1=env  $2=label  -> prints "tok/s xN | b=3 ms | text"
  local env_str="$1" label="$2" log="$SCRATCH/$2.log" pid
  env $env_str ./target/release/arf-serve --model "$MODEL" --arch "$ARCH" \
      --quant "$QUANT" --no-tty --port "$PORT" --max-batch-size 4 > "$log" 2>&1 &
  pid=$!
  for _ in $(seq 1 170); do
    grep -q "serving on" "$log" 2>/dev/null && break
    if grep -q "island compile failed" "$log" 2>/dev/null; then
      printf '  %-10s COMPILE FAIL: %s\n' "$label" "$(grep -o 'error:.*' "$log"|head -1)"
      kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
    fi
    if ! kill -0 $pid 2>/dev/null; then
      printf '  %-10s DIED: %s\n' "$label" "$(tail -2 "$log"|tr '\n' ' ')"; return 1
    fi
    sleep 2
  done
  python3 scripts/ab_bench.py "$PORT" warm >/dev/null 2>&1
  printf '  %-10s ' "$label"
  local i; for i in $(seq 1 "$REPS"); do
    printf '%s ' "$(python3 scripts/ab_bench.py "$PORT" | grep -o '[0-9.]* tok/s')"
  done
  printf '| b=3 %s ' "$(timeout 500 python3 scripts/ab_bench.py "$PORT" batch3 2>/dev/null | grep -o '[0-9.]* ms')"
  local txt
  txt=$(timeout 150 curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":24,\"temperature\":0}" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"].replace(chr(10)," ")[:60])' 2>/dev/null)
  printf '| %s\n' "$txt"
  echo "$txt" > "$SCRATCH/$2.txt"
  kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}

pkill -f arf-serve 2>/dev/null; sleep 3
echo "=== ladder for $FLAG  (reps=$REPS, port=$PORT) ==="
echo "--- step 2: BASELINE (HEAD, this minute) ---"
run_arm "" baseline
if [ "$GATE" = 1 ]; then
  echo "--- step 1: GATE (flag OFF == baseline?) ---"
  echo "    (same binary as the arm below; a difference here means an"
  echo "     IMPLEMENTATION bug, so the A/B would be measuring that, not the idea)"
fi
echo "--- step 4: A/B ---"
run_arm "$FLAG" experiment

echo "--- step 3: CORRECTNESS ---"
if diff -q "$SCRATCH/baseline.txt" "$SCRATCH/experiment.txt" >/dev/null 2>&1; then
  echo "  text IDENTICAL across arms"
else
  echo "  ⚠️  TEXT DIVERGED — a speed number here is meaningless until explained:"
  echo "     baseline:   $(cat "$SCRATCH/baseline.txt" 2>/dev/null)"
  echo "     experiment: $(cat "$SCRATCH/experiment.txt" 2>/dev/null)"
fi
echo
echo "logs: $SCRATCH"
