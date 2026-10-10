#!/usr/bin/env bash
# soak_conc64.sh — THE SUSTAINED number (vs the burst number every other bench reports).
#
# WHY: every H2H/canonical figure we quote comes from a ~30s steady window. Over a long session
# the machine sags (MEASURED 2026-07-28: same-config canonical 209 -> 172 across an hour of
# continuous benching). A number you publish, or gate a nightly on, must be the one the
# engine HOLDS — not the one it touches once on a cold machine.
#
# HOW: serve_loop_bench prints its steady-p50 CANONICAL only at the END of a run, so a soak is
# built from BACK-TO-BACK measured slices (default 8 slices x ~2.5 min). Each slice is a full,
# honest canonical; the SERIES is the decay shape; the LAST slices are the sustained truth.
# The model is reloaded per slice (warm mmap cache, ~15s) — that cost is outside the measured
# window, and reloading is itself realistic (it is what a restarted daemon pays).
#
# USAGE:  scripts/soak_conc64.sh [SLICES] [CONC]
#   env:  PREFIX=512 GEN=32 REQS_PER_SLICE=1024 GGUF=<blob> OUT=<dir> STAMP=<tag>
#   sudo (optional) enables the GPU-clock column.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SLICES="${1:-8}"; CONC="${2:-64}"
PREFIX="${PREFIX:-512}"; GEN="${GEN:-32}"
REQS="${REQS_PER_SLICE:-1024}"
GGUF="${GGUF:-${QWEN_GGUF:-models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf}}"
OUT="${OUT:-target/soak}"; STAMP="${STAMP:-soak}"
CSV="$OUT/$(date '+%Y-%m-%d')-conc${CONC}-soak-${STAMP}.csv"

cleanup(){ pkill -TERM -f 'serve_loop_bench' 2>/dev/null; exit 130; }
trap cleanup INT TERM

echo "════════ conc$CONC SOAK · $SLICES slices · REQS=$REQS PREFIX=$PREFIX GEN=$GEN ════════"
pmset -g batt 2>/dev/null | grep -q "AC Power" || echo "  ⚠️  ON BATTERY — absolutes capped; plug in for a real baseline."
pgrep -f 'serve_loop_bench|arf-serve|batched_mega_parity' >/dev/null && { echo "🔴 another GPU proc live — one 30B load at a time."; exit 2; }
mkdir -p "$OUT"
echo "slice,elapsed_min,tok_s,p50_ms,p99_ms,gpu_mhz,rss_gb" > "$CSV"
printf "  %5s  %8s  %7s  %8s  %8s  %8s  %7s\n" slice elapsed tok/s p50_ms p99_ms GPU_MHz RSS_GB

T0=$(date +%s)
for s in $(seq 1 "$SLICES"); do
  LOG=$(mktemp /tmp/soak_slice.XXXXXX)
  # THE WIN CONFIG (what we race with). No GPUSTATS — its per-chunk waits distort the step.
  env ARF_NO_WARM_PIPELINES=1 ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARF_BATCH_MEGA_KSTEP=8 \
      ARF_ATTN_COALESCED=1 ARF_KV_F16=1 ARF_ATTN_KVHEAD=1 \
      QUANT=q4ks ARCH=qwen3-coder GGUF="$GGUF" CONC="$CONC" REQS="$REQS" GEN="$GEN" PREFIX="$PREFIX" \
      ./target/release/examples/serve_loop_bench >"$LOG" 2>&1 &
  BENCH=$!
  # Sample the clock + RSS mid-slice (once the run is in steady decode).
  ( sleep 45; MHZ=$( (sudo -n powermetrics --samplers gpu_power -i 400 -n 4 2>/dev/null \
        | grep -oE 'GPU HW active frequency: [0-9]+' | grep -oE '[0-9]+' \
        | awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}') )
    RSS=$(ps -o rss= -p "$BENCH" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
    echo "${MHZ:-NA} ${RSS:-NA}" > "$LOG.env" ) &
  wait "$BENCH" 2>/dev/null
  read -r MHZ RSS < "$LOG.env" 2>/dev/null || { MHZ=NA; RSS=NA; }
  TOKS=$(grep -oE "CANONICAL conc[0-9]+ = [0-9]+" "$LOG" | grep -oE '[0-9]+$' | tail -1)
  # NOTE: grep -oE 'p50 [0-9.]+ms' matches BOTH the number and the "50"/"99" of the label when
  # piped again, so extract the ms value directly (tr -d newline guards the CSV row).
  P50=$(grep -oE 'p50 [0-9.]+ms' "$LOG" | tail -1 | sed -E 's/p50 ([0-9.]+)ms/\1/' | tr -d '\n')
  P99=$(grep -oE 'p99 [0-9.]+ms' "$LOG" | tail -1 | sed -E 's/p99 ([0-9.]+)ms/\1/' | tr -d '\n')
  MIN=$(awk "BEGIN{printf \"%.1f\", ($(date +%s)-$T0)/60}")
  printf "  %5d  %8s  %7s  %8s  %8s  %8s  %7s\n" "$s" "$MIN" "${TOKS:-ERR}" "${P50:-NA}" "${P99:-NA}" "${MHZ:-NA}" "${RSS:-NA}"
  echo "$s,$MIN,${TOKS:-},${P50:-},${P99:-},${MHZ:-},${RSS:-}" >> "$CSV"
  rm -f "$LOG" "$LOG.env"
done

echo ""
echo "════════════════════ SUSTAINED VERDICT ════════════════════"
python3 - "$CSV" <<'PY'
import csv, sys, statistics
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['tok_s'].strip().isdigit()]
if len(rows) < 2:
    print("  (too few slices — nothing to conclude)"); raise SystemExit
v = [int(r['tok_s']) for r in rows]
half = max(1, len(v)//2)
first, last = v[0], v[-1]
tail = v[half:]
print(f"  slices: {v}")
print(f"  PEAK (best slice)      : {max(v)} tok/s")
print(f"  SUSTAINED (2nd-half med): {int(statistics.median(tail))} tok/s   <-- the honest number")
print(f"  decay first->last      : {100*(last-first)/first:+.1f}%   (spread {min(v)}..{max(v)})")
p99 = [float(r['p99_ms']) for r in rows if r['p99_ms'].strip()]
if p99: print(f"  p99 step ms            : first {p99[0]:.0f} -> last {p99[-1]:.0f}")
rss = [float(r['rss_gb']) for r in rows if r['rss_gb'].strip() and r['rss_gb'] != 'NA']
if rss:
    print(f"  RSS GB                 : first {rss[0]:.1f} -> last {rss[-1]:.1f}" +
          ("   ⚠️ GROWTH — check for a leak" if rss[-1] - rss[0] > 1.0 else "   (flat — no leak)"))
PY
echo "  📝 series → $CSV"
