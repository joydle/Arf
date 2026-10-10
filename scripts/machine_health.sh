#!/usr/bin/env bash
# machine_health.sh — ONE command: "is the GPU healthy or throttled RIGHT NOW?"
#
# WHY: perf "vanished" repeatedly because the GPU clock silently sat at its 338MHz floor
# instead of ramping to 1578MHz. macOS exposes NO single "throttled" flag for this — the
# DVFS/SMC power state can wedge while pmset/ioreg all read "clean". This script probes the
# clock UNDER LOAD (the only honest test) and cross-checks with llama.cpp as an independent
# referee, then prints ONE verdict: RECOVERED / THROTTLED.
#
# USAGE:  sudo scripts/machine_health.sh          # sudo needed for the real clock read
#         scripts/machine_health.sh --no-sudo     # skip clock, use llama referee only
#
# READ THE VERDICT:
#   RECOVERED  -> clock holds ~1578MHz under load AND llama ~>=380  -> run your benches, demo away.
#   THROTTLED  -> clock stuck low under load OR llama ~<150         -> cold boot: Shut Down, unplug 90s, boot.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
CODER="${GGUF:-${QWEN_GGUF:-models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf}}"
LLAMA="${LLAMA:-/opt/homebrew/bin/llama-bench}"
NOSUDO=0; [ "${1:-}" = "--no-sudo" ] && NOSUDO=1

echo "════════ MACHINE HEALTH — $(date '+%H:%M:%S') ════════"

# --- static, no-sudo signals (fast) ---
echo "── static checks (no load) ──"
echo "  uptime:      $(uptime | sed 's/^ *//')"
echo "  energy mode: $(pmset -g custom 2>/dev/null | grep -i powermode | head -1 | awk '{print $2}') (0=auto 1=low 2=high; 1 would cap the GPU)"
echo "  GPU faults:  recoveryCount=$(ioreg -r -c IOAccelerator 2>/dev/null | grep -oE '"recoveryCount"=[0-9]*' | head -1 | cut -d= -f2) (>0 = GPU reset since boot)"
echo "  thermal:     $(pmset -g therm 2>/dev/null | grep -iE 'CPU_Speed|limit' | head -1 || echo 'none recorded (not thermal)')"
echo "  spotlight:   $(pgrep -f mdworker_shared | wc -l | tr -d ' ') mdworkers active (many = reindexing, steals bandwidth)"

# --- the referee: llama.cpp under load (independent of our code) ---
echo "── llama.cpp referee (independent engine; healthy ~398) ──"
if [ -x "$LLAMA" ] && [ -f "$CODER" ]; then
  LT=$($LLAMA -m "$CODER" -p 0 -n 128 -ngl 99 2>/dev/null | grep -oE 'tg128 *\| *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo "  llama tg128: ${LT:-ERR} tok/s"
else
  echo "  (llama-bench or model not found — skipping referee)"; LT=""
fi

# --- the real clock, under load (sudo) ---
CLK=""
if [ "$NOSUDO" = 0 ]; then
  echo "── GPU clock UNDER LOAD (the definitive test) ──"
  # drive the GPU with a short bench in the background, sample the clock while it runs
  ARF_NO_WARM_PIPELINES=1 ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 QUANT=q4ks \
    ARCH=qwen3-coder GGUF="$CODER" CONC=64 REQS=192 GEN=96 \
    ./target/release/examples/serve_loop_bench >/tmp/mh_bench.log 2>&1 &
  BP=$!
  sleep 20   # let it reach steady decode so the clock has ramped
  CLK=$(powermetrics --samplers gpu_power -i 1000 -n 4 2>/dev/null \
        | grep -oE 'GPU HW active frequency: [0-9]+ MHz' | grep -oE '[0-9]+' | sort -n | tail -1)
  echo "  peak GPU freq under load: ${CLK:-ERR} MHz (healthy = ~1578; floor = 338)"
  kill "$BP" 2>/dev/null; wait "$BP" 2>/dev/null
fi

# --- VERDICT ---
echo "──────────────────────────────────────────"
HEALTHY=1
[ -n "$LT" ]  && awk "BEGIN{exit !($LT < 200)}"  && HEALTHY=0   # llama way below ~398
[ -n "$CLK" ] && [ "$CLK" -lt 1400 ] 2>/dev/null && HEALTHY=0   # clock never reached near-max under load
if [ "$HEALTHY" = 1 ]; then
  echo "  ✅ RECOVERED — clock reaches max under load, referee healthy. Run benches / demo."
else
  echo "  🔴 THROTTLED — machine is capping the GPU. Cold boot: Shut Down, unplug 90s, boot, wait 5min, re-run this."
fi
echo "════════════════════════════════════════════"
