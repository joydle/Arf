#!/usr/bin/env bash
# spec_ab.sh — the interleaved greedy-vs-speculation A/B this repo has owed since L313.
#
# WHY THIS EXISTS. Every earlier speculation timing was labelled "not
# quotable", for one of two reasons: it used a FRESH PROCESS PER ARM (so the arms saw different
# cache and clock states), or the box was loaded. Rule 3 is explicit — arms must be interleaved
# in ONE session, because this box drifts 4-8% between sessions and far worse once swap grows.
# A ratio against llama.cpp cannot do this job: a different question entirely.
#
# WHY ONE DAEMON AT A TIME, ALTERNATING IN ROUNDS. `spec_k()` in arf-serve is a process-wide
# OnceLock read from the environment at startup, so ONE daemon CANNOT serve both arms — asking it
# for a "spec" run would silently return another greedy run and report a meaningless ~1.00.
# Two RESIDENT daemons would be ideal, but this model is ~22 GB resident on a 39 GB box: two at
# once forces swap, and swap is precisely what makes a number worthless here (L325 measured the
# same work running ~1000x slower once the box paged). So the arms run in ROUNDS — greedy daemon
# up, N samples, down; spec daemon up, N samples, down; repeat — and the ROUND is the unit that
# interleaves. That still cancels the session-scale drift rule 3 is aimed at (both arms see the
# same box, minutes apart, repeatedly), while never having 44 GB in flight. The cost is honest and
# stated: a model reload sits between arms, so per-round noise is higher than a shared-daemon A/B
# would give — which is why the MEDIAN over rounds is reported, not a single pair.
#
# CONTRACT — a number is emitted ONLY if all of these hold:
#   * load average < LOAD_MAX at start AND at end
#   * swap used == 0 at start AND at end   (swap growth is the L325 killer: same work, ~1000x slower)
#   * every pair's two arms produced IDENTICAL text (speculation is byte-identical to greedy by
#     construction; if it is not, the speed number is worthless — rule 4, correctness gates speed)
#
# USAGE:  scripts/spec_ab.sh [PAIRS] [K] [GEN]
#   e.g.  scripts/spec_ab.sh 5 1 128
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PAIRS="${1:-5}"; K="${2:-1}"; GEN="${3:-128}"
PORT="${PORT:-8127}"
MODEL="${MODEL:-models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf}"
ARCH="${ARCH:-qwen3.8-27b}"
QUANT="${QUANT:-q4ks}"
LOAD_MAX="${LOAD_MAX:-4.0}"
PROMPT="${PROMPT:-Write a haiku about the sea, then explain it in two sentences.}"

load_now(){ uptime | sed 's/.*load averages: //' | awk '{print $1}'; }
swap_now(){ sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/'; }

L0="$(load_now)"; S0="$(swap_now)"
echo "box at start: load $L0, swap ${S0}M"
if awk -v l="$L0" -v m="$LOAD_MAX" 'BEGIN{exit !(l+0 > m+0)}'; then
  echo "REFUSE: load $L0 exceeds $LOAD_MAX. A number taken here is noise (rule 3)."; exit 2
fi

cleanup(){ trap - INT TERM EXIT; for p in "${DPID:-}" "${DPID2:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; }
trap cleanup INT TERM EXIT

wait_up(){ for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

PJ=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")
req(){ # $1 = port -> "ms<TAB>md5"
  local t0 t1 body
  t0=$(python3 -c 'import time;print(time.time())')
  body=$(curl -s "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":$PJ}],\"max_tokens\":$GEN,\"temperature\":0}")
  t1=$(python3 -c 'import time;print(time.time())')
  printf '%s\t%s\n' \
    "$(python3 -c "print(f'{($t1-$t0)*1000:.1f}')")" \
    "$(printf '%s' "$body" | python3 -c 'import json,sys,hashlib;d=json.load(sys.stdin);print(hashlib.md5(d["choices"][0]["message"]["content"].encode()).hexdigest()[:12])' 2>/dev/null || echo ERR)"
}

# Run one arm: bring the daemon up at spec-k=$1, take SAMPLES timings, bring it down.
# Echoes "median_ms<TAB>md5". The first request of each arm is discarded (pipeline warm).
SAMPLES="${SAMPLES:-3}"
run_arm(){
  local k="$1" port="$2" pid ms md5 first=1
  # QUANT is passed as an ENV VAR, not only as --quant: weight_cache::flags_hash() keys the
  # transcode sidecar on the ENV `QUANT`, so running with the flag alone (env unset) hashes
  # differently and forces a full ~22 GB rebuild — a guaranteed cache MISS on every arm, which
  # both wastes minutes and leaves a `.building` orphan if the run is interrupted.
  #
  # 🔴 L334 — assignments are EXPLICIT PREFIXES, never `env $string`: zsh does not word-split
  # an unquoted parameter, so `env $envs cmd` passes ONE garbage assignment and silently sets
  # NOTHING else. That exact bug made a four-arm A/B measure four identical configs and nearly
  # published a fabricated +20%. If you refactor this launch, keep the assignments literal.
  QUANT="$QUANT" ARF_MSL_GEMV=1 ARF_SPEC_K="$k" \
    ./target/release/arf serve "$MODEL" --port "$port" \
    --arch "$ARCH" --quant "$QUANT" >"/tmp/spec_ab_k$k.log" 2>&1 &
  pid=$!
  if ! wait_up "$port"; then
    echo "REFUSE: daemon (k=$k) never came up." >&2; tail -20 "/tmp/spec_ab_k$k.log" >&2; kill $pid 2>/dev/null; return 1
  fi
  local acc=()
  for _ in $(seq 1 $((SAMPLES+1))); do
    IFS=$'\t' read -r ms md5 < <(req "$port")
    if [ "$first" = 1 ]; then first=0; continue; fi   # discard the warm request
    acc+=("$ms")
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  printf '%s\t%s\n' "$(printf '%s\n' "${acc[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')" "$md5"
}

echo
echo "  round |   greedy ms |     spec ms |  ratio | md5 greedy   | md5 spec"
echo "  ------+-------------+-------------+--------+--------------+-------------"
RATIOS=(); MISMATCH=0
for i in $(seq 1 "$PAIRS"); do
  IFS=$'\t' read -r GMS GMD5 < <(run_arm 0 "$PORT") || exit 3
  IFS=$'\t' read -r SMS SMD5 < <(run_arm "$K" "$PORT") || exit 3
  R=$(python3 -c "print(f'{$GMS/$SMS:.3f}')" 2>/dev/null || echo "-")
  RATIOS+=("$R")
  printf '  %5s | %11s | %11s | %6s | %-12s | %-12s\n' "$i" "$GMS" "$SMS" "$R" "$GMD5" "$SMD5"
  [ "$GMD5" != "$SMD5" ] && { echo "  !! md5 MISMATCH on round $i"; MISMATCH=1; }
done

L1="$(load_now)"; S1="$(swap_now)"
echo
echo "box at end:   load $L1, swap ${S1}M"
# What matters is swap GROWTH during the run, not its absolute level: macOS keeps pages in the
# swap file after the pressure that caused them is gone, so demanding 0 would refuse on a box
# that is actually idle. Growth is the signal that THIS run started paging (L325).
SWAP_GROWTH_MAX="${SWAP_GROWTH_MAX:-64}"
if awk -v a="$S0" -v b="$S1" -v m="$SWAP_GROWTH_MAX" 'BEGIN{exit !((b+0)-(a+0) > m+0)}'; then
  echo "REFUSE: swap grew ${S0}M -> ${S1}M during the run (> ${SWAP_GROWTH_MAX}M)."
  echo "        The run paged; discard these numbers (see L325)."; exit 4
fi
if [ "$MISMATCH" = 1 ]; then
  echo "REFUSE: the arms produced different text. Speculation must be byte-identical to greedy;"
  echo "        a speed number from a path that emits wrong output is worthless (rule 4)."; exit 5
fi

MED=$(printf '%s\n' "${RATIOS[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
echo
echo "MEDIAN RATIO (greedy/spec; >1 means speculation is FASTER): $MED"
echo "NOTE: arms alternate in ROUNDS (one daemon resident at a time — 2x22 GB does not fit 39 GB);"
echo "      $SAMPLES timed samples per arm per round, first discarded; md5 equal on every round."
