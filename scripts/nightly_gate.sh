#!/usr/bin/env bash
# nightly_gate.sh — THE ANTI-ROT HARNESS.
#
# WHY THIS EXISTS (two measured failures, one week):
#   1. Spec-decode's VERIFY COST silently decayed 2.4x -> 4.5x a decode over weeks. Nobody
#      noticed, because nothing watched it. That rot is why the spec ceiling is 1.26x and chat
#      stays blocked.
#   2. A "22x single-stream regression" was escalated as CRITICAL and then FULLY RETRACTED — the
#      metric lied (DECODE-mean undercounts the burst path ~8x). A conc1 gate would have caught
#      the error in seconds instead of costing hours.
#   Parity gates CORRECTNESS. The H2H scripts race the BATCHED path. Single-stream SPEED had
#   ZERO coverage. This closes that.
#
# CONTRACT: emits PASS only if every check clears. Any failure => non-zero exit + a named reason.
# A refused run is the gate WORKING.
#
# USAGE:  scripts/nightly_gate.sh [--quick]
#   --quick  skip the 36-min conc64 soak (use for pre-merge; the nightly should run it)
#   env:  GGUF=<blob>  CONC1_FLOOR=80  SOAK_FLOOR=420  H2H_TOL=0.05
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
GGUF="${GGUF:-${QWEN_GGUF:-models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf}}"
# L89/L90 (2026-08-07): serve_loop_bench now defaults ARF_KV_F16 (the daemon has since L5b), so
# this arm measures ~4% faster than when the floor was set. Re-raced conc1 = 89.5 ± 2.2 (4 runs,
# 93/89/87/89) vs llama 96.67 ± 3.19 = 0.926x. Floor raised 80 -> 84 to keep the SAME real margin
# below the measured mean (~6%): leaving it at 80 would have silently loosened the guard by the
# exact amount the KV fix gained, which is how a regression slips through unnoticed.
CONC1_FLOOR="${CONC1_FLOOR:-84}"      # tok/s (measured 89.5 ± 2.2; llama 96.67 ± 3.19)
SOAK_FLOOR="${SOAK_FLOOR:-420}"       # tok/s sustained conc64 (measured 441/443)
H2H_TOL="${H2H_TOL:-0.05}"            # allowed fractional drop vs the recorded baseline
BENCH=./target/release/examples/serve_loop_bench
FAILED=(); PASSED=()

say(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
ok(){   printf '  ✅ %s\n' "$*"; PASSED+=("$1"); }
bad(){  printf '  ❌ %s\n' "$*"; FAILED+=("$1"); }

# ── 0. PREFLIGHT: a gate on a sagging machine is worse than no gate ────────────────────────────
# MEASURED 2026-07-28: the same config read 209 vs 172 tok/s across one hot hour. Ratios cancel
# drift; absolutes do not — so the absolute floors below are only meaningful on a settled box.
say "0. PREFLIGHT"
ON_AC=1
if pmset -g batt 2>/dev/null | grep -q "AC Power"; then ok "on AC power"; else
  ON_AC=0
  echo "  ⚠️  ON BATTERY — absolute floors (conc1, soak) will be SKIPPED; H2H ratios still valid (the ratio cancels the cap)."
fi
if pgrep -f 'serve_loop_bench|arf-serve|llama-batched-bench|batched_mega_parity' >/dev/null; then
  bad "another GPU proc is live (one 30B load at a time)"
else ok "no competing GPU proc"; fi
LOAD=$(uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/')
awk "BEGIN{exit !($LOAD < 8)}" && ok "load settled ($LOAD)" || bad "load too high ($LOAD) — let the machine settle"
[ -f "$GGUF" ] && ok "GGUF present" || bad "GGUF missing: $GGUF"
[ -f "$GGUF.arf-q4ks-cache" ] && ok "q4ks weight cache warm" || echo "  ⚠️  cold weight cache — first run pays a one-time transcode"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '\n🔴 PREFLIGHT FAILED — not measuring on this machine state.\n'; exit 2
fi

# ── 0.5 ENV SURFACE RATCHET — cheap, non-GPU, runs even on battery ────────────────
# The config surface grew 187 -> 238 knobs in the weeks after the first count, because nothing
# watched it. Every knob is an untested branch. This ratchet may only ever be lowered; a new
# knob must replace one or carry a measured justification in the pull request. The rule
# and the full categorisation were in the 2026-08-05 env audit (removed).
say "0.5 ENV SURFACE RATCHET"
if EOUT=$(bash scripts/env_surface_guard.sh 2>&1); then
  ok "env surface: $(echo "$EOUT" | head -1)"
else
  bad "env surface GREW: $(echo "$EOUT" | grep -E 'GUARD FAILED' | head -1)"
fi

cargo build --release -q --example serve_loop_bench --example batched_mega_parity -p arf-gpu \
  || { echo "🔴 build failed"; exit 1; }

# ── 1. PARITY (correctness gates everything: a fast wrong answer is worthless) ─────────────────
say "1. PARITY (both MoE paths)"
POUT=$(make parity-conc 2>&1); PRC=$?
# NOTE: do NOT grep -i 'fail' — parity's PASSING lines contain the words "not a fail".
# Anchor on the harness's real failure signals: a non-zero exit, an explicit ": FAIL", or a panic.
if [ $PRC -ne 0 ] || echo "$POUT" | grep -qE ': FAIL|panicked|assertion.*failed'; then
  bad "parity FAILED (exit $PRC): $(echo "$POUT" | grep -E ': FAIL|panicked' | head -2 | tr '\n' ' ')"
else
  ok "parity all-PASS ($(echo "$POUT" | grep -c 'PASS') lines, exit 0)"
fi

# ── 2. conc1 FLOOR — the check that did not exist, and the metric that must be used ────────────
# ⚠️ THE RULE: read p90 PER GPU ROUND-TRIP and divide by the tokens that round-trip produced.
# NEVER DECODE-mean, NEVER p50 — reading DECODE-mean is exactly what produced the retracted
# "22x regression" (2026-07-29).
#
# L30 (2026-08-05) — THIS STAGE HARDCODED ARF_MEGA_KTOK=8 AND WOULD HAVE HIDDEN ITS OWN
# SUBJECT'S REGRESSION. The burst default flipped 8 -> 1 (+26.9% conc1, 2026-08-05: the
# burst became a pessimization once L16 proved the ~20ms inter-buffer gap it existed to
# amortize was gone). A gate pinning KTOK=8 would have gone on certifying the PESSIMIZED path
# forever — green while the shipped default rotted, which is the exact failure mode L30 found
# in the burst itself. So: DO NOT pin tuning flags here. Inherit the shipped defaults, and
# divide by KTOK so the metric follows whatever the default is.
KTOK_GATE="${ARF_MEGA_KTOK:-1}"          # inherit the shipped default (L30: 1)
say "2. SINGLE-STREAM FLOOR (p90 per round-trip / KTOK=$KTOK_GATE tokens)"
if [ "$ON_AC" = "0" ]; then echo "  ⏭  skipped (on battery — an absolute floor needs AC)"; C1=""; else
C1=$(env ARF_NO_WARM_PIPELINES=1 ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARF_MEGA_SINGLEQ=1 \
     ARF_MEGA_BLOCKING=1 QUANT=q4ks ARCH=qwen3-coder GGUF="$GGUF" \
     CONC=1 REQS=1 GEN=96 PREFIX=0 timeout 900 "$BENCH" 2>&1 \
     | grep -oE 'p90 [0-9.]+ms' | tail -1 | sed -E 's/p90 ([0-9.]+)ms/\1/')
if [ -n "$C1" ]; then
  RATE=$(awk "BEGIN{printf \"%.0f\", 1000*$KTOK_GATE/$C1}")
  awk "BEGIN{exit !($RATE >= $CONC1_FLOOR)}" \
    && ok "conc1 $RATE tok/s (p90 ${C1}ms/${KTOK_GATE}-tok round-trip) >= floor $CONC1_FLOOR" \
    || bad "conc1 $RATE tok/s BELOW floor $CONC1_FLOOR (p90 ${C1}ms/round-trip)"
elif [ "$ON_AC" = "1" ]; then bad "conc1 produced no p90 — run did not reach steady state"; fi
fi

# ── 3. H2H RATIOS (the drift-cancelling claim: ours/llama, alternating, same window) ───────────
say "3. LONG-CTX H2H RATIOS vs llama.cpp"
# Baselines measured 2026-07-28/29. NOTE: macOS bash is 3.2 — NO associative arrays (`declare
# -A` fails silently-ish and every lookup becomes empty), so use a portable case.
# RE-ANCHORED 2026-08-04. The old baselines (8:1.004 16:0.957 32:1.037 64:1.099) were set before
# L5b/L6/L7b/L8/L12 reshaped the engine, and conc16+conc64 then failed EVERY run — three
# consecutive gates read conc16 0.8527/0.8535/0.8543 and conc64 1.0010/0.9987/0.9950, i.e. tight
# and reproducible, which is drift-free measurement of a MOVED baseline, not a regression. A gate
# that always fails gets disabled, so these are re-anchored to the measured present. NOTE this is
# the SYNTHETIC long-ctx H2H; the REAL-path arms in stage 3.5 are the numbers we quote.
baseline_for(){ case "$1" in 8) echo 1.010;; 16) echo 0.854;; 32) echo 0.996;; 64) echo 0.998;; *) echo 1.0;; esac; }
for C in 8 16 32 64; do
  B=$(baseline_for "$C")
  R=$(bash a head-to-head run "$C" 3 2>&1 | grep -oE 'median ratio ours/llama = [0-9.]+' | grep -oE '[0-9.]+$')
  if [ -z "$R" ]; then bad "conc$C H2H produced no ratio"; continue; fi
  MIN=$(awk "BEGIN{printf \"%.4f\", $B*(1-$H2H_TOL)}")
  awk "BEGIN{exit !($R >= $MIN)}" \
    && ok "conc$C ratio $R (baseline $B, min $MIN)" \
    || bad "conc$C ratio $R BELOW $MIN (baseline $B) — regression or machine drift"
done

# ── 3.5 REAL-PATH FLOOR (2026-08-04) — the stage every earlier gate LACKED. All the campaign's
# false headlines (synthetic 89 tok/s single-stream, "441/1.10x conc64") passed the in-process
# stages above while the REAL serving path stalled 40s per prompt. This stage boots the actual
# daemon with DEFAULT config (no flags — defaults ARE the win config since 2026-08-03) and drives
# real 200-word prompts through HTTP, streamed, text verified. Floors are set ~10% under the
# post-campaign clean-race numbers (conc8 183/0.35s, conc16 184/0.48s).
say "3.5 REAL-PATH FLOOR (default daemon, real prompts over HTTP)"
SWAP_GB=$(sysctl -n vm.swapusage | awk '{print $3}' | tr -d 'M' | awk '{printf "%.0f", $1/1024}')
if [ "${SWAP_GB:-0}" -gt 10 ]; then
  bad "swap ${SWAP_GB}GB > 10GB — machine is not settled; real-path numbers would be garbage (skip is a FAIL: reboot first)"
else
  pkill -9 -f 'arf-serve|llama-server' 2>/dev/null; sleep 3
  # L133: this arm ran with NO env flags and --num-blocks 2048 (9 GB of pinned KV pools), which
  # is exactly the configuration that OOM-rebooted the Mac twice on 2026-08-12. ARF_ATTN_COALESCED
  # + ARF_KV_F16 are REQUIRED for batch.rs:1498 to mark the f32 pool volatile.
  . "$(dirname "$0")/lib/arf_env.sh"
  ram_preflight 2048 "nightly real-path" || bad "preflight refused --num-blocks 2048 — not enough free RAM"
  "${ARF_ENV[@]}" ./target/release/arf-serve --model "$GGUF" --arch qwen3-coder --quant q4ks --no-tty     --port 18080 --max-batch-size 64 --num-blocks 2048 >/tmp/gate_real.log 2>&1 &
  GPID=$!
  for i in $(seq 1 600); do curl -fsS localhost:18080/healthz >/dev/null 2>&1 && break; sleep 1; done
  # L30 (2026-08-05): conc1 added to the REAL-path arm. Stage 2 above measures conc1 through
  # serve_loop_bench (in-process, synthetic) — which is precisely the class of metric that let
  # the daemon stall 40s/prompt while the gate stayed green. conc1 is now a defended number
  # (64.9 -> 82.3, 2026-08-05) and the thing to defend is what a USER gets over HTTP.
  # Floor 72 ≈ 12% under the measured 81.8-82.5, the same margin the 8/16 floors use.
  RB=$(GEN=64 CTX_WORDS=200 bash scripts/real_conc_baseline.sh 18080 1 8 16 2>/dev/null | grep -E "^     ")
  # DEPTH ARM (added 2026-08-04 after L12): the 200-word arm above CANNOT see ctx-slope wins —
  # L12's routing lever measured +5.8% at 200 words and +19% at ~1560 tokens, and a short-prompt
  # gate would have discarded it. IDE/agent workloads live at 1-4k ctx, so the gate must watch
  # a deep arm too. GEN=192 keeps the long prefill amortised (a short GEN here measures PREFILL,
  # not decode — two runs were voided learning that).
  # VERIFIED end-to-end 2026-08-04 — and the first version of this arm FAILED (read 9.8 vs a
  # floor of 40) because real_conc_baseline.sh warms only a SHORT prefix: at CTX_WORDS=1200 a
  # single request still pays the full ~4.6s prefill, so the harness measured PREFILL, not the
  # decode slope this arm exists to watch. Warm the EXACT deep prompt first, then time a long
  # generation directly — the same shape as the measurement that justified the L12 default flip.
  # OWN DAEMON for the depth arm (2026-08-04, second gate bug): bumping the SHARED daemon to
  # 4096 blocks to fit deep contexts gave 12.9GB of KV + 18.4GB weights on a 39GB box — the
  # 8/16-stream shallow arms then tipped into swap and read 2.8 tok/s / 11.7s TTFT. The two arms
  # want different pools, so they get different daemons. Shallow keeps 2048; depth gets 4096 on
  # its own port, started only after the shallow arms are done and their daemon is stopped.
  # --max-batch-size stays 64 here to match the production default. NOTE (L13, 2026-08-04): the
  # older claim in this comment — that mbs=8 reads 33.9 vs 41.8 at mbs=64, "a 23% swing on B=1
  # work from a flag that should not touch it" — was investigated and the PREMISE DID NOT SURVIVE.
  # `max_batch_size` is consumed at exactly ONE site in the engine (scheduler.rs:176, the
  # admit_waiting bound) and is NEVER passed to the GPU backend; every GPU scratch/arena is sized
  # from the ACTUAL row count. At B=1 the predicate is `1 < 8` vs `1 < 64` — same plan, same
  # kernels. The 33.9/41.8 pair was recorded ad hoc (not under the ABBA/fresh-daemon protocol) and
  # is most likely machine drift; L13 measured the SAME mbs=64 config at 14.8 and 22.1 on a box
  # that had silently gone to battery. Re-confirm on AC before treating it as real. (Was "See
  # ATTACK_PLAN L13" — that plan was removed in the doc cleanup, 2026-08-28.)
  # ALSO (L13): this arm's prompt is 1200 "context" words => the engine sees past_len=1200 EXACTLY
  # (probe-verified), not the "ctx~1560" quoted below and in the L12 section.
  kill $GPID 2>/dev/null; pkill -9 -f 'arf-serve' 2>/dev/null; sleep 4
  # L133: --num-blocks 4096 = 18.4 GB of KV pools. Unarmed and unattended this is a guaranteed
  # OOM on a 36 GB box. Preflight it and arm volatility.
  ram_preflight 4096 "nightly depth" || bad "preflight refused --num-blocks 4096 — not enough free RAM"
  "${ARF_ENV[@]}" ./target/release/arf-serve --model "$GGUF" --arch qwen3-coder --quant q4ks --no-tty \
    --port 18081 --max-batch-size 64 --num-blocks 4096 >/tmp/gate_depth.log 2>&1 &
  GPID2=$!
  for i in $(seq 1 600); do curl -fsS localhost:18081/healthz >/dev/null 2>&1 && break; sleep 1; done
  C1D=$(python3 - <<'PYGATE'
import json, time, urllib.request
url = "http://127.0.0.1:18081/v1/completions"
prompt = " ".join(["context"] * 1200)
def run(n):
    body = json.dumps({"prompt": prompt, "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    urllib.request.urlopen(req, timeout=600).read()
    return time.time() - t0
try:
    run(8)                      # warm THIS prompt (amortise the deep prefill)
    print(f"{192/run(192):.1f}")
except Exception:
    print("0")
PYGATE
)
  kill $GPID2 2>/dev/null; pkill -9 -f 'arf-serve' 2>/dev/null
  awk "BEGIN{exit !(${C1D:-0} >= 40)}" && ok "real conc1 @ctx~1560 $C1D >= 40 (V2-default measured ~53)" \
    || bad "real conc1 @ctx~1560 ${C1D:-none} BELOW 40 — the ctx-slope regression floor (L12)"
  kill $GPID 2>/dev/null; pkill -9 -f arf-serve 2>/dev/null
  C1A=$(echo "$RB" | awk '$1==1{print $2}')
  C8A=$(echo "$RB" | awk '$1==8{print $2}');  C8T=$(echo "$RB" | awk '$1==8{print $4}')
  C16A=$(echo "$RB" | awk '$1==16{print $2}')
  # grep -c EXITS 1 on zero matches, so `|| echo 0` appended a SECOND "0" => "0\n0", which
  # failed the string compare and reported a zero-panic run as a FAILURE (caught by the first
  # end-to-end gate run, 2026-08-04). tr -d strips any stray newline.
  PANICS=$(grep -c panicked /tmp/gate_real.log 2>/dev/null | head -1 | tr -d '[:space:]')
  PANICS=${PANICS:-0}
  [ "$PANICS" = "0" ] && ok "real-path: zero panics" || bad "real-path: $PANICS PANICS in daemon log"
  # L30: conc1 over the REAL HTTP path. Guards BOTH defaults the 2026-08-05 strike measured —
  # ARF_MEGA_KTOK=1 (+26.9%) and ARF_KV_F16 (+31.6%). Either one silently reverting drops
  # this well under 72: KTOK=8 reads ~64.9, NO_KV_F16 reads ~62.4. That is the point — the burst
  # rotted for weeks because nothing watched the shipped default on the shipped path.
  awk "BEGIN{exit !(${C1A:-0} >= 72)}"   && ok "real conc1 agg $C1A >= 72 (KTOK=1 + f16 KV; measured 81.8-82.5)" \
    || bad "real conc1 agg ${C1A:-none} BELOW 72 — check ARF_MEGA_KTOK (8 reads ~64.9) and ARF_KV_F16 (off reads ~62.4)"
  awk "BEGIN{exit !(${C8A:-0} >= 165)}"  && ok "real conc8 agg $C8A >= 165"  || bad "real conc8 agg ${C8A:-none} BELOW 165 (clean-race 183)"
  awk "BEGIN{exit !(${C8T:-9} <= 0.60)}" && ok "real conc8 TTFT ${C8T}s <= 0.60s" || bad "real conc8 TTFT ${C8T:-none}s ABOVE 0.60s (clean-race 0.35)"
  awk "BEGIN{exit !(${C16A:-0} >= 165)}" && ok "real conc16 agg $C16A >= 165" || bad "real conc16 agg ${C16A:-none} BELOW 165 (clean-race 184)"
fi

# ── 4. SUSTAINED conc64 (the number we quote publicly; peak != sustained) ──────────────────────
if [ "$QUICK" = "1" ] || [ "$ON_AC" = "0" ]; then
  say "4. SUSTAINED SOAK — skipped (--quick or on battery)"
else
  say "4. SUSTAINED conc64 SOAK (2 slices, ~36 min)"
  S=$(STAMP=gate bash scripts/soak_conc64.sh 2 64 2>&1 | grep -oE 'SUSTAINED \(2nd-half med\): [0-9]+' | grep -oE '[0-9]+$')
  if [ -n "$S" ]; then
    awk "BEGIN{exit !($S >= $SOAK_FLOOR)}" \
      && ok "sustained conc64 $S tok/s >= floor $SOAK_FLOOR" \
      || bad "sustained conc64 $S tok/s BELOW floor $SOAK_FLOOR"
  else bad "soak produced no sustained number"; fi
fi

# ── 5. VERDICT + the trend record (rot is only visible over time) ──────────────────────────────
say "VERDICT"
STAMP_DATE=$(date '+%Y-%m-%d %H:%M')
mkdir -p target/nightly   # the trend record: local and gitignored
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '  🟢 GATE PASS — %d checks clear (%s)\n' "${#PASSED[@]}" "$STAMP_DATE"
  echo "| $STAMP_DATE | PASS | conc1 ${RATE:-?} | sustained ${S:-skipped} | $(git rev-parse --short HEAD) |" >> target/nightly/nightly-gate-log.md
  exit 0
else
  printf '  🔴 GATE FAIL — %d check(s) failed:\n' "${#FAILED[@]}"
  for f in "${FAILED[@]}"; do printf '     - %s\n' "$f"; done
  echo "| $STAMP_DATE | FAIL | ${FAILED[*]} | $(git rev-parse --short HEAD) |" >> target/nightly/nightly-gate-log.md
  exit 1
fi
