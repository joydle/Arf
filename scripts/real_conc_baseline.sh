#!/usr/bin/env bash
# real_conc_baseline.sh — THE CONCURRENCY BASELINE, with REAL prompts and REAL text.
#
# WHY: every conc number this project has ever quoted (441 tok/s @conc64, 1.10x vs llama, etc.)
# came from `serve_loop_bench` — synthetic token IDs, no text generated, prefill amortised or
# skipped, HTTP path bypassed entirely. This drives the SAME OpenAI API a real client uses, with
# realistic prompts, and reports what a user/operator actually gets.
#
# ⚠️ ONE ENGINE AT A TIME on a 39 GB box (a 30B q4ks model is ~18 GB resident; two = swap death,
# measured 2026-08-03). Start the engine you want, run this, stop it, start the other.
#
# USAGE:  scripts/real_conc_baseline.sh [PORT] [CONCS...]
#   e.g.  scripts/real_conc_baseline.sh 8080 1 8 16 32 64
#   env:  GEN=64  CTX_WORDS=200   (200 words ~ 260 tokens = a realistic chat/IDE context)
#
# Reports per tier: aggregate tok/s (all streams), per-stream tok/s, mean TTFT, and #ok.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PORT="${1:-8080}"; shift || true
CONCS=("${@:-1 8 16 32 64}")
[ $# -eq 0 ] && CONCS=(1 8 16 32 64)
GEN="${GEN:-64}"
# REPS (2026-08-04, after L22/L23): ONE rep per tier leaves a -5.7%..-7.0% conc32 same-config
# drift floor, and TWO consecutive strikes burned their whole ABBA budget to learn "inside the
# noise". REPS>1 measures each tier n times on the SAME warm daemon so a 3-5% lever becomes
# decidable. Read the SPREAD across reps as the floor, and judge a delta only if it exceeds it.
REPS="${REPS:-1}"
CTX_WORDS="${CTX_WORDS:-200}"
URL="http://127.0.0.1:$PORT"

curl -fsS --max-time 5 "$URL/v1/models" >/dev/null 2>&1 || curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1 \
  || { echo "🔴 nothing serving on $URL — start ONE engine first"; exit 2; }

# 🔴 L148 — REPORT THE DAEMON'S CONFIG. A ladder run against ONE daemon at --max-batch-size 64
# silently measures the low tiers with concurrency-conditional defaults resolved for batch 64. The
# real board did exactly that and understated conc1 by 35% (57.5 vs the true 77.8) because L135
# turns the f16 KV pool OFF above batch 8. Print what the daemon actually resolved so a row can
# never be read as "conc1 performance" when it was measured in a conc64 configuration.
CFG=$(curl -fsS --max-time 5 "$URL/hud" 2>/dev/null \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('max_batch=%s' % d.get('max_batch_size','?'))" 2>/dev/null || echo "config unknown")
echo "════════ REAL CONCURRENCY BASELINE · $URL · GEN=$GEN · ctx≈${CTX_WORDS} words ════════"
echo "  ⚠️  daemon config: $CFG — low tiers measured here inherit THIS daemon's conditional"
echo "      defaults (L148). For a true per-tier number, launch each tier at its own batch size."
printf '\n  %5s %12s %12s %10s %8s\n' conc agg_tok/s per_strm mean_TTFT ok

for C in "${CONCS[@]}"; do
  for _rep in $(seq 1 "$REPS"); do
  python3 - "$URL" "$C" "$GEN" "$CTX_WORDS" <<'PY'
import json, sys, time, urllib.request, threading, random
url, conc, gen, ctxw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

# Realistic context: a chunk of prose + a unique task tail per stream (shared-prefix shaped,
# which is exactly what an agent swarm / IDE fleet sends).
#
# 🔴 BUG FIXED 2026-08-12 (L138): the source string was a FIXED `* 8` = 176 words, then sliced to
# ctxw. Any CTX_WORDS > 176 therefore produced THE SAME 176-WORD PROMPT — the slice cannot exceed
# what is there. A crossover sweep at 200/400/800/1600 words returned four identical rows
# (throughput 218.1/218.2/218.1, reuse 980 at every point) because it measured ONE prefix length
# four times. Repeat enough times to COVER ctxw, then slice.
_UNIT = ("You are a precise coding assistant working in a large Rust repository. "
         "Follow the instructions exactly, cite line numbers, and prefer small diffs. ")
_reps = max(1, -(-ctxw // len(_UNIT.split())))          # ceil(ctxw / words_per_unit)
BASE = " ".join((_UNIT * _reps).split()[:ctxw])
assert len(BASE.split()) == ctxw, f"prefix is {len(BASE.split())} words, asked for {ctxw}"

res = [None] * conc
def worker(i):
    prompt = f"{BASE} Task {i}: write a function that reverses a string."
    body = json.dumps({"prompt": prompt, "max_tokens": gen, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    # 🔴 L141 — COUNT TOKENS, NOT chars/4. Measured 2026-08-12 on the SAME prompt with BOTH
    # engines generating EXACTLY 128 tokens (llama's usage block proves it):
    #     arf 704 chars -> chars/4 says 176 tokens   (37% over)
    #     llama  548 chars -> chars/4 says 137 tokens   (7%  over)
    # chars/4 therefore credited us 176 vs llama 137 for IDENTICAL work — a 28% bias, and it
    # rewards verbose text rather than speed. Every tok/s this harness ever printed carried it.
    # Each streamed SSE chunk is one token, so counting chunks is exact and engine-neutral.
    t0 = time.time(); ttft = None; chars = 0; ntok = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                p = line[5:].strip()
                if p == "[DONE]": break
                try: piece = json.loads(p)["choices"][0].get("text", "")
                except Exception: continue
                if piece:
                    if ttft is None: ttft = time.time() - t0
                    chars += len(piece); ntok += 1
    except Exception:
        res[i] = None; return
    res[i] = (ttft if ttft is not None else time.time() - t0, time.time() - t0, chars, ntok)

# WARM the shared prefix ONCE before timing. Without this we measure conc x COLD PREFILL, not
# decode — the exact artifact that made this script report 2.1 tok/s @conc1 when the same server
# does 57-65 (measured 2026-08-03).
try:
    wb = json.dumps({"prompt": BASE + " warm", "max_tokens": 4, "temperature": 0}).encode()
    urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data=wb,
        headers={"Content-Type": "application/json"}), timeout=600).read()
except Exception:
    pass

t_all = time.time()
ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
[t.start() for t in ths]; [t.join() for t in ths]
wall = time.time() - t_all

ok = [r for r in res if r]
if not ok:
    print(f"  {conc:5d} {'ERR':>12} {'ERR':>12} {'ERR':>10} {0:>8}"); raise SystemExit
toks = sum(n for _, _, _, n in ok)              # REAL streamed tokens (one SSE chunk = one token)
agg = toks / wall                               # what the OPERATOR sees
per = agg / len(ok)                             # what EACH USER feels
ttft = sum(t for t, _, _, _ in ok) / len(ok)
# WORK-PARITY GATE (L141): tok/s is only comparable if both engines did the SAME work. An engine
# that stops early looks fast. Require every stream to have produced `gen` tokens; flag if not.
short = [n for _, _, _, n in ok if n < gen]
flag = "" if not short else f"  ⚠️ {len(short)}/{len(ok)} streams SHORT (min {min(short)}/{gen}) — NOT comparable"
print(f"  {conc:5d} {agg:12.1f} {per:12.1f} {ttft:10.3f} {len(ok):>8}{flag}")
PY
  done
done
echo ""
echo "  agg_tok/s = total throughput (operator view) · per_strm = what ONE user feels"
echo "  TTFT is the felt latency. Tokens are COUNTED from streamed SSE chunks (L141), not chars/4."
