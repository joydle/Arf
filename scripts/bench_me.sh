#!/usr/bin/env bash
# bench_me.sh — "how fast is MY box?"
#
# Reports decode-only tok/s against YOUR machine's memory roofline, because the absolute
# number is meaningless without it: this engine is bandwidth-bound, so the honest question
# is not "how many tokens" but "what fraction of what your memory bus can deliver".
#
# Decode-only means first-token→last-token, excluding prefill. That is the number the
# roofline actually bounds; a wall-clock figure that includes prefill is not comparable
# to it (a mistake this project made and corrected).
#
# WHY IT REFUSES rather than printing a bad number: this engine's own measurement rules
# (the project's rules) say a figure taken while the box is paging or loaded is noise. So
# is one taken here.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:-llama3.2:1b}"
PORT="${PORT:-8127}"
TOKENS="${TOKENS:-128}"
SAMPLES="${SAMPLES:-5}"
BIN=./target/release/arf

command -v curl >/dev/null || { echo "need curl"; exit 1; }
[ -x "$BIN" ] || { echo "→ building the CLI first"; cargo build --release -p arf-cli -q || exit 1; }

# --- box check: refuse rather than report noise -------------------------------------
if [ "$(uname)" = "Darwin" ]; then
  LOAD=$(uptime | sed 's/.*load averages*: //' | awk '{print $1}')
  SWAP=$(sysctl -n vm.swapusage 2>/dev/null | sed 's/.*used = \([0-9.]*\)M.*/\1/')
  NCPU=$(sysctl -n hw.ncpu)
  echo "box: load ${LOAD}, swap ${SWAP}M, ${NCPU} cores"
  if awk -v l="$LOAD" -v n="$NCPU" 'BEGIN{exit !(l+0 > n/2)}'; then
    echo "REFUSING: load ${LOAD} is over half your core count — the number would be noise."
    echo "          Close what is busy and re-run. (This engine's own rule: the project's rules)"
    exit 2
  fi
fi

echo "→ starting $MODEL on :$PORT"
"$BIN" serve "$MODEL" --port "$PORT" >/tmp/bench_me.log 2>&1 &
DPID=$!
trap 'kill $DPID 2>/dev/null' EXIT INT TERM
for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
  echo "daemon never came up; see /tmp/bench_me.log"; tail -20 /tmp/bench_me.log; exit 3; }

python3 - "$PORT" "$TOKENS" "$SAMPLES" <<'PYEOF'
import json, sys, time, urllib.request, statistics as st
port, ntok, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def one():
    body = json.dumps({"model":"local","stream":True,"temperature":0,"max_tokens":ntok,
        "messages":[{"role":"user","content":"Write a haiku about the sea, then explain it in two sentences."}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type":"application/json"})
    tf=tl=None; n=0
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode("utf-8","replace").strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            try: d = json.loads(p)
            except json.JSONDecodeError: continue
            if d.get("choices",[{}])[0].get("delta",{}).get("content"):
                now = time.perf_counter()
                if tf is None: tf = now
                tl = now; n += 1
    # n-1 intervals span first->last: that is the honest decode rate.
    return (n-1)/(tl-tf) if tf and n > 1 else None

one()  # discard: the first request pays JIT and cache warm
tps = [t for t in (one() for _ in range(reps)) if t]
if not tps:
    print("no samples"); sys.exit(1)
med = st.median(tps)
spread = 100*(max(tps)-min(tps))/med
print(f"\n  decode-only: {med:.2f} tok/s   (median of {len(tps)}, spread {spread:.1f}%)")
if spread > 10:
    print("  ⚠ spread over 10% — something else is using the machine; treat this as indicative.")
print(f"\n  To turn this into a fraction of YOUR roofline:")
print(f"    bytes_per_token = the model's on-disk size (weights are streamed once per token)")
print(f"    ceiling         = your memory bandwidth / bytes_per_token")
print(f"  For reference, this project measures Qwen3.8-27B Q4_K_M at 18.6 GB/token, and on an")
print(f"  M4 Max (410 GB/s) that is a 22.0 tok/s ceiling — reached at 21.1 (96%).")
print(f"  Compare with docs/PERFORMANCE.md; every number there carries its date and method.")
PYEOF
