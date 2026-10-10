#!/usr/bin/env bash
# refactor_gate.sh — THE SEAM REFACTOR GATE.
#
# Records, then re-checks, the exact bytes this engine produces. Run it ONCE on a known-good
# tree to capture a baseline; run it after every refactor step to prove nothing moved.
#
# WHY THIS EXISTS. The backend-seam work  restructures the code every token
# passes through. The whole plan rests on one claim — "byte-identical output" — and a claim
# nobody can check is a wish. This makes it checkable in ~2 minutes.
#
# WHAT IT PINS, and why each:
#   * md5 of generated text, per model      — the refactor must not change a single token
#   * decode-only tok/s, per model          — a structural change must not cost speed
#   * the dispatch-lever line               — proves the FAST PATH still engaged (rule 7:
#                                             a flag that changes nothing looks like one that works)
#
# USAGE
#   scripts/refactor_gate.sh --record     # capture baseline into .refactor-baseline.json
#   scripts/refactor_gate.sh              # compare against it; non-zero exit on any drift
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BASELINE=".refactor-baseline.json"
MODE="${1:-check}"
PORT="${PORT:-8131}"
TOK="${TOK:-64}"
PROMPT="Write a haiku about the sea, then explain it in two sentences."
# tok/s may drift with box conditions; text may NOT drift at all. Separate tolerances.
TPS_TOL="${TPS_TOL:-12}"   # percent

# model  path  arch  quant
# Each model needs its OWN warm transcode cache under THIS flag set, or its run is a ~20 min
# rebuild that the wait loop reports as DAEMON_DOWN. ONLY="name" records a subset — use it to
# capture a real baseline for whatever is warm rather than nothing at all.
MODELS=(
  "qwen3.8-27b|models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf|qwen3.8-27b|q4ks"
  "qwen3-coder-30b|models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf|qwen3-coder|q4ks"
  "gemma-4-12b|models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf|gemma-4-12b|q4ks"
)
ONLY="${ONLY:-}"

load=$(uptime | sed 's/.*load averages*: //' | awk '{print $1}')
ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
# L363c — text-only mode (TPS_TOL=999) skips the LOAD refusal: the md5 does not drift with load,
# and a busy box (a headless browser on 12 cores, twice in one week) must not block a
# correctness check. The RAM refusal below still applies — eviction can corrupt text.
if [ "${TPS_TOL:-}" != "999" ] && awk -v l="$load" -v n="$ncpu" 'BEGIN{exit !(l+0 > n/2)}'; then
  echo "REFUSING: load $load is over half of $ncpu cores. A tok/s number here is noise."
  echo "          (Text md5 would still be valid — re-run with TPS_TOL=999 to check text only.)"
  exit 2
fi

# L358c — THE SECOND HALF OF THE INTERLEAVING RULE, on the third attempt. The rule names BOTH
# `vm.swapusage` and load average; this gate read only load, so a box with a quiet CPU and no
# memory headroom sailed through. That is not hypothetical: on 2026-09-03 two orphaned 4.2 GB
# `arf_gpu` test binaries outlived their suite and pinned swap at 6.1 GB of 7.2 GB while load
# sat at 5.1 — under the threshold. A 23 GB model measured there pages, and paging looks exactly
# like a regression.
#
# 🔴 THE FIRST TWO VERSIONS OF THIS CHECK WERE WRONG, both in the same direction — measuring the
# swap FILE instead of the memory. Recorded because the second one looked obviously right:
#
#   v1: refuse when `used` swap is high.   Wrong: a 16 GB swap file 60% consumed has more
#       headroom than a 2 GB file 60% free. Absolute `used` says nothing without the total.
#   v2: refuse when `free` swap < 2 GB.    Wrong in the OPPOSITE direction, and it fired on a
#       healthy box: after macOS reclaimed and SHRANK the swap file 7168 -> 3072 MB, `free` fell
#       to 1672 MB *because the machine got better*. Free swap goes DOWN as swap health improves.
#
# What actually matters is whether the weights fit in RAM without evicting. So: compare free
# RAM (free + inactive — inactive is reclaimable) against the largest model this gate runs, and
# treat heavy swap USAGE AS A FRACTION OF ITS OWN TOTAL as the corroborating signal.
page=16384
# L359b — v4 of this figure, and the reason is measured rather than guessed. v3 read
# `free + inactive`, which OMITS `speculative` — read-ahead pages the kernel drops on the first
# hint of pressure. On a box that had just run the suite that was a 4.4 GB undercount:
#
#   physical    36.0     wired   4.3 (unreclaimable)    active     14.5
#   inactive    11.6     free    1.1                    speculative 3.1
#   free+inactive          = 12.7 GB   <- what the guard saw, and refused on
#   physical-wired-active  = 17.1 GB   <- what was actually available
#
# The gate then blocked indefinitely on a machine that was fine. Availability is
# `physical - wired - active`: everything except what the kernel cannot release and what a
# running process is actively touching. That is the same quantity Activity Monitor calls
# "available", and it is the one that decides whether the weights fit.
free_ram_gb=$(vm_stat | awk -v p=$page -v tot="$(sysctl -n hw.memsize)" '
  /Pages wired down/{gsub(/\./,"",$4); w=$4}
  /Pages active/{gsub(/\./,"",$3); a=$3}
  END{printf "%.1f", (tot - (w+a)*p)/1073741824}')
swap_used_mb=$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')
swap_tot_mb=$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*total = \([0-9.]*\)M.*/\1/p')
NEED_GB="${NEED_GB:-20}"   # largest model the gate runs; override for a smaller ONLY= subset

# Wired pages are the ones that matter most here and are invisible to `free + inactive`: Metal
# keeps model weights WIRED, so a 23 GB model that was loaded minutes ago is still held even
# after the process exits and even though nothing shows in `ps` RSS. That is a real refusal
# reason, and one worth NAMING — the first time this fired, `ps -Ao rss` showed no offender at
# all and it read like a false positive. It was not.
wired_gb=$(vm_stat | awk '/Pages wired down/{gsub(/\./,"",$4); printf "%.1f", $4*16384/1073741824}')
if awk -v r="$free_ram_gb" -v n="$NEED_GB" 'BEGIN{exit !(r+0 < n+0)}'; then
  echo "REFUSING: only ${free_ram_gb}GB RAM free, and the gate needs ~${NEED_GB}GB resident."
  echo "          ${wired_gb}GB is WIRED — Metal holds model weights wired, so a previous run's"
  echo "          model can still be resident with nothing visible in \`ps -Ao rss\`. Wait for it"
  echo "          to drain (a minute or two) rather than hunting for a process that isn't there."
  echo "          Measuring here pages, and paging looks exactly like a regression."
  echo "          (Text md5 would still be valid — re-run with TPS_TOL=999 to check text only.)"
  echo "          (Running a subset? NEED_GB=8 ONLY=<model> bash scripts/refactor_gate.sh)"
  exit 2
fi
# L362e — swap ratio is a WARNING, not a refusal. v3 (L358c) refused above 85% of the swap file's
# own total. That fired for 30+ minutes on a box with 8.2 GB of RAM FREE, wired 3.9 GB, no process
# over 0.8 GB RSS: the "used" swap was stale pages from an earlier four-daemon run, and macOS swaps
# pages back in only on access, never proactively — so the ratio can sit at 85% forever on a
# machine that is fine. Pressure is swap high AND free RAM low; the free-RAM check above already
# refuses the second half. The same L359b mistake in a new coat: a number reasoned about instead
# of the whole vm_stat breakdown looked at. Warn loudly, print the figures, do not refuse.
if [ -n "$swap_tot_mb" ] && awk -v u="$swap_used_mb" -v t="$swap_tot_mb" 'BEGIN{exit !(t+0 > 0 && u/t > 0.85)}'; then
  echo "  WARNING: swap ${swap_used_mb}/${swap_tot_mb}MB (>85%) — stale pages unless free RAM is also short;"
  echo "           free RAM is ${free_ram_gb}GB, which passed the refusal above. Proceeding."
fi
echo "  machine: load $load / $ncpu cores | ${free_ram_gb}GB RAM free | swap ${swap_used_mb}/${swap_tot_mb}MB"

run_one() { # $1=path $2=arch $3=quant -> "md5<TAB>tps<TAB>levers"
  local log=/tmp/rg_$$.log
  # 🔴 THE HANDSHAKE BUG (L351b, four failed baselines). This function is called inside
  # `$(...)`, so its stdout is a pipe the caller reads. A daemon backgrounded here INHERITS
  # that pipe and holds it open — but worse, `arf serve` EXECs arf-serve, which re-parses
  # and can exit before the loop's first curl. Redirecting the daemon's stdin/stdout/stderr
  # AWAY from the captured pipe (and off the controlling terminal with setsid where available)
  # makes the child fully independent of this function's output capture.
  #
  # Everything else here — the wait loop, the timing, the md5 — was always correct. The daemon
  # was simply never alive to answer.
  QUANT="$3" ARF_MSL_GEMV=1 nohup ./target/release/arf serve "$1" \
    --arch "$2" --quant "$3" --port "$PORT" >"$log" 2>&1 </dev/null &
  local pid=$!
  disown 2>/dev/null || true
  # THE HANDSHAKE (L351c). Four baselines failed here; `bash -x` finally showed why — the loop
  # ran ONE iteration then fell through to DAEMON_DOWN. The `for _ in $(seq 1 600)` form was
  # being cut short under this shell; a `while` on an explicit counter is not.
  #
  # Budget: 600s. A 17 GB model on a WARM transcode cache answers in ~32-40s (measured); a cold
  # cache rebuild takes ~20 min and will legitimately exhaust this — /tmp/refactor_gate_fail.log
  # distinguishes the two, since a rebuild logs `[weight-cache] MISS ... FLAGS differ`.
  local up=0 waited=0
  while [ "$waited" -lt 600 ]; do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
    sleep 1
    waited=$((waited + 1))
  done
  if [ "$up" != 1 ]; then
    pkill -f "arf-serve.*--port $PORT" 2>/dev/null; kill $pid 2>/dev/null
    # Preserve the log HERE too — this early return is the path that actually fires, and
    # returning without it is why DAEMON_DOWN stayed undiagnosed across three runs.
    cp "$log" /tmp/refactor_gate_fail.log 2>/dev/null
    echo -e "DAEMON_DOWN\t0\t"; return
  fi
  local out
  out=$(python3 - "$PORT" "$TOK" "$PROMPT" <<'PY'
import json,sys,time,urllib.request,hashlib
port,ntok,prompt=sys.argv[1],int(sys.argv[2]),sys.argv[3]
def go():
    b=json.dumps({"model":"local","stream":True,"temperature":0,"max_tokens":ntok,
                  "messages":[{"role":"user","content":prompt}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=b,
                             headers={"Content-Type":"application/json"})
    tf=tl=None;n=0;txt=[]
    with urllib.request.urlopen(r) as resp:
        for raw in resp:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data:"): continue
            p=l[5:].strip()
            if p=="[DONE]": break
            try: d=json.loads(p)
            except json.JSONDecodeError: continue
            c=d.get("choices",[{}])[0].get("delta",{}).get("content")
            if c:
                t=time.perf_counter()
                if tf is None: tf=t
                tl=t;n+=1;txt.append(c)
    return (hashlib.md5("".join(txt).encode()).hexdigest()[:12],
            (n-1)/(tl-tf) if tf and n>1 else 0.0)
go()  # discard the warm request
res=[go() for _ in range(3)]
md5s={m for m,_ in res}
tps=sorted(t for _,t in res)[1]
print(f"{(md5s.pop() if len(md5s)==1 else 'UNSTABLE')}\t{tps:.2f}")
PY
)
  local lev; lev=$(grep -m1 'fast-path levers' "$log" | sed 's/.*levers: //' || true)
  pkill -f "arf-serve.*--port $PORT" 2>/dev/null; kill $pid 2>/dev/null
  wait $pid 2>/dev/null; sleep 3
  # Keep the log when the daemon never answered: deleting it unconditionally hid a
  # `[weight-cache] MISS ... FLAGS differ` (a ~20 min rebuild, not a crash) behind a bare
  # DAEMON_DOWN for two full runs. On success it is noise; on failure it is the diagnosis.
  if [ "$up" = 1 ]; then rm -f "$log"; else mv "$log" "/tmp/refactor_gate_fail.log"; fi
  printf '%s\t%s\n' "$out" "$lev"
}

[ -x ./target/release/arf ] || { echo "build first: cargo build --release -p arf-cli"; exit 1; }

# WARM THE CACHE FIRST, or the gate times out on a rebuild instead of measuring anything.
# The transcode sidecar is keyed on the ENV vars QUANT / ARF_MSL_GEMV / ARF_MOE_MM_F32 /
# ARF_Q3K_FFN (weight_cache::flags_hash) — a --quant flag alone hashes differently and forces
# a ~23 GB rebuild that takes far longer than any sane wait loop. This gate exports the same
# values it launches with, so the hash matches whatever the last real run built. If you see
# `[weight-cache] MISS ... FLAGS differ` in /tmp/rg_*.log, that is this trap, not a gate bug.
echo "note: first run per model may rebuild the transcode cache (~20 min). Re-run after."

declare -a results
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name path arch quant <<<"$spec"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  [ -e "$path" ] || { echo "SKIP  $name (weights absent)"; continue; }
  IFS=$'\t' read -r md5 tps lev < <(run_one "$path" "$arch" "$quant")
  results+=("$name|$md5|$tps|$lev")
  printf "  %-18s md5=%s  %6s tok/s\n" "$name" "$md5" "$tps"
done

if [ "$MODE" = "--record" ]; then
  { echo "{"; echo "  \"recorded\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"commit\": \"$(git rev-parse --short HEAD)\","; echo "  \"models\": {"
    for i in "${!results[@]}"; do
      IFS='|' read -r n m t l <<<"${results[$i]}"
      sep=","; [ $i -eq $((${#results[@]}-1)) ] && sep=""
      echo "    \"$n\": {\"md5\": \"$m\", \"tps\": $t, \"levers\": \"$l\"}$sep"
    done
    echo "  }"; echo "}"; } > "$BASELINE"
  echo "✓ baseline recorded -> $BASELINE  (commit the file; it IS the contract)"
  exit 0
fi

[ -f "$BASELINE" ] || { echo "no baseline; run: scripts/refactor_gate.sh --record"; exit 1; }
fail=0
for r in "${results[@]}"; do
  IFS='|' read -r n m t l <<<"$r"
  bm=$(python3 -c "import json;print(json.load(open('$BASELINE'))['models'].get('$n',{}).get('md5',''))")
  bt=$(python3 -c "import json;print(json.load(open('$BASELINE'))['models'].get('$n',{}).get('tps',0))")
  bl=$(python3 -c "import json;print(json.load(open('$BASELINE'))['models'].get('$n',{}).get('levers',''))")
  [ -z "$bm" ] && { echo "  ?  $n: not in baseline"; continue; }
  if [ "$m" != "$bm" ]; then
    echo "  ✗  $n TEXT CHANGED: $bm -> $m   <<< the refactor is not byte-identical"; fail=1; continue
  fi
  if [ "$l" != "$bl" ]; then
    echo "  ✗  $n LEVERS CHANGED: '$bl' -> '$l'   <<< the fast path stopped engaging"; fail=1; continue
  fi
  d=$(python3 -c "print(f'{100*($t-$bt)/$bt:+.1f}')")
  slow=$(python3 -c "print(1 if 100*($bt-$t)/$bt > $TPS_TOL else 0)")
  if [ "$slow" = 1 ]; then echo "  ✗  $n SLOWER: $bt -> $t tok/s ($d%)"; fail=1
  else echo "  ✓  $n  md5 stable · levers stable · $d% tok/s"; fi
done
[ $fail -eq 0 ] && echo "GATE PASS — output byte-identical, fast path engaged, no regression."
exit $fail
