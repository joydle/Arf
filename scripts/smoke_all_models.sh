#!/usr/bin/env bash
# smoke_all_models.sh — does every supported model actually RUN?
#
# WHY THIS EXISTS. The docs claim eight architectures. "Loads and runs" was an assertion, not a
# test: nothing in CI opens a 37 GB GGUF, and the per-model correctness gates cover two models.
# The failure that matters most is not a slow model, it is a model that does not start.
#
# So: every model in the supported set, default config, one short greedy generation each, and a
# PASS/FAIL per row. Text out, not tok/s — speed is the benchmarks' job, correctness is
# two_model_gate.sh's. This answers only "can a person run this at all".
#
# Usage:  scripts/smoke_all_models.sh [--quick]     (--quick = 8 tokens instead of 24)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TOK=24; [ "${1:-}" = "--quick" ] && TOK=8
CLI=./target/release/arf
[ -x "$CLI" ] || { echo "build first: cargo build --release -p arf-cli"; exit 1; }

PROMPT="Say hello in exactly five words."
# L363k — a prompt LONGER than 64 tokens with a checkable answer. The short prompt above cannot see
# a prompt-row fault: L363j dropped every prompt row past --max-batch-size in the batched GEMV and
# the sweep stayed green because "Say hello…" is 8 tokens. This one is ~75 tokens (two 64-row GEMV
# tiles at --max-batch-size 4) and the answer must name the birds that were IN the prompt.
LONG_PROMPT="Here is a list of animals: cat, dog, horse, sheep, cow, goat, duck, goose, hen, pig, fox, wolf, bear, deer, elk, moose, otter, seal, whale, shark, crab, squid, frog, toad, newt, owl, hawk, crow, robin, wren, finch, lark, swan, stork, heron, eagle, gull, tern, dove, quail. Which of them are birds? Answer briefly."
pass=0; fail=0; rows=()

try() { # name  model-path  arch  extra-args...
  local name="$1" path="$2" arch="$3"; shift 3
  [ -e "$path" ] || { rows+=("SKIP  $name  (weights not present)"); return; }
  local out rc
  out=$(timeout 900 "$CLI" generate --model "$path" --arch "$arch" --quant q4ks --device gpu \
        --max-tokens "$TOK" --prompt "$PROMPT" "$@" 2>&1); rc=$?
  # a pass is: exit 0, and at least a few non-control characters of output
  local text; text=$(printf '%s' "$out" | tr -d '[:space:]' | wc -c | tr -d ' ')
  # L350 — DEGENERATE-OUTPUT CHECK. "exit 0 and >8 characters" is not a correctness gate:
  # `!!!!!!!!!!!!` satisfies it. A collapsed decode emits ONE token forever, so the tell is that
  # the whole reply is a single repeated character — which is how Llama-3.2-1B passed this sweep
  # while emitting nothing but `!` (L349).
  #
  # `uniq_chars` counts DISTINCT non-space characters in the generated text. Verified against
  # real samples: `!!!!!!!!!!!!` -> FAIL (1 distinct), a real sentence -> PASS, and short output
  # still fails on the length check as before. The threshold is <=2 rather than <=1 so a
  # two-character loop (`ababab`) is caught too.
  local uniq_chars
  uniq_chars=$(printf '%s' "$out" | tail -1 | tr -d '[:space:]' | fold -w1 | sort -u | wc -l | tr -d ' ')
  if [ $rc -eq 0 ] && [ "$text" -gt 8 ] && [ "${uniq_chars:-0}" -le 2 ]; then
    rows+=("FAIL  $name  (DEGENERATE: ${uniq_chars} distinct chars — collapsed decode) :: $(printf '%s' "$out" | tail -1 | cut -c1-60)")
    fail=$((fail+1))
  elif [ $rc -eq 0 ] && [ "$text" -gt 8 ]; then
    rows+=("PASS  $name"); pass=$((pass+1))
  # The memory guard refusing a model too big for this box is CORRECT behaviour, not a
  # failure of the model: it prints the largest --num-blocks that fits and refuses rather
  # than letting the KV pool evict the weights and emit '!!!!' (L146/L147).
  elif printf '%s' "$out" | grep -q "refusing to load"; then
    rows+=("GUARD $name  (too large for this box — the memory guard refused it, correctly)")
  else
    rows+=("FAIL  $name  (rc=$rc, ${text} chars) :: $(printf '%s' "$out" | tail -1 | cut -c1-90)")
    fail=$((fail+1))
  fi
}

# ── L360 — THE SERVER ARM ───────────────────────────────────────────────────────────────────
# `try` above exercises `arf generate` ONLY. That is not the path anyone serves from, and the
# gap was not theoretical: gemma-4-12b passed this sweep green on 2026-09-03 while
# `arf-serve` emitted `'Count'` and then empty deltas forever on the same weights. The CLI's
# short single-shot run and the server's batched decode take different branches, and the
# batched megakernel bound ONE attention geometry for a model that has two (L360).
#
# The comment above says generate drives "the same one-token megakernel step the server uses".
# That was the assumption that let this ship. It is close enough to be persuasive and wrong
# where it matters, so the sweep now checks the server itself rather than a proxy for it.
try_serve() { # name  model-path  arch
  local name="$1" path="$2" arch="$3" port=8321 log=/tmp/smoke_serve_$$.log
  [ -e "$path" ] || { rows+=("SKIP  $name (serve)  (weights not present)"); return; }
  # --max-batch-size 4: the STRICT shape (ACC_ROWS=4), so any prompt over 4 tokens exercises the
  # wide GEMV twin (L363j). The default 32 would hide a regression on every prompt under 32 tokens.
  ./target/release/arf-serve --model "$path" --arch "$arch" --quant q4ks \
    --port $port --max-batch-size 4 "${@:4}" >"$log" 2>&1 </dev/null &
  local pid=$! w=0 up=0
  while [ "$w" -lt 900 ]; do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1; w=$((w+1))
  done
  if [ "$up" != 1 ]; then
    kill "$pid" 2>/dev/null
    rows+=("FAIL  $name (serve)  (daemon never answered in ${w}s) :: $(tail -1 "$log" | cut -c1-70)")
    fail=$((fail+1)); return
  fi
  local out
  out=$(curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"local\",\"temperature\":0,\"max_tokens\":$TOK,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
        | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception as e: print("PARSE_ERROR", e)' 2>&1)
  local long_out birds
  long_out=$(curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"local\",\"temperature\":0,\"max_tokens\":160,\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_PROMPT\"}]}" \
        | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception as e: print("PARSE_ERROR", e)' 2>&1)
  # 2026-09-19 — THE SAME REQUEST AGAIN MUST RETURN THE SAME TEXT. On Qwen3.8-27B a prefix-cache
  # hit restored KV for the 16 attention layers and nothing for the 48 recurrent ones: request 0
  # was correct, requests 1-4 returned 14 tokens of unrelated Chinese text, 4 of 4, and tok/s got
  # BETTER (TTFT 0.85 s -> 0.1 s), so no speed gate could see it. Every check above sends each
  # prompt once, which is exactly why this shipped. Same daemon, same prompt, temperature 0.
  local out_again
  out_again=$(curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"local\",\"temperature\":0,\"max_tokens\":$TOK,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
        | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception as e: print("PARSE_ERROR", e)' 2>&1)
  if [ "$out_again" != "$out" ]; then
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    rows+=("FAIL  $name (serve)  (REPEATED PROMPT ANSWERED DIFFERENTLY — prefix-cache hit on a model whose state it cannot restore?) :: 1st=$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-40) | 2nd=$(printf '%s' "$out_again" | tr '\n' ' ' | cut -c1-40)")
    fail=$((fail+1)); return
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # L363m — the long prompt is judged against the REFERENCE PATH, not on content: a second daemon
  # with ARF_BATCH_MEGA_NO_PREFILL=1 runs the same prompt with prefill on the wgpu path (the
  # path the md5 gate certifies) and the texts must be identical. Content judging failed on the
  # thinking models (Qwen3.8 named 3 birds in 160 tokens; muse spent them all on "to=self…" while
  # its text was byte-identical to the reference). A near-tie can legitimately flip a token late in
  # a long answer, so a mismatch falls back to the content check (>= 4 birds named) before failing.
  local ref_out
  ARF_BATCH_MEGA_NO_PREFILL=1 ./target/release/arf-serve --model "$path" --arch "$arch" --quant q4ks \
    --port $port --max-batch-size 4 "${@:4}" >"$log.ref" 2>&1 </dev/null &
  local rpid=$! rw=0 rup=0
  while [ "$rw" -lt 900 ]; do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { rup=1; break; }
    kill -0 "$rpid" 2>/dev/null || break
    sleep 1; rw=$((rw+1))
  done
  if [ "$rup" = 1 ]; then
    ref_out=$(curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"local\",\"temperature\":0,\"max_tokens\":160,\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_PROMPT\"}]}" \
        | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception as e: print("PARSE_ERROR", e)' 2>&1)
  else
    ref_out="(reference daemon never answered)"
  fi
  kill "$rpid" 2>/dev/null; wait "$rpid" 2>/dev/null
  local birds
  birds=$(printf '%s' "$long_out" | tr 'A-Z' 'a-z' | grep -oE 'duck|goose|hen|owl|hawk|crow|robin|wren|finch|lark|swan|stork|heron|eagle|gull|tern|dove|quail' | sort -u | wc -l | tr -d ' ')
  # L363m — judge on a LONG IDENTICAL PREFIX (160 chars, ~40 tokens), not full identity: a 171-row
  # muse answer matched its reference for 365 chars and then split on a near-tie inside a thinking
  # ramble. Every real prompt-row fault seen today ('1.1.1', 'is is is', "you haven't provided…")
  # diverged at the FIRST token, so 160 identical characters after a >64-row prefill is the signal.
  if [ "$long_out" = "$ref_out" ] || { [ "${#long_out}" -ge 160 ] && [ "${long_out:0:160}" = "${ref_out:0:160}" ]; }; then
    :  # record prefill == wgpu prefill (fully, or for its first 160 characters)
  elif [ "${birds:-0}" -ge 4 ]; then
    rows+=("NOTE  $name (serve)  long prompt differs from the wgpu-prefill reference but names ${birds} birds (near-tie?) :: $(printf '%s' "$long_out" | tr '\n' ' ' | cut -c1-60)")
  else
    rows+=("FAIL  $name (serve)  (LONG PROMPT != wgpu-prefill reference, ${birds} birds named — prompt rows wrong? L363j/l) :: rec=$(printf '%s' "$long_out" | tr '\n' ' ' | cut -c1-50) | ref=$(printf '%s' "$ref_out" | tr '\n' ' ' | cut -c1-50)")
    fail=$((fail+1)); return
  fi
  local n uniq
  n=$(printf '%s' "$out" | tr -d '[:space:]' | wc -c | tr -d ' ')
  uniq=$(printf '%s' "$out" | tr -d '[:space:]' | fold -w1 | sort -u | wc -l | tr -d ' ')
  # The gemma-4 failure was ONE real token then empty deltas — 5 chars, 5 distinct. Neither the
  # length check nor the degenerate check alone would catch it, so require BOTH a real length
  # and real variety. `$TOK` tokens of correct text is always far more than 12 characters.
  if [ "$n" -le 12 ]; then
    rows+=("FAIL  $name (serve)  (COLLAPSED: only ${n} chars from $TOK tokens) :: $(printf '%s' "$out" | cut -c1-60)")
    fail=$((fail+1))
  elif [ "${uniq:-0}" -le 2 ]; then
    rows+=("FAIL  $name (serve)  (DEGENERATE: ${uniq} distinct chars) :: $(printf '%s' "$out" | cut -c1-60)")
    fail=$((fail+1))
  else
    rows+=("PASS  $name (serve)"); pass=$((pass+1))
  fi
}

echo "smoke: $TOK tokens per model, default config"
echo

# L322 — hybrids run here too now: generate_hybrid drives the same one-token megakernel step
# the server uses, which is the only shape a recurrent state admits.
#
# L360 — that last clause ("the same ... step the server uses") is NOT true in general, and
# believing it is what let a broken model pass this sweep green. The CLI's single-shot run and
# the server's batched decode take DIFFERENT branches; gemma-4-12b was correct here and emitted
# 'Count' then empty deltas over HTTP. The claim is left standing above because it is true of the
# recurrent STATE shape, which is what L322 was about — but it does not extend to the whole
# decode path, and the try_serve arm below now checks the server instead of trusting the proxy.
try "Qwen3.8-27B"      models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf                 qwen3.8-27b
try "Qwen3-Coder-30B"  models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf     qwen3-coder-30b
try "muse-glimmer-30B" models/muse-glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf muse-glimmer-30b
try "Gemma-4-12B"      models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf   gemma-4-12b
try "Gemma-4-31B"      models/gemma-4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf       gemma-4-31b
# L363n — Gemma-3-4B text: the GGUF embeds a unigram tokenizer (needs --tokenizer) and has a
# 262208-row vocab (config fixed the same day). It raced at 0.81x; it must stay loadable.
try "Gemma-3-4B"       models/gemma-3-4b-vision/gemma-3-4b-it-Q4_0.gguf           gemma-3-4b --tokenizer models/gemma-3-4b-vision/tokenizer.json

# L360 — the SERVER arm, same models. Slower (a daemon start each), and worth it: this is the
# path that ships. gemma-4-12b passed the CLI arm and failed here.
echo
echo "smoke: server path (arf-serve over HTTP)"
try_serve "Qwen3.8-27B"      models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf                 qwen3.8-27b
try_serve "muse-glimmer-30B" models/muse-glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf muse-glimmer-30b
try_serve "Gemma-4-12B"      models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf   gemma-4-12b
try_serve "Gemma-3-4B"       models/gemma-3-4b-vision/gemma-3-4b-it-Q4_0.gguf           gemma-3-4b --tokenizer models/gemma-3-4b-vision/tokenizer.json
# L350 — Llama-3.2-1B REMOVED from the sweep. It is the only SAFETENSORS model here (every
# other is a GGUF) and the only one with head_dim=64 (the rest are 128 or 256), and on the
# Metal fast path it emits `!!!!!!!!` — bisected across seven levers in L349/L349b without
# finding the mechanism. It was "passing" this sweep only because the old check accepted any
# 8+ characters; the degenerate-output check added above now catches exactly that.
#
# It is not a perf target (it was the CPU correctness model), so the honest move is to stop
# advertising a broken path rather than ship a workaround. If someone roots out the bf16 /
# head_dim=64 fast-path bug, put this line back — it is the regression test for that fix.

printf '%s\n' "${rows[@]}"
echo
echo "text models: $pass passed, $fail failed"
echo "(FLUX and Gemma-3 vision are separate entry points)"
[ "$fail" -eq 0 ]
