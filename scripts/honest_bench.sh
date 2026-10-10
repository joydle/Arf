#!/usr/bin/env bash
# Honest decode-throughput bench: warm-up (discarded) + median-of-3, reported per
# config. Run ONLY on a quiet machine (load < ~3, no background compiles) — absolute
# tok/s is CPU-contention-sensitive. Ratios (batch-N vs batch-1) are robust either way.
set -euo pipefail
cd "$(dirname "$0")/.."

GG=models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
GTOK=models/gemma-4-31b-it/tokenizer.json
LTOK=models/llama-3.2-1b/tokenizer.json
GP="2,3689,563,506,5279,529,7001"   # gemma chat-ish tokens
LP="128000,791,6864,315,9822,374"   # llama tokens

# median of 3 timed tok/s for a given arg set; $1 = label, rest = arf args
med() {
  local label="$1"; shift
  local runs=()
  # warm-up (discard)
  cargo run -q --release -p arf-cli -- generate "$@" >/dev/null 2>&1 || true
  for i in 1 2 3; do
    local v
    v=$(cargo run -q --release -p arf-cli -- generate "$@" 2>&1 | grep -oE '[0-9]+\.[0-9]+ tok/s( aggregate)?' | grep -oE '^[0-9.]+' | head -1)
    runs+=("${v:-0}")
  done
  local sorted; sorted=$(printf '%s\n' "${runs[@]}" | sort -n)
  local m; m=$(echo "$sorted" | sed -n '2p')
  printf '%-34s median=%-8s runs=[%s]\n' "$label" "$m" "$(echo "${runs[*]}" | tr ' ' ',')"
}

echo "### machine: load=$(uptime | grep -oE 'load averages?: [0-9.]+' | grep -oE '[0-9.]+$')  $(date)"
echo "### llama-3.2-1B Q4"
med "llama-1B single"  --device gpu --quant q4k --model models/llama-3.2-1b --tokenizer "$LTOK" --tokens "$LP" --max-tokens 64
med "llama-1B batch=8" --device gpu --quant q4k --model models/llama-3.2-1b --tokenizer "$LTOK" --tokens "$LP" --batch 8 --max-tokens 32
echo "### gemma-4-12B Q4 (the centerpiece)"
med "gemma-12B single"  --arch gemma-4-12b --device gpu --quant q4k --model "$GG" --tokenizer "$GTOK" --tokens "$GP" --max-tokens 48
med "gemma-12B batch=4" --arch gemma-4-12b --device gpu --quant q4k --model "$GG" --tokenizer "$GTOK" --tokens "$GP" --batch 4 --max-tokens 16
med "gemma-12B batch=8" --arch gemma-4-12b --device gpu --quant q4k --model "$GG" --tokenizer "$GTOK" --tokens "$GP" --batch 8 --max-tokens 16
echo "### done — ratios (batch-N agg / single) are the lead; absolutes valid only if load was low."
