#!/usr/bin/env bash
# arf_env.sh — the ONE definition of how a benchmark launches arf-serve, plus the RAM
# preflight that stops a launch from taking the machine down.
#
# WHY THIS FILE EXISTS (2026-08-12): the Mac OOM-rebooted TWICE in one session. The durable
# value here is the RAM PREFLIGHT below — no launch should be able to take the box down.
#
# ⚠️ CORRECTED DIAGNOSIS (L133). The first version of this file claimed the benches had to set
# ARF_ATTN_COALESCED + ARF_KV_F16 to arm L8's f32-pool volatility. THAT WAS WRONG:
# arf-serve/main.rs:296-323 sets both ITSELF whenever ARF_MSL_GEMV is present, and every
# daemon printed "[serve] concurrent win config defaulted ON" + "[m1-kv-f16] engaged=true".
# Listing them here is redundant — kept only so the launch env is explicit at the call site.
# (Caught by a control arm that logged engaged=true with the flags supposedly unset.)
#
# THE REAL COST: f16 KV allocated a SECOND (half-size) pool ALONGSIDE the f32 staging pool
# (weights.rs:1700) — 6.0 GB f32 + 3.0 GB f16 = 9.0 GB at blocks 2048 on a 36 GB box. The f32
# pool cannot be dropped instead: prefill scatters into AND attends it (batch.rs:1485).
#
# ✅ RESOLVED IN THE ENGINE (L135): the f16 pool is now OFF BY DEFAULT — it measured a TIE at
# conc8 (-0.9%) and conc16 (-0.15%) against L5b's claimed +7.0%/+1.4%. No env var needed here,
# and the ARF_LEAN knob this file briefly carried is GONE. Opt back in with ARF_KV_F16=1.
# ⚠️ Polarity trap: downstream gates are is_some()-based, so ARF_KV_F16=0 still ENABLES it.
#
# Source this, then launch with "${ARF_ENV[@]}" "$OURS_BIN" ...

# Override with ARF_ENV_EXTRA for experiment-specific flags.
#
# NOTE: this is an ARRAY, not a string. A bare `$ARF_ENV="env A=1 B=1"` expands to a single
# command WORD under zsh ("command not found: env A=1 B=1"), which kills the daemon instantly.
# Launch with the array form: "${ARF_ENV[@]}" "$OURS_BIN" ...
# The daemon derives the rest of the win config from ARF_MSL_GEMV itself (main.rs:296) — do
# NOT re-list those flags here; that redundancy is what produced L133's wrong diagnosis.
# ── MODEL PATHS ───────────────────────────────────────────────────────────────────────────
# The ONE place a benchmark names a model. Previously all 24 bench scripts pasted the raw
# ollama blob path (`~/.ollama/models/blobs/sha256-1194192c…`), which is unreadable, unstable
# across machines, and impossible to grep for intent.
#
# `models/qwen3-coder-30b/` is a HARDLINK to that blob — same inode, so it costs ZERO extra
# bytes on a 17 GB file (verified: 265 Gi free before and after). Override either var to race
# a different build.
: "${QWEN_GGUF:=models/qwen3-coder-30b/qwen3-coder-30b-a3b-q4_k_s.gguf}"
: "${MUSE_GGUF:=models/muse-glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"

ARF_ENV=(env ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1)
# shellcheck disable=SC2206
[ -n "${ARF_ENV_EXTRA:-}" ] && ARF_ENV+=(${ARF_ENV_EXTRA})

# KV pool bytes for a given --num-blocks, from the real geometry (qwen3-coder: 48 layers,
# 4 kv_heads, head_dim 128). f32 keys+values = slots*kv_dim*4*2*layers; the f16 pool adds half
# that again when ARF_KV_F16 is on (it is ADDITIVE — it does not replace the f32 pool).
# f32 pool always; the f16 pool ONLY when ARF_KV_F16 is explicitly opted back in (L135).
kv_pool_mb() { # $1 num_blocks  [$2 layers] [$3 kv_heads] [$4 head_dim] [$5 block_size]
  awk -v nb="$1" -v L="${2:-48}" -v KH="${3:-4}" -v HD="${4:-128}" -v BS="${5:-16}" \
      -v f16on="${ARF_KV_F16:-}" 'BEGIN{
    n = nb*BS*KH*HD; f32 = n*4*2*L; f16 = (f16on=="" ? 0 : n*2*2*L)
    printf "%.0f", (f32+f16)/1048576 }'
}

free_mb() { # physical RAM the OS could hand out right now (free + inactive + speculative)
  vm_stat 2>/dev/null | awk -F'[:.]' '
    /page size of/ {for(i=1;i<=NF;i++) if($i+0>1000) ps=$i+0}
    /Pages free/ {f=$2} /Pages inactive/ {ia=$2} /Pages speculative/ {sp=$2}
    END{ if(!ps) ps=16384; printf "%.0f", (f+ia+sp)*ps/1048576 }'
}

# PREFLIGHT — refuse a launch that cannot fit. This is the guard that was missing when
# `llama-server -np 64 -c 196608` was started on a box with 794 MB free and took it down.
# WEIGHTS_MB defaults to the qwen3-coder-30B Q4_K_S resident size; HEADROOM_MB is what we
# leave the rest of the system.
ram_preflight() { # $1 num_blocks  $2 label
  local nb="$1" label="${2:-engine}" need have weights headroom
  weights="${WEIGHTS_MB:-3300}"; headroom="${HEADROOM_MB:-4096}"
  need=$(( $(kv_pool_mb "$nb") + weights ))
  have=$(free_mb)
  printf '  preflight %s: need ~%s MB (KV %s + weights %s), free %s MB, headroom %s MB\n' \
    "$label" "$need" "$(kv_pool_mb "$nb")" "$weights" "$have" "$headroom"
  if [ "$(( need + headroom ))" -gt "$have" ]; then
    echo "  🔴 REFUSING TO LAUNCH $label — would leave < ${headroom} MB free and drive the box"
    echo "     into swap (this OOM-rebooted the Mac twice on 2026-08-12)."
    echo "     Lower --num-blocks, close apps, or set HEADROOM_MB=<smaller> to override."
    return 1
  fi
  return 0
}

# Same guard for llama-server, whose KV is sized by -c (context) x per-token bytes.
llama_preflight() { # $1 total_ctx  $2 label
  local ctx="$1" label="${2:-llama}" need have
  # 48 layers x 4 kv_heads x 128 head_dim x 2 (K+V) x 2 bytes (f16) = 98304 B/token
  need=$(awk -v c="$ctx" -v w="${WEIGHTS_MB:-3300}" 'BEGIN{printf "%.0f", c*98304/1048576 + w}')
  have=$(free_mb)
  printf '  preflight %s: need ~%s MB (ctx %s), free %s MB\n' "$label" "$need" "$ctx" "$have"
  if [ "$(( need + ${HEADROOM_MB:-4096} ))" -gt "$have" ]; then
    echo "  🔴 REFUSING TO LAUNCH $label — ctx $ctx does not fit in $have MB free."
    return 1
  fi
  return 0
}
