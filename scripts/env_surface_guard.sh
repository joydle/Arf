#!/usr/bin/env bash
# env_surface_guard.sh — THE RATCHET.
#
# WHY THIS EXISTS:
#   The first count recorded 187 distinct ARF_* env vars. By L29 the count was 238. Every strike
#   adds knobs and NOTHING removes them. Each knob is a branch a future reader must reason about,
#   a code path parity does not cover, and a config the daemon can be launched wrong with.
#
#   The project's standing rule: A MEASURED WINNER SHOULD SIMPLY *BE* THE BEHAVIOR. If a lever
#   won, delete the flag and make it the default. If it lost, delete the flag AND the dead branch.
#   The only knobs that earn their keep are diagnostics (default-off), NO_* escape hatches for
#   defaults that already won, and genuinely environment-dependent tuning (paths, ports, sizes).
#
# WHAT IT DOES:
#   Counts distinct ARF_* identifiers in crates/**/*.rs and FAILS if the count exceeds
#   ENV_CEILING. The ceiling is a RATCHET: it may only ever be lowered. Lowering it is the
#   deliverable of a cleanup strike; raising it requires the rule below.
#
# THE RULE (enforced by review, not by this script):
#   A NEW ARF_* KNOB MUST EITHER
#     (a) REPLACE an existing one (net-zero surface — lower or hold the ceiling), OR
#     (b) come with a MEASURED JUSTIFICATION in the pull request naming the strike, the
#         A/B numbers, and the reason it cannot simply be the default.
#   "I might want to try this later" is not a justification. Try it, measure it, then make the
#   winner the default and delete the flag.
#
# USAGE:  scripts/env_surface_guard.sh [--list] [--ceiling N]
#   --list        print the full sorted inventory (var + first file:line) and exit 0
#   --ceiling N   override the pinned ceiling (for local experiments only; do NOT commit a raise)
#   exit 0 = at or under ceiling; exit 1 = OVER ceiling (surface grew); exit 2 = usage error
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# ── THE RATCHET ────────────────────────────────────────────────────────────────────────────────
# Pinned 2026-08-05 (L29) at the post-cleanup count. HISTORY — this number may only go DOWN:
#   187  first count
#   238  L29 start (pre-cleanup) — the ratchet is installed HERE, then lowered as L29 deletes.
#   233  L29 batch 1: the L10 multicol lever (3 vars + its 168-line shader) and the two NSG=2
#        occupancy levers (measured flat / -5% conc16) deleted, branches and all.
#   232  L29 batch 2: ARF_GQA_ATTN (conc64 -13%, premise REFUTED) + its 149-line kernel.
#   227  L29 batch 3: the MoE OVERLAP family (DBUF/PREFETCH/LOWSMEM/DEQSHORTCUT — "the whole
#        overlap family is ruled out") + the skip-all-barriers probe.
#   228  L44 (2026-08-06): net +1 for ARF_OCCUPANCY, justified under rule (b). It is a
#        default-OFF, read-only diagnostic (maxTotalThreadsPerThreadgroup / threadExecutionWidth /
#        staticThreadgroupMemoryLength at PSO build) and it FOUND A REAL BUG: attention reserved
#        16448 B of a 32 KiB core (1 threadgroup resident) because MAXHD=512 sized the accumulator
#        for gemma's hd=512 while we run qwen3's hd=128. It cannot "simply be the default" — it
#        prints a line per pipeline at load. Paid for in the same strike by deleting
#        ARF_ATTN_KVH_TILE (L40 flat), ARF_NO_ATTN_MAXHD_FIT (fit made unconditional), and
#        the DARK ARF_BATCH_MEGA_KSTEP setting in the daemon win config (L34: the daemon set it
#        and printed "KSTEP=8" while nothing in the engine read it).
#   218  2026-08-06: -10 from deleting the Qwen3.5-SSM engine, the MTP modules and their
#        examples/probes. These vars vanished with the code that read them — the ratchet is
#        lowered to lock the win in.
ENV_CEILING="${ENV_CEILING:-218}"

LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list)    LIST=1; shift;;
    --ceiling) ENV_CEILING="${2:?--ceiling needs a number}"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# The canonical count. Keep this expression identical to the one used for the first count so
# the numbers are comparable across strikes.
INVENTORY=$(grep -rhoE 'ARF_[A-Z0-9_]+' --include='*.rs' crates/ | sort -u)
COUNT=$(printf '%s\n' "$INVENTORY" | grep -c . )

if [ "$LIST" = "1" ]; then
  printf '%s\n' "$INVENTORY" | while read -r v; do
    [ -z "$v" ] && continue
    loc=$(grep -rnoE "\b$v\b" --include='*.rs' crates/ | head -1 | cut -d: -f1,2)
    printf '%-40s %s\n' "$v" "$loc"
  done
  printf '\n%d distinct ARF_* vars (ceiling %d)\n' "$COUNT" "$ENV_CEILING"
  exit 0
fi

printf 'env surface: %d distinct ARF_* vars (ceiling %d)\n' "$COUNT" "$ENV_CEILING"

if [ "$COUNT" -gt "$ENV_CEILING" ]; then
  cat >&2 <<EOF

🔴 ENV SURFACE GUARD FAILED — $COUNT > ceiling $ENV_CEILING (grew by $((COUNT - ENV_CEILING)))

  The config surface may not grow. A new ARF_* knob must either REPLACE an existing one,
  or carry a measured justification in the pull request (strike, A/B numbers, why it cannot be
  the default). If your lever WON: make it the default and delete the flag. If it LOST: delete
  the flag and its branch.

  See what you added:   scripts/env_surface_guard.sh --list
EOF
  exit 1
fi

if [ "$COUNT" -lt "$ENV_CEILING" ]; then
  printf '  ✅ under ceiling by %d — LOWER THE RATCHET: set ENV_CEILING=%d in %s\n' \
    "$((ENV_CEILING - COUNT))" "$COUNT" "scripts/env_surface_guard.sh"
else
  printf '  ✅ at ceiling\n'
fi
exit 0
