#!/usr/bin/env bash
# get_qwen38.sh — build the Qwen3.8-27B bundle from PUBLIC files, for when `arf pull qwen3.8:27b`
# cannot reach the published group-64 repo (or you would rather build it yourself).
#
#   models/qwen3.8-27b-arf/
#     model.gguf       OUR group-64 weights, built here by scripts/g64/quantize_g64.py
#     draft/           the DFlash 2 block draft (incoai/Qwen3.8-27B-DFlash2)
#     mmproj.gguf      the vision projector (unsloth/Qwen3.8-27B-GGUF mmproj-BF16.gguf)
#     arf-bundle.txt   arch=qwen3.8-27b, quant=q4ks — read by arf-serve (crates/arf-core/src/bundle.rs)
#
# then:   arf serve qwen3.8:27b        (or: target/release/arf-serve --model models/qwen3.8-27b-arf)
#
# Inputs, all public, all sha256-checked against the values pinned below:
#   unsloth/Qwen3.8-27B-GGUF  Qwen3.8-27B-Q8_0.gguf     29.05 GB  the source the projections are fit to
#                             Qwen3.8-27B-Q4_K_M.gguf   17.11 GB  the TEMPLATE: every non-projection
#                                                                 tensor and all metadata are copied
#                             imatrix_unsloth.gguf      0.01 GB   the importance weights
#                             mmproj-BF16.gguf          0.93 GB
#   incoai/Qwen3.8-27B-DFlash2  config.json + model.safetensors  3.85 GB
#
# THE TEMPLATE IS PINNED TO AN OLD REVISION, ON PURPOSE. The shipped file was built against
# unsloth's plain Q4_K_M (17,106,775,008 bytes, sha256 7e78da5d...), which unsloth deleted from
# `main` on 2026-08-19 (commit e1d8a267). `main` now has only UD-Q4_K_M, and that is NOT a
# drop-in: its header differs from the shipped file's template on 104 tensors outside the
# projections (ssm_alpha/ssm_beta are Q8_0 there, F32 here; the MTP layer's types differ), so a
# UD template builds a different model than the one measured. Checked 2026-09-27 by reading both
# headers. Commit 4121cb19 is the last one that carries the plain Q4_K_M; HF still serves it.
#
# Disk: ~64 GB free while converting (Q8_0 + template + imatrix + output = 63.3 GB), then the Q8_0
# and the template are deleted (KEEP_Q8=1 / KEEP_TEMPLATE=1 keep them) and the bundle is 21.9 GB.
# The first `arf serve` adds a ~19 GB weight cache beside model.gguf (model.gguf.arf-q4ks-cache).
#
# Conversion needs Apple silicon (the quantizer runs on the GPU through MLX) and Python 3 with
# numpy, gguf and mlx. It took ~13 min on an M4 Max (measured 2026-09-25).
#
# Idempotent: each step is skipped when its output exists and checks out, and downloads resume,
# so re-running after an interruption continues where it stopped.
#
# env: OUT=<bundle dir>         default models/qwen3.8-27b-arf
#      SRC=<inputs dir>         default <OUT>.src; a SRC you name is never deleted from
#      PYTHON=<interpreter>     default python3
#      KEEP_Q8=1 KEEP_TEMPLATE=1  keep the conversion inputs afterwards
#      HF_TOKEN                 honoured by `hf` if set; never printed; none of these repos need it
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/models/qwen3.8-27b-arf}"
DEFAULT_SRC="$OUT.src"
SRC="${SRC:-$DEFAULT_SRC}"
PYTHON="${PYTHON:-python3}"
KEEP_Q8="${KEEP_Q8:-0}"
KEEP_TEMPLATE="${KEEP_TEMPLATE:-0}"

SRC_REPO="unsloth/Qwen3.8-27B-GGUF"
TEMPLATE_REV="4121cb19390a7984a0e6dc0f46bea9177b846f15"   # last commit with the plain Q4_K_M
DRAFT_REPO="incoai/Qwen3.8-27B-DFlash2"

# name  size  sha256  (sizes and hashes are HF's x-linked-size / x-linked-etag, read 2026-09-27)
Q8="Qwen3.8-27B-Q8_0.gguf"
Q8_SIZE=29047086048
Q8_SHA=a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348
TEMPLATE="Qwen3.8-27B-Q4_K_M.gguf"
TEMPLATE_SIZE=17106775008
TEMPLATE_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
IMAT="imatrix_unsloth.gguf"
IMAT_SIZE=13642656
IMAT_SHA=0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1
MMPROJ_SRC="mmproj-BF16.gguf"
MMPROJ_SIZE=931146432
MMPROJ_SHA=83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53
DRAFT_WEIGHTS="model.safetensors"
DRAFT_SIZE=3848817896
DRAFT_SHA=67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c

# The output: the shipped file's shape. 866 tensors (the template's), 401 of them Q4_1 (the 400
# trunk projections + the lm_head), 17,119,616,032 bytes — sizes follow from shapes and types
# alone, so the same pinned inputs give exactly this size.
MODEL="model.gguf"
MODEL_SIZE=17119616032
EXPECT_TENSORS=866
EXPECT_Q4_1=401

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }
size_of() { if [ -f "$1" ]; then wc -c <"$1" | tr -d ' '; else echo 0; fi; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# verified <file> <size> <sha>: true when the file has that size and hash. The hash of a big file
# costs a minute, so a match is remembered in <file>.sha256 and trusted while the size agrees.
verified() {
  local f=$1 want_size=$2 want_sha=$3
  [ -f "$f" ] || return 1
  [ "$(size_of "$f")" = "$want_size" ] || return 1
  if [ -f "$f.sha256" ] && [ "$(cat "$f.sha256")" = "$want_sha" ]; then return 0; fi
  say "checking sha256 of $(basename "$f") ($(gb "$want_size"))"
  local have
  have=$(sha256_of "$f")
  if [ "$have" != "$want_sha" ]; then
    printf 'error: %s: sha256 %s, expected %s\n' "$f" "$have" "$want_sha" >&2
    return 1
  fi
  printf '%s' "$want_sha" >"$f.sha256"
}

HF_CLI=""
if command -v hf >/dev/null 2>&1; then HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF_CLI=huggingface-cli; fi

# fetch <repo> <revision> <file> <dir> <size>: download <dir>/<file>, resuming a partial one.
fetch() {
  local repo=$1 rev=$2 file=$3 dir=$4 size=$5
  mkdir -p "$dir"
  say "downloading $repo/$file ($(gb "$size"))"
  if [ -n "$HF_CLI" ]; then
    "$HF_CLI" download "$repo" "$file" --revision "$rev" --local-dir "$dir"
  else
    local part="$dir/$file.part"
    if [ "$(size_of "$part")" != "$size" ]; then
      curl -fL --retry 5 --retry-delay 5 -C - -o "$part" \
        "https://huggingface.co/$repo/resolve/$rev/$file"
    fi
    mv "$part" "$dir/$file"
  fi
  [ -f "$dir/$file" ] || die "download of $repo/$file produced no file"
}

# get <repo> <revision> <file> <dir> <size> <sha>: fetch unless already verified, then verify.
get() {
  local repo=$1 rev=$2 file=$3 dir=$4 size=$5 sha=$6
  if verified "$dir/$file" "$size" "$sha"; then
    say "have $file"
    return 0
  fi
  if [ -f "$dir/$file" ] && [ "$(size_of "$dir/$file")" = "$size" ]; then
    rm -f "$dir/$file" "$dir/$file.sha256"   # right size, wrong bytes: start that file over
  fi
  fetch "$repo" "$rev" "$file" "$dir" "$size"
  verified "$dir/$file" "$size" "$sha" || die "$file failed verification (removed nothing; delete it and re-run)"
}

# tensor_count <gguf>: the header's tensor count (plain struct read, no Python packages needed).
tensor_count() {
  "$PYTHON" - "$1" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    b = f.read(24)
print(struct.unpack("<4sIQQ", b)[2] if b[:4] == b"GGUF" else -1)
PY
}

model_ok() {
  local f="$OUT/$MODEL"
  [ -f "$f" ] && [ "$(size_of "$f")" = "$MODEL_SIZE" ] && [ "$(tensor_count "$f")" = "$EXPECT_TENSORS" ]
}

# the nearest existing directory at or above <path>
existing() { local d=$1; while [ ! -d "$d" ]; do d=$(dirname "$d"); done; echo "$d"; }
# free bytes, and the mount point, of the filesystem holding <path>
free_bytes() { df -Pk "$(existing "$1")" | awk 'NR == 2 { printf "%.0f", $4 * 1024 }'; }
mount_of() { df -Pk "$(existing "$1")" | awk 'NR == 2 { print $6 }'; }

# need_disk <path> <bytes> <what>: refuse, politely, when the filesystem holding <path> is short.
need_disk() {
  local have
  have=$(free_bytes "$1")
  if [ "$have" -lt "$2" ]; then
    die "not enough disk for $3: it needs $(gb "$2") free and $(existing "$1") has $(gb "$have").
    Free some space, or put the inputs on a bigger disk with SRC=/big/disk/qwen38-src (the bundle with OUT=)."
  fi
  say "disk: $(gb "$have") free for $3, $(gb "$2") needed"
}

missing() {  # bytes still to download for <file> <size> (a partial download counts as missing)
  if [ "$(size_of "$1")" = "$2" ]; then echo 0; else echo "$2"; fi
}

command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found (set PYTHON=/path/to/python3)"
[ -n "$HF_CLI" ] || command -v curl >/dev/null 2>&1 || die "need \`hf\` (pip install -U huggingface_hub) or curl"
# OUT is created only once there is something to put in it: an empty models/qwen3.8-27b-arf would
# make `arf serve qwen3.8:27b` find a bundle with no weights.

# ---- 1. the weights -------------------------------------------------------------------------
if model_ok; then
  say "have $MODEL ($EXPECT_TENSORS tensors, $(gb "$MODEL_SIZE"))"
else
  # Python packages FIRST: finding out after a 46 GB download is the wrong order.
  if ! "$PYTHON" -c 'import numpy, gguf' >/dev/null 2>&1; then
    die "the quantizer needs numpy and gguf for $PYTHON:
    $PYTHON -m pip install numpy gguf mlx"
  fi
  if ! "$PYTHON" -c 'import mlx.core' >/dev/null 2>&1; then
    die "the quantizer runs on the GPU through MLX, which needs Apple silicon and:
    $PYTHON -m pip install mlx"
  fi

  # Disk: the conversion peak is every input plus the output at once (2 GB spare). The draft and
  # the projector come down AFTER the Q8_0 and the template are deleted, so they only add to the
  # peak when both are kept.
  inputs=$(( $(missing "$SRC/$Q8" "$Q8_SIZE") + $(missing "$SRC/$TEMPLATE" "$TEMPLATE_SIZE") \
           + $(missing "$SRC/$IMAT" "$IMAT_SIZE") ))
  output=$(( MODEL_SIZE + 2000000000 ))
  if [ "$KEEP_Q8" = 1 ] && [ "$KEEP_TEMPLATE" = 1 ]; then
    output=$(( output + DRAFT_SIZE + MMPROJ_SIZE ))
  fi
  if [ "$(mount_of "$SRC")" = "$(mount_of "$OUT")" ]; then
    need_disk "$SRC" $(( inputs + output )) "the inputs and the output while converting"
  else
    need_disk "$SRC" $(( inputs + 2000000000 )) "the inputs"
    need_disk "$OUT" "$output" "the output"
  fi

  # imatrix_unsloth.gguf is NOT in the pinned revision (unsloth uploaded it on 2026-08-20): it
  # comes from main, held to the hash pinned above like everything else.
  get "$SRC_REPO" main "$IMAT" "$SRC" "$IMAT_SIZE" "$IMAT_SHA"
  get "$SRC_REPO" "$TEMPLATE_REV" "$TEMPLATE" "$SRC" "$TEMPLATE_SIZE" "$TEMPLATE_SHA"
  get "$SRC_REPO" main "$Q8" "$SRC" "$Q8_SIZE" "$Q8_SHA"

  mkdir -p "$OUT"
  building="$OUT/$MODEL.building"
  rm -f "$building"
  say "quantizing: group-64 projections + lm_head from the Q8_0 (scripts/g64/quantize_g64.py)"
  # G64_LM_HEAD=1: the shipped file quantizes the lm_head too. G64_3BIT unset: all 4-bit.
  env -u G64_3BIT G64_LM_HEAD=1 "$PYTHON" "$ROOT/scripts/g64/quantize_g64.py" \
    "$SRC/$Q8" "$SRC/$TEMPLATE" "$SRC/$IMAT" "$building"

  # Verify the thing, not the exit code: count, Q4_1 count, size.
  read -r n q41 < <("$PYTHON" - "$building" <<'PY'
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
print(len(r.tensors), sum(t.tensor_type.name == "Q4_1" for t in r.tensors))
PY
)
  size=$(size_of "$building")
  if [ "$n" != "$EXPECT_TENSORS" ] || [ "$q41" != "$EXPECT_Q4_1" ] || [ "$size" != "$MODEL_SIZE" ]; then
    die "$building is not the expected file: $n tensors ($EXPECT_TENSORS expected), $q41 Q4_1 ($EXPECT_Q4_1), $size bytes ($MODEL_SIZE). Left in place for inspection."
  fi
  mv "$building" "$OUT/$MODEL"
  say "built $MODEL: $n tensors, $q41 group-64 Q4_1, $(gb "$size")"
fi

# The inputs are only needed to build the weights. Delete the big two unless asked not to — and
# never from a SRC you pointed at yourself.
if model_ok; then
  if [ "$SRC" = "$DEFAULT_SRC" ]; then
    if [ "$KEEP_Q8" != 1 ] && [ -f "$SRC/$Q8" ]; then
      rm -f "$SRC/$Q8" "$SRC/$Q8.sha256"; say "deleted $Q8 ($(gb "$Q8_SIZE")); KEEP_Q8=1 keeps it"
    fi
    if [ "$KEEP_TEMPLATE" != 1 ] && [ -f "$SRC/$TEMPLATE" ]; then
      rm -f "$SRC/$TEMPLATE" "$SRC/$TEMPLATE.sha256"; say "deleted $TEMPLATE ($(gb "$TEMPLATE_SIZE")); KEEP_TEMPLATE=1 keeps it"
    fi
  elif [ -f "$SRC/$Q8" ] || [ -f "$SRC/$TEMPLATE" ]; then
    say "SRC=$SRC is yours: its inputs were left in place"
  fi
fi

# ---- 2. the vision projector ------------------------------------------------------------------
if verified "$OUT/mmproj.gguf" "$MMPROJ_SIZE" "$MMPROJ_SHA" 2>/dev/null; then
  say "have mmproj.gguf"
else
  get "$SRC_REPO" main "$MMPROJ_SRC" "$SRC" "$MMPROJ_SIZE" "$MMPROJ_SHA"
  mv "$SRC/$MMPROJ_SRC" "$OUT/mmproj.gguf"
  mv "$SRC/$MMPROJ_SRC.sha256" "$OUT/mmproj.gguf.sha256"
fi

# ---- 3. the block draft -----------------------------------------------------------------------
if verified "$OUT/draft/$DRAFT_WEIGHTS" "$DRAFT_SIZE" "$DRAFT_SHA" 2>/dev/null \
   && grep -q DFlash2DraftModel "$OUT/draft/config.json" 2>/dev/null; then
  say "have draft/"
else
  mkdir -p "$SRC/draft" "$OUT/draft"
  get "$DRAFT_REPO" main "$DRAFT_WEIGHTS" "$SRC/draft" "$DRAFT_SIZE" "$DRAFT_SHA"
  [ -s "$SRC/draft/config.json" ] || fetch "$DRAFT_REPO" main config.json "$SRC/draft" 1239
  grep -q DFlash2DraftModel "$SRC/draft/config.json" \
    || die "$SRC/draft/config.json is not a DFlash 2 draft config"
  mv "$SRC/draft/config.json" "$OUT/draft/config.json"
  mv "$SRC/draft/$DRAFT_WEIGHTS" "$OUT/draft/$DRAFT_WEIGHTS"
  mv "$SRC/draft/$DRAFT_WEIGHTS.sha256" "$OUT/draft/$DRAFT_WEIGHTS.sha256"
fi

# ---- 4. what arf-serve needs to know ----------------------------------------------------------
cat >"$OUT/arf-bundle.txt" <<'EOF'
# written by scripts/get_qwen38.sh; read by arf-serve when --model points at this directory.
# A flag given on the command line wins over the value here.
arch=qwen3.8-27b
quant=q4ks
EOF

# `hf download --local-dir` keeps its resume bookkeeping in <dir>/.cache; SRC's only.
if [ "$SRC" = "$DEFAULT_SRC" ] && [ "$KEEP_Q8" != 1 ] && [ "$KEEP_TEMPLATE" != 1 ]; then
  rm -rf "$SRC"
fi

say "bundle ready: $OUT"
if [ "$OUT" = "$ROOT/models/qwen3.8-27b-arf" ]; then
  printf '\n    arf serve qwen3.8:27b\n\n'
else
  printf '\n    arf serve %s\n\n' "$OUT"
fi
printf 'or, without the arf front door:  %s/target/release/arf-serve --model %s\n' "$ROOT" "$OUT"
