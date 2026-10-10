#!/usr/bin/env bash
# Fetch (anonymously, no HF token) and sha256-verify the MIT TRELLIS checkpoints the M0 harness
# uses. DINOv3 (gated, Meta licence) and RMBG-2.0 (CC BY-NC) are deliberately NOT fetched.
#
#   scripts/ref/trellis2/fetch_weights.sh [512|all] [--verify-only]
#
# Weights land in ${TRELLIS2_WEIGHTS:-/tmp/trellis2-ref/weights} (never in the repo).
# Hashes are the LFS oids from the Hugging Face tree API, read 2026-09-27.
set -uo pipefail
W="${TRELLIS2_WEIGHTS:-/tmp/trellis2-ref/weights}"
SET="${1:-512}"
VERIFY_ONLY="${2:-}"
mkdir -p "$W"

# repo  path  sha256  bytes  set
FILES=$(cat <<'EOF'
TRELLIS.2-4B pipeline.json - - 512
TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.json - - 512
TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.safetensors 1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a 147591972 512
TRELLIS.2-4B ckpts/ss_flow_img_dit_1_3B_64_bf16.json - - 512
TRELLIS.2-4B ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6 2584426920 512
TRELLIS.2-4B ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json - - 512
TRELLIS.2-4B ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f 2584574424 512
TRELLIS.2-4B ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json - - 512
TRELLIS.2-4B ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors 8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d 2584672728 512
TRELLIS.2-4B ckpts/shape_dec_next_dc_f16c32_fp16.json - - 512
TRELLIS.2-4B ckpts/shape_dec_next_dc_f16c32_fp16.safetensors e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581 948490494 512
TRELLIS.2-4B ckpts/tex_dec_next_dc_f16c32_fp16.json - - 512
TRELLIS.2-4B ckpts/tex_dec_next_dc_f16c32_fp16.safetensors 97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c 948458812 512
TRELLIS.2-4B ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json - - all
TRELLIS.2-4B ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors 07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd 2584574424 all
TRELLIS.2-4B ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json - - all
TRELLIS.2-4B ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors 580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686 2584672728 all
EOF
)

rc=0
while read -r repo path sha bytes set; do
  [ "$SET" = "512" ] && [ "$set" = "all" ] && continue
  out="$W/$repo/$path"
  mkdir -p "$(dirname "$out")"
  if [ "$VERIFY_ONLY" != "--verify-only" ] && [ ! -f "$out.verified" ]; then
    # DNS for huggingface.co was intermittent on this box (2026-09-27): retry until it lands.
    for attempt in $(seq 1 100); do
      env -u HF_TOKEN curl -sSL --retry 10 --retry-all-errors -C - -o "$out" \
        "https://huggingface.co/microsoft/$repo/resolve/main/$path" && break
      echo "retry $attempt $path"; sleep 5
    done
  fi
  if [ "$sha" = "-" ]; then
    [ -s "$out" ] && echo "ok     $repo/$path" || { echo "MISSING $repo/$path"; rc=1; }
    continue
  fi
  size=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$size" != "$bytes" ]; then echo "SIZE   $repo/$path: $size != $bytes"; rc=1; continue; fi
  if [ -f "$out.verified" ]; then echo "ok     $repo/$path (verified earlier)"; continue; fi
  got=$(shasum -a 256 "$out" | cut -d' ' -f1)
  if [ "$got" = "$sha" ]; then touch "$out.verified"; echo "sha256 $repo/$path"; else echo "HASH   $repo/$path: $got"; rc=1; fi
done <<< "$FILES"
exit $rc
