#!/usr/bin/env bash
# L264 — fail the build if a developer's home directory is committed in live code.
#
# Everything a user compiles or runs must be path-agnostic.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -I skips binary files: a stale __pycache__/*.pyc embeds the build path and is not
# something a human wrote. (Finding one is also a signal it should be gitignored.)
hits=$(grep -rlI -E "/Users/[a-z]+/|/home/[a-z]+/" crates/ scripts/ 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "error: hardcoded home directory in live code:" >&2
  echo "$hits" | sed 's/^/  /' >&2
  echo "use an env var or a path relative to the repo root instead" >&2
  exit 1
fi
echo "no hardcoded home paths in crates/ or scripts/"
