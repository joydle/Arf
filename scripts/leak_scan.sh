#!/usr/bin/env bash
# Leak scan for the public repository: run before every push (the pre-push hook does) and in CI.
#
#   scripts/leak_scan.sh            # scan the tracked tree of this checkout
#   scripts/leak_scan.sh DIR        # scan another checkout
#   scripts/leak_scan.sh --strings FILE...   # scan the printable strings of built binaries (#56):
#                                   # a name split across string pieces in the source is whole in
#                                   # the binary; the release job runs this on the published tarball
#
# Two layers:
#   1. GENERIC patterns, below — credentials, home paths, IP addresses, ssh targets, email
#      addresses. Public, because they name nothing.
#   2. A LOCAL DENYLIST, never committed here: `$ARF_DENYLIST`, else `~/.config/arf/denylist`.
#      One `label|extended-regex` per line, `#` comments allowed. It holds the names of things that
#      must not appear in this repository; publishing the list would publish the names.
# CI runs layer 1 only. A maintainer's pre-push hook runs both. Exit 1 on any hit.
set -euo pipefail
STRINGS=()
if [ "${1:-}" = "--strings" ]; then
  shift
  [ "$#" -gt 0 ] || { echo "--strings needs at least one file" >&2; exit 2; }
  STRINGS=("$@")
else
  cd "${1:-$(git rev-parse --show-toplevel)}"
fi

# Reviewed false positives: loopback / any-address and the router's 10.0.0.x doc examples; the
# code of conduct's contact address; SSH remotes of this project's own repositories (git@github.com:joydle/...,
# not an email); this script's own pattern table.
ALLOW='127\.0\.0\.1|0\.0\.0\.0|http://10\.0\.0\.[0-9]:|contact@joydle\.dev|git@github\.com:joydle/|^scripts/leak_scan\.sh:'

GENERIC='API token|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|xox[bp]-|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY
home path|/Users/[a-z]+/|/home/[a-z]+/
IPv4 address|(^|[^0-9.])([0-9]{1,3}\.){3}[0-9]{1,3}([^0-9.]|$)
ssh to a box|ssh +(-[a-zA-Z] *[^ ]+ +)*[a-z0-9_]+@|ssh -p|scp -P|root@[0-9]
email address|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|dev|io|net|org|ai)([^A-Za-z]|$)'

# ARF_INDEX_EXCLUDE (optional) names a file of paths, one per line, that this checkout does not
# scan — the same exclusion list the doc generators honour. Unset or empty: scan everything.
EXCL=(':!Cargo.lock')
if [ -n "${ARF_INDEX_EXCLUDE:-}" ]; then
  [ -f "$ARF_INDEX_EXCLUDE" ] || { echo "ARF_INDEX_EXCLUDE names a missing file: $ARF_INDEX_EXCLUDE" >&2; exit 2; }
  while IFS= read -r p; do
    case "$p" in ''|'#'*) continue ;; esac
    EXCL+=(":!${p%/}")
  done < "$ARF_INDEX_EXCLUDE"
fi

# In a binary: the CI runner's build paths (panic locations of dependencies) and the OpenSSL
# authors' address in vendored crypto code.
ALLOW_STRINGS='/Users/runner/|appro@openssl\.org'

scan() {
  local hit=0 label pat out f
  while IFS='|' read -r label pat; do
    case "$label" in ''|'#'*) continue ;; esac
    if [ "${#STRINGS[@]}" -gt 0 ]; then
      out="$(for f in "${STRINGS[@]}"; do strings -n 6 "$f" | sed "s|^|$(basename "$f"): |"; done | grep -E "$pat" | grep -v -E "$ALLOW|$ALLOW_STRINGS" | head -8 || true)"
    else
      out="$(git grep -n -I -E "$pat" -- . "${EXCL[@]}" 2>/dev/null | grep -v -E "$ALLOW" | head -8 || true)"
    fi
    if [ -n "$out" ]; then
      echo "LEAK? $label"; echo "$out" | cut -c1-200 | sed 's/^/    /'; hit=1
    fi
  done
  return $hit
}

fail=0
printf '%s\n' "$GENERIC" | scan || fail=1
deny="${ARF_DENYLIST:-$HOME/.config/arf/denylist}"
if [ -f "$deny" ]; then
  scan < "$deny" || fail=1
  echo "leak scan: generic + local denylist ($(grep -cv -E '^(#|$)' "$deny") patterns)"
else
  echo "leak scan: generic only (no local denylist at $deny)"
fi
[ "$fail" = 0 ] && echo "leak scan: clean" || { echo "leak scan: FAILED" >&2; exit 1; }
