#!/usr/bin/env bash
# Point the Homebrew formula at a release: rewrite its `url` and `sha256` in place.
#
#   scripts/update_tap_formula.sh FORMULA VERSION SHA256
#
# FORMULA is a checkout's Formula/arf.rb (joydle/homebrew-tap). The release workflow runs this
# after publishing, so the formula never points at an older tarball than the newest release.
# Refuses anything but a dotted version and a 64-hex sha, and checks the result names both.
set -euo pipefail
formula=$1 version=$2 sha=$3
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not a version: $version" >&2; exit 1; }
[[ $sha =~ ^[0-9a-f]{64}$ ]] || { echo "not a sha256: $sha" >&2; exit 1; }
url="https://github.com/joydle/Arf/releases/download/v$version/arf-$version-aarch64-apple-darwin.tar.gz"
tmp=$(mktemp)
sed -E -e "s|^  url \".*\"$|  url \"$url\"|" -e "s|^  sha256 \"[0-9a-f]{64}\"$|  sha256 \"$sha\"|" \
  "$formula" > "$tmp"
grep -qF "  url \"$url\"" "$tmp" && grep -qF "  sha256 \"$sha\"" "$tmp" \
  || { rm -f "$tmp"; echo "$formula: no url/sha256 line to rewrite" >&2; exit 1; }
mv "$tmp" "$formula"
