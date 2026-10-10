#!/usr/bin/env bash
# Rule 8 made executable: PROVE THE INSTRUMENT BEFORE YOU BELIEVE THE NUMBER.
#
# Every measurement harness in this repo has produced a confident wrong answer at least once, and
# always the same way: the thing under test never ran, and silence looked like success. This runs
# the three checks that would have caught every one of them (2026-09-21).
#
# usage: verify_instrument.sh <logfile> <marker> [port]
#   <marker>  a string the log MUST contain if the feature actually dispatched
#             (e.g. "[state-snapshot]", "[chunkprof]", "[dflash]")
#   [port]    if given, assert OUR server owns it — not a stale daemon from hours ago
set -euo pipefail
log="${1:?usage: verify_instrument.sh <logfile> <marker> [port]}"
marker="${2:?missing marker}"
port="${3:-}"
fail() { echo "INSTRUMENT NOT PROVEN: $1" >&2; exit 1; }

[ -s "$log" ] || fail "$log is empty — the run produced nothing"
grep -qF -- "$marker" "$log" || fail "no '$marker' in $log — the feature never dispatched, so a green result is not evidence"
if grep -qiE "already serving|bootstrap failed|resource_assembly" "$log"; then
  fail "the server did not start; the numbers belong to something else"
fi

if [ -n "$port" ]; then
  owner=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
  [ -n "$owner" ] || fail "nothing is listening on :$port"
  age=$(ps -o etimes= -p "$owner" 2>/dev/null | tr -d ' ' || echo 99999)
  [ "$age" -lt 1800 ] || fail "the process on :$port has been up ${age}s — probably a stale daemon, not this run"
fi

echo "instrument proven: '$marker' present, server clean${port:+, :$port owned by a process ${age}s old}"
