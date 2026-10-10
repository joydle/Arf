# Releasing

A release is a tag on `main` that the release workflow turns into a tarball, a GitHub release and a
Homebrew formula pointing at it. Nothing is tagged until the GPU session for its numbers has passed.

1. **Measure.** On a quiet Mac: the prefix-cache gate (`scripts/prefix_cache_hybrid.py`), the
   benchmarks (`scripts/bench_engines.py`, `scripts/speed_bench.py`), with the rules in
   [`PERFORMANCE.md`](PERFORMANCE.md). A run whose log does not show the feature ran is not a result.
2. **Notes.** `docs/RELEASE_NOTES.md` names only what has a recorded measurement or a test, and says _pending_
   for what has neither. Replace each _pending_ the session settled; set the date.
3. **Version.** Bump `version` in the workspace `Cargo.toml`, `cargo build --workspace` (the lockfile
   follows), `make check`, `make leak-scan`, commit.
4. **Tag.** `git tag vX.Y.Z && git push origin vX.Y.Z`. The workflow checks the tag against the
   version, builds generic aarch64 binaries and publishes the release with the notes.
5. **Test the published binary before Homebrew.** Download the tarball, check its `.sha256`, and run
   against the extracted folder: `python3 scripts/agent_e2e.py --bin <folder>` (Claude Code and
   OpenCode through `arf launch`: cold, warm, two agents, restart) and
   `python3 scripts/model_matrix.py --bin <folder>` (every model the site and README list, started
   with the command they advertise and asked something whose answer is known), and
   `AGENT_BENCH_SERVE=<folder>/arf-serve python3 scripts/agent_bench/run.py 1 arf claude,opencode`
   (each agent fixes failing tests in three small repositories; a run passes only if the tests pass
   afterwards and the session ends inside its limit — an early build passed the first two scripts and could
   not finish one of these). A row that fails is
   fixed, or its model is relabelled or removed where it is listed; a skipped row (files not
   downloaded) is not a pass. The tap does not move until all three pass.
6. **Homebrew.** Point `joydle/homebrew-tap` at the new tarball: `scripts/update_tap_formula.sh Formula/arf.rb X.Y.Z <sha256>`
   in a tap checkout (the sha is in the `.sha256` file next to the tarball). The workflow does this itself
   only when a `TAP_DEPLOY_KEY` secret is configured; without one it prints the command.
7. **Verify.** On a Mac: `brew update && brew upgrade joydle/tap/arf` (or `brew install`),
   `arf --version` names the tag's commit, `arf doctor` is clean.
