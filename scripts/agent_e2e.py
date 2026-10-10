#!/usr/bin/env python3
"""What a developer does with Arf, end to end: `arf launch` with Claude Code and OpenCode in a repository with
a large AGENTS.md, timed — the test a release candidate passes before it ships.

    python3 scripts/agent_e2e.py                      # this checkout's target/release
    python3 scripts/agent_e2e.py --bin ~/Downloads/arf-1.0.0-aarch64-apple-darwin   # a release tarball
    python3 scripts/agent_e2e.py --engine splash      # the same steps on Splash (`splash serve`), to compare

Steps, each through the real `arf launch` (which execs the agent):
  1. cold      — a fresh background server; Claude Code's first message (`-p hey`) carries its system
                 prompt, its tools and the AGENTS.md: tens of thousands of tokens read before the answer.
                 `arf status` is polled meanwhile and must show the read progressing.
  2. warm      — the same again: the prompt comes from the prefix cache.
  3. two agents — Claude Code and OpenCode at the same moment, as a developer with both open.
  4. restart   — `arf stop`, then Claude Code again: the saved prompt state loads from disk.
  5. real task — each agent fixes the failing tests of a small repository (scripts/agent_bench/tasks/calc):
                 it must call tools, edit files, and END its session. Passes only if the tests print OK
                 afterwards. An early build passed steps 1-4 and could not finish this with Claude Code in auto mode.
Then the server log is read for progress lines, saved/loaded prompt states and errors, and each step is held
to a limit. Runs on its own port (ARF_PORT, default 8091) so a model already serving on 8080 is untouched.
Needs `claude` and `opencode` on PATH and the model (`--model`, default this checkout's models/qwen3.8-27b-arf).
The machine should be quiet: the times are the result.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Limits a release candidate is held to (M4 Max class; the cold step has none — it is the prompt's size).
WARM_MAX_S = 30
RESTART_MAX_S = 60
REAL_TASK_MAX_S = 900
STEP_TIMEOUT_S = 1800


def agents_md(kb):
    """A realistic AGENTS.md of about `kb` kilobytes: rules, conventions, a layout — repeated sections."""
    parts = ["# AGENTS.md\n\nThis repository is a web service with a worker queue. Read this before changing code.\n"]
    i = 0
    while sum(len(p) for p in parts) < kb * 1024:
        i += 1
        parts.append(
            f"\n## Area {i}: conventions\n\n"
            f"- Module `svc/area_{i}` owns the {i}th domain. Keep its public API in `api.py`; private helpers in `_impl.py`.\n"
            f"- Every handler validates input with the schema in `schemas/area_{i}.json` before touching the database.\n"
            f"- Tests live in `tests/area_{i}/`; run `make test AREA={i}` before a commit. A failing test blocks the merge.\n"
            f"- Log with `log.info` at boundaries only; never log a token, a password or a full request body.\n"
            f"- Migrations: one file per change, named `{1000 + i}_<what>.sql`, reversible, reviewed by the owner of area {i}.\n"
        )
    return "".join(parts)


def run(cmd, cwd, env, timeout=STEP_TIMEOUT_S):
    t = time.time()
    # PWD as a shell sets it: OpenCode takes its working directory from PWD, not from the process's
    # own directory, and ran its commands where this script was started (2026-10-07: it found an
    # older copy of the task elsewhere on disk, ran its tests there and answered DONE).
    env = {**env, "PWD": cwd}
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A failed step, not a crash of the run (2026-10-08: OpenCode sent no request for 1,800 s
        # in step 3, and the timeout ended the whole run before steps 4 and 5).
        return time.time() - t, -1, f"timed out after {timeout} s"
    # The agent's answer is its stdout; its stderr carries its own notices, which are not an answer.
    return time.time() - t, p.returncode, p.stdout.strip()


def watch_status(arf, env, stop, seen):
    """Poll `arf status` every 5 s while a step runs; keep the reading lines it printed."""
    while not stop.is_set():
        out = subprocess.run([arf, "status"], env=env, capture_output=True, text=True).stdout
        seen += [l.strip() for l in out.splitlines() if "tokens (" in l]
        stop.wait(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release"), help="directory with arf and arf-serve")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "qwen3.8-27b-arf"))
    ap.add_argument("--port", default=os.environ.get("ARF_PORT", "8091"))
    ap.add_argument("--agents-md-kb", type=int, default=46, help="size of the generated AGENTS.md (default 46, a real one's)")
    ap.add_argument("--no-mcp", action="store_true",
                    help="without the three MCP servers (default: filesystem, memory and everything, via npx — "
                         "their tool definitions ride on every first message, as in a real setup)")
    ap.add_argument("--keep-state", action="store_true",
                    help="keep ~/.arf/anchors and ~/.arf/prefix in place (default: moved aside for the run and "
                         "restored, so step 1 is a first-ever message, not a replay of an earlier session's prompt)")
    ap.add_argument("--engine", choices=["arf", "splash"], default="arf",
                    help="splash: the same steps against `splash serve` on :8000, the agents pointed at it directly")
    a = ap.parse_args()

    arf = os.path.join(a.bin, "arf")
    for tool in [arf, os.path.join(a.bin, "arf-serve")]:
        if not os.path.isfile(tool):
            sys.exit(f"agent_e2e: {tool} not found (--bin)")
    for agent in ["claude", "opencode"]:
        if not shutil.which(agent):
            sys.exit(f"agent_e2e: `{agent}` is not on PATH")
    env = {**os.environ, "ARF_PORT": str(a.port)}
    log = os.path.expanduser(f"~/.arf/logs/serve-{a.port}.log")
    version = subprocess.run([arf, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"agent_e2e: {version} · model {a.model} · port {a.port} · AGENTS.md {a.agents_md_kb} KB")
    print(f"load {os.getloadavg()[0]:.1f}")

    work = tempfile.mkdtemp(prefix="arf-agent-e2e-")
    subprocess.run(["git", "init", "-q", work], check=True)
    with open(os.path.join(work, "AGENTS.md"), "w") as f:
        f.write(agents_md(a.agents_md_kb))
    # Three real MCP servers, as a developer's setup has: their tools are part of every first message.
    mcp_args = []
    if not a.no_mcp:
        servers = {
            "filesystem": ["npx", "-y", "@modelcontextprotocol/server-filesystem", work],
            "memory": ["npx", "-y", "@modelcontextprotocol/server-memory"],
            "everything": ["npx", "-y", "@modelcontextprotocol/server-everything"],
        }
        mcp = os.path.join(work, "mcp.json")
        with open(mcp, "w") as f:
            json.dump({"mcpServers": {n: {"command": c[0], "args": c[1:]} for n, c in servers.items()}}, f)
        with open(os.path.join(work, "opencode.json"), "w") as f:  # OpenCode's project config
            json.dump({"$schema": "https://opencode.ai/config.json",
                       "mcp": {n: {"type": "local", "command": c, "enabled": True} for n, c in servers.items()}}, f)
        mcp_args = ["--mcp-config", mcp]
    print(f"MCP servers: {'none' if a.no_mcp else 'filesystem, memory, everything'}")
    if a.engine == "arf":
        claude = lambda msg: ([arf, "launch", "claude", "--model", a.model, "--", "-p", msg] + mcp_args, env)
        opencode = lambda msg: ([arf, "launch", "opencode", "--model", a.model, "--", "run", msg], env)
        stop_server = lambda: run([arf, "stop"], work, env)
        start_server = lambda: None  # `arf launch` starts it
    else:
        splash = Splash(work)
        claude = lambda msg: (["claude", "-p", msg] + mcp_args, splash.claude_env(env))
        opencode = lambda msg: (["opencode", "run", msg], splash.opencode_env(env))
        stop_server, start_server = splash.stop, splash.start

    # A first-ever message: earlier sessions' recorded prompts (replayed at start) and saved prompt
    # states (loaded at start) would make step 1 something else. Moved aside, restored in `finally`.
    aside, fresh = [], []
    if a.engine == "arf" and not a.keep_state:
        for d in ["anchors", "prefix"]:
            src = os.path.expanduser(f"~/.arf/{d}")
            if os.path.exists(src):
                dst = f"{src}.agent_e2e-{os.getpid()}"
                os.rename(src, dst)
                aside.append((dst, src))
            else:
                fresh.append(src)  # this run's alone: removed afterwards (prompt states are GBs)
    before = ""
    results = []  # (step, seconds, limit or None, ok, note)
    last = lambda out: out.splitlines()[-1][:60] if out else "no answer"
    step = lambda cmd_env: run(cmd_env[0], work, cmd_env[1])
    try:
        stop_server()
        start_server()

        # 1. cold, with `arf status` watched
        stop, seen = threading.Event(), []
        if a.engine == "arf":
            w = threading.Thread(target=watch_status, args=(arf, env, stop, seen), daemon=True)
            w.start()
        s, rc, out = step(claude("hey"))
        stop.set()
        results.append(("1 cold first message", s, None, rc == 0 and bool(out), last(out)))
        if a.engine == "arf":
            results.append(("  arf status during it", None, None, bool(seen), seen[-1] if seen else "showed no read"))

        # 2. warm
        s, rc, out = step(claude("hey"))
        results.append(("2 warm, same prompt", s, WARM_MAX_S, rc == 0 and bool(out) and s <= WARM_MAX_S, last(out)))

        # 3. two agents at once
        both = {}
        def go(name, cmd_env):
            both[name] = step(cmd_env)
        ts = [threading.Thread(target=go, args=("claude", claude("what is 2+2?"))),
              threading.Thread(target=go, args=("opencode", opencode("hey")))]
        [t.start() for t in ts]; [t.join() for t in ts]
        for name in ["claude", "opencode"]:
            s, rc, out = both[name]
            # Claude Code was asked what 2+2 is: the model's own answer has to be in what it printed.
            ok = rc == 0 and bool(out) and (name != "claude" or "4" in out)
            results.append((f"3 two agents: {name}", s, None, ok, last(out)))

        # 4. restart: the saved prompt state loads from disk (the server rewrites its log at start:
        # keep the first server's part)
        before = open(log).read() if a.engine == "arf" and os.path.exists(log) else ""
        stop_server()
        start_server()
        s, rc, out = step(claude("hey"))
        results.append(("4 after a restart", s, RESTART_MAX_S, rc == 0 and bool(out) and s <= RESTART_MAX_S, last(out)))

        # 5. a real task, one agent after the other, each on a fresh copy
        task = os.path.join(HERE, "agent_bench", "tasks", "calc")
        prompt = open(os.path.join(task, "PROMPT")).read().strip()
        # Each agent gets a fresh copy of the task at the project root (`work`, a git repository
        # with the large AGENTS.md), where a developer's files are. In a subfolder OpenCode took
        # the git root as its project and worked there: it answered DONE with the subfolder's
        # tests still failing (2026-10-07), a fault of the setup, not of the agent.
        for name, mk in [("claude", claude), ("opencode", opencode)]:
            repo = work
            shutil.copytree(task, repo, ignore=shutil.ignore_patterns("PROMPT", "__pycache__"),
                            dirs_exist_ok=True)
            cmd, cmd_env = mk(prompt)
            if name == "claude":
                cmd = cmd + ["--allowedTools", "Bash(python3:*)", "Read", "Edit"]
            try:
                s, rc, out = run(cmd, repo, cmd_env, timeout=REAL_TASK_MAX_S)
            except subprocess.TimeoutExpired:
                s, rc, out = float(REAL_TASK_MAX_S), 1, "the session did not end"
            fixed = subprocess.run([sys.executable, "test_calc.py"], cwd=repo, capture_output=True,
                                   text=True).stdout.strip().endswith("OK")
            results.append((f"5 real task: {name}", s, REAL_TASK_MAX_S, rc == 0 and fixed,
                            "tests pass" if fixed else f"tests FAIL · {last(out)}"))
    finally:
        stop_server()
        shutil.rmtree(work, ignore_errors=True)
        for dst, src in aside:  # this run's own recordings go, the user's come back
            shutil.rmtree(src, ignore_errors=True)
            os.rename(dst, src)
        for src in fresh:
            shutil.rmtree(src, ignore_errors=True)

    if a.engine == "splash":
        report(results, [])
        return
    text = before + (open(log).read() if os.path.exists(log) else "")
    progress = len(re.findall(r"^\[prefill\]", text, re.M))
    saved = len(re.findall(r"\[prefix-disk\] saved", text))
    loaded = len(re.findall(r"\[prefix-disk\] loaded", text))
    # `[safety]` is the server saying it took care after an earlier crash report, not an error.
    errors = [l for l in text.splitlines()
              if re.search(r"panicked|^Error|FATAL|fatal", l) and not l.startswith("[safety]")]
    results.append(("log: progress lines", None, None, progress > 0, f"{progress}"))
    results.append(("log: prompt state loaded after restart", None, None, loaded > 0, f"saved {saved}, loaded {loaded}"))
    results.append(("log: errors", None, None, not errors, errors[0][:60] if errors else "none"))
    # `arf launch claude` reads the safety check ahead, as background work (`arf warm-claude`):
    # its reads must have finished, or the first tool call reads ~30K tokens itself.
    if a.engine == "arf":
        warmed = re.findall(r"^\[done\] input (\d+) .*· background$", text, re.M)
        results.append(("log: safety check read ahead", None, None, bool(warmed),
                        f"{len(warmed)} read(s): {', '.join(warmed)} tokens" if warmed else "none finished"))
    dones = re.findall(r"\[done\] input (\d+) · cached (\d+) · output \d+ · TTFT ([\d.]+) s", text)

    report(results, dones[2:])  # past the two warm-up requests
    print(f"log: {log}")


def report(results, dones):
    print(f"\n{'step':42} {'time':>8} {'limit':>6}  result")
    for step, s, lim, ok, note in results:
        print(f"{step:42} {(f'{s:.1f} s' if s is not None else ''):>8} {(f'{lim} s' if lim else ''):>6}  {'PASS' if ok else 'FAIL'}  {note}")
    if dones:
        print("\nrequests the server saw (input · cached · first token):")
        for i, c, t in dones:
            print(f"  {int(i):>6} · {int(c):>6} · {float(t):6.1f} s")
    if not all(r[3] for r in results):
        sys.exit(1)


class Splash:
    """`splash serve` on :8000, and the environments that point Claude Code and OpenCode at it — the
    same variables `arf launch` sets, so the agents send exactly the same requests."""

    PORT = 8000

    def __init__(self, work):
        self.work, self.p, self.model = work, None, "incoai/Qwen3.8-27B-Splash"

    def start(self):
        import urllib.request
        logf = open(os.path.join(tempfile.gettempdir(), "agent_e2e_splash.log"), "a")
        self.p = subprocess.Popen(["splash", "serve", "--model", "incoai/Qwen3.8-27B-Splash", "--no-webui"],
                                  stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(1200):
            if self.p.poll() is not None:
                sys.exit("agent_e2e: splash exited while loading")
            try:
                v = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.PORT}/v1/models", timeout=2))
                self.model = v["data"][0]["id"]
                return
            except Exception:
                time.sleep(1)
        sys.exit("agent_e2e: splash did not come up")

    def stop(self):
        import signal
        if self.p and self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGINT)
            try:
                self.p.wait(timeout=60)
            except Exception:
                os.killpg(self.p.pid, signal.SIGKILL)
            time.sleep(3)

    def claude_env(self, env):
        e = {k: v for k, v in env.items() if k != "ANTHROPIC_API_KEY"}
        base, m = f"http://127.0.0.1:{self.PORT}", self.model
        e.update({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": "splash", "ANTHROPIC_MODEL": m,
                  "ANTHROPIC_SMALL_FAST_MODEL": m, "ANTHROPIC_DEFAULT_OPUS_MODEL": m,
                  "ANTHROPIC_DEFAULT_SONNET_MODEL": m, "ANTHROPIC_DEFAULT_HAIKU_MODEL": m,
                  "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
        return e

    def opencode_env(self, env):
        cfg = {"$schema": "https://opencode.ai/config.json", "model": f"splash/{self.model}",
               "small_model": f"splash/{self.model}",
               "provider": {"splash": {"npm": "@ai-sdk/openai-compatible", "name": "splash (local)",
                                       "options": {"baseURL": f"http://127.0.0.1:{self.PORT}/v1", "apiKey": "splash",
                                                   "timeout": False},
                                       "models": {self.model: {"name": self.model, "tool_call": True}}}}}
        return {**env, "OPENCODE_CONFIG_CONTENT": json.dumps(cfg)}


if __name__ == "__main__":
    main()
