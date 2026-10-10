#!/usr/bin/env python3
"""What a user of a coding agent feels: wall time from prompt to PASSING tests, per engine x agent x task.

Every engine serves the same model name (`qwen3.8-27b`) on the same port (8080), so each agent's isolated
config is identical across engines; one engine resident at a time, the engine order rotating per round.
Each run gets a fresh copy of its task (scripts/agent_bench/tasks/<task>: a failing test_*.py + PROMPT) and
PASSES only if the task's test prints OK afterwards — the agent saying DONE counts for nothing.

Agents run isolated (their own config dirs under AGENT_HOME, default $TMPDIR/arf-agents: Claude Code
`cc/config`, Pi `pi/agent`, OpenCode via OPENCODE_CONFIG_CONTENT) — never the invoking user's own configs.

usage: run.py [ROUNDS] [ENGINES arf,splash,llamacpp] [AGENTS claude,pi,opencode] [TASKS calc,inventory,lru]

AGENT_BENCH_SERVE=<path to arf-serve> runs the `arf` engine from another build, e.g. a published release's
extracted tarball (docs/RELEASING.md): `AGENT_BENCH_SERVE=<folder>/arf-serve run.py 1 arf claude,opencode`.

AGENT_CONCURRENCY=N runs N agent x task runs at once against the one resident engine (default 1, one at a
time). Issue #12's measurement is `AGENT_CONCURRENCY=4 run.py 1 arf`: afterwards the arf log is searched for
"while seq ... still holds" (a snapshot evicted while the request that took it was still running) and the
count is printed per round.
"""
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AH = os.environ.get("AGENT_HOME", os.path.join(tempfile.gettempdir(), "arf-agents"))
PORT = 8080
# The arf-serve to test: this checkout's build, or a release tarball's (AGENT_BENCH_SERVE=<folder>/arf-serve).
ARF_SERVE = os.environ.get("AGENT_BENCH_SERVE", "./target/release/arf-serve")
ENGINES = {
    "arf": ([ARF_SERVE, "--model", "models/qwen3.8-27b-arf/model.gguf",
             "--arch", "qwen3.8-27b", "--quant", "q4ks", "--port", str(PORT), "--draft", "models/qwen3.8-27b-dflash2",
             "--max-context", "65536"], {"QUANT": "q4ks"}),
    # The same server with EARLY ANCHORS (ARF_EARLY_ANCHOR=1, opt-in): declared here because up() strips
    # every ARF_* variable from the parent environment.
    "arf+early": ([ARF_SERVE, "--model", "models/qwen3.8-27b-arf/model.gguf",
                   "--arch", "qwen3.8-27b", "--quant", "q4ks", "--port", str(PORT), "--draft", "models/qwen3.8-27b-dflash2",
                   "--max-context", "65536"], {"QUANT": "q4ks", "ARF_EARLY_ANCHOR": "1"}),
    "splash": (["splash", "serve", "--model", "incoai/Qwen3.8-27B-Splash", "--no-webui", "--port", str(PORT),
                "--served-model-name", "qwen3.8-27b"], {}),
    "llamacpp": (["llama-server", "-m", f"{ROOT}/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf", "-ngl", "99", "-fa", "on",
                  "-c", "65536", "--jinja", "--port", str(PORT), "--alias", "qwen3.8-27b"], {}),
}
PY = "Bash(python3:*)"


def agent_cmd(agent, work, prompt):
    if agent == "claude":
        c = f"{AH}/cc/config"
        env = {"HOME": c, "PATH": os.environ["PATH"], "TERM": "dumb", "CLAUDE_CONFIG_DIR": c,
               "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}", "ANTHROPIC_API_KEY": "local",
               "ANTHROPIC_MODEL": "qwen3.8-27b", "ANTHROPIC_SMALL_FAST_MODEL": "qwen3.8-27b",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1"}
        return ([os.path.expanduser("~/.local/bin/claude"), "-p", "--model", "qwen3.8-27b", "--strict-mcp-config",
                 "--allowedTools", PY, "Read", "Edit", "--output-format", "stream-json", "--verbose", prompt], env)
    if agent == "pi":
        # Pi's isolated config needs the `arf` provider (its docs/models.md format). Without it every
        # run failed in under a second, "Unknown provider", and counted as a failed task (2026-10-05).
        os.makedirs(f"{AH}/pi/agent", exist_ok=True)
        json.dump({"providers": {"arf": {
            "baseUrl": f"http://127.0.0.1:{PORT}/v1", "api": "openai-completions", "apiKey": "local",
            "compat": {"supportsDeveloperRole": False},
            "models": [{"id": "qwen3.8-27b", "reasoning": True}]}}},
            open(f"{AH}/pi/agent/models.json", "w"), indent=1)
        env = dict(os.environ, PI_CODING_AGENT_DIR=f"{AH}/pi/agent", PI_OFFLINE="1", PI_TELEMETRY="0")
        return (["pi", "--provider", "arf", "--model", "qwen3.8-27b", "--thinking", "medium", "--tools",
                 "read,edit,bash", "--no-session", "-nc", "-ns", "-np", "-ne", "--mode", "json", "-p", prompt], env)
    if agent == "opencode":
        cfg = {"$schema": "https://opencode.ai/config.json", "enabled_providers": ["arf"], "model": "arf/qwen3.8-27b",
               "small_model": "arf/qwen3.8-27b", "autoupdate": False, "share": "disabled", "snapshot": False,
               "provider": {"arf": {"npm": "@ai-sdk/openai-compatible", "name": "local",
                                    "options": {"baseURL": f"http://127.0.0.1:{PORT}/v1", "apiKey": "local",
                                                "timeout": False},
                                    "models": {"qwen3.8-27b": {"name": "Qwen3.8-27B", "tool_call": True,
                                                               "reasoning": True, "temperature": False,
                                                               "limit": {"context": 65536, "output": 16384}}}}},
               "permission": {"skill": "deny", "webfetch": "deny", "task": "deny", "todowrite": "deny"},
               "agent": {"title": {"disable": True}}}
        o = f"{AH}/oc"
        env = dict(os.environ, OPENCODE_CONFIG_CONTENT=json.dumps(cfg), XDG_CONFIG_HOME=f"{o}/xdg/config",
                   XDG_DATA_HOME=f"{o}/xdg/data", XDG_CACHE_HOME=f"{o}/xdg/cache", XDG_STATE_HOME=f"{o}/xdg/state",
                   OPENCODE_DB=f"{o}/oc.db", OPENCODE_DISABLE_CLAUDE_CODE="1", OPENCODE_DISABLE_EXTERNAL_SKILLS="1",
                   OPENCODE_DISABLE_MODELS_FETCH="1", OPENCODE_DISABLE_AUTOUPDATE="1", OPENCODE_DISABLE_SHARE="1",
                   OPENCODE_DISABLE_DEFAULT_PLUGINS="1")
        return ([os.path.expanduser("~/.opencode/bin/opencode"), "run", "--dir", work, "-m", "arf/qwen3.8-27b",
                 "--format", "json", prompt], env)
    raise ValueError(agent)


def up(engine, tag):
    argv, extra = ENGINES[engine]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARF_")}
    env.update(extra)
    p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=open(f"/tmp/agentbench_{engine}_{tag}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(1200):
        if p.poll() is not None:
            raise RuntimeError(f"{engine} exited during load")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"{engine} never came up")


def down(p):
    try:
        os.killpg(p.pid, signal.SIGINT)
        p.wait(timeout=60)
    except Exception:
        os.killpg(p.pid, signal.SIGKILL)
    time.sleep(3)


def one(engine, agent, task, rnd):
    work = f"/tmp/agentbench_work/{engine}_{agent}_{task}_{rnd}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(f"{HERE}/tasks/{task}", work, ignore=shutil.ignore_patterns("PROMPT", "__pycache__"))
    prompt = open(f"{HERE}/tasks/{task}/PROMPT").read().strip()
    argv, env = agent_cmd(agent, work, prompt)
    t = time.time()
    try:
        r = subprocess.run(argv, cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=int(os.environ.get("AGENT_TIMEOUT", "900")))
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    wall = time.time() - t
    test = next(f for f in os.listdir(work) if f.startswith("test_") and f.endswith(".py"))
    ok = subprocess.run(["python3", test], cwd=work, capture_output=True, text=True).stdout.strip().endswith("OK")
    tools = out.count('"tool_use"') + out.count('"toolName"') + out.count('"type":"tool"')
    return {"engine": engine, "agent": agent, "task": task, "round": rnd, "wall": round(wall, 1), "pass": ok,
            "tool_events": tools, "timeout": out == "TIMEOUT"}


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    engines = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ENGINES)
    agents = sys.argv[3].split(",") if len(sys.argv) > 3 else ["claude", "pi", "opencode"]
    tasks = sys.argv[4].split(",") if len(sys.argv) > 4 else ["calc", "inventory", "lru"]
    print(f"start: load {os.getloadavg()[0]:.1f}", flush=True)
    res = []
    for rnd in range(rounds):
        k = rnd % len(engines)
        for e in engines[k:] + engines[:k]:
            p = up(e, rnd)
            try:
                runs = [(a, tk) for a in agents for tk in tasks]
                with concurrent.futures.ThreadPoolExecutor(int(os.environ.get("AGENT_CONCURRENCY", "1"))) as ex:
                    for r in ex.map(lambda at: one(e, at[0], at[1], rnd), runs):
                        res.append(r)
                        print(json.dumps(r), flush=True)
            finally:
                down(p)
            if e.startswith("arf"):
                log = open(f"/tmp/agentbench_{e}_{rnd}.log", errors="replace").read()
                print(f"round {rnd}: {log.count('still holds it')} snapshot(s) evicted while held, "
                      f"{log.count('resumes from')} 'resumes from' line(s)", flush=True)
    json.dump(res, open("/tmp/agent_bench.json", "w"), indent=1)
    print("\n| engine | agent | " + " | ".join(tasks) + " | all passed, total |")
    print("|---|---|" + "---:|" * (len(tasks) + 1))
    for e in engines:
        for a in agents:
            cells, tot, allok = [], 0.0, True
            for tk in tasks:
                rs = [r for r in res if (r["engine"], r["agent"], r["task"]) == (e, a, tk)]
                w = sorted(r["wall"] for r in rs)[len(rs) // 2]
                ok = all(r["pass"] for r in rs)
                allok &= ok
                tot += w
                cells.append(f"{w:.0f} s" + ("" if ok else " ✗"))
            print(f"| {e} | {a} | " + " | ".join(cells) + f" | {'yes' if allok else 'NO'}, {tot:.0f} s |")


if __name__ == "__main__":
    main()
