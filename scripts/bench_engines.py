#!/usr/bin/env python3
"""Five engines, ONE client, the same Qwen3.8-27B workloads — the README's comparison table.

Every engine is driven through its OpenAI-compatible `/v1/chat/completions` with streaming and the same
bodies, one engine resident at a time, ROUNDS rounds with the engine order rotating (rule 3: never compare
against a number from another session). Each engine runs its OWN default ~4-bit build of Qwen3.8-27B and
its own default speculation — that is what its users get, and the table says which is which.

Workloads, per engine per round:
  prefill   TTFT of one ~2,450-token prompt (a unique first line: no prefix-cache hit), max_tokens 1
  greedy    6 prompts x 512 tokens, temperature 0; decode tok/s = sum(tokens - 1) / sum(last - first token)
  sampled   the same 6 prompts, temperature 0.7 / top_p 0.95 / top_k 20, fixed seeds
  conc4     4 greedy requests at once x 256 tokens; aggregate = all tokens / wall of the burst

Token counts are the server's `usage.completion_tokens` (stream_options.include_usage); a server that sends
no usage is counted by streamed pieces and marked `~` (a piece may be more than one token).

usage: bench_engines.py [ROUNDS] [ENGINES e.g. arf,splash,llamacpp,ollama,mlx]    -> JSON + a markdown table
"""
import json
import os
import signal
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conc_prompt_fidelity import CASES  # noqa: E402

Q4KM = f"{ROOT}/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf"
ENGINES = {
    # name: (port, model field, argv, env extras, how it is built)
    "arf": (8233, "local", ["./target/release/arf-serve", "--model",
                            "models/qwen3.8-27b-arf/model.gguf", "--arch", "qwen3.8-27b",
                            "--quant", "q4ks", "--port", "8233", "--draft", "models/qwen3.8-27b-dflash2"],
            {"QUANT": "q4ks"}, "group-64 4-bit + DFlash-2 draft"),
    "splash": (8000, "", ["splash", "serve", "--model", "incoai/Qwen3.8-27B-Splash", "--no-webui"], {},
               "its 4-bit package + DFlash-2 draft"),
    "llamacpp": (8081, "q4km", ["llama-server", "-m", Q4KM, "-ngl", "99", "-fa", "on", "-c", "32768",
                                "--jinja", "--port", "8081", "--alias", "q4km", "-np", "4"], {},
                 "Q4_K_M, no speculation"),
    "ollama": (11434, "qwen38-27b-q4km", ["ollama", "serve"], {"OLLAMA_HOST": "127.0.0.1:11434",
                                                                "OLLAMA_NUM_PARALLEL": "4",
                                                                "OLLAMA_FLASH_ATTENTION": "1"},
               "the same Q4_K_M GGUF, no speculation"),
    "mlx": (8082, "mlx-community/Qwen3.8-27B-4bit", ["mlx_lm.server", "--model", "mlx-community/Qwen3.8-27B-4bit",
                                                     "--port", "8082"], {}, "mlx-community 4-bit, no speculation"),
}
PROMPTS = [c[0] for c in CASES[:6]]
# 8,000 characters of a public doc (a unique first line keeps it a cache miss); until 2026-10-05 the text
# came from a doc that is no longer in this repository, so TTFTs from before then are not comparable.
LONG = open(f"{ROOT}/docs/ENGINE.md").read()[:8000]


def ask(port, model, prompt, max_tokens, sampled=False, seed=0, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    b = {"messages": msgs, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    b.update({"temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": seed} if sampled else {"temperature": 0})
    if model:
        b["model"] = model
    t0 = time.time()
    first = last = None
    usage = None
    pieces = 0
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                                      json.dumps(b).encode(), {"Content-Type": "application/json"}),
                               timeout=3600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        j = json.loads(line[5:])
        if j.get("usage"):
            usage = j["usage"]
        for c in j.get("choices", []):
            d = c.get("delta") or {}
            if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                pieces += 1
                now = time.time()
                first = first or now
                last = now
    n = usage["completion_tokens"] if usage and usage.get("completion_tokens") else pieces
    return {"tokens": n, "counted": bool(usage), "ttft": (first or time.time()) - t0,
            "decode_s": (last - first) if first and last else 0.0, "wall": time.time() - t0}


def up(name, rnd):
    port, model, argv, extra, _ = ENGINES[name]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARF_")}
    env.update(extra)
    p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=open(f"/tmp/bench_{name}_r{rnd}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(1200):
        if p.poll() is not None:
            raise RuntimeError(f"{name} exited during load — /tmp/bench_{name}_r{rnd}.log")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            break
        except Exception:
            time.sleep(1)
    ask(port, model, "Say hi.", 8)  # warm (discarded); for ollama this also loads the model
    return p


def down(p):
    try:
        os.killpg(p.pid, signal.SIGINT)
        p.wait(timeout=60)
    except Exception:
        os.killpg(p.pid, signal.SIGKILL)
    time.sleep(3)


def run(name, rnd):
    port, model, *_ = ENGINES[name]
    p = up(name, rnd)
    try:
        out = {}
        pf = ask(port, model, "Summarize this document.\n\n" + LONG, 1, system=f"Request bench-{name}-{rnd}.")
        out["prefill_ttft"] = pf["ttft"]
        for mode in ("greedy", "sampled"):
            rs = [ask(port, model, pr, 512, sampled=mode == "sampled", seed=100 + i) for i, pr in enumerate(PROMPTS)]
            out[mode] = sum(r["tokens"] - 1 for r in rs) / sum(r["decode_s"] for r in rs)
            out[mode + "_counted"] = all(r["counted"] for r in rs)
        res = [None] * 4
        t = time.time()
        th = [threading.Thread(target=lambda i=i: res.__setitem__(i, ask(port, model, PROMPTS[i], 256)))
              for i in range(4)]
        [x.start() for x in th]
        [x.join() for x in th]
        out["conc4"] = sum(r["tokens"] for r in res) / (time.time() - t)
        return out
    finally:
        down(p)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ENGINES)
    load = os.getloadavg()[0]
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    print(f"start: load {load:.1f}, swap {swap}", flush=True)
    acc = {n: [] for n in names}
    for rnd in range(rounds):
        k = rnd % len(names)
        for n in names[k:] + names[:k]:
            o = run(n, rnd)
            acc[n].append(o)
            print(f"round {rnd} {n:9} prefill {o['prefill_ttft']:6.2f} s | greedy {o['greedy']:6.1f} | "
                  f"sampled {o['sampled']:6.1f} | conc4 {o['conc4']:6.1f} tok/s  [load {os.getloadavg()[0]:.1f}]",
                  flush=True)
    json.dump(acc, open("/tmp/bench_engines.json", "w"), indent=1)
    med = lambda n, k: st.median(o[k] for o in acc[n])
    print("\n| engine | build | prefill TTFT, 2.45K tokens | greedy decode | sampled decode | 4 at once |")
    print("|---|---|---:|---:|---:|---:|")
    for n in names:
        mark = lambda k: "" if all(o[k + "_counted"] for o in acc[n]) else " ~"
        print(f"| {n} | {ENGINES[n][4]} | {med(n, 'prefill_ttft'):.2f} s | {med(n, 'greedy'):.1f}{mark('greedy')} | "
              f"{med(n, 'sampled'):.1f}{mark('sampled')} | {med(n, 'conc4'):.1f} |")


if __name__ == "__main__":
    main()
