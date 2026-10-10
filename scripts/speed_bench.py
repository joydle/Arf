#!/usr/bin/env python3
"""SPEED-Bench coding prompts, the four measures other engines publish for Qwen3.8-27B — on THIS Mac.

Another engine's README number is a rumour until it is measured on the same machine and workload
(CLAUDE.md rule 9). The published shape: "selected SPEED-Bench coding prompts over HTTP, a 1,024-token
output limit, reasoning on", reported as

  decode   short prompt, one stream          tok/s = sum(tokens - 1) / sum(last - first token time)
  prefill  one ~32K-token prompt, cold        tok/s = usage.prompt_tokens / TTFT (max_tokens 1)
  cached   the same 32K request again         TTFT
  conc4    4 short prompts at once            tok/s = sum(tokens - 1) / (last token - first token, any)

"Selected" is not specified anywhere we could find, so the selection here is fixed and stated: the
`qualitative` split's `coding` rows that are single-turn and carry their text (some rows are
placeholders reading "FULL BENCHMARK DATA SHOULD BE FETCHED ..."), sorted by question_id, the first
SPEED_N (default 8); the 32K prompt is the first `throughput_32k` `code_completion` row by question_id.
Every request is greedy (temperature 0) with `reasoning_effort: "medium"` and
`chat_template_kwargs.enable_thinking: true`; an engine that ignores either is still measured.

The dataset (nvidia/SPEED-Bench, NVIDIA evaluation dataset license) is downloaded at run time into
SPEED_BENCH_DIR (default ~/.cache/arf-bench/speed-bench) and never committed. Reading the parquet files
needs pyarrow; without it the script runs itself under `uv run --with pyarrow`.

Engines and their builds are bench_engines.py's, with the context raised so a 32K prompt fits (and
llama.cpp's slots sharing one KV pool, `-kvu`). One engine resident at a time, the order rotating per
round (rule 3). Each round prints load and swap before and after; a round in which swap grew more than
512 MB is marked and left out of the medians (rule 3: this box drifts once swap grows).

Correctness gates speed (rule 4): every decode answer must be non-empty and reach at least 64 tokens,
and the cached replay's first token must equal the cold request's. A round that fails either is marked.

usage: speed_bench.py [ROUNDS] [ENGINES e.g. arf,splash,llamacpp,mlx]   env: SPEED_N, SPEED_OUT
       speed_bench.py --prompts      download if needed, print the selection, start no engine
"""
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import pyarrow.parquet as pq  # noqa: F401
except ImportError:
    if os.environ.get("SPEED_BENCH_REEXEC"):
        sys.exit("speed_bench: pyarrow is needed to read SPEED-Bench (pip install pyarrow, or install uv)")
    os.environ["SPEED_BENCH_REEXEC"] = "1"
    os.execvp("uv", ["uv", "run", "-q", "--with", "pyarrow", "python3", os.path.abspath(__file__)] + sys.argv[1:])

import bench_engines as be  # noqa: E402

DATA = "https://huggingface.co/datasets/nvidia/SPEED-Bench/resolve/main"
CACHE = os.environ.get("SPEED_BENCH_DIR", os.path.expanduser("~/.cache/arf-bench/speed-bench"))
PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"
N = int(os.environ.get("SPEED_N", "8"))
MAX_TOKENS = 1024
OUT = os.environ.get("SPEED_OUT", "/tmp/speed_bench.json")

# Context for a ~32K prompt plus its answer; llama.cpp's 4 slots share one pool (-kvu).
CTX = "49152"
ENGINES = dict(be.ENGINES)
_p, _m, _argv, _env, _d = ENGINES["arf"]
ENGINES["arf"] = (_p, _m, _argv + ["--max-context", CTX], _env, _d)
_p, _m, _argv, _env, _d = ENGINES["llamacpp"]
_argv = list(_argv)
_argv[_argv.index("-c") + 1] = CTX
ENGINES["llamacpp"] = (_p, _m, _argv + ["-kvu"], _env, _d + ", unified KV")


def fetch(split):
    path = os.path.join(CACHE, f"{split}.parquet")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        print(f"downloading SPEED-Bench {split} ...", flush=True)
        urllib.request.urlretrieve(f"{DATA}/{split}/test-00000-of-00001.parquet", path + ".part")
        os.rename(path + ".part", path)
    return pq.read_table(path).to_pylist()


def prompts():
    rows = [r for r in fetch("qualitative") if r["category"] == "coding" and not r["multiturn"]
            and not r["turns"][0].startswith(PLACEHOLDER)]
    short = sorted(rows, key=lambda r: r["question_id"])[:N]
    longs = [r for r in fetch("throughput_32k") if r["sub_category"] == "code_completion"
             and not r["turns"][0].startswith(PLACEHOLDER)]
    long = min(longs, key=lambda r: r["question_id"])
    ids = [r["question_id"] for r in short] + [long["question_id"]]
    return [r["turns"][0] for r in short], long["turns"][0], ids


def ask(port, model, prompt, max_tokens, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    b = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}, "reasoning_effort": "medium",
         "chat_template_kwargs": {"enable_thinking": True}}
    if model:
        b["model"] = model
    t0 = time.time()
    first = last = None
    usage, pieces, text = None, 0, []
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(b).encode(),
                                 {"Content-Type": "application/json"})
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        j = json.loads(line[5:])
        if j.get("usage"):
            usage = j["usage"]
        for c in j.get("choices", []):
            d = c.get("delta") or {}
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
            if piece:
                pieces += 1
                text.append(piece)
                now = time.time()
                first = first or now
                last = now
    n = usage["completion_tokens"] if usage and usage.get("completion_tokens") else pieces
    return {"tokens": n, "counted": bool(usage), "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "t0": t0, "first": first, "last": last, "ttft": (first or time.time()) - t0,
            "text": "".join(text)}


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    try:
        return float(s.split("used =")[1].split("M")[0])
    except (IndexError, ValueError):
        return float("nan")


def run(name, rnd, short, long):
    port, model, *_ = ENGINES[name]
    be.ENGINES[name] = ENGINES[name]  # be.up reads its own table
    s0, l0 = swap_mb(), os.getloadavg()[0]
    p = be.up(name, rnd)
    try:
        o = {"problems": []}
        rs = [ask(port, model, pr, MAX_TOKENS) for pr in short]
        for i, r in enumerate(rs):
            if not r["text"].strip() or r["tokens"] < 64:
                o["problems"].append(f"decode prompt {i}: {r['tokens']} tokens")
        o["decode"] = sum(r["tokens"] - 1 for r in rs) / sum((r["last"] - r["first"]) for r in rs if r["first"])
        o["decode_counted"] = all(r["counted"] for r in rs)
        o["answers"] = [hashlib.sha256(r["text"].encode()).hexdigest()[:12] for r in rs]
        tag = f"Request speed-{name}-{rnd}-{time.time_ns()}."  # a unique first line: no cache hit
        cold = ask(port, model, long, 1, system=tag)
        o["prompt_tokens"] = cold["prompt_tokens"]
        o["prefill"] = (cold["prompt_tokens"] or float("nan")) / cold["ttft"]
        o["prefill_ttft"] = cold["ttft"]
        warm = ask(port, model, long, 1, system=tag)
        o["cached_ttft"] = warm["ttft"]
        if warm["text"] != cold["text"]:
            o["problems"].append(f"cached replay's first token {warm['text']!r} != cold {cold['text']!r}")
        res = [None] * 4
        th = [threading.Thread(target=lambda i=i: res.__setitem__(i, ask(port, model, short[i % len(short)],
                                                                         MAX_TOKENS))) for i in range(4)]
        [x.start() for x in th]
        [x.join() for x in th]
        span = max(r["last"] for r in res) - min(r["first"] for r in res)
        o["conc4"] = sum(r["tokens"] - 1 for r in res) / span
    finally:
        be.down(p)
    o["swap_grew_mb"] = swap_mb() - s0
    o["load"] = [l0, os.getloadavg()[0]]
    return o


def main():
    if sys.argv[1:2] == ["--prompts"]:
        short, long, ids = prompts()
        for q, pr in zip(ids, short):
            print(f"{q}  {len(pr):6} chars  {pr[:70]!r}")
        print(f"{ids[-1]}  {len(long):6} chars  (32K code completion)")
        return
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["arf", "splash", "llamacpp", "mlx"]
    short, long, ids = prompts()
    print(f"SPEED-Bench: {len(short)} coding prompts + one 32K code-completion prompt "
          f"({len(long)} chars); question ids {ids}", flush=True)
    print(f"start: load {os.getloadavg()[0]:.1f}, swap {swap_mb():.0f} MB", flush=True)
    acc = {n: [] for n in names}
    for rnd in range(rounds):
        k = rnd % len(names)
        for n in names[k:] + names[:k]:
            o = run(n, rnd, short, long)
            acc[n].append(o)
            flag = " SWAP" if o["swap_grew_mb"] > 512 else ""
            flag += " PROBLEMS: " + "; ".join(o["problems"]) if o["problems"] else ""
            print(f"round {rnd} {n:9} decode {o['decode']:6.1f} | prefill {o['prefill']:6.0f} tok/s "
                  f"({o['prompt_tokens']} tok) | cached TTFT {o['cached_ttft'] * 1000:6.0f} ms | conc4 "
                  f"{o['conc4']:6.1f} tok/s  [load {o['load'][0]:.1f}->{o['load'][1]:.1f}, swap "
                  f"{o['swap_grew_mb']:+.0f} MB]{flag}", flush=True)
    json.dump({"question_ids": ids, "engines": {n: ENGINES[n][4] for n in names}, "rounds": acc},
              open(OUT, "w"), indent=1)
    good = {n: [o for o in acc[n] if o["swap_grew_mb"] <= 512 and not o["problems"]] for n in names}
    print(f"\nmedians over clean rounds (raw: {OUT})\n")
    print("| engine | build | clean rounds | decode | prefill 32K | cached TTFT 32K | 4 at once |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for n in names:
        g = good[n]
        if not g:
            print(f"| {n} | {ENGINES[n][4]} | 0 of {len(acc[n])} | — | — | — | — |")
            continue
        med = lambda k: st.median(o[k] for o in g)
        mark = "" if all(o["decode_counted"] for o in g) else " ~"
        print(f"| {n} | {ENGINES[n][4]} | {len(g)} of {len(acc[n])} | {med('decode'):.1f}{mark} | "
              f"{med('prefill'):.0f} tok/s | {med('cached_ttft') * 1000:.0f} ms | {med('conc4'):.1f} |")


if __name__ == "__main__":
    main()
