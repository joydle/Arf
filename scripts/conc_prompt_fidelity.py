#!/usr/bin/env python3
"""Does each answer address ITS OWN prompt under concurrency?

Identity against a reference cannot gate concurrency on this model: at 16 streams the shipped path
does not reproduce itself (measured 2026-09-19). So judge the thing that matters instead.
Each prompt carries key terms a faithful reading repeats in its first ~60 tokens ("17", "23";
"haiku"; "Japan"). The score is the share of prompts whose answer contains ALL of its terms.
Crude on purpose: it catches "the product of 17 and 13" for "17 times 23", which is the failure
seen, and it needs no second model to judge.

usage: conc_prompt_fidelity.py N[,N...] [ENV=VALUE ...]      (one daemon start per N)
"""
import sys, os, json, time, signal, subprocess, threading, urllib.request
PORT = 8137
CASES = [
    ("Name the three primary colours and say one thing about each.", ["primary", "colo"]),
    ("Write a haiku about autumn rain.", ["haiku"]),
    ("What is 17 times 23? Show the working.", ["17", "23"]),
    ("Explain in two sentences why the sky is blue.", ["sky", "blue"]),
    ("List four planets and one fact about each.", ["four planets"]),
    ("Write a two-line rhyme about a cat.", ["cat"]),
    ("What is the capital of Japan, and what is it known for?", ["Japan"]),
    ("Give three tips for writing clear code.", ["clear", "code"]),
    ("Describe how a bicycle stays upright, briefly.", ["bicycle"]),
    ("What are the first five prime numbers and why is 1 not one of them?", ["prime"]),
    # Not "limerick": a thinking model repeats the word in its reasoning, a non-thinking one just
    # writes the limerick. qwen3-coder-30b scored 15 of 16 on EVERY path for exactly that.
    ("Write a limerick about a programmer.", ["programmer"]),
    ("Summarise the water cycle in three steps.", ["water cycle"]),
    ("Translate 'good morning, how are you?' into French and Spanish.", ["French", "Spanish"]),
    ("What is the difference between a list and a tuple in Python?", ["list", "tuple"]),
    ("Name two causes of the seasons on Earth.", ["season"]),
    ("Suggest a name for a bakery and explain it in one sentence.", ["bakery"]),
]
MODEL = os.environ.get("MPP_MODEL", "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf")
ARCH = os.environ.get("MPP_ARCH", "qwen3.8-27b")


def run(n, extra):
    env = {**os.environ, "QUANT": "q4ks", "ARF_SPEC_K": "0", **extra}
    p = subprocess.Popen(["./target/release/arf-serve", "--model", MODEL, "--arch", ARCH, "--quant", "q4ks", "--port", str(PORT)] + os.environ.get("MPP_EXTRA", "").split(),
                         env=env, stdout=open("/tmp/conc_fidelity.log", "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2); break
            except Exception:
                time.sleep(1)
        bursts = []
        for _ in range(int(os.environ.get("TRIALS", "1"))):
            bursts.append(burst(n))
        return bursts
    finally:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=40)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(4)


def burst(n):
    """One burst of n concurrent requests against the running daemon."""
    if True:
        out = [None] * n

        def ask(i):
            body = json.dumps({"model": "local", "temperature": 0, "max_tokens": 60,
                               "messages": [{"role": "user", "content": CASES[i][0]}]}).encode()
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1200))
            m = r["choices"][0]["message"]
            out[i] = (m.get("reasoning_content") or "") + (m.get("content") or "")
        ts = [threading.Thread(target=ask, args=(i,)) for i in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        return out


if __name__ == "__main__":
    counts = [int(x) for x in sys.argv[1].split(",")]
    extra = dict(kv.split("=", 1) for kv in sys.argv[2:])
    for n in counts:
        scores = []
        for out in run(n, extra):
            bad = [i for i in range(n) if not all(t.lower() in (out[i] or "").lower() for t in CASES[i][1])]
            scores.append(n - len(bad))
            # SAY WHICH ONE. A bare "15 of 16" cannot tell a corrupted stream from an answer that
            # simply never repeats the prompt's key term — and those need opposite responses.
            if os.environ.get("FIDELITY_SHOW") and bad:
                for i in bad:
                    print(f"    unfaithful [{i}] wants {CASES[i][1]} | {' '.join((out[i] or '').split())[:150]!r}", flush=True)
        clean = sum(1 for x in scores if x == n)
        print(f"{n:2} concurrent {extra}: per-burst faithful answers {scores} of {n} | fully clean bursts {clean}/{len(scores)}", flush=True)
