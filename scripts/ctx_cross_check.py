#!/usr/bin/env python3
"""Does the server survive a sequence CROSSING a context boundary while decoding?

Found 2026-09-20: after one request crossed 2,048 tokens mid-decode, every later request came
back as `!!!!` (NaN from its first token) — fluent-speed garbage, invisible to tok/s. Request 1
itself looked fine, so only a FOLLOW-UP request can see it. This sends a prompt that ends just
under the boundary, lets the answer cross it, then asks a short factual question and a long one.

usage: ctx_cross_check.py [BOUNDARY=2048] [ENV=VALUE ...]     env: MPP_MODEL, MPP_ARCH, CROSS_FLAGS
"""
import json, os, signal, subprocess, sys, time, urllib.request
PORT = 8181
MODEL = os.environ.get("MPP_MODEL", "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf")
ARCH = os.environ.get("MPP_ARCH", "qwen3.8-27b")
FILL = ("The harbour town woke slowly that morning. Fishing boats rocked against the quay while gulls "
        "argued over scraps, and the baker on the corner propped his door open so the smell of bread "
        "drifted down the hill toward the water. ")  # ~45 tokens


def ask(prompt, n):
    body = json.dumps({"model": "local", "temperature": 0, "max_tokens": n,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or ""), r["usage"], time.time() - t


def sane(text):
    return len(text) > 0 and sum(c.isascii() and (c.isalnum() or c in " .,'\n") for c in text) / len(text) > 0.8


def main():
    args = [a for a in sys.argv[1:] if "=" not in a]
    boundary = int(args[0]) if args else 2048
    extra = dict(kv.split("=", 1) for kv in sys.argv[1:] if "=" in kv)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARF_")}
    env.update(extra); env["QUANT"] = "q4ks"
    p = subprocess.Popen(["./target/release/arf-serve", "--model", MODEL, "--arch", ARCH, "--quant", "q4ks",
                          "--port", str(PORT)] + os.environ.get("CROSS_FLAGS", "").split(), env=env,
                         stdout=open("/tmp/ctx_cross.log", "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    ok = True
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2); break
            except Exception:
                time.sleep(1)
        reps = (boundary - 60) // 45
        long_prompt = "Story. " + FILL * reps + "\nContinue this story for several paragraphs."
        t, u, dt = ask(long_prompt, 160)
        end = u["prompt_tokens"] + u["completion_tokens"]
        crossed = u["prompt_tokens"] < boundary <= end
        print(f"1 crossing : {u['prompt_tokens']} + {u['completion_tokens']} tokens -> ends at {end} "
              f"({'CROSSED' if crossed else 'DID NOT CROSS'} {boundary}) {dt:5.1f}s sane={sane(t)} | {' '.join(t.split())[-70:]!r}")
        ok &= crossed and sane(t)
        t, u, dt = ask("What is the capital of Japan? Answer in one word.", 60)
        hit = "tokyo" in t.lower()
        print(f"2 short    : {'OK  ' if hit else 'FAIL'} {dt:5.1f}s | {' '.join(t.split())[-70:]!r}")
        ok &= hit
        t, u, dt = ask("Tale. " + FILL * 12 + "\nIn one sentence: what does the baker do?", 120)
        hit = "door" in t.lower() or "bread" in t.lower()
        print(f"3 long     : {'OK  ' if hit else 'FAIL'} {dt:5.1f}s | {' '.join(t.split())[-70:]!r}")
        ok &= hit
    except Exception as e:  # a server that DIED is the finding, not a harness error
        print(f"REQUEST FAILED: {e} | server alive: {p.poll() is None}")
        ok = False
    finally:
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=40)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(5)
    print(f"{extra or 'defaults'} boundary {boundary}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
