#!/usr/bin/env python3
"""Did the engine READ the prompt? Facts planted at three depths of a long passage must come back.

A prefill that drops, reorders or corrupts prompt rows still produces fluent text at full speed,
so a speed number for a prefill change means nothing without this. Three planted facts (start,
middle, end), one question each, temperature 0; the answer must contain the planted string.
Also prints time-to-first-token per request (wall of the request / its short answer).

RECALL_BUSY=N starts N other chats first and asks while they are MID-ANSWER — the mixed path: a
long prompt prefilled next to live decoders. Their answers are checked too (each must still
address its own prompt), because a prefill that tramples a neighbour's state breaks THEM, not itself.

usage: prefill_recall.py [ENV=VALUE ...]   env: MPP_MODEL, MPP_ARCH, RECALL_REPS (12), RECALL_BUSY (0), RECALL_FLAGS (extra server flags)
"""
import json, os, signal, subprocess, sys, threading, time, urllib.request
PORT = 8171
MODEL = os.environ.get("MPP_MODEL", "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf")
ARCH = os.environ.get("MPP_ARCH", "qwen3.8-27b")
FILL = ("The harbour town woke slowly that morning. Fishing boats rocked against the quay while gulls "
        "argued over scraps, and the baker on the corner propped his door open so the smell of bread "
        "drifted down the hill toward the water. ")
BUSY = [("Explain in detail how a bicycle stays upright, step by step.", "bicycle"),
        ("Write a long, detailed description of the water cycle.", "water"),
        ("Describe the history of the printing press in detail.", "print"),
        ("Explain how photosynthesis works, in detail.", "photosynth")]
FACTS = [("The harbour master was a woman named Oswin Trelawney. ", "What was the harbour master's name?", "Trelawney"),
         ("The oldest boat in the harbour was called the Grey Heron. ", "What was the oldest boat called?", "Heron"),
         ("That year the lighthouse was painted green and white. ", "What colours was the lighthouse painted?", "green")]


def main():
    extra = dict(kv.split("=", 1) for kv in sys.argv[1:])
    reps = int(os.environ.get("RECALL_REPS", "12"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARF_")}
    env.update(extra); env["QUANT"] = "q4ks"
    p = subprocess.Popen(["./target/release/arf-serve", "--model", MODEL, "--arch", ARCH, "--quant", "q4ks",
                          "--port", str(PORT)] + os.environ.get("RECALL_FLAGS", "").split(), env=env, stdout=open("/tmp/prefill_recall.log", "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    ok = 0
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2); break
            except Exception:
                time.sleep(1)
        third = reps // 3
        passage = FACTS[0][0] + FILL * third + FACTS[1][0] + FILL * third + FILL * (reps - 2 * third) + FACTS[2][0]
        busy_n = int(os.environ.get("RECALL_BUSY", "0"))
        busy_ok, busy_total = [0], [0]

        def chat(prompt, want):
            b = json.dumps({"model": "local", "temperature": 0, "max_tokens": 320,
                            "messages": [{"role": "user", "content": prompt}]}).encode()
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=1800))
            m = r["choices"][0]["message"]
            t = (m.get("reasoning_content") or "") + (m.get("content") or "")
            ascii_share = sum(c.isascii() for c in t) / max(1, len(t))
            busy_total[0] += 1
            busy_ok[0] += want.lower() in t.lower() and ascii_share > 0.95

        for i, (_, q, want) in enumerate(FACTS):
            ths = [threading.Thread(target=chat, args=BUSY[j % len(BUSY)]) for j in range(busy_n)]
            [t.start() for t in ths]
            if ths:
                time.sleep(4)  # let them get past their own prefill and into decode
            # a distinct opening per request so no prefix cache can serve the passage
            body = json.dumps({"model": "local", "temperature": 0, "max_tokens": 200, "messages": [
                {"role": "user", "content": f"Text {i + 1}. {passage}\n\nQuestion: {q} Answer in one short sentence."}]}).encode()
            t = time.time()
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
            dt = time.time() - t
            m = r["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + (m.get("content") or "")
            hit = want.lower() in text.lower()
            ok += hit
            u = r["usage"]
            [t.join() for t in ths]
            print(f"  [{'OK ' if hit else 'MISS'}] {u['prompt_tokens']} prompt tokens, {u['completion_tokens']} out, {dt:6.1f} s | "
                  f"{' '.join(text.split())[-110:]!r}", flush=True)
    finally:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=40)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(5)
    print(f"{extra or 'defaults'}: recalled {ok} of {len(FACTS)}"
          + (f"; neighbours still faithful {busy_ok[0]} of {busy_total[0]}" if busy_total[0] else ""))
    return 0 if ok == len(FACTS) and busy_ok[0] == busy_total[0] else 1


if __name__ == "__main__":
    sys.exit(main())
