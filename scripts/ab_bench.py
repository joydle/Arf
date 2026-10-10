#!/usr/bin/env python3
"""ab_bench.py — streamed decode-rate probe for kernel A/Bs.

Modes:
  ab_bench.py PORT           single stream, p50 inter-token gap -> tok/s   (b=1)
  ab_bench.py PORT warm      one throwaway run to warm caches, prints nothing
  ab_bench.py PORT batch3    3 CONCURRENT streams, per-seq p50 gap          (b=3)
  ab_bench.py PORT batchN    N concurrent streams

MEASURES INTER-TOKEN GAP, NOT TOTAL WALL TIME. Total time folds in prefill and
TTFT, which move independently of the decode kernels an A/B is usually about.
p50 (not mean) because a single scheduler hiccup skews a mean and this box has them.

TRAPS THIS ENCODES (each one cost a retracted measurement earlier):
  L179  every concurrent stream gets a DISTINCT prompt. Streams sharing a
        tokeniser prefix batch differently and the "b=3" number is then a lie.
  L182  never pool gaps across arms with different b — population counts that
        span b=1 and b=2 report a number belonging to neither.
  L176  never `tail -1` a stream's gaps; the last gap includes teardown.
"""
import json, sys, time, threading, statistics, urllib.request

def stream_gaps(port, prompt, maxtok):
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maxtok, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    gaps, last = [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            # L224 — count BOTH `content` and `reasoning_content`. llama-server emits a
            # thinking model's chain-of-thought under `reasoning_content` and only the final
            # answer under `content`, so counting `content` alone sees almost no tokens on a
            # reasoning model and reports a wildly wrong rate. Both are decoded tokens and both
            # cost a forward pass, so both belong in an inter-token gap.
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if last is not None:
                    gaps.append((now - last) * 1000)
                last = now
    # drop the final gap: it is contaminated by stream teardown (L176).
    return gaps[:-1] if len(gaps) > 2 else gaps

def main():
    port = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "single"
    if mode == "warm":
        stream_gaps(port, "Count to twenty", 24)
        return
    if mode.startswith("batch"):
        n = int(mode[5:] or 3)
        res = {}
        def run(i):
            # DISTINCT prompt per stream (L179) — no shared tokeniser prefix.
            g = stream_gaps(port, f"Count from {i*100} to {i*100+60}, one per line.", 120)
            res[i] = statistics.median(g) if g else 0
        ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in ts: t.start()
        for t in ts: t.join()
        good = [v for v in res.values() if v]
        if not good:
            print("  no tokens"); return
        print(f"  b={n}: per-seq p50 gap {statistics.median(good):.1f} ms")
        return
    g = stream_gaps(port, "Count to fifty", 48)
    if not g:
        print("  no tokens"); return
    p50 = statistics.median(g)
    print(f"p50 {p50:.1f} ms = {1000.0/p50:.2f} tok/s")

if __name__ == "__main__":
    main()
