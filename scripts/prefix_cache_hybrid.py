#!/usr/bin/env python3
"""The prefix cache on a HYBRID (recurrent) model must change how long an answer takes and
NOTHING about what it says.

On 2026-09-19 it was refused on Qwen3.8-27B: a hit restored the KV blocks and left the 48
recurrent layers with another sequence's state — request 0 correct, requests 1-4 unrelated text,
and tok/s IMPROVED. It is lifted by recurrent-state snapshots (metal/state_snapshots.rs); this is
the gate that says the lift is sound:

  A. the same long prompt five times           -> five identical answers, facts recalled
  B. turn 2 of that conversation (turn 1 + its answer + a new question)
  C. a different conversation sharing nothing  -> must not be served from anyone's state

each run on a cache-ON server and a `--no-prefix-cache` server: every answer must be
CHARACTER-IDENTICAL between the two, and the ON server must show hits (wall time of a repeat
well under the first). Exit 1 otherwise.

CORRECTED 2026-10-05: "character-identical" is stricter than the engine's guarantee, and the gate
failed on a sound cache. On the published bundle (group-64 weights, DFlash-2 draft, MTP k=1) A[1-4]
answered "The Grey Heron" and the cache-off server "The oldest boat was called the Grey Heron." —
because the cold SPECULATIVE run left greedy at the 4th generated token, where plain greedy's own
margin is 0.067 nats (' from' -0.734 vs ' user' -0.801; measured 2026-10-05). The verify
pass computes logits in another batch shape, and a near-tie flips; the cache hit had in fact
matched plain greedy. With speculation off, one answer (C) still differed between the servers.
What the gate exists for — a hit serving ANOTHER sequence's state, which reads as unrelated text
with large margins — is a different failure. So the identity check now runs where both servers
report their distributions: every request is asked twice, once on the production path (timing,
facts, and text compared but only reported) and once with logprobs (the non-speculative path).
Those token streams must be identical, or part at a NEAR-TIE: the other server's token within
NEAR_TIE nats of the top choice on BOTH servers. Anything else fails, as before.

usage: prefix_cache_hybrid.py            env: MPP_MODEL, MPP_ARCH, PCH_FLAGS (extra server flags),
       PCH_ENV ("ARF_X=1 ARF_Y=2": declared for both servers, since every ARF_* of the caller is stripped),
       PCH_CONCURRENT=1 (also case D: two requests sharing a long system prompt, sent at the same moment —
       the path ARF_EARLY_ANCHOR changes; compared with the cache-off server like every other case)
"""
import json, os, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_recall import FACTS, FILL, MODEL, ARCH  # noqa: E402
PORT = 8173
# PCH_EXTRA_FILL=N lengthens the passage by N fills (2026-09-26): the default 734-token prompt snapshots
# at 512, a multiple of the 256-row prefill window, so it cannot see a snapshot that lands INSIDE a
# window (an odd multiple of the 128 default alignment, e.g. 640 for a 769-896-token prompt).
PASSAGE = (FACTS[0][0] + FILL * 6 + FACTS[1][0] + FILL * 6 + FACTS[2][0] + FILL * 2
           + FILL * int(os.environ.get("PCH_EXTRA_FILL", "0")))
Q1 = PASSAGE + "\n\nAnswer from the passage only. " + FACTS[1][1]
Q2 = "And one more from the same passage: " + FACTS[0][1]
# Turn 2's assistant reply is FIXED, so both servers get the same conversation whatever turn 1 said.
ANSWER1 = "The oldest boat was called the Grey Heron."
# A first difference within this many nats of the top choice on both servers is a near-tie flip.
NEAR_TIE = 0.1
OTHER = ("Here is a note about a mountain village. The innkeeper was a man named Corvin Hale. " + FILL * 8
         + "\n\nAnswer from the note only. What was the innkeeper's name?")


def server(flags):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARF_")}
    env["QUANT"] = "q4ks"
    env.update(kv.split("=", 1) for kv in os.environ.get("PCH_ENV", "").split() if "=" in kv)
    log = open("/tmp/prefix_cache_hybrid.log", "w")
    p = subprocess.Popen(["./target/release/arf-serve", "--model", MODEL, "--arch", ARCH, "--quant", "q4ks",
                          "--port", str(PORT)] + flags + os.environ.get("PCH_FLAGS", "").split(),
                         env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(900):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2); return p
        except Exception:
            time.sleep(1)
    sys.exit("server did not start")


def ask(messages, n=120, logprobs=False):
    body = {"model": "local", "temperature": 0, "max_tokens": n, "messages": messages}
    if logprobs:
        body.update(logprobs=True, top_logprobs=4)
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]
    m = c["message"]
    toks = [(x["token"], {y["token"]: y["logprob"] for y in x["top_logprobs"]})
            for x in (c.get("logprobs") or {}).get("content") or []]
    return ((m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), time.time() - t,
            r["usage"]["prompt_tokens"], toks)


CASES = [("A", [{"role": "user", "content": Q1}])] * 5 + [
    ("B", [{"role": "user", "content": Q1}, {"role": "assistant", "content": ANSWER1},
           {"role": "user", "content": Q2}]),
    ("C", [{"role": "user", "content": OTHER}]),
    ("A", [{"role": "user", "content": Q1}]),  # and A again after B and C
]


# Case D: a shared system prompt long enough for an anchor (>= 1,024 tokens; with FILL * 6 it was 1,010 and
# took none, measured 2026-10-05), two different questions.
SYSTEM_D = "You answer questions about the passage below, briefly.\n\n" + PASSAGE + FILL * 9
CONCURRENT = [
    ("D", [{"role": "system", "content": SYSTEM_D}, {"role": "user", "content": FACTS[0][1]}]),
    ("D", [{"role": "system", "content": SYSTEM_D}, {"role": "user", "content": FACTS[2][1]}]),
]


def run(flags):
    """The cases on the production path, then again with logprobs; a fresh server per pass, so
    each pass starts cold and the first A is a miss in both."""
    passes = []
    for lp in (False, True):
        p = server(flags)
        try:
            ask([{"role": "user", "content": "Say hi."}], 8)
            out = [(tag, *ask(msgs, logprobs=lp)) for tag, msgs in CASES]
            if os.environ.get("PCH_CONCURRENT"):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(2) as ex:
                    futs = [ex.submit(ask, msgs, 120, lp) for _, msgs in CONCURRENT]
                    out += [(tag, *f.result()) for (tag, _), f in zip(CONCURRENT, futs)]
            passes.append(out)
        finally:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=60)
            except Exception:
                os.killpg(p.pid, signal.SIGKILL)
            time.sleep(5)
    return passes


def parting(a, b):
    """Where two token streams first differ (None if identical), and whether that is a near-tie:
    each side's token within NEAR_TIE nats of the top choice in the OTHER side's distribution."""
    for k, ((ta, da), (tb, db)) in enumerate(zip(a, b)):
        if ta != tb:
            near = (tb in da and max(da.values()) - da[tb] <= NEAR_TIE
                    and ta in db and max(db.values()) - db[ta] <= NEAR_TIE)
            gap = max(da.values()) - da.get(tb, float("-inf"))
            return k, near, f"token {k}: {ta!r} vs {tb!r}, {gap:.3f} nats apart"
    if len(a) != len(b):
        return min(len(a), len(b)), False, f"lengths {len(a)} vs {len(b)}"
    return None, True, ""


if __name__ == "__main__":
    (off, off_lp), (on, on_lp) = run(["--no-prefix-cache"]), run([])
    log = open("/tmp/prefix_cache_hybrid.log", errors="replace").read()
    bad = ties = 0
    for i, (a, b, alp, blp) in enumerate(zip(on, off, on_lp, off_lp)):
        same = a[1] == b[1]
        k, near, why = parting(alp[4], blp[4])
        if k is not None:
            bad += not near
            ties += near
        verdict = "identical" if k is None else ("NEAR-TIE " if near else "DIFFERENT ") + why
        print(f"{a[0]}[{i}] {a[3]:4d} prompt tokens | cache ON {a[2]:6.2f} s, OFF {b[2]:6.2f} s | production text "
              f"identical: {same} | logprob path: {verdict} | {a[1].split(chr(0))[1][:50]!r}", flush=True)
    facts = [FACTS[1][2] in on[0][1], FACTS[0][2] in on[5][1], "Corvin" in on[6][1] or "Hale" in on[6][1]]
    facts_lp = [FACTS[1][2] in on_lp[0][1], FACTS[0][2] in on_lp[5][1],
                "Corvin" in on_lp[6][1] or "Hale" in on_lp[6][1]]
    if os.environ.get("PCH_CONCURRENT"):  # case D: each concurrent answer recalls its own fact
        facts += [FACTS[0][2] in on[8][1], FACTS[2][2] in on[9][1]]
        facts_lp += [FACTS[0][2] in on_lp[8][1], FACTS[2][2] in on_lp[9][1]]
    print(f"facts recalled (A, B, C[, D, D]): production {facts}, logprob path {facts_lp}")
    repeats_on = sorted(x[2] for x in on[1:5])
    print(f"A repeated: cache ON first {on[0][2]:.2f} s, repeats median {repeats_on[len(repeats_on)//2]:.2f} s; "
          f"cache OFF first {off[0][2]:.2f} s, repeats median {sorted(x[2] for x in off[1:5])[2]:.2f} s")
    print("server said:", [l.strip()[:110] for l in log.splitlines() if "prefix caching" in l or "state-snapshot" in l][:3])
    hit = repeats_on[len(repeats_on) // 2] < on[0][2] * 0.8
    if not all(len(x[4]) for x in on_lp + off_lp):
        sys.exit("FAIL: the logprob pass returned no token distributions — the instrument did not run")
    if bad or not all(facts) or not all(facts_lp) or not hit:
        sys.exit(f"FAIL: {bad} answers part from the cache-off server at more than a near-tie, facts {facts} / "
                 f"{facts_lp}, repeats faster: {hit}")
    print(f"HYBRID PREFIX CACHE GATE: PASS ({ties} near-tie partings of {len(on)} answers)")
