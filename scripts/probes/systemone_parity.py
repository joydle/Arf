#!/usr/bin/env python3
"""System One (`POST /v1/systemone`) parity probe: replay a request set against a server and
record or diff the answers.

The reference is llama.cpp's own server (System One landed in a4cb4c61f; the committed
reference `systemone_reference.json` was recorded at 050439614, Metal build), run on
the tiny openjev test model `ggml-org/tinyopenjev-for-testing-gguf`
(`tinyopenjev-for-testing-Q8_0.gguf`). That model is licensed **CC-BY-NC-4.0**
(non-commercial): it is downloaded for testing and never committed to this repository.

Record the reference (llama.cpp, one slot, verbose log so the rendered prompts can be read back):

    llama-server -m tinyopenjev-for-testing-Q8_0.gguf -np 1 --port 8091 -v > llama.log 2>&1 &
    python3 scripts/probes/systemone_parity.py record --url http://127.0.0.1:8091 \\
        --llama-log llama.log --out scripts/probes/systemone_reference.json

`record` stores, per case: the HTTP status, the response body, and (with `--llama-log`) every
prompt llama.cpp rendered for the case (read from the `launching slot` lines of its verbose log)
plus that prompt's token ids from the server's `/tokenize` (`add_special: false`,
`parse_special: true`, which is how the decision prompt is tokenized). The token count of the
prompts is checked against `usage.input_tokens`.

Diff any server (Arf, or llama.cpp at another commit) against the recorded reference:

    python3 scripts/probes/systemone_parity.py compare --url http://127.0.0.1:8080 \\
        --ref scripts/probes/systemone_reference.json [--tol 1e-3]

`compare` checks the status, the answer keys and their order, `choice`, `legend`, and every
probability, `score`, `noul` and `confidence` within `--tol` (absolute), and `usage` exactly.
Error responses are compared by status only (the messages are free text).

Arf's own gate is the Rust test `systemone_parity_tiny` in `crates/arf-serve/tests/`, which
reads the same reference file and runs the model on Arf's CPU reference forward.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REQUESTS = os.path.join(HERE, "systemone_requests.json")


def post(url, body):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}


def body_of(case):
    if "raw_body" in case:
        return case["raw_body"].encode()
    return json.dumps(case["request"], ensure_ascii=False).encode()


def launched_prompts(text):
    """id_task -> prompt, from the `launching slot` lines of a llama-server verbose log. A slot
    prints that line (with the task it last ran) when it is launched again, so the prompt of a
    case's last task appears only once a later task starts."""
    out = {}
    marker = "launching slot : "
    for line in text.splitlines():
        i = line.find(marker)
        if i < 0:
            continue
        try:
            d = json.loads(line[i + len(marker):])
            out[d["id_task"]] = d["prompt"]
        except (json.JSONDecodeError, KeyError):
            pass
    return out


def read_from(path, off):
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(off)
        return f.read()


def record(args):
    cases = json.load(open(args.requests))
    results = []
    for case in cases:
        off = os.path.getsize(args.llama_log) if args.llama_log else 0
        status, body = post(args.url + "/v1/systemone", body_of(case))
        entry = {"name": case["name"], "status": status, "response": body}
        if args.llama_log and status == 200:
            # the tasks of this case: their "new prompt" lines are logged before the response
            text = read_from(args.llama_log, off)
            entry["tasks"] = [int(t) for t in re.findall(r"task (\d+) \| new prompt", text)]
        print(f"{case['name']}: {status}")
        results.append(entry)

    if args.llama_log:
        # launch one more task so the last one's prompt is printed, then read every prompt
        post(args.url + "/v1/systemone", body_of(cases[0]))
        time.sleep(0.5)
        prompts = launched_prompts(read_from(args.llama_log, 0))
        for entry in results:
            tasks = entry.pop("tasks", None)
            if tasks is None:
                continue
            entry["prompts"] = [prompts[t] for t in tasks]
            entry["prompt_tokens"] = []
            for p in entry["prompts"]:
                _, tok = post(args.url + "/tokenize", json.dumps(
                    {"content": p, "add_special": False, "parse_special": True}).encode())
                entry["prompt_tokens"].append(tok["tokens"])
            n = sum(len(t) for t in entry["prompt_tokens"])
            if n != entry["response"]["usage"]["input_tokens"]:
                print(f"{entry['name']}: WARNING re-tokenized prompts = {n} tokens, "
                      f"usage.input_tokens = {entry['response']['usage']['input_tokens']}",
                      file=sys.stderr)
    for entry in results:
        # the server echoes the model path; keep only the file name
        if isinstance(entry["response"], dict) and "model" in entry["response"]:
            entry["response"]["model"] = os.path.basename(entry["response"]["model"])
    with open(args.out, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in results) + "\n]\n")


def close(a, b, tol):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol


def diff_answer(qid, ref, got, tol):
    errs = []
    if ref.get("type") != got.get("type"):
        return [f"{qid}: type {got.get('type')} != {ref.get('type')}"]
    for k in ("choice",):
        if k in ref and ref[k] != got.get(k):
            errs.append(f"{qid}: {k} {got.get(k)!r} != {ref[k]!r}")
    if "legend" in ref and ref["legend"] != got.get("legend"):
        errs.append(f"{qid}: legend differs")
    for k in ("score", "noul", "confidence"):
        if k in ref and not close(ref[k], got.get(k), tol):
            errs.append(f"{qid}: {k} {got.get(k)} vs {ref[k]}")
    if "probabilities" in ref:
        rp, gp = ref["probabilities"], got.get("probabilities", {})
        if list(rp) != list(gp):
            errs.append(f"{qid}: probability keys {list(gp)} != {list(rp)}")
        for k, v in rp.items():
            if not close(v, gp.get(k), tol):
                errs.append(f"{qid}: p[{k}] {gp.get(k)} vs {v}")
    return errs


def compare(args):
    ref = {e["name"]: e for e in json.load(open(args.ref))}
    cases = json.load(open(args.requests))
    failed = 0
    for case in cases:
        r = ref.get(case["name"])
        if r is None:
            print(f"{case['name']}: no reference, skipped")
            continue
        status, body = post(args.url + "/v1/systemone", body_of(case))
        errs = []
        if status != r["status"]:
            errs.append(f"status {status} != {r['status']}")
        elif status == 200:
            ra, ga = r["response"]["answers"], body.get("answers", {})
            if list(ra) != list(ga):
                errs.append(f"answer keys {list(ga)} != {list(ra)}")
            for qid in ra:
                errs += diff_answer(qid, ra[qid], ga.get(qid, {}), args.tol)
            if body.get("usage") != r["response"]["usage"]:
                errs.append(f"usage {body.get('usage')} != {r['response']['usage']}")
        print(f"{case['name']}: {'ok' if not errs else 'FAIL'}")
        for e in errs:
            print(f"    {e}")
        failed += bool(errs)
    print(f"{len(cases) - failed}/{len(cases)} cases match")
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", required=True)
    r.add_argument("--requests", default=DEFAULT_REQUESTS)
    r.add_argument("--llama-log", help="llama-server -v log, to read back the rendered prompts")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--url", required=True)
    c.add_argument("--requests", default=DEFAULT_REQUESTS)
    c.add_argument("--ref", required=True)
    c.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    (record if args.cmd == "record" else compare)(args)


if __name__ == "__main__":
    main()
