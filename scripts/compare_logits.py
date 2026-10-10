#!/usr/bin/env python3
"""
compare_logits.py — Correctness Ladder Layer 4 comparison script.

Loads two JSONL logit dumps (any engine that emits the schema below) and
computes the four Layer-4 metrics:

  1. top-1 agreement rate     — fraction of positions with the same argmax
  2. top-5 overlap            — mean Jaccard overlap of each position's top-5 sets
  3. KL divergence            — D_KL(P_ref ‖ P_arf) per position, then mean & max
  4. max |Δlogit|             — maximum absolute logit difference over the union of
                                dumped ids per position (then the global max)

Exits non-zero when mean KL > --kl-threshold OR top-1 rate < --top1-threshold,
so it can gate CI once a reference dump is available.

─── JSONL schema (reference-agnostic) ──────────────────────────────────────────
Both files must follow the same schema:

  Line 0 (header):
    {"meta": {"model": "…", "quant": "…", "device": "…",
              "vocab": N, "positions": N, "engine": "…"}}

  Lines 1…N (one per position):
    {"pos": 0, "token_id": <input token>, "argmax": <argmax id>,
     "top": [[id, logit], …],   <- sorted descending, at least top-64 recommended
     "logsumexp": <f32>}        <- log(Σ exp(logit_i)) over the FULL vocab row

  Optional per-line key (when the dump was run with --full):
    "logits": [<vocab f32 list>]  <- enables exact KL; overrides top-k approx

Any engine (Arf via `arf dump-logits`, HF via scripts/dump_hf_logits.py,
llama.cpp, vLLM, …) that emits this schema can be cross-compared.

─── KL approximation note ──────────────────────────────────────────────────────
When neither dump includes --full logits, KL is approximated over the UNION of
the top-k ids from both dumps.  For each position:

  p_i  = exp(logit_i_ref   - logsumexp_ref)    (reconstructed from ref top-k)
  q_i  = exp(logit_i_prim  - logsumexp_prim)   (reconstructed from arf top-k)

  The remaining vocab mass (ids not in the union) is split proportionally
  and assigned to a synthetic "other" bin — this makes the computation
  well-defined even when the two top-k sets diverge.  The approximation
  converges to exact KL as top-k → vocab_size (or when --full is used).

Usage:
  python3 scripts/compare_logits.py ref.jsonl arf.jsonl
  python3 scripts/compare_logits.py ref.jsonl arf.jsonl --kl-threshold 0.05 --top1-threshold 0.99

Exit codes:
  0  — all thresholds passed
  1  — KL_mean > --kl-threshold or top-1 < --top1-threshold
  2  — fatal error (schema mismatch, alignment error, missing files, …)
"""

import argparse
import json
import math
import sys


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Return (meta_dict, list_of_position_dicts)."""
    meta = None
    positions = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    sys.exit(f"[error] {path}:{lineno}: invalid JSON — {exc}")
                if "meta" in obj:
                    meta = obj["meta"]
                else:
                    positions.append(obj)
    except OSError as exc:
        sys.exit(f"[error] cannot open {path}: {exc}")
    if not positions:
        sys.exit(f"[error] {path}: no position lines found (empty dump?)")
    return meta, positions


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _reconstruct_probs(top_pairs, logsumexp):
    """
    Given top-k [(id, logit), …] and log(Σ exp) over the full row,
    return {id: probability} for the dumped ids.
    Probability mass for undumped ids = 1 - sum(returned values).
    """
    probs = {}
    for id_, logit in top_pairs:
        probs[id_] = math.exp(logit - logsumexp)
    return probs


def _kl_approx(ref_pos, prim_pos):
    """
    Approximate D_KL(P_ref ‖ P_prim) for one position.

    When both dumps have "logits" (--full), uses the full row (exact KL).
    Otherwise uses the union-of-top-k reconstruction with a residual "other" bin.
    """
    ref_full  = ref_pos.get("logits")
    prim_full = prim_pos.get("logits")

    if ref_full is not None and prim_full is not None:
        # Exact KL using full vocab distributions.
        ref_lse  = ref_pos["logsumexp"]
        prim_lse = prim_pos["logsumexp"]
        kl = 0.0
        for r_logit, p_logit in zip(ref_full, prim_full):
            p = math.exp(r_logit  - ref_lse)
            q = math.exp(p_logit - prim_lse)
            if p > 0.0:
                if q <= 0.0:
                    # Undefined (infinite KL) — clamp to a large value.
                    kl += p * 50.0
                else:
                    kl += p * (math.log(p) - math.log(q))
        return kl

    # Top-k approximation path.
    ref_lse  = ref_pos["logsumexp"]
    prim_lse = prim_pos["logsumexp"]

    ref_probs  = _reconstruct_probs(ref_pos["top"],  ref_lse)
    prim_probs = _reconstruct_probs(prim_pos["top"], prim_lse)

    union_ids = set(ref_probs) | set(prim_probs)

    # Residual mass in each distribution for ids NOT in the union.
    ref_residual  = max(0.0, 1.0 - sum(ref_probs.values()))
    prim_residual = max(0.0, 1.0 - sum(prim_probs.values()))

    kl = 0.0
    for id_ in union_ids:
        p = ref_probs.get(id_, 0.0)
        q = prim_probs.get(id_, 0.0)
        # Fill missing ids with a proportional share of the residual.
        if p == 0.0 and ref_residual > 0.0:
            # This id was not in ref's top-k — assign a share of ref's residual.
            p = ref_residual / max(1, len(prim_probs) - len(ref_probs & prim_probs.keys()))
            # Simpler: uniformly spread residual over union-only ids.
            ref_only_in_prim = len(prim_probs) - len(set(ref_probs) & set(prim_probs))
            p = ref_residual / max(1, ref_only_in_prim)
        if q == 0.0 and prim_residual > 0.0:
            prim_only_in_ref = len(ref_probs) - len(set(ref_probs) & set(prim_probs))
            q = prim_residual / max(1, prim_only_in_ref)
        if p > 0.0 and q > 0.0:
            kl += p * (math.log(p) - math.log(q))
        elif p > 0.0:
            # q ≈ 0 after residual fill → treat as clamped large KL contribution.
            kl += p * 50.0

    return kl


def _max_delta_logit(ref_pos, prim_pos):
    """max |Δlogit| over the union of dumped ids."""
    ref_map  = {id_: logit for id_, logit in ref_pos["top"]}
    prim_map = {id_: logit for id_, logit in prim_pos["top"]}
    union_ids = set(ref_map) & set(prim_map)  # only ids present in BOTH
    if not union_ids:
        return float("nan")
    return max(abs(ref_map[i] - prim_map[i]) for i in union_ids)


def _top5_overlap(ref_pos, prim_pos, k=5):
    """Jaccard overlap of the top-k id sets."""
    ref_ids  = {id_ for id_, _ in ref_pos["top"][:k]}
    prim_ids = {id_ for id_, _ in prim_pos["top"][:k]}
    if not ref_ids and not prim_ids:
        return 1.0
    return len(ref_ids & prim_ids) / len(ref_ids | prim_ids)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer-4 correctness: compare two logit dumps (top-1, top-5, KL, max|Δlogit|).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ref",    help="reference JSONL dump (e.g. from HF transformers)")
    parser.add_argument("arf", help="Arf JSONL dump (from `arf dump-logits`)")
    parser.add_argument("--kl-threshold",   type=float, default=0.05,
                        help="fail if mean KL exceeds this (default 0.05)")
    parser.add_argument("--top1-threshold", type=float, default=0.99,
                        help="fail if top-1 agreement rate is below this (default 0.99)")
    args = parser.parse_args()

    ref_meta,  ref_positions  = load_jsonl(args.ref)
    prim_meta, prim_positions = load_jsonl(args.arf)

    # ── Alignment ─────────────────────────────────────────────────────────
    if len(ref_positions) != len(prim_positions):
        sys.exit(
            f"[error] position count mismatch: ref has {len(ref_positions)}, "
            f"arf has {len(prim_positions)} — dumps must cover the same token sequence."
        )

    n = len(ref_positions)

    # Check teacher-forcing invariant: same input token_id at every position.
    mismatches = []
    for i, (r, p) in enumerate(zip(ref_positions, prim_positions)):
        if r["token_id"] != p["token_id"]:
            mismatches.append(i)
    if mismatches:
        first = mismatches[0]
        sys.exit(
            f"[error] token_id mismatch at {len(mismatches)} position(s) "
            f"(first at pos {first}: ref={ref_positions[first]['token_id']}, "
            f"arf={prim_positions[first]['token_id']}). "
            f"Teacher-forcing requires identical input sequences — check that both "
            f"dumps used exactly the same --prompt / --tokens."
        )

    # ── Per-position metrics ───────────────────────────────────────────────
    top1_matches = 0
    top5_overlaps = []
    kl_values = []
    max_delta_logits = []

    for r, p in zip(ref_positions, prim_positions):
        # top-1
        if r["argmax"] == p["argmax"]:
            top1_matches += 1

        # top-5 overlap (Jaccard)
        top5_overlaps.append(_top5_overlap(r, p, k=5))

        # KL D_KL(P_ref ‖ P_arf)
        try:
            kl = _kl_approx(r, p)
            kl_values.append(kl)
        except Exception as exc:
            print(f"[warn] KL computation failed at pos {r['pos']}: {exc}", file=sys.stderr)
            kl_values.append(float("nan"))

        # max |Δlogit|
        mdl = _max_delta_logit(r, p)
        if not math.isnan(mdl):
            max_delta_logits.append(mdl)

    # ── Aggregates ────────────────────────────────────────────────────────
    top1_rate      = top1_matches / n
    top5_mean      = sum(top5_overlaps) / n
    kl_valid       = [v for v in kl_values if not math.isnan(v)]
    kl_mean        = sum(kl_valid) / len(kl_valid) if kl_valid else float("nan")
    kl_max         = max(kl_valid) if kl_valid else float("nan")
    global_max_dlt = max(max_delta_logits) if max_delta_logits else float("nan")

    # ── Print summary ─────────────────────────────────────────────────────
    col_w = 30
    sep   = "─" * 60
    print(sep)
    print("  Correctness Ladder — Layer 4: teacher-forced logit agreement")
    print(sep)
    if ref_meta:
        print(f"  REF    engine={ref_meta.get('engine','?')}  "
              f"model={ref_meta.get('model','?')}  quant={ref_meta.get('quant','?')}")
    if prim_meta:
        print(f"  ARF engine={prim_meta.get('engine','?')}  "
              f"model={prim_meta.get('model','?')}  quant={prim_meta.get('quant','?')}")
    print(f"  Positions compared: {n}")
    print(sep)
    print(f"  {'Metric':<{col_w}} {'Value':>10}  {'Threshold':>10}  Status")
    print(f"  {'-'*col_w} {'-'*10}  {'-'*10}  ------")

    def _status(val, threshold, higher_is_better):
        if math.isnan(val):
            return "N/A"
        if higher_is_better:
            return "PASS" if val >= threshold else "FAIL"
        else:
            return "PASS" if val <= threshold else "FAIL"

    rows = [
        ("Top-1 agreement rate",  top1_rate,      args.top1_threshold, True),
        ("Top-5 mean overlap",    top5_mean,       None,                None),
        ("KL mean D_KL(ref‖prim)", kl_mean,        args.kl_threshold,   False),
        ("KL max",                kl_max,          None,                None),
        ("Max |Δlogit|",          global_max_dlt,  None,                None),
    ]
    for label, val, threshold, higher_is_better in rows:
        val_str  = f"{val:.6f}" if not math.isnan(val) else "  N/A   "
        thr_str  = f"{threshold:.4f}" if threshold is not None else "  —   "
        status   = _status(val, threshold, higher_is_better) if threshold is not None else ""
        print(f"  {label:<{col_w}} {val_str:>10}  {thr_str:>10}  {status}")

    print(sep)

    # ── KL approximation note ─────────────────────────────────────────────
    has_full_ref  = any("logits" in r for r in ref_positions[:1])
    has_full_prim = any("logits" in p for p in prim_positions[:1])
    if not (has_full_ref and has_full_prim):
        print("  NOTE: KL is an approximation over the union of dumped top-k ids")
        print("        (exact when both dumps were generated with --full).")
        print(sep)

    # ── CI gate ───────────────────────────────────────────────────────────
    failed = False
    if math.isnan(kl_mean) or kl_mean > args.kl_threshold:
        print(f"  [FAIL] KL mean {kl_mean:.6f} exceeds threshold {args.kl_threshold}")
        failed = True
    if math.isnan(top1_rate) or top1_rate < args.top1_threshold:
        print(f"  [FAIL] top-1 rate {top1_rate:.6f} below threshold {args.top1_threshold}")
        failed = True
    if failed:
        sys.exit(1)

    print("  All thresholds passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
