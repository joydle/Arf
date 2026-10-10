#!/usr/bin/env python3
"""Aggregate ARF_DUMP_ROUTING=1 server stderr into a routing-skew report.

The batched-MoE probe in gpu.rs emits one line per decode step:

    [ROUTE] pos=<int> ids=[e0, e1, ..., e{top_k-1}]

These are the LAST MoE layer's top-k expert selections for that step. Across
many steps the distribution reveals whether Qwen3-30B routing is near-uniform
(sort-by-expert pays only at large batch) or skewed toward hot experts (it pays
even at B=8). This script reads such lines (from a file arg or stdin) and prints:

  - per-expert hit counts + a sorted histogram
  - coverage: how many of the 128 experts ever fire
  - skew metrics: Gini, top-k mass, normalized entropy
  - the money number: EXPECTED DISTINCT EXPERTS at B=8/16/32 under the OBSERVED
    routing (Monte-Carlo over the empirical step-distribution), vs the uniform
    baseline — i.e. the real weight-amortization sort-by-expert would deliver.

Usage:
    arf-serve ... 2> route.log         # capture server stderr
    python3 scripts/routing_histogram.py route.log
    # or:  grep '\\[ROUTE\\]' route.log | python3 scripts/routing_histogram.py
"""
import sys
import re
import math
from collections import Counter

NE = 128  # Qwen3-30B num_experts (informational; actual count taken from data)

LINE = re.compile(r"\[ROUTE\]\s+pos=(\d+)\s+ids=\[([0-9,\s]*)\]")


def parse(lines):
    """Return list of per-step expert-id lists."""
    steps = []
    for ln in lines:
        m = LINE.search(ln)
        if not m:
            continue
        ids_str = m.group(2).strip()
        if not ids_str:
            continue
        ids = [int(x) for x in ids_str.split(",") if x.strip() != ""]
        steps.append(ids)
    return steps


def gini(counts):
    """Gini coefficient of a list of counts (0 = uniform, 1 = all one expert)."""
    vals = sorted(counts)
    n = len(vals)
    if n == 0:
        return 0.0
    cum = 0
    total = sum(vals)
    if total == 0:
        return 0.0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


def expected_distinct(steps, B, trials_stride=1):
    """Expected distinct experts touched by a batch of B steps, under the
    OBSERVED routing — averaged over every length-B window of the step stream
    (deterministic, no RNG: scripts can't use random anyway). Each window is B
    consecutive real decode steps, which is what a real batch looks like."""
    if len(steps) < B:
        return None
    distinct_counts = []
    for start in range(0, len(steps) - B + 1, trials_stride):
        window = steps[start:start + B]
        touched = set()
        for ids in window:
            touched.update(ids)
        distinct_counts.append(len(touched))
    return sum(distinct_counts) / len(distinct_counts)


def uniform_distinct(ne, top_k, B):
    """Expected distinct experts for uniform top_k-of-ne routing at batch B."""
    return ne * (1 - (1 - top_k / ne) ** B)


def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    steps = parse(src)
    if not steps:
        print("no [ROUTE] lines found — was ARF_DUMP_ROUTING=1 set on the "
              "server, and did it actually decode any tokens?", file=sys.stderr)
        sys.exit(1)

    top_k = max(len(s) for s in steps)
    flat = Counter()
    for ids in steps:
        flat.update(ids)
    observed_ne = max(flat) + 1 if flat else 0
    ne = max(NE, observed_ne)

    counts = [flat.get(e, 0) for e in range(ne)]
    total_hits = sum(counts)
    fired = sum(1 for c in counts if c > 0)

    print(f"=== Routing-skew report ===")
    print(f"steps (decode tokens sampled): {len(steps)}")
    print(f"top_k per step:                {top_k}")
    print(f"experts that ever fired:       {fired}/{ne}")
    print(f"total expert-hits:             {total_hits}")
    print()

    # Skew metrics.
    g = gini([c for c in counts if True])
    probs = [c / total_hits for c in counts if c > 0]
    ent = -sum(p * math.log(p) for p in probs)
    ent_norm = ent / math.log(ne) if ne > 1 else 0.0
    print(f"Gini coefficient:        {g:.3f}   (0=uniform, 1=one hot expert)")
    print(f"normalized entropy:      {ent_norm:.3f}   (1=uniform, 0=degenerate)")

    ranked = sorted(enumerate(counts), key=lambda kv: -kv[1])
    for frac_label, frac in (("top-8 experts", 8), ("top-16 experts", 16),
                             ("top-32 experts", 32)):
        mass = sum(c for _, c in ranked[:frac]) / total_hits
        print(f"{frac_label:>16} hold {mass*100:5.1f}% of all routing mass")
    print()

    # Top hot experts.
    print("hottest experts (id: hits):")
    for eid, c in ranked[:12]:
        bar = "#" * int(60 * c / ranked[0][1]) if ranked[0][1] else ""
        print(f"  e{eid:<3} {c:6d}  {bar}")
    print()

    # The money number: amortization sort-by-expert would actually deliver.
    print("=== Weight-amortization potential (observed vs uniform) ===")
    print("  B   loads_now  distinct_obs  distinct_unif  amort_obs  amort_unif")
    for B in (2, 4, 8, 16, 32, 64):
        obs = expected_distinct(steps, B)
        if obs is None:
            print(f" {B:3d}   (need >= {B} steps; have {len(steps)})")
            continue
        unif = uniform_distinct(ne, top_k, B)
        loads = B * top_k
        print(f" {B:3d}   {loads:7d}   {obs:10.1f}    {unif:10.1f}   "
              f"{loads/obs:7.2f}x   {loads/unif:7.2f}x")
    print()
    print("amort = weight-loads-now / distinct-experts-touched. This is roughly")
    print("the throughput multiplier sort-by-expert buys on the MoE-weight-bound")
    print("kernels (gate/up/down). >~1.5x at your target batch => clearly worth it.")


if __name__ == "__main__":
    main()
