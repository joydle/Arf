#!/usr/bin/env python3
"""READ THE MISSES. `ARF_SPEC_MISS=1` logs, per block window, the anchor, what the draft proposed
at position 0 and what greedy actually wanted there.

Everything measured on the 26% zero-accept rate so far has been a COUNT — rate, mean, permutation
test — and every mechanism proposed from counts has been eliminated by measurement (state damage:
the ARF_BLOCK_AFTER_FULL manipulation; text autocorrelation: the MTP control over 680 windows;
the committed rows: another engine commits the same ones). This looks at CONTENT instead: which tokens,
and whether the misses concentrate anywhere.

usage: spec_miss_report.py <server.log> [tokenizer.json]
"""
import collections, json, re, sys

LINE = re.compile(r"\[miss\] accepted=(\d+) anchor=(\d+) pos0_proposed=(\d+) pos0_wanted=(\d+)")


def main():
    rows = []
    for ln in open(sys.argv[1], errors="replace"):
        m = LINE.search(ln)
        if m:
            rows.append(tuple(int(g) for g in m.groups()))
    if not rows:
        sys.exit("no [miss] lines — was ARF_SPEC_MISS=1 set, and did the block draft run?")

    dec = {}
    if len(sys.argv) > 2:
        # {"<id>": "<text>"} as written by `example dump_vocab`.
        dec = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
    show = lambda t: repr(dec.get(t, f"<{t}>").replace("Ġ", " ").replace("Ċ", "\\n"))

    zero = [r for r in rows if r[0] == 0]
    print(f"{len(rows)} block windows, {len(zero)} zero-accept ({len(zero)/len(rows)*100:.1f}%)")
    print()

    # THE QUESTION: on a zero-accept, is the draft proposing something systematic?
    print("most common ZERO-ACCEPT (proposed -> wanted) pairs:")
    pairs = collections.Counter((p, w) for _, _, p, w in zero)
    for (p, w), n in pairs.most_common(15):
        print(f"  {n:4}x  proposed {show(p):<22} wanted {show(w)}")
    print()
    print("most common ANCHORS on a zero-accept (what the draft was conditioned on):")
    anc = collections.Counter(a for _, a, _, _ in zero)
    allanc = collections.Counter(a for _, a, _, _ in rows)
    for a, n in anc.most_common(15):
        tot = allanc[a]
        print(f"  {n:4}/{tot:<4} ({n/tot*100:5.1f}% miss)  anchor {show(a)}")
    print()
    # Does the draft repeat the anchor, or propose the token BEFORE the wanted one?
    same = sum(1 for _, a, p, _ in zero if p == a)
    print(f"zero-accepts where the draft proposed the ANCHOR ITSELF: {same} ({same/max(len(zero),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
