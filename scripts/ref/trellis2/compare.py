"""Compare two run_ref.py fixture directories tensor by tensor.

  python compare.py RUN_A RUN_B            # determinism: every tensor must be BIT-identical
  python compare.py RUN_F32 RUN_NATIVE --envelope
      # precision envelope: rel = |a-b|/|b| against the FIRST (f32) run, per tensor, for tensors
      # whose shapes match; tensors downstream of a discrete decision that diverged are reported
      # as not comparable rather than as a number.

Exit status is non-zero in bitwise mode if anything differs or is missing.
"""
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import load_file


def file_md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_run(d):
    with open(os.path.join(d, 'manifest.json')) as f:
        man = json.load(f)
    return man, {st: load_file(os.path.join(d, info['file'])) for st, info in man['stages'].items()}


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    envelope = '--envelope' in sys.argv
    ma, ta = load_run(a_dir)
    mb, tb = load_run(b_dir)
    bad = 0
    n = 0
    for st in ta:
        if st not in tb:
            print(f'MISSING stage {st} in {b_dir}')
            bad += 1
            continue
        for k, x in ta[st].items():
            n += 1
            y = tb[st].get(k)
            if y is None:
                print(f'MISSING {st}/{k}')
                bad += 1
                continue
            if envelope:
                if x.shape != y.shape:
                    print(f'  {st}/{k}: shapes {list(x.shape)} vs {list(y.shape)}: not comparable')
                    continue
                if not x.is_floating_point():
                    eq = torch.equal(x, y.to(x.dtype))
                    print(f'  {st}/{k}: integer/bool, identical={eq}')
                    continue
                xd, yd = x.double(), y.double()
                den = xd.norm().item()
                r = (xd - yd).norm().item() / den if den > 0 else float('nan')
                print(f'  {st}/{k}: rel={r:.3e} max_abs={(xd - yd).abs().max().item():.3e}')
            else:
                same = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.view(torch.uint8) if x.is_floating_point() else x,
                                                                                    y.view(torch.uint8) if y.is_floating_point() else y)
                if not same:
                    bad += 1
                    print(f'DIFF {st}/{k}: {x.dtype}{list(x.shape)} vs {y.dtype}{list(y.shape)}')
    for st in tb:
        if st not in ta:
            print(f'MISSING stage {st} in {a_dir}')
            bad += 1
    if not envelope:
        # md5 of the files ON DISK (not the manifests' records, which a tampered file would not update)
        md5_same = all(file_md5(os.path.join(a_dir, ma['stages'][s]['file'])) == file_md5(os.path.join(b_dir, ma['stages'][s]['file']))
                       for s in ma['stages'] if s in mb['stages'])
        print(f'{n} tensors compared, {bad} differ or missing; per-stage file md5s identical: {md5_same}')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
