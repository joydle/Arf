"""Prove the shim before trusting anything it produces (rule 8). CPU only.

  1. sparse submanifold conv (shim) vs a DENSE F.conv3d in f64 evaluated at the active sites,
     on a random sparse tensor with two batch items, through the reference's own
     `trellis2.modules.sparse.SparseConv3d` (so the weight layout the checkpoints use is covered);
  2. the same conv vs FlexGEMM's OWN torch explicit-GEMM path, executed from its source file
     (`$FLEXGEMM_SRC/flex_gemm/ops/spconv/submanifold_conv3d.py`) with its CUDA kernels stubbed;
  3. varlen sparse attention (shim) vs per-item dense attention computed by hand in f64;
  4. the o_voxel hashmap stand-in vs a Python dict, including out-of-range probes.

Exit status is non-zero if any check fails. Usage:
  TRELLIS2_SRC=/tmp/trellis2-src FLEXGEMM_SRC=/tmp/flexgemm-src python check_shim.py
"""
import os
import sys
import types
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shim  # noqa: E402  (must precede trellis2)

TRELLIS2_SRC = os.environ.get('TRELLIS2_SRC', '/tmp/trellis2-src')
FLEXGEMM_SRC = os.environ.get('FLEXGEMM_SRC', '/tmp/flexgemm-src')
shim.install(TRELLIS2_SRC)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402

FAIL = []


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def report(name, r, bound):
    ok = r <= bound
    print(f"{'PASS' if ok else 'FAIL'} {name}: rel={r:.3e} (bound {bound:.0e})")
    if not ok:
        FAIL.append(name)


def random_sparse(g, B=2, R=24, density=0.15, Ci=16):
    coords = []
    for b in range(B):
        occ = torch.rand(R, R, R, generator=g) < density
        xyz = torch.argwhere(occ).int()
        coords.append(torch.cat([torch.full((xyz.shape[0], 1), b, dtype=torch.int32), xyz], dim=1))
    coords = torch.cat(coords, 0).contiguous()
    feats = torch.randn(coords.shape[0], Ci, generator=g)
    return coords, feats


def dense_submconv_f64(coords, feats, w_dense, bias, B, R):
    Ci = feats.shape[1]
    grid = torch.zeros(B, Ci, R, R, R, dtype=torch.float64)
    c = coords.long()
    grid[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = feats.double()
    out = F.conv3d(grid, w_dense.double(), bias.double(), padding=1)
    return out[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]]


def check_conv():
    g = torch.Generator().manual_seed(7)
    B, R, Ci, Co = 2, 24, 16, 24
    coords, feats = random_sparse(g, B, R, 0.15, Ci)
    print(f"random sparse tensor: {coords.shape[0]} active sites in {B}x{R}^3, Ci={Ci}, Co={Co}")
    conv = sp.SparseConv3d(Ci, Co, 3)       # reference module, flex_gemm backend = shim
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=g) * 0.1)
        conv.bias.copy_(torch.randn(Co, generator=g) * 0.1)
    x = sp.SparseTensor(feats=feats, coords=coords)
    before = shim.COUNTERS['sparse_conv']
    with torch.no_grad():
        y = conv(x).feats
    assert shim.COUNTERS['sparse_conv'] == before + 1, 'shim conv did not dispatch'
    w_dense = conv.weight.detach().permute(0, 4, 1, 2, 3)      # [Co,K0,K1,K2,Ci] -> [Co,Ci,K0,K1,K2]
    ref = dense_submconv_f64(coords, feats, w_dense, conv.bias.detach(), B, R)
    report('sparse conv (shim, f32) vs dense conv3d (f64)', rel(y, ref), 1e-6)
    # Negative control: the same comparison with the kernel's x and z axes swapped MUST fail,
    # or the check above cannot tell a layout bug from a pass.
    ref_bad = dense_submconv_f64(coords, feats, w_dense.transpose(2, 4), conv.bias.detach(), B, R)
    r_bad = rel(y, ref_bad)
    ok = r_bad > 1e-2
    print(f"{'PASS' if ok else 'FAIL'} negative control (kernel x<->z swapped) is detected: rel={r_bad:.3e}")
    if not ok:
        FAIL.append('negative control not detected')

    # Forcing tiny chunks must not change a bit (chunking is only a memory device).
    old = shim.CONV_CHUNK_BYTES
    shim.CONV_CHUNK_BYTES = 1
    with torch.no_grad():
        y_small = conv(sp.SparseTensor(feats=feats, coords=coords)).feats
    shim.CONV_CHUNK_BYTES = old
    same = torch.equal(y, y_small)
    print(f"{'PASS' if same else 'FAIL'} chunk size 1024 rows vs default: bit-identical={same}")
    if not same:
        FAIL.append('chunking changes bits')

    # f16 inputs (the decoders run use_fp16): output within f16 rounding of the f64 truth.
    with torch.no_grad():
        conv_h = sp.SparseConv3d(Ci, Co, 3)
        conv_h.weight.copy_(conv.weight.half())
        conv_h.bias.copy_(conv.bias.half())
        conv_h.half()
        yh = conv_h(sp.SparseTensor(feats=feats.half(), coords=coords)).feats
    ref_h = dense_submconv_f64(coords, feats.half().float(), w_dense.half().float(), conv.bias.detach().half().float(), B, R)
    report('sparse conv (shim, f16 in/out) vs dense conv3d (f64 on the f16 values)', rel(yh, ref_h), 1e-3)

    # FlexGEMM's own torch explicit-GEMM path, from its source file.
    fg_file = os.path.join(FLEXGEMM_SRC, 'flex_gemm', 'ops', 'spconv', 'submanifold_conv3d.py')
    if not os.path.exists(fg_file):
        print(f"SKIP FlexGEMM torch path: {fg_file} not found")
        FAIL.append('flexgemm source missing')
        return
    saved = {k: sys.modules.get(k) for k in ['flex_gemm', 'flex_gemm.ops', 'flex_gemm.ops.spconv', 'flex_gemm.kernels', 'flex_gemm.ops.utils']}
    try:
        pkg = types.ModuleType('flex_gemm'); pkg.__path__ = []
        ops = types.ModuleType('flex_gemm.ops'); ops.__path__ = []
        spc = types.ModuleType('flex_gemm.ops.spconv'); spc.__path__ = []
        ker = types.ModuleType('flex_gemm.kernels')
        uspec = importlib.util.spec_from_file_location('flex_gemm.ops.utils', os.path.join(FLEXGEMM_SRC, 'flex_gemm', 'ops', 'utils.py'))
        utils = importlib.util.module_from_spec(uspec)
        uspec.loader.exec_module(utils)

        class Algorithm:
            EXPLICIT_GEMM = 'explicit_gemm'
        spc.Algorithm = Algorithm
        spc.ALGORITHM = Algorithm.EXPLICIT_GEMM
        spc.HASHMAP_RATIO = 2.0
        pkg.kernels, pkg.ops, ops.spconv, ops.utils = ker, ops, spc, utils
        sys.modules.update({'flex_gemm': pkg, 'flex_gemm.ops': ops, 'flex_gemm.ops.spconv': spc,
                            'flex_gemm.kernels': ker, 'flex_gemm.ops.utils': utils})
        spec = importlib.util.spec_from_file_location('flex_gemm.ops.spconv.submanifold_conv3d', fg_file)
        fgm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fgm)
        fn = fgm.SubMConv3dFunction
        shape = torch.Size([B, Ci, R, R, R])
        cache = fn._compute_neighbor_cache_torch(coords, shape, (3, 3, 3), (1, 1, 1))
        with torch.no_grad():
            y_fg = fn._sparse_submanifold_conv_forward(feats.contiguous(), cache, conv.weight.detach(), conv.bias.detach())
        nm_fg = cache['neighbor_map'].long()
        nm_fg = torch.where(nm_fg == 0xffffffff, torch.full_like(nm_fg, -1), nm_fg)
        nm_shim = shim.build_neighbor_map(coords, shape)
        nm_same = torch.equal(nm_fg, nm_shim)
        print(f"{'PASS' if nm_same else 'FAIL'} neighbour map: shim == FlexGEMM torch path: {nm_same}")
        if not nm_same:
            FAIL.append('neighbour map differs from FlexGEMM')
        report('sparse conv (shim) vs FlexGEMM torch explicit-GEMM path (f32)', rel(y, y_fg), 1e-6)
        report('FlexGEMM torch explicit-GEMM path (f32) vs dense conv3d (f64)', rel(y_fg, ref), 1e-6)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def check_attention():
    g = torch.Generator().manual_seed(11)
    H, C = 4, 32
    lens = [37, 81]
    q = torch.randn(sum(lens), H, C, generator=g)
    k = torch.randn(sum(lens), H, C, generator=g)
    v = torch.randn(sum(lens), H, C, generator=g)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32)
    out = shim.flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens))
    refs = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        qq, kk, vv = (t[a:b].double().permute(1, 0, 2) for t in (q, k, v))
        p = torch.softmax(qq @ kk.transpose(1, 2) / C ** 0.5, dim=-1)
        refs.append((p @ vv).permute(1, 0, 2))
    report('varlen attention (shim SDPA) vs per-item dense softmax attention (f64)', rel(out, torch.cat(refs)), 1e-6)
    # cross: dense cond of length 29 per item
    kv = torch.randn(2 * 29, 2, H, C, generator=g)
    cuk = torch.tensor([0, 29, 58], dtype=torch.int32)
    out2 = shim.flash_attn_varlen_kvpacked_func(q, kv, cu, cuk, max(lens), 29)
    refs = []
    for (a, b), (c, d) in zip(zip(cu[:-1].tolist(), cu[1:].tolist()), zip(cuk[:-1].tolist(), cuk[1:].tolist())):
        qq = q[a:b].double().permute(1, 0, 2)
        kk = kv[c:d, 0].double().permute(1, 0, 2)
        vv = kv[c:d, 1].double().permute(1, 0, 2)
        p = torch.softmax(qq @ kk.transpose(1, 2) / C ** 0.5, dim=-1)
        refs.append((p @ vv).permute(1, 0, 2))
    report('varlen cross-attention (shim) vs dense (f64)', rel(out2, torch.cat(refs)), 1e-6)


def check_hashmap():
    g = torch.Generator().manual_seed(3)
    R = 16
    xyz = torch.argwhere(torch.rand(R, R, R, generator=g) < 0.3).int()
    xyz = xyz[torch.randperm(xyz.shape[0], generator=g)]
    coords = torch.cat([torch.zeros(xyz.shape[0], 1, dtype=torch.int32), xyz], 1)
    keys = torch.full((2 * coords.shape[0],), 0xffffffff, dtype=torch.uint32)
    vals = torch.empty((2 * coords.shape[0],), dtype=torch.uint32)
    shim.hashmap_insert_3d_idx_as_val_cuda(keys, vals, coords, R, R, R)
    table = {tuple(c): i for i, c in enumerate(xyz.tolist())}
    probe = torch.randint(-1, R + 1, (5000, 3), generator=g).int()
    got = shim.hashmap_lookup_3d_cuda(keys, vals, torch.cat([torch.zeros(5000, 1, dtype=torch.int32), probe], 1), R, R, R)
    want = [table.get(tuple(p), 0xffffffff) for p in probe.tolist()]
    ok = got.long().tolist() == want
    n_oob = int(((probe < 0) | (probe >= R)).any(1).sum())
    print(f"{'PASS' if ok else 'FAIL'} hashmap stand-in vs dict: 5000 probes ({n_oob} out of range)")
    if not ok:
        FAIL.append('hashmap')


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('TRELLIS2_THREADS', '8')))
    check_conv()
    check_attention()
    check_hashmap()
    from transformers.utils import is_flash_attn_2_available
    fa = is_flash_attn_2_available()
    print(f"{'PASS' if not fa else 'FAIL'} transformers does not mistake the flash_attn stand-in for the real package: {not fa}")
    if fa:
        FAIL.append('transformers sees flash_attn')
    print('shim counters:', dict(shim.COUNTERS))
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL SHIM CHECKS PASS')
