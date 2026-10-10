"""CPU shim for the CUDA-only pieces of the TRELLIS.2 reference (M0 of the TRELLIS.2 port).

Import this module BEFORE anything from `trellis2`. It installs stand-in modules in
`sys.modules` so the reference source runs UNMODIFIED on a CPU-only torch:

  flash_attn   varlen self/cross attention -> torch SDPA, one call per batch segment.
               (The dense DiT path uses the reference's own `sdpa` backend via ATTN_BACKEND.)
  flex_gemm    submanifold 3x3x3 sparse conv -> sort + searchsorted neighbour map and a
               CHUNKED explicit GEMM. The neighbour map is FlexGEMM's own torch path
               (`flex_gemm/ops/spconv/submanifold_conv3d.py`, `_compute_neighbor_cache_torch`,
               MIT, (c) 2025 Jianfeng Xiang); the GEMM is chunked so the last decoder level does
               not materialise an 8 GB im2col, and accumulates in f32 (the CUDA Triton kernels
               accumulate in f32; torch's CPU f16 addmm is not guaranteed to).
  o_voxel      only `convert.flexible_dual_grid_to_mesh`, executed from the reference's own
               source file with a torch stand-in for the CUDA hashmap `_C` (sorted keys +
               searchsorted; same flat key and same "missing / out of range -> UINT32_MAX" rule
               as `o-voxel/src/hash/hash.cu`).
  cumesh       NOT shimmed: every call raises. `Mesh.fill_holes`, simplify, remesh and the whole
               `to_glb` path therefore cannot run on this Mac (plan §5).

Nothing here is copied from nvdiffrast, CuMesh or Hunyuan3D.

Every shim counts its own calls in `COUNTERS` so a harness can prove the shimmed path was the
one that ran (rule 8: a green run in which the feature did not dispatch is not evidence).
"""
import os
import sys
import types
import importlib.util
import importlib.machinery
from collections import Counter

import torch
import torch.nn.functional as F

COUNTERS = Counter()

# ---------------------------------------------------------------- backend selection
# Dense attention: the reference ships an `sdpa` backend; select it. Sparse attention only
# accepts xformers / flash_attn / flash_attn_3, so it stays on `flash_attn` = our stand-in.
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
os.environ['SPARSE_CONV_BACKEND'] = 'flex_gemm'

# Rows per GEMM chunk in the sparse conv. 65536 rows x 27 x 1024 ch x 4 B = 7.2 GB at the widest
# level, so the chunk is sized from the channel count instead (see _chunk_rows).
CONV_CHUNK_BYTES = int(os.environ.get('TRELLIS2_CONV_CHUNK_BYTES', str(512 << 20)))


# ---------------------------------------------------------------- flash_attn (varlen)
def _segments(cu):
    cu = cu.tolist()
    return list(zip(cu[:-1], cu[1:]))


def _sdpa_segment(q, k, v):
    # q [Lq,H,C], k/v [Lk,H,C] -> [Lq,H,C]; default scale 1/sqrt(C) as flash_attn.
    out = F.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0), k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0))
    return out.squeeze(0).permute(1, 0, 2)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, **kw):
    assert not kw.get('causal', False)
    COUNTERS['sparse_attn'] += 1
    outs = [_sdpa_segment(q[a:b], k[c:d], v[c:d])
            for (a, b), (c, d) in zip(_segments(cu_seqlens_q), _segments(cu_seqlens_k))]
    return torch.cat(outs, dim=0)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, **kw):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, **kw)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, **kw):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, **kw)


def _dense_unavailable(*a, **k):
    raise RuntimeError('shim: dense flash_attn is not shimmed; ATTN_BACKEND=sdpa should have been used')


_fa = types.ModuleType('flash_attn')
_fa.flash_attn_varlen_func = flash_attn_varlen_func
_fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
_fa.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
_fa.flash_attn_func = _fa.flash_attn_qkvpacked_func = _fa.flash_attn_kvpacked_func = _dense_unavailable
_fa.__shim__ = True
# transformers probes optional packages with find_spec(), which raises ValueError on a module whose
# __spec__ is None. Give the stand-in a spec; with no dist-info, transformers still reports
# flash_attn as unavailable (checked in check_shim.py), so nothing in transformers routes to it.
_fa.__spec__ = importlib.machinery.ModuleSpec('flash_attn', loader=None)
sys.modules['flash_attn'] = _fa


# ---------------------------------------------------------------- flex_gemm (submanifold conv)
INVALID = 0xffffffff


class NeighborCache(dict):
    """Stands in for FlexGEMM's SubMConv3dNeighborCache (the reference only stores and reuses it)."""


def build_neighbor_map(coords: torch.Tensor, shape, kernel_size=(3, 3, 3), dilation=(1, 1, 1)) -> torch.Tensor:
    """[L, K^3] int64 neighbour indices, -1 where absent. FlexGEMM's torch path (see header)."""
    N, C, W, H, D = shape
    L = coords.shape[0]
    assert N * W * H * D <= 2 ** 32
    M = torch.tensor([W * H * D, H * D, D, 1], dtype=torch.int64)
    c64 = coords.long()
    keys = (c64 * M[None]).sum(dim=-1)
    sorted_keys, indices = torch.sort(keys)
    offset = torch.meshgrid(
        *[torch.arange(-(kernel_size[i] // 2) * dilation[i], kernel_size[i] // 2 * dilation[i] + 1, dilation[i])
          for i in range(3)], indexing='ij')
    offset = torch.stack(offset, dim=-1).reshape(-1, 3).long()
    V = offset.shape[0]
    nb = c64.unsqueeze(1).repeat(1, V, 1)
    nb[:, :, 1:] += offset.unsqueeze(0)
    nb = nb.reshape(-1, 4)
    valid = (nb[:, 1] >= 0) & (nb[:, 1] < W) & (nb[:, 2] >= 0) & (nb[:, 2] < H) & (nb[:, 3] >= 0) & (nb[:, 3] < D)
    nkeys = (nb * M[None]).sum(dim=-1)
    pos = torch.searchsorted(sorted_keys, nkeys).clamp_(0, L - 1)
    valid &= sorted_keys[pos] == nkeys
    out = torch.full((L * V,), -1, dtype=torch.int64)
    out[valid] = indices[pos[valid]]
    return out.reshape(L, V)


def _chunk_rows(v_ci: int) -> int:
    return max(1024, CONV_CHUNK_BYTES // (4 * v_ci))


def submconv_apply(feats, neighbor_map, weight, bias):
    """out[i] = bias + sum_v W[:, v, :] @ feats[nmap[i, v]] (absent neighbours contribute 0).

    weight [Co, K0, K1, K2, Ci] (FlexGEMM layout). Computed in f32, returned in feats.dtype.
    """
    L, V = neighbor_map.shape
    Co, Ci = weight.shape[0], weight.shape[-1]
    w = weight.reshape(Co, V * Ci).float().t().contiguous()      # [V*Ci, Co]
    b = bias.float() if bias is not None else None
    f32 = feats.float()
    zero_row = torch.zeros(1, Ci, dtype=torch.float32)
    src = torch.cat([f32, zero_row], dim=0)                      # index L = the zero row
    nm = torch.where(neighbor_map < 0, torch.full_like(neighbor_map, L), neighbor_map)
    out = torch.empty(L, Co, dtype=torch.float32)
    step = _chunk_rows(V * Ci)
    for s in range(0, L, step):
        e = min(L, s + step)
        cols = src[nm[s:e].reshape(-1)].reshape(e - s, V * Ci)
        if b is not None:
            torch.addmm(b, cols, w, out=out[s:e])
        else:
            torch.mm(cols, w, out=out[s:e])
    COUNTERS['sparse_conv'] += 1
    COUNTERS['sparse_conv_rows'] += L
    return out.to(feats.dtype)


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias=None, neighbor_cache=None, dilation=(1, 1, 1)):
    Co, K0, K1, K2, Ci = weight.shape
    assert feats.shape[-1] == Ci
    if neighbor_cache is None:
        neighbor_cache = NeighborCache(neighbor_map=build_neighbor_map(coords, shape, (K0, K1, K2), tuple(dilation)))
        COUNTERS['neighbor_map_built'] += 1
    out = submconv_apply(feats.contiguous(), neighbor_cache['neighbor_map'], weight, bias)
    return out, neighbor_cache


def _flex_noop(*a, **k):
    return None


def _grid_sample_unavailable(*a, **k):
    raise RuntimeError('shim: flex_gemm.ops.grid_sample is not shimmed (only used by to_glb / query_attrs)')


_fg = types.ModuleType('flex_gemm')
_fg_ops = types.ModuleType('flex_gemm.ops')
_fg_spconv = types.ModuleType('flex_gemm.ops.spconv')
_fg_gs = types.ModuleType('flex_gemm.ops.grid_sample')
_fg_spconv.sparse_submanifold_conv3d = sparse_submanifold_conv3d
_fg_spconv.set_algorithm = _flex_noop
_fg_spconv.set_hashmap_ratio = _flex_noop
_fg_gs.grid_sample_3d = _grid_sample_unavailable
_fg.ops = _fg_ops
_fg_ops.spconv = _fg_spconv
_fg_ops.grid_sample = _fg_gs
_fg.__shim__ = True
sys.modules.update({'flex_gemm': _fg, 'flex_gemm.ops': _fg_ops,
                    'flex_gemm.ops.spconv': _fg_spconv, 'flex_gemm.ops.grid_sample': _fg_gs})


# ---------------------------------------------------------------- cumesh (not available)
class _CuMeshUnavailable:
    def __init__(self, *a, **k):
        raise RuntimeError('shim: CuMesh is CUDA-only and has no CPU path (fill_holes / simplify / to_glb)')


_cm = types.ModuleType('cumesh')
_cm.CuMesh = _CuMeshUnavailable
_cm.__shim__ = True
sys.modules['cumesh'] = _cm


# ---------------------------------------------------------------- o_voxel (mesh extraction only)
def _flat_key(coords, W, H, D):
    c = coords.long()
    return ((c[:, 0] * W + c[:, 1]) * H + c[:, 2]) * D + c[:, 3]


def hashmap_insert_3d_idx_as_val_cuda(keys, vals, coords, W, H, D):
    # Stateless stand-in: the "hashmap" becomes a sorted key array in the first N slots; the rest
    # keep UINT32_MAX, which sorts last, so searchsorted over the whole buffer still works.
    k = _flat_key(coords, W, H, D)
    sk, idx = torch.sort(k)
    assert sk.numel() == torch.unique(sk).numel(), 'duplicate coords inserted into hashmap'
    n = sk.numel()
    keys[:n] = sk.to(keys.dtype)
    vals[:n] = idx.to(vals.dtype)
    COUNTERS['hashmap_insert'] += 1


def hashmap_lookup_3d_cuda(keys, vals, coords, W, H, D):
    c = coords.long()
    inb = (c[:, 1] >= 0) & (c[:, 1] < W) & (c[:, 2] >= 0) & (c[:, 2] < H) & (c[:, 3] >= 0) & (c[:, 3] < D)
    k = _flat_key(coords, W, H, D)
    kk = keys.long()
    pos = torch.searchsorted(kk, k).clamp_(0, kk.numel() - 1)
    hit = inb & (kk[pos] == k)
    out = torch.full((coords.shape[0],), INVALID, dtype=torch.int64)
    out[hit] = vals.long()[pos[hit]]
    COUNTERS['hashmap_lookup'] += 1
    return out.to(vals.dtype)


def install_o_voxel(trellis2_src: str):
    """Expose `o_voxel.convert.flexible_dual_grid_to_mesh` from the reference's own source file."""
    ov = types.ModuleType('o_voxel')
    ov.__path__ = []
    c = types.ModuleType('o_voxel._C')
    c.hashmap_insert_3d_idx_as_val_cuda = hashmap_insert_3d_idx_as_val_cuda
    c.hashmap_lookup_3d_cuda = hashmap_lookup_3d_cuda
    conv = types.ModuleType('o_voxel.convert')
    conv.__path__ = []
    ov._C = c
    ov.convert = conv
    sys.modules.update({'o_voxel': ov, 'o_voxel._C': c, 'o_voxel.convert': conv})
    path = os.path.join(trellis2_src, 'o-voxel', 'o_voxel', 'convert', 'flexible_dual_grid.py')
    spec = importlib.util.spec_from_file_location('o_voxel.convert.flexible_dual_grid', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    conv.flexible_dual_grid_to_mesh = mod.flexible_dual_grid_to_mesh
    conv.mesh_to_flexible_dual_grid = mod.mesh_to_flexible_dual_grid


def install(trellis2_src: str):
    """Put the reference on sys.path behind the shims. Returns nothing; raises if the source is wrong."""
    if not os.path.isdir(os.path.join(trellis2_src, 'trellis2')):
        raise SystemExit(f'TRELLIS.2 source not found at {trellis2_src}')
    install_o_voxel(trellis2_src)
    if trellis2_src not in sys.path:
        sys.path.insert(0, trellis2_src)
