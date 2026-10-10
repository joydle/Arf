"""TRELLIS.2 reference harness, CPU only (M0 of the TRELLIS.2 port).

Runs the reference pipeline's OWN stage methods (`Trellis2ImageTo3DPipeline.sample_sparse_structure`,
`sample_shape_slat[_cascade]`, `sample_tex_slat`, `decode_shape_slat`, `decode_tex_slat`) in the
order `run()` calls them, behind `shim.py`, and dumps every stage boundary to safetensors.

What differs from `run()` and why:
  * No DINOv3 (gated; its licence is not accepted for this reference) and no RMBG (CC BY-NC). The image
    conditioning is a FIXED tensor of the right shape, drawn from its own generator
    (`--cond-seed`) and passed through the extractor's final non-affine `layer_norm`, so it has the
    statistics the DiT expects per token. It does not consume the global RNG, so the noise stream
    after `torch.manual_seed(seed)` is exactly the one `run()` draws. A stage's parity given
    identical inputs does not need real image features; the dumped cond IS the input.
    Token count 1+4+32*32 = 1029 / 1+4+64*64 = 4101 assumes the 4 DINOv3 registers (the gated
    config.json could not be read); the DiTs are length-agnostic in the cond.
  * `decode_latent` is not called: its `Mesh.fill_holes()` is CuMesh (CUDA only). The mesh is
    dumped as `flexible_dual_grid_to_mesh` returns it, BEFORE hole filling.
  * Models are loaded on first use and dropped at stage boundaries (the pipeline's low_vram idea).
  * `from_pretrained` loads with strict=False; this harness loads the same way and then FAILS if
    any checkpoint tensor is unused or any parameter was left at its random init.

Dumps (per run directory): cond.safetensors, ss.safetensors, shape.safetensors (+ shape_lr /
upsample for the cascade), tex.safetensors, decode.safetensors, and manifest.json with md5s, shapes,
per-stage CPU wall time, shim call counters, thread count, versions and host load.

Usage:
  python run_ref.py --pipeline 512 --out /tmp/trellis2-ref/fixtures/p512_f32_a
  python compare.py /tmp/trellis2-ref/fixtures/p512_f32_a /tmp/trellis2-ref/fixtures/p512_f32_b
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shim  # noqa: E402  (must precede trellis2)

TRELLIS2_SRC = os.environ.get('TRELLIS2_SRC', '/tmp/trellis2-src')
WEIGHTS = os.environ.get('TRELLIS2_WEIGHTS', '/tmp/trellis2-ref/weights')
EXPECTED_SRC_COMMIT = '75fbf0183001ed9876c8dbb35de6b68552ee08bd'
shim.install(TRELLIS2_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from trellis2 import models as t2models  # noqa: E402
from trellis2.pipelines import samplers  # noqa: E402
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline  # noqa: E402
from trellis2.modules.sparse import SparseTensor  # noqa: E402

PIPELINE_JSON = os.path.join(WEIGHTS, 'TRELLIS.2-4B', 'pipeline.json')


# ------------------------------------------------------------------ recording
class Recorder:
    def __init__(self):
        self.stage = None
        self.tensors = {}          # stage -> {name: tensor}
        self.randn_calls = []      # (stage, shape)
        self.forward_idx = 0
        self.recording = True

    def put(self, name, t):
        d = self.tensors.setdefault(self.stage, {})
        assert name not in d, f'duplicate tensor name {self.stage}/{name}'
        d[name] = t.detach().clone().contiguous()


REC = Recorder()
_orig_randn = torch.randn


def _recording_randn(*args, **kwargs):
    out = _orig_randn(*args, **kwargs)
    if REC.recording and kwargs.get('generator') is None:   # only the global-RNG draws are the pipeline's noise
        i = len(REC.randn_calls)
        REC.randn_calls.append((REC.stage, list(out.shape)))
        REC.put(f'noise_{i}', out)
    return out


torch.randn = _recording_randn


def feats_of(x):
    return x.feats if isinstance(x, SparseTensor) else x


def flow_hook(module, args, kwargs, output):
    x, t, cond = args[0], args[1], args[2]
    i = REC.forward_idx
    REC.forward_idx += 1
    is_neg = bool((cond == 0).all())
    REC.put(f'fwd{i:02d}_x', feats_of(x))
    REC.put(f'fwd{i:02d}_t1000', t)
    REC.put(f'fwd{i:02d}_{"neg" if is_neg else "pos"}_v', feats_of(output))
    if 'concat_cond' in kwargs and kwargs['concat_cond'] is not None:
        if 'concat_cond' not in REC.tensors.get(REC.stage, {}):
            REC.put('concat_cond', feats_of(kwargs['concat_cond']))


# ------------------------------------------------------------------ model loading
ALLOWED_MISSING = {'rope_phases', 'pos_emb'}    # buffers computed in __init__, not stored


def load_model(rel_path, precision):
    base = os.path.join(WEIGHTS, rel_path)
    # fetch_weights.sh leaves `.verified` only after the sha256 matched the HF LFS oid; a file still
    # downloading loads fine if its header happens to be complete, so refuse anything unverified.
    if not os.path.exists(base + '.safetensors.verified') and os.environ.get('TRELLIS2_ALLOW_UNVERIFIED') != '1':
        raise SystemExit(f'{base}.safetensors is not sha256-verified; run scripts/ref/trellis2/fetch_weights.sh')
    with open(base + '.json') as f:
        cfg = json.load(f)
    t0 = time.perf_counter()
    # The constructors draw from the GLOBAL RNG (e.g. every DiT block's `modulation` parameter is
    # torch.randn(6*C)). run() builds all models in from_pretrained BEFORE torch.manual_seed(seed);
    # this harness loads lazily AFTER it, so construction must not touch the global stream.
    # MEASURED-OUT 2026-09-27: without fork_rng the first try recorded 30 [9216] draws ahead of the
    # SS noise -- the noise tensor was not the one run() would have drawn.
    REC.recording = False
    with torch.random.fork_rng(devices=[]):
        model = getattr(t2models, cfg['name'])(**cfg['args'])
    REC.recording = True
    sd = load_file(base + '.safetensors')
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if k.split('.')[-1] not in ALLOWED_MISSING]
    if missing or res.unexpected_keys:
        raise SystemExit(f'{rel_path}: checkpoint does not cover the model: missing={missing[:5]} '
                         f'unexpected={res.unexpected_keys[:5]}')
    del sd
    if precision == 'f32':
        if hasattr(model, 'convert_to'):
            model.convert_to(torch.float32)
        if hasattr(model, 'convert_to_fp32'):
            model.convert_to_fp32()
            model.dtype = torch.float32
            model.use_fp16 = False
    model.eval()
    if cfg['name'] in ('SparseStructureFlowModel', 'SLatFlowModel'):
        model.register_forward_hook(flow_hook, with_kwargs=True)
    dt = time.perf_counter() - t0
    print(f'[load] {rel_path} ({cfg["name"]}) in {dt:.1f}s, precision={precision}', flush=True)
    return model


class LazyModels(dict):
    def __init__(self, paths, precision):
        super().__init__()
        self.paths = paths
        self.precision = precision
        self.load_seconds = 0.0

    def __getitem__(self, k):
        if not dict.__contains__(self, k):
            t0 = time.perf_counter()
            dict.__setitem__(self, k, load_model(self.paths[k], self.precision))
            self.load_seconds += time.perf_counter() - t0
        return dict.__getitem__(self, k)

    def __contains__(self, k):
        return k in self.paths

    def evict(self):
        self.clear()


def model_paths(args_json):
    out = {}
    for k, v in args_json['models'].items():
        if v.startswith('microsoft/TRELLIS-image-large/'):
            out[k] = os.path.join('TRELLIS-image-large', v[len('microsoft/TRELLIS-image-large/'):])
        else:
            out[k] = os.path.join('TRELLIS.2-4B', v)
    return out


def build_pipeline(precision):
    with open(PIPELINE_JSON) as f:
        a = json.load(f)['args']
    pipe = Trellis2ImageTo3DPipeline(models={})
    pipe.models = LazyModels(model_paths(a), precision)
    # Same field-by-field setup as Trellis2ImageTo3DPipeline.from_pretrained, minus DINOv3 / RMBG.
    for key in ('sparse_structure', 'shape_slat', 'tex_slat'):
        s = a[f'{key}_sampler']
        setattr(pipe, f'{key}_sampler', getattr(samplers, s['name'])(**s['args']))
        setattr(pipe, f'{key}_sampler_params', s['params'])
    pipe.shape_slat_normalization = a['shape_slat_normalization']
    pipe.tex_slat_normalization = a['tex_slat_normalization']
    pipe.low_vram = True
    pipe._device = torch.device('cpu')
    return pipe, a


def schedule_f64(steps, rescale_t):
    t = np.linspace(1, 0, steps + 1)
    return rescale_t * t / (1 + (rescale_t - 1) * t)


# ------------------------------------------------------------------ main
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def host_state():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f'n/a ({e})'
    return {'loadavg': os.getloadavg(), 'swap': sh(['sysctl', '-n', 'vm.swapusage']),
            'cpu': sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='512', choices=['512', '1024_cascade'])
    # Default f32: the plan's gates are all against the f32 reference, and torch 2.6's CPU bf16 GEMM
    # measured 0.13 TFLOP/s vs 2.15 for f32 on this box (4096x1536 @ 1536x8192, 8 threads,
    # 2026-09-27) -- a native bf16 run costs ~16x more CPU time per DiT forward.
    ap.add_argument('--precision', default='f32', choices=['native', 'f32'],
                    help='f32 = every module upcast (the "truth" of plan §5); native = the checkpoints\' '
                         'dtypes as the reference runs them (bf16 DiT blocks, f16 decoders), for the envelope')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    # 2: of the three seeds probed on 2026-09-27 (1234 -> N0 = 10, 1 -> 142, 2 -> 261 occupied 32^3
    # voxels), the one whose sparse structure is least degenerate. A random cond is not an image.
    ap.add_argument('--cond-seed', type=int, default=2)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('TRELLIS2_THREADS', '8')))
    ap.add_argument('--stop-after', default='decode', choices=['ss', 'shape', 'tex', 'decode'])
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    src_commit = subprocess.run(['git', '-C', TRELLIS2_SRC, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if src_commit != EXPECTED_SRC_COMMIT:
        raise SystemExit(f'TRELLIS.2 source is at {src_commit}, expected {EXPECTED_SRC_COMMIT}')

    manifest = {'argv': sys.argv, 'pipeline': args.pipeline, 'precision': args.precision, 'seed': args.seed,
                'cond_seed': args.cond_seed, 'threads': args.threads, 'torch': torch.__version__,
                'trellis2_commit': src_commit, 'host_start': host_state(), 'stages': {}}
    manifest['native_deviations'] = []
    if args.precision == 'native':
        # MEASURED 2026-09-27: torch 2.6's CPU f16 Conv3d is a single-threaded slow path -- 128->128 ch
        # at 32^3 took 44.8 s in f16 vs 0.04 s in f32 (0.6 vs 732 GFLOP/s); the first native run sat
        # 20+ min in the SS decoder at 100% of ONE core and was killed. So in the native arm the dense
        # SS decoder's f16 Conv3d computes in f32 from the f16 values and rounds the output to f16
        # (what an f32-accumulating GPU conv does). Recorded in the manifest.
        _conv3d_fwd = torch.nn.Conv3d._conv_forward

        def _conv3d_f32acc(self, x, w, b):
            if x.dtype == torch.float16:
                return _conv3d_fwd(self, x.float(), w.float(), None if b is None else b.float()).half()
            return _conv3d_fwd(self, x, w, b)
        torch.nn.Conv3d._conv_forward = _conv3d_f32acc
        manifest['native_deviations'].append('dense Conv3d with f16 input computed in f32, output rounded to f16')
    pipe, pj = build_pipeline(args.precision)
    timings = {}

    def begin(stage):
        REC.stage = stage
        REC.forward_idx = 0
        timings[stage] = {'t0': time.perf_counter(), 'load0': pipe.models.load_seconds}

    def end(stage):
        tm = timings[stage]
        wall = time.perf_counter() - tm['t0']
        load = pipe.models.load_seconds - tm['load0']
        tm.clear()
        tm.update({'wall_s': round(wall, 2), 'model_load_s': round(load, 2), 'compute_s': round(wall - load, 2)})
        print(f'[time] {stage}: wall {wall:.1f}s = load {load:.1f}s + compute {wall - load:.1f}s (CPU)', flush=True)
        pipe.models.evict()
        # Written now as well as at the end, so a run killed in a later stage keeps what it finished.
        save_file(REC.tensors.get(stage, {}), os.path.join(args.out, f'{stage}.safetensors'))

    # --- fixed conditioning (not the global RNG) ---
    REC.stage = 'cond'
    g = torch.Generator().manual_seed(args.cond_seed)
    c512 = torch.randn(1, 1 + 4 + 32 * 32, 1024, generator=g)
    c512 = F.layer_norm(c512, c512.shape[-1:])
    cond_512 = {'cond': c512, 'neg_cond': torch.zeros_like(c512)}
    REC.put('cond_512', c512)
    cond_1024 = None
    if args.pipeline != '512':
        c1024 = torch.randn(1, 1 + 4 + 64 * 64, 1024, generator=g)
        c1024 = F.layer_norm(c1024, c1024.shape[-1:])
        cond_1024 = {'cond': c1024, 'neg_cond': torch.zeros_like(c1024)}
        REC.put('cond_1024', c1024)
    for key in ('sparse_structure', 'shape_slat', 'tex_slat'):
        p = pj[f'{key}_sampler']['params']
        REC.put(f'schedule_f64_{key}', torch.from_numpy(schedule_f64(p['steps'], p['rescale_t'])))

    torch.manual_seed(args.seed)      # exactly where run() seeds, relative to the noise draws

    # --- stage 2+3: sparse structure flow + SS decoder ---
    begin('ss')
    ss_dec_out = {}
    orig_ss_sample = pipe.sparse_structure_sampler.sample

    def ss_sample(*a, **k):
        out = orig_ss_sample(*a, **k)
        REC.put('z_s', out.samples)
        return out
    pipe.sparse_structure_sampler.sample = ss_sample
    dec = pipe.models['sparse_structure_decoder']
    dec.register_forward_hook(lambda m, a, o: ss_dec_out.setdefault('logits', o.detach().clone()))
    coords = pipe.sample_sparse_structure(cond_512, {'512': 32, '1024_cascade': 32}[args.pipeline], 1, {})
    REC.put('logits', ss_dec_out['logits'])
    REC.put('coords', coords)
    print(f'[ss] occupied voxels N0 = {coords.shape[0]} (32^3 grid)', flush=True)
    end('ss')
    if coords.shape[0] == 0:
        raise SystemExit('sparse structure is empty: nothing downstream can run')

    if args.stop_after != 'ss':
        begin('shape')
        if args.pipeline == '512':
            slat = pipe.sample_shape_slat(cond_512, pipe.models['shape_slat_flow_model_512'], coords, {})
            res = 512
        else:
            upsample_out = {}
            dec_s = pipe.models['shape_slat_decoder']
            orig_up = dec_s.upsample

            def up(x, upsample_times):
                REC.put('lr_slat_denorm', x.feats)
                REC.put('lr_coords', x.coords)
                hr = orig_up(x, upsample_times)
                REC.put('upsample_coords_512', hr)
                upsample_out['n'] = hr.shape[0]
                return hr
            dec_s.upsample = up
            slat, res = pipe.sample_shape_slat_cascade(
                cond_512, cond_1024, pipe.models['shape_slat_flow_model_512'], pipe.models['shape_slat_flow_model_1024'],
                512, 1024, coords, {}, 49152)
        REC.put('shape_slat_denorm', slat.feats)
        REC.put('shape_slat_coords', slat.coords)
        print(f'[shape] tokens = {slat.feats.shape[0]}, res = {res}', flush=True)
        end('shape')

    if args.stop_after not in ('ss', 'shape'):
        begin('tex')
        tex_key = 'tex_slat_flow_model_512' if args.pipeline == '512' else 'tex_slat_flow_model_1024'
        tex_slat = pipe.sample_tex_slat(cond_512 if args.pipeline == '512' else cond_1024, pipe.models[tex_key], slat, {})
        REC.put('tex_slat_denorm', tex_slat.feats)
        end('tex')

    if args.stop_after == 'decode':
        begin('decode')
        dshape = pipe.models['shape_slat_decoder']
        h_out = {}
        dshape.output_layer.register_forward_hook(lambda m, a, o: h_out.setdefault('h', o))
        meshes, subs = pipe.decode_shape_slat(slat, res)
        REC.put('shape_h', h_out['h'].feats)
        REC.put('shape_h_coords', h_out['h'].coords)
        for i, s in enumerate(subs):
            REC.put(f'subs{i}_feats', s.feats)
            REC.put(f'subs{i}_coords', s.coords)
        m = meshes[0]
        REC.put('mesh_vertices_pre_fill_holes', m.vertices)
        REC.put('mesh_faces_pre_fill_holes', m.faces)
        print(f'[decode] N4 = {h_out["h"].feats.shape[0]}, mesh V = {m.vertices.shape[0]}, F = {m.faces.shape[0]}', flush=True)
        tex_vox = pipe.decode_tex_slat(tex_slat, subs)
        REC.put('tex_attrs', tex_vox.feats)
        REC.put('tex_coords', tex_vox.coords)
        end('decode')

    # --- dump ---
    for stage, d in REC.tensors.items():
        path = os.path.join(args.out, f'{stage}.safetensors')
        save_file(d, path)
        manifest['stages'][stage] = {
            'file': os.path.basename(path), 'md5': md5(path), 'timing': timings.get(stage),
            'tensors': {k: [str(v.dtype).replace('torch.', ''), list(v.shape)] for k, v in d.items()}}
    manifest['randn_calls'] = REC.randn_calls
    # Replay: the recorded noise must be EXACTLY the first draws of a fresh seed-`seed` stream, in
    # order. Any hidden consumer of the global RNG between draws (a constructor, a randn_like) fails this.
    g = torch.Generator().manual_seed(args.seed)
    replay_ok = True
    i = 0
    for st, d in REC.tensors.items():
        for k in sorted((k for k in d if k.startswith('noise_')), key=lambda k: int(k.split('_')[1])):
            if not torch.equal(_orig_randn(*d[k].shape, generator=g), d[k]):
                replay_ok = False
                print(f'[rng] {st}/{k} is NOT draw #{i} of the seed-{args.seed} stream', flush=True)
            i += 1
    manifest['noise_replays_from_seed'] = replay_ok
    print(f'[rng] {i} noise tensors replay bit-exactly from torch.manual_seed({args.seed}): {replay_ok}', flush=True)
    manifest['shim_counters'] = dict(shim.COUNTERS)
    manifest['host_end'] = host_state()
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'[done] {args.out}; shim counters {dict(shim.COUNTERS)}', flush=True)


if __name__ == '__main__':
    with torch.no_grad():      # run() and decode_latent() are @torch.no_grad; the stage methods are not
        main()
