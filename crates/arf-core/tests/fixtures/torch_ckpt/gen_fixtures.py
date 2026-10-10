"""Generate the PyTorch-checkpoint fixtures for `tests/torch_ckpt.rs`.

Every fixture here is created from scratch with CPU torch (`torch.save` of tensors built in this
script) — no third-party checkpoint is read. The committed `.pt` files are what the Rust reader
parses; `expected.json` is what torch itself says each tensor is (dtype, shape, stride, storage
offset, and the hex of `t.contiguous()`'s bytes), taken from the in-memory tensors, NOT from
re-reading the files — so the test compares the reader against torch, not against itself.

The malicious fixture is written as raw bytes (a hand-assembled pickle referencing `os.system`).
It is never unpickled, by this script or anything else.

Regenerate (the committed files were made with torch 2.14.0):
    <venv>/bin/python crates/arf-core/tests/fixtures/torch_ckpt/gen_fixtures.py
"""

import collections
import io
import json
import os
import zipfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)


def hexbytes(t):
    t = t.detach().contiguous()
    if t.numel() == 0:
        return ""
    return bytes(t.reshape(-1).view(torch.uint8).tolist()).hex()


def describe(t):
    t = t.detach()
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "offset": t.storage_offset(),
        "hex": hexbytes(t),
    }


def flatten(obj, prefix, out):
    if isinstance(obj, torch.Tensor):
        out[prefix] = describe(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)) and not isinstance(obj, torch.Size):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}.{i}" if prefix else str(i), out)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)
        self.norm = torch.nn.BatchNorm1d(3)  # adds an int64 num_batches_tracked buffer
        self.emb = torch.nn.Embedding(5, 2)


def all_dtypes():
    base = torch.linspace(-3.5, 3.5, 12)
    return collections.OrderedDict(
        f32=base.reshape(3, 4),
        f64=base.double(),
        f16=base.half().reshape(2, 6),
        bf16=base.bfloat16().reshape(4, 3),
        i64=torch.arange(-5, 5, dtype=torch.int64),
        i32=torch.arange(-7, 5, dtype=torch.int32).reshape(3, 4),
        i16=torch.arange(-3, 3, dtype=torch.int16),
        i8=torch.arange(-4, 4, dtype=torch.int8).reshape(2, 2, 2),
        u8=torch.arange(250, 256, dtype=torch.uint8),
        bool=torch.tensor([True, False, False, True]),
        c64=torch.complex(base[:3], base[3:6]),
        scalar=torch.tensor(2.5),
        empty=torch.zeros(0, 3),
    )


def views():
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return collections.OrderedDict(
        base=base,
        transposed=base.t(),  # non-contiguous, same storage
        row2=base[2],  # contiguous, storage_offset 12
        col_slice=base[1:3, ::2],  # non-contiguous with offset
        expanded=torch.arange(3, dtype=torch.int32).reshape(3, 1).expand(3, 4),  # stride 0
        narrow=base.view(-1)[5:9],  # contiguous view into shared storage
    )


def main():
    expected = {}

    model = Tiny()
    sd = model.state_dict()  # OrderedDict carrying `_metadata` (pickled via BUILD)
    sd.update(all_dtypes())
    torch.save(sd, os.path.join(HERE, "state_dict.pt"))
    flat = {}
    flatten(sd, "", flat)
    expected["state_dict.pt"] = flat

    v = views()
    torch.save(v, os.path.join(HERE, "views.pt"))
    flat = {}
    flatten(v, "", flat)
    expected["views.pt"] = flat

    shared = torch.randn(2, 3)
    ema = collections.OrderedDict((k, t.clone()) for k, t in Tiny().state_dict().items())
    nested = {
        "model": sd,
        "ema": ema,
        "optimizer": {"state": {}, "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999)}]},
        "step": 1234,
        "big": 2**70,
        "neg": -(2**40),
        "lr": 3.5e-4,
        "name": "tiny",
        "done": False,
        "nothing": None,
        "dtype": torch.bfloat16,
        "size": torch.Size([2, 3]),
        "tags": {"x", "y"},
        "blob": b"\x00\x01ab",
        "list": [shared, shared[1]],
        "tied_a": shared,
        "tied_b": shared,  # same object: pickled as a memo reference
        "param": torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float16)),
    }
    torch.save(nested, os.path.join(HERE, "nested.pt"))
    flat = {}
    flatten(nested, "", flat)
    expected["nested.pt"] = flat

    # Higher pickle protocols (FRAME, MEMOIZE, SHORT_BINUNICODE, STACK_GLOBAL, ...).
    torch.save(nested, os.path.join(HERE, "nested_proto4.pt"), pickle_protocol=4)
    expected["nested_proto4.pt"] = expected["nested.pt"]

    # The legacy (pre-1.6, non-zip) format: magic, protocol, sys_info, pickle, keys, raw storages.
    torch.save(v, os.path.join(HERE, "legacy_views.pt"), _use_new_zipfile_serialization=False)
    expected["legacy_views.pt"] = expected["views.pt"]
    torch.save(sd, os.path.join(HERE, "legacy_state_dict.pt"), _use_new_zipfile_serialization=False)
    expected["legacy_state_dict.pt"] = expected["state_dict.pt"]

    # A zip whose entries are DEFLATED: torch never writes this; the reader must refuse clearly.
    buf = io.BytesIO()
    torch.save(all_dtypes(), buf)
    src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    with zipfile.ZipFile(os.path.join(HERE, "compressed.pt"), "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info.filename))

    # A checkpoint-shaped zip whose data.pkl calls `os.system("echo pwned")`. Assembled by hand
    # as bytes (PROTO 2, GLOBAL 'os system', BINUNICODE, TUPLE1, REDUCE, STOP); never loaded.
    evil = (
        b"\x80\x02"
        + b"cos\nsystem\n"
        + b"X\x0a\x00\x00\x00echo pwned"
        + b"\x85"
        + b"R"
        + b"."
    )
    with zipfile.ZipFile(os.path.join(HERE, "malicious.pt"), "w", zipfile.ZIP_STORED) as z:
        z.writestr("malicious/data.pkl", evil)
        z.writestr("malicious/byteorder", "little")
        z.writestr("malicious/version", "3\n")

    with open(os.path.join(HERE, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
