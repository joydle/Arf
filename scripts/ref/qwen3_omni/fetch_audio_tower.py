#!/usr/bin/env python3
"""Fetch ONLY Qwen3-Omni's audio encoder (`thinker.audio_tower.*`, 525 tensors, 1.30 GB bf16) out
of shard 1 of Qwen/Qwen3-Omni-30B-A3B-Instruct with HTTP range requests, and write it as a
standalone safetensors file with the ORIGINAL tensor names and byte-identical tensor data.

Anonymous: no token is read or sent. Stdlib only.

    python3 fetch_audio_tower.py OUT_DIR

Writes OUT_DIR/audio_tower.safetensors (and OUT_DIR/audio_tower.sha256). Every tensor's bytes are
the shard's bytes, copied verbatim; the header is rebuilt with offsets into the new file. The
shard's own header is also saved (OUT_DIR/shard1_header.json) so the offsets can be audited.
"""

import concurrent.futures
import hashlib
import http.client
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REV = "26291f793822fb6be9555850f06dfe95f2d7e695"
SHARD = "model-00001-of-00015.safetensors"
PREFIX = "thinker.audio_tower."
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"
# Plan section 1: exact bf16 byte count of the audio encoder, from the headers.
EXPECT_BYTES = 1_295_854_336


def get_range(start, end_incl):
    # Retried: this network drops DNS lookups intermittently (seen 2026-09-27, twice).
    for attempt in range(20):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end_incl}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise SystemExit(f"expected 206 Partial Content, got {r.status}")
                data = r.read()
            if len(data) != end_incl - start + 1:
                raise OSError(f"short read {len(data)}")
            return data
        except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
            print(f"\n  retry {attempt + 1} after {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise SystemExit("giving up")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    (hlen,) = struct.unpack("<Q", get_range(0, 7))
    header = json.loads(get_range(8, 8 + hlen - 1))
    with open(os.path.join(out, "shard1_header.json"), "w") as f:
        json.dump(header, f, indent=0, sort_keys=True)
    base = 8 + hlen
    names = sorted(k for k in header if k.startswith(PREFIX))
    spans = sorted((header[k]["data_offsets"][0], header[k]["data_offsets"][1], k) for k in names)
    total = sum(e - s for s, e, _ in spans)
    lo, hi = spans[0][0], spans[-1][1]
    print(f"{len(names)} tensors, {total} bytes, span [{lo}, {hi}) = {hi - lo} bytes")
    if total != EXPECT_BYTES:
        raise SystemExit(f"byte total {total} != plan's {EXPECT_BYTES}")
    if any(header[k]["dtype"] != "BF16" for k in names):
        raise SystemExit("non-BF16 tensor in the audio tower")
    if hi - lo != total:
        raise SystemExit("audio tower is not contiguous in the shard; fetch per tensor instead")
    # One contiguous span: fetch it in 16 MiB pieces, several connections at once, each piece
    # kept in OUT_DIR/parts/ so an interrupted run resumes (this network ran at ~130 KB/s per
    # connection with DNS drops on 2026-09-27; a single stream would have taken hours).
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    step = 16 << 20
    starts = list(range(lo, hi, step))

    def piece(s):
        e = min(hi, s + step)
        path = os.path.join(parts, f"{s}.bin")
        if os.path.exists(path) and os.path.getsize(path) == e - s:
            return
        data = get_range(base + s, base + e - 1)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("FETCH_CONNS", "8"))) as ex:
        for _ in ex.map(piece, starts):
            done += 1
            print(f"  piece {done}/{len(starts)}", file=sys.stderr, flush=True)
    blob = bytearray()
    for s in starts:
        with open(os.path.join(parts, f"{s}.bin"), "rb") as f:
            blob += f.read()
    assert len(blob) == hi - lo
    # New header: same names/dtypes/shapes, offsets rebased to 0.
    new_header = {"__metadata__": {"source": f"{REPO}@{REV}/{SHARD}", "prefix": PREFIX}}
    for k in names:
        s, e = header[k]["data_offsets"]
        new_header[k] = {
            "dtype": header[k]["dtype"],
            "shape": header[k]["shape"],
            "data_offsets": [s - lo, e - lo],
        }
    hb = json.dumps(new_header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    path = os.path.join(out, "audio_tower.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 24), b""):
            h.update(piece)
    with open(os.path.join(out, "audio_tower.sha256"), "w") as f:
        f.write(f"{h.hexdigest()}  audio_tower.safetensors\n")
    for s in starts:
        os.remove(os.path.join(parts, f"{s}.bin"))
    os.rmdir(parts)
    print(path, h.hexdigest())


if __name__ == "__main__":
    main()
