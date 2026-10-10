#!/usr/bin/env python3
"""Every model the site and README advertise, started with the command they advertise and asked something
whose answer is known — the check a release passes before a model is listed.

    python3 scripts/model_matrix.py                    # every row whose files are present
    python3 scripts/model_matrix.py --only gemma3:4b   # one row
    ARF_MODELS_DIR=/path python3 scripts/model_matrix.py   # where `arf pull` put the models

Each row: start the server (`arf serve <alias>`, or `arf-serve` with the README's flags), wait for it, run its
checks, stop it. A row whose files are not downloaded is SKIPPED (pull it first: `arf pull <alias>`), never
counted as passed. Checks: a factual question (the answer must contain the expected word); image input for the
models the site lists with images (a generated red square must be called red; a model serving text only must refuse it); audio input for Qwen3-Omni (a
sentence spoken by macOS `say` must be heard); `/v1/systemone` for openjev; `/v1/images/generations` for FLUX.
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = int(os.environ.get("MATRIX_PORT", "8092"))
LOAD_TIMEOUT_S = 1800  # a first load can quantize a safetensors checkpoint


def red_png(n=448):
    """A solid red n x n PNG, as a data URL — the image check's input."""
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * n for _ in range(n))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def spoken_wav():
    """A short sentence spoken by macOS `say`, 16 kHz mono wav, base64 — the audio check's input."""
    path = os.path.join(tempfile.gettempdir(), "model_matrix_say.wav")
    subprocess.run(["say", "-o", path, "--data-format=LEI16@16000", "The capital of France is Paris."], check=True)
    return base64.b64encode(open(path, "rb").read()).decode()


def post(path, body, timeout=600):
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout)
    return json.load(r)


def chat(content, max_tokens=64):
    r = post("/v1/chat/completions", {"model": "x", "temperature": 0, "max_tokens": max_tokens,
                                      "chat_template_kwargs": {"enable_thinking": False},
                                      "messages": [{"role": "user", "content": content}]})
    m = r["choices"][0]["message"]
    return (m.get("content") or "") + " " + (m.get("reasoning_content") or "")


def check_text():
    a = chat("What is the capital of France? Answer with one word.")
    return "paris" in a.lower(), a.strip()[:60]


def check_image():
    a = chat([{"type": "image_url", "image_url": {"url": red_png()}},
              {"type": "text", "text": "What colour is this image? Answer with one word."}])
    return "red" in a.lower(), a.strip()[:60]


def check_image_refused():
    """A model serving text only must refuse an image, not answer from the text alone."""
    try:
        a = chat([{"type": "image_url", "image_url": {"url": red_png()}},
                  {"type": "text", "text": "What colour is this image? Answer with one word."}])
    except urllib.error.HTTPError as e:
        return e.code == 400, f"HTTP {e.code}"
    return False, f"answered {a.strip()[:40]!r} without seeing the image"


def check_audio():
    a = chat([{"type": "input_audio", "input_audio": {"data": spoken_wav(), "format": "wav"}},
              {"type": "text", "text": "Which city does the speaker name? Answer with one word."}])
    return "paris" in a.lower(), a.strip()[:60]


def check_systemone():
    # The README's request shape: a state and typed questions; a `noul` answer is the probability of yes.
    r = post("/v1/systemone", {"state": "I was charged twice for my order last week and nobody has replied.",
                               "questions": {"billing": {"type": "noul", "instructions": "Is this about a payment?"},
                                             "weather": {"type": "noul", "instructions": "Is this about the weather?"}}})
    text = json.dumps(r)
    conf = {k: v.get("noul") for k, v in (r.get("answers") or {}).items() if isinstance(v, dict)}
    ok = (isinstance(conf.get("billing"), (int, float)) and isinstance(conf.get("weather"), (int, float))
          and conf["billing"] > 0.5 > conf["weather"])
    return ok, (str(conf) if conf else text)[:60]


def check_images():
    r = post("/v1/images/generations", {"prompt": "a red apple on a white table", "size": 512}, timeout=1200)
    b = base64.b64decode(r["data"][0]["b64_json"])
    return b[:8] == b"\x89PNG\r\n\x1a\n" and len(b) > 10_000, f"{len(b)} bytes PNG"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release"))
    ap.add_argument("--only", help="one alias / row name")
    a = ap.parse_args()
    arf, serve = os.path.join(a.bin, "arf"), os.path.join(a.bin, "arf-serve")
    models = os.environ.get("ARF_MODELS_DIR") or os.path.join(ROOT, "models")
    omni = os.path.join(models, "qwen3-omni")
    gv = os.path.join(models, "gemma-3-4b-vision")
    # (row, how to start it, the files that must exist, checks)
    rows = [
        ("qwen3.8:27b", [arf, "serve", "qwen3.8:27b"], [os.path.join(models, "qwen3.8-27b-arf")], [check_text, check_image]),
        ("qwen3:30b", [arf, "serve", "qwen3:30b"], [os.path.join(models, "qwen3-coder-30b-a3b-instruct")], [check_text]),
        ("gemma3:4b", [arf, "serve", "gemma3:4b"], [os.path.join(models, "gemma-3-4b-it")], [check_text, check_image_refused]),
        ("gemma3 + mmproj", [serve, "--model", os.path.join(gv, "gemma-3-4b-it-Q4_0.gguf"),
                             "--mmproj", os.path.join(gv, "mmproj-F16.gguf"), "--arch", "gemma-3-4b",
                             "--tokenizer", os.path.join(gv, "tokenizer.json")],
         [os.path.join(gv, "mmproj-F16.gguf")], [check_text, check_image]),
        ("llama3.2:1b", [arf, "serve", "llama3.2:1b"], [os.path.join(models, "llama-3.2-1b-instruct")], [check_text]),
        ("openjev", [serve, "--model", os.path.join(models, "openjev", "OpenJev-Q4_K_M.gguf"), "--arch", "qwen3.8-27b"],
         [os.path.join(models, "openjev")], [check_systemone]),
        ("qwen3-omni", [serve, "--model", os.path.join(omni, "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"),
                        "--audio-tower", os.path.join(omni, "audio_tower.safetensors")],
         [os.path.join(omni, "audio_tower.safetensors")], [check_text, check_audio]),
        ("flux-schnell", None, [os.path.join(models, "flux-schnell")], [check_images]),
    ]
    results = []
    for name, cmd, need, checks in rows:
        if a.only and a.only != name:
            continue
        if not all(os.path.exists(p) for p in need):
            results.append((name, "SKIP", f"not downloaded ({need[0]})"))
            continue
        if cmd is None:  # served by any running model's server (the FLUX actor loads on first use)
            cmd = [arf, "serve", "llama3.2:1b"]
        log = open(os.path.join(tempfile.gettempdir(), f"model_matrix_{name.replace(':', '_')}.log"), "w")
        env = {**os.environ, "ARF_MODELS_DIR": models}  # FLUX is found there too (no ARF_FLUX_DIR)
        p = subprocess.Popen(cmd + ["--port", str(PORT)], stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        t0 = time.time()
        up = False
        while time.time() - t0 < LOAD_TIMEOUT_S and p.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
                up = True
                break
            except Exception:
                time.sleep(2)
        if not up:
            results.append((name, "FAIL", f"did not start ({time.time() - t0:.0f} s) — {log.name}"))
        else:
            for c in checks:
                try:
                    ok, note = c()
                except Exception as e:
                    ok, note = False, f"{type(e).__name__}: {e}"[:60]
                results.append((f"{name} · {c.__name__[6:]}", "PASS" if ok else "FAIL",
                                f"{note!r} ({time.time() - t0:.0f} s from start)"))
        try:
            os.killpg(p.pid, 2)
            p.wait(timeout=60)
        except ProcessLookupError:
            pass  # it exited on its own (a row that did not start)
        except Exception:
            os.killpg(p.pid, 9)
        time.sleep(3)
    print(f"\n{'row':32} result  note")
    for r, res, note in results:
        print(f"{r:32} {res:6}  {note}")
    if any(res == "FAIL" for _, res, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
