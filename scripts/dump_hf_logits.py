#!/usr/bin/env python3
"""
dump_hf_logits.py — Reference logit dumper using HuggingFace transformers (f32).

This is the **reference of record** for Correctness Ladder Layer 4.  It teacher-forces a fixed token sequence
through an HF AutoModelForCausalLM at f32 precision and writes per-position
next-token logits to a JSONL file in the same schema as `arf dump-logits`.

The output can be compared against a Arf dump using:
  python3 scripts/compare_logits.py ref.jsonl arf.jsonl

─── Schema ─────────────────────────────────────────────────────────────────────
See compare_logits.py docstring for the full schema definition.

─── Requirements ────────────────────────────────────────────────────────────────
  pip install torch transformers

Run on CPU or GPU.  For exact f32 reference, use:
  --dtype float32   (default; matches the Layer-4 "HF at f32" spec)

For a fast approximate run (matches common f16 serving):
  --dtype float16

─── Usage ───────────────────────────────────────────────────────────────────────
  # Dump logits for a text prompt using a local HF model directory:
  python3 scripts/dump_hf_logits.py \\
      --model /path/to/meta-llama/Llama-3.2-1B \\
      --prompt "The capital of France is" \\
      --out ref_logits.jsonl

  # Or with raw token ids (exact mirror of `arf dump-logits --tokens`):
  python3 scripts/dump_hf_logits.py \\
      --model /path/to/meta-llama/Llama-3.2-1B \\
      --tokens 128000,1,791,6864,315,9822,374 \\
      --out ref_logits.jsonl

  # Dump full logit rows (enables exact KL in compare_logits.py):
  python3 scripts/dump_hf_logits.py --model … --prompt "…" --out ref.jsonl --full

  # Then compare against Arf:
  python3 scripts/compare_logits.py ref.jsonl arf.jsonl
"""

import argparse
import json
import math
import sys

# ─── Guard heavy imports ──────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    sys.exit(
        "[error] PyTorch not found.  Install it with:\n"
        "  pip install torch\n"
        "See https://pytorch.org/get-started/locally/ for platform-specific instructions."
    )

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    sys.exit(
        "[error] transformers not found.  Install it with:\n"
        "  pip install transformers\n"
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _logsumexp(logits_tensor):
    """Numerically stable log(Σ exp(x_i)) for a 1-D tensor."""
    m = logits_tensor.max().item()
    return m + math.log(sum(math.exp(v.item() - m) for v in logits_tensor))


def _json_str(s):
    """Minimal JSON string escaping."""
    return json.dumps(s)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HF transformers reference logit dumper for Correctness Ladder Layer 4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model id (e.g. meta-llama/Llama-3.2-1B) or local path."
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Prompt text.  The tokenizer adds BOS; teacher-forcing runs over all tokens."
    )
    parser.add_argument(
        "--tokens", default=None,
        help="Comma-separated raw token ids (bypasses the tokenizer)."
    )
    parser.add_argument(
        "--out", default="ref_logits.jsonl",
        help="Output JSONL path (default: ref_logits.jsonl)."
    )
    parser.add_argument(
        "--top", type=int, default=64,
        help="Number of top-(id,logit) pairs to dump per position (default 64; --full dumps all)."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Also dump the full logit vector per position (enables exact KL)."
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
        help="Model dtype (default float32 — the Layer-4 reference standard)."
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device (cpu/cuda/mps; default: auto-detect)."
    )
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] using device: {device}", file=sys.stderr)

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── load tokenizer & model ────────────────────────────────────────────
    print(f"[info] loading tokenizer from {args.model!r} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[info] loading model at {args.dtype} …", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=str(device),
    )
    model.eval()

    # ── resolve prompt tokens ─────────────────────────────────────────────
    if args.tokens is not None:
        token_ids = [int(t.strip()) for t in args.tokens.split(",")]
    elif args.prompt is not None:
        enc = tokenizer(args.prompt, add_special_tokens=True, return_tensors="pt")
        token_ids = enc["input_ids"][0].tolist()
    else:
        sys.exit("[error] provide --prompt or --tokens")

    n = len(token_ids)
    if n == 0:
        sys.exit("[error] empty token sequence")
    print(f"[info] teacher-forcing over {n} tokens …", file=sys.stderr)

    # ── teacher-forced forward pass ───────────────────────────────────────
    # Feed the full sequence as input_ids; HF returns logits for all positions.
    # logits shape: [1, n, vocab_size]
    input_ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids_t)
    logits_all = outputs.logits[0]  # [n, vocab_size], dtype=torch_dtype

    # Convert to f32 numpy for stable arithmetic.
    logits_f32 = logits_all.to(torch.float32).cpu()

    vocab_size = logits_f32.shape[1]
    top_k = max(args.top, 64)  # minimum 64 for KL approximation quality

    # ── write JSONL ───────────────────────────────────────────────────────
    quant_str  = args.dtype
    device_str = str(device)
    model_str  = args.model

    print(f"[info] writing JSONL to {args.out!r} …", file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as fh:
        # Header line
        meta = {
            "meta": {
                "model":     model_str,
                "quant":     quant_str,
                "device":    device_str,
                "vocab":     vocab_size,
                "positions": n,
                "engine":    "hf-transformers",
            }
        }
        fh.write(json.dumps(meta) + "\n")

        for pos in range(n):
            row = logits_f32[pos]  # [vocab_size] float32 tensor

            # logsumexp over full row
            lse = _logsumexp(row)

            # argmax
            argmax_id = int(row.argmax().item())

            # top-k
            topk_vals, topk_ids = torch.topk(row, min(top_k, vocab_size))
            top_pairs = [
                [int(topk_ids[i].item()), float(topk_vals[i].item())]
                for i in range(len(topk_ids))
            ]

            record = {
                "pos":       pos,
                "token_id":  token_ids[pos],
                "argmax":    argmax_id,
                "top":       top_pairs,
                "logsumexp": float(lse),
            }
            if args.full:
                record["logits"] = row.tolist()

            fh.write(json.dumps(record) + "\n")

    print(f"[info] done: {n} positions written to {args.out!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
