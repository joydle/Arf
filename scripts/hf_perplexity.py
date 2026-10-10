#!/usr/bin/env python3
"""HF transformers reference perplexity for Correctness Ladder Layer 5.

Computes PPL = exp(mean NLL per token) over a fixed text file using
HF transformers at float32, teacher-forced — same algorithm as
`arf perplexity`.

Usage (inside the HF env):
    source /tmp/hf_l4_env/bin/activate
    python scripts/hf_perplexity.py \
        --model models/llama-3.2-1b \
        --text docs/fixtures/ppl_sample.txt

The script prints the PPL and per-token NLL so you can directly compare
to `arf perplexity --model models/llama-3.2-1b --text-file docs/fixtures/ppl_sample.txt`.

Ratio < 1.02 is the acceptance bar (~1-2% of reference).
"""

import argparse
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_ppl(model_path: str, text: str, device: str = "cpu") -> dict:
    """Teacher-forced PPL over `text` using HF transformers float32."""
    print(f"loading tokenizer from {model_path}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"loading model from {model_path} (float32)...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    model.to(device)

    # Tokenize.
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [1, N]

    # Prepend BOS if absent.
    bos_id = tok.bos_token_id
    if bos_id is not None and input_ids[0, 0].item() != bos_id:
        bos_tensor = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    n_tokens = input_ids.shape[1]
    print(f"corpus: {n_tokens} tokens", file=sys.stderr)

    if n_tokens < 2:
        raise ValueError("corpus too short — need at least 2 tokens")

    with torch.no_grad():
        # Teacher-forced forward: pass all tokens, get logits at every position.
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, N, vocab]

    # Compute NLL of each actual next token.
    # logits[:, i, :] → distribution after consuming input_ids[:, i]
    # target: input_ids[:, i+1]
    log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)  # [N-1, vocab]
    targets = input_ids[0, 1:]  # [N-1]
    token_nlls = -log_probs[torch.arange(len(targets)), targets]  # [N-1]

    scored = len(targets)
    mean_nll = token_nlls.mean().item()
    ppl = math.exp(mean_nll)

    return {
        "n_tokens": n_tokens,
        "scored": scored,
        "mean_nll": mean_nll,
        "ppl": ppl,
    }


def main():
    parser = argparse.ArgumentParser(description="HF transformers reference PPL (Layer 5)")
    parser.add_argument("--model", required=True, help="HF model dir (safetensors)")
    parser.add_argument("--text", help="Inline text to score")
    parser.add_argument("--text-file", help="File containing text to score")
    parser.add_argument("--device", default="cpu", help="torch device (cpu / cuda / mps)")
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("provide --text or --text-file")
        return

    result = compute_ppl(args.model, text, device=args.device)
    print(
        f"perplexity [hf-f32] over {result['scored']} tokens: "
        f"{result['ppl']:.4f} (mean NLL: {result['mean_nll']:.6f} nats)"
    )


if __name__ == "__main__":
    main()
