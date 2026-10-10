# Teacher-forced perplexity of an MLX model over a text, in ppl_windows' protocol: the text encoded
# WITHOUT special tokens, fed from token 0 in chunks through a KV cache, every next token scored
# (n-1 predictions), mean NLL in nats -> exp. Also dumps the token ids for a tokenizer cross-check.
import sys, math, json, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
path, text_file = sys.argv[1], sys.argv[2]
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model, tok = load(path)
text = open(text_file).read()
ids = tok.encode(text, add_special_tokens=False)
n = len(ids)
json.dump(ids, open(text_file + ".mlx_ids.json", "w"))
print(f"{n} tokens", flush=True)
cache = make_prompt_cache(model)
nll, hits, scored = 0.0, 0, 0
t = time.time()
at = 0
while at + 1 < n:
    k = min(chunk, n - 1 - at)
    x = mx.array(ids[at:at + k])[None]
    logits = model(x, cache=cache)[0].astype(mx.float32)          # [k, vocab]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(ids[at + 1:at + 1 + k])
    picked = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
    top = mx.argmax(logits, axis=-1)
    mx.eval(picked, top)
    nll -= float(picked.sum().item()); hits += int((top == tgt).sum().item()); scored += k
    at += k
print(f"{scored} tokens scored in {time.time()-t:.1f} s, chunk {chunk}: perplexity {math.exp(nll/scored):.4f} (mean NLL {nll/scored:.5f} nats), top-1 agreement with the text {100*hits/scored:.2f}%")
