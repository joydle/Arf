#!/usr/bin/env python3
"""Speculative decoding must reproduce GREEDY text exactly — including a string copied from the prompt.

Greedy speculation accepts a draft only when the target would have produced it, so at temperature 0
every arm below must be character-identical to ARF_SPEC_K=0. The first prompt plants a serial
number that has to be copied verbatim: that is the exact failure of 2026-08-30 ("with the
shared-prefix verify kernel both rows lose the prompt"), whose cause was found on 2026-09-20 — the
kernel's staged K/V tile was sized for head_dim 128 and indexed by the 27B's 256. With that shader
fix reverted this prints "identical to greedy: [False, False] ... serial ok: False".

usage: spec_identity_check.py          (exit 1 if any SHIPPED arm differs from greedy)
"""
import json, os, signal, subprocess, sys, time, urllib.request, hashlib
GGUF=os.environ.get("IDCHECK_GGUF","models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf"); PORT=8197  # IDCHECK_GGUF: another model (2026-09-25: the group-64 file)
CODE = """fn parse_config(path: &str) -> Result<Config, ConfigError> {
    let raw_text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
    let mut cfg = Config::default();
    for (line_no, line) in raw_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let (key, value) = line.split_once('=').ok_or(ConfigError::Syntax(line_no))?;
        match key.trim() {
            "port" => cfg.port = value.trim().parse().map_err(|_| ConfigError::Value(line_no))?,
            "host" => cfg.host = value.trim().to_string(),
            other => return Err(ConfigError::UnknownKey(other.to_string(), line_no)),
        }
    }
    Ok(cfg)
}"""
PROMPTS=["The serial number is ZX-4471-QB. Repeat the serial number exactly, then explain in two sentences what a serial number is for.",
         "Write a Rust function that reverses a singly linked list, then explain it briefly.",
         # a COPY-heavy request: this is what prompt-lookup windows (8 rows, rollback) fire on
         "Rename the variable `raw_text` to `contents` in this function and return the complete function, nothing else.\n\n```rust\n" + CODE + "\n```"]
ARMS=[("greedy",{"ARF_SPEC_K":"0"}),("DEFAULTS (mtp + prompt lookup)",{}),("mtp k=1, lookup off",{"ARF_SPEC_K":"1","ARF_NO_PROMPT_LOOKUP":"1"}),("mtp k=1 + shared prefix",{"ARF_SPEC_K":"1","ARF_SPEC_SHARED_PREFIX":"1"}),("mtp k=3 + shared prefix",{"ARF_SPEC_K":"3","ARF_SPEC_SHARED_PREFIX":"1"})]
# The block draft (DFlash 2, the port plan) is a draft SOURCE: it may only change how
# many tokens a window lands, never which. Its arms run when the checkpoint is on disk.
DRAFT="models/qwen3.8-27b-dflash2"
DRAFT_ARMS={"block draft (DFlash 2)":{}, "block draft, lookup off":{"ARF_NO_PROMPT_LOOKUP":"1"},
            # 2026-09-23: the GPU-selected draft (selector on the GPU, the verify reads the window
            # from a GPU buffer, the CPU never waits for the draft) — gated here before any default
            # 2026-09-25: GPU select is the DEFAULT now — this arm keeps the CPU selector covered
            "block draft, CPU select":{"ARF_NO_DFLASH_GPU_SELECT":"1"},
            # 2026-09-23: verify windows through the flash prefill attention kernel
            "block draft, verify flash attn":{"ARF_VERIFY_FA":"1"}}
if os.path.isdir(DRAFT): ARMS+=list(DRAFT_ARMS.items())
# IDCHECK_ARMS="greedy,mtp k=1, lookup off": run only these arms (names as printed; greedy is always
# the reference and must be included). 2026-09-25, to bisect a failure without the full ~25 min run.
if os.environ.get("IDCHECK_ARMS"):
    keep=[a.strip() for a in os.environ["IDCHECK_ARMS"].split("|")]
    ARMS=[a for a in ARMS if a[0].strip() in keep]
ref=None
bad=0
def resident():
    return subprocess.run(["pgrep","-fl","arf-serve|splash serve|launcher.py serve"],capture_output=True,text=True).stdout.strip()
for name,extra in ARMS:
    # 2026-09-23: two ~20 GB engines co-resident on 36 GB is an OOM (it happened). Never start an
    # arm while ANY server is resident — including one this script failed to stop.
    if resident(): sys.exit(f"an inference server is already resident before the {name} arm — refusing (OOM risk):\n{resident()}")
    env={k:v for k,v in os.environ.items() if not k.startswith("ARF_")}; env.update(extra); env["QUANT"]="q4ks"
    # SG_V1 / other ARF_ A/B knobs set by the CALLER are re-applied per arm via IDCHECK_ENV
    for kv in os.environ.get("IDCHECK_ENV","").split():
        k,v=kv.split("=",1); env[k]=v
    p=subprocess.Popen(["./target/release/arf-serve","--model",GGUF,"--arch","qwen3.8-27b","--quant","q4ks","--port",str(PORT)]+(["--draft",DRAFT] if name in DRAFT_ARMS else []),env=env,stdout=open("/tmp/shared_prefix.log","w"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    try:
        for _ in range(900):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=2); break
            except Exception: time.sleep(1)
        outs=[]
        for pr in PROMPTS:
            b=json.dumps({"model":"local","temperature":0,"max_tokens":320,"messages":[{"role":"user","content":pr}]}).encode(); t=time.time()
            r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",b,{"Content-Type":"application/json"}),timeout=900))
            m=r["choices"][0]["message"]; outs.append(((m.get("reasoning_content") or "")+(m.get("content") or ""), r["usage"]["completion_tokens"]/(time.time()-t)))
        if ref is None: ref=[o[0] for o in outs]
        same=[o[0]==r for o,r in zip(outs,ref)]
        # The shared-prefix verify kernel is OPT-IN and has a KNOWN OPEN BUG at longer context (it
        # diverges on the ~300-token code-edit prompt; the head_dim overrun fixed 2026-09-20 was
        # only part of it). Report it, but only SHIPPED paths decide the exit code.
        experimental = "shared prefix" in name
        if not experimental:
            bad += (not all(same)) or ('ZX-4471-QB' not in outs[0][0])
        # where each differing prompt first departs from greedy (a character offset; a near-tie
        # flip shows up late and in one place, a broken path early and everywhere) — 2026-09-25
        first=[None if a==r else next((i for i,(x,y) in enumerate(zip(a,r)) if x!=y),min(len(a),len(r))) for (a,_),r in zip(outs,ref)]
        print(f"{name:32} identical to greedy: {same}  first diff at {first}  tok/s {[round(o[1],1) for o in outs]}  md5 {hashlib.md5(outs[0][0].encode()).hexdigest()[:12]} | serial ok: {'ZX-4471-QB' in outs[0][0]}",flush=True)
        if os.environ.get("IDCHECK_DUMP"):
            json.dump([o[0] for o in outs],open(os.path.join(os.environ["IDCHECK_DUMP"],name.replace(" ","_").replace(",","").replace("(","").replace(")","")+".json"),"w"))
    finally:
        try: os.killpg(p.pid,signal.SIGINT); p.wait(timeout=40)
        except Exception:
            try: os.killpg(p.pid,signal.SIGKILL); p.wait(timeout=30)
            except Exception: pass
        for _ in range(60):
            if not resident(): break
            time.sleep(1)
        else:
            sys.exit(f"a server outlived the {name} arm — refusing to continue (OOM risk)")
        time.sleep(3)
sys.exit(1 if bad else 0)
