import os
"""Correctness first: the same 4 distinct prompts, sent CONCURRENTLY (so they batch), temperature 0,
with the MPP switch off and on. Prints each answer's head and where the two arms first differ."""
import sys, os, json, time, signal, subprocess, threading, urllib.request
PORT=8129
# MODEL/ARCH/EXTRA come from the environment so ONE harness serves every model. Defaults: the 27B.
MODEL=os.environ.get("MPP_MODEL","models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf")
ARCH=os.environ.get("MPP_ARCH","qwen3.8-27b")
EXTRA=os.environ.get("MPP_EXTRA","").split()
PROMPTS=["Name the three primary colours and say one thing about each.",
         "Write a haiku about autumn rain.",
         "What is 17 times 23? Show the working.",
         "Explain in two sentences why the sky is blue.",
         "List four planets and one fact about each.",
         "Write a two-line rhyme about a cat.",
         "What is the capital of Japan, and what is it known for?",
         "Give three tips for writing clear code.",
         "Describe how a bicycle stays upright, briefly.",
         "What are the first five prime numbers and why is 1 not one of them?",
         "Write a limerick about a programmer.",
         "Summarise the water cycle in three steps.",
         "Translate 'good morning, how are you?' into French and Spanish.",
         "What is the difference between a list and a tuple in Python?",
         "Name two causes of the seasons on Earth.",
         "Suggest a name for a bakery and explain it in one sentence."]
def ask(p, out, i):
    body=json.dumps({"model":"local","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":p}]}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=600))
    m=r["choices"][0]["message"]; out[i]=(m.get("reasoning_content") or "")+(m.get("content") or "")
_arm_tag=["off"]
def arm(env_extra, n):
    env={**os.environ,"QUANT":"q4ks",**env_extra}
    log=f"/tmp/mpp_e2e_{_arm_tag[0]}.log"; _arm_tag[0]="on"
    p=subprocess.Popen(["./target/release/arf-serve","--model",MODEL,"--arch",ARCH,"--quant","q4ks","--port",str(PORT)]+EXTRA,
                       env=env,stdout=open(log,"w"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(600):
        if p.poll() is not None: break
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=2); break
        except Exception: time.sleep(1)
    out=[None]*n; ts=[threading.Thread(target=ask,args=(PROMPTS[i],out,i)) for i in range(n)]
    t0=time.time(); [t.start() for t in ts]; [t.join() for t in ts]; dt=time.time()-t0
    single=[None]; ask(PROMPTS[0],single,0)          # b=1 afterwards: must be the shipped path either way
    os.killpg(p.pid,signal.SIGINT)
    try: p.wait(timeout=40)
    except Exception: os.killpg(p.pid,signal.SIGKILL)
    time.sleep(4)
    mpp=[l.strip() for l in open(log,errors="replace") if "[mpp]" in l or "MPP Q4" in l or "simdgroup_matrix" in l]
    return out, single[0], dt, mpp
n=int(sys.argv[1]) if len(sys.argv)>1 else 4
# The two arms. Default: MPP off vs on (what this harness was written for). MPP_ARM_OFF/ON are
# JSON env dicts, so the SAME harness A/Bs any switch on the real serving path — e.g.
#   MPP_ARM_OFF='{}' MPP_ARM_ON='{"ARF_SG_Q4":"1"}'   (matmul2d vs the simdgroup_matrix kernel)
ARM_OFF=json.loads(os.environ.get("MPP_ARM_OFF",'{"ARF_NO_MPP_Q4":"1"}'))
ARM_ON=json.loads(os.environ.get("MPP_ARM_ON","{}"))
off, off1, t_off, _ = arm(ARM_OFF, n)
on, on1, t_on, mpp = arm(ARM_ON, n)
for l in mpp: print("   ", l[:170])
print(f"wall for {n} concurrent x 64 tokens: OFF {t_off:.1f}s  ON {t_on:.1f}s")
for i in range(n):
    a,b=off[i] or "",on[i] or ""
    d=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    # An identical pair used to print "first difference at char 282 of 282/282", which reads as a
    # failure. Say the verdict in words, and refuse to call two EMPTY answers identical.
    verdict = "EMPTY" if not a or not b else "IDENTICAL" if a == b else f"DIFFERS at char {d} of {len(a)}/{len(b)}"
    print(f"[{i}] {verdict} | OFF: {a[:70]!r}")
    if d<min(len(a),len(b)): print(f"      ...OFF {a[max(0,d-25):d+35]!r}\n      ....ON {b[max(0,d-25):d+35]!r}")
print(f"IDENTICAL {sum(bool(a) and a == b for a, b in zip(off, on))} of {n}")
if os.environ.get("MPP_DUMP"):
    json.dump({"prompts":PROMPTS[:n],"off":off,"on":on}, open(os.environ["MPP_DUMP"],"w"), ensure_ascii=False, indent=1)
print("b=1 after the batch identical OFF vs ON:", off1==on1, "|", (on1 or "")[:60].replace("\n"," "))
