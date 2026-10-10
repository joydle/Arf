import sys, os, json, time, signal, subprocess, threading, urllib.request
PORT=8131
PROMPTS=["Name the three primary colours.","Write a haiku about autumn rain.","What is 17 times 23?","Why is the sky blue? One sentence.",
         "List four planets.","Write a two-line rhyme about a cat.","What is the capital of Japan?","Give three tips for clear code."]
binary=sys.argv[1]; counts=[int(x) for x in sys.argv[2].split(",")]; extra=sys.argv[3:] 
env={**os.environ,"QUANT":"q4ks","ARF_SPEC_K":"0"}
p=subprocess.Popen([binary,"--model","models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf","--arch","qwen3.8-27b","--quant","q4ks","--port",str(PORT)]+extra,
                   env=env,stdout=open("/tmp/conc_sanity.log","w"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
for _ in range(600):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=2); break
    except Exception: time.sleep(1)
def ask(pr,out,i):
    body=json.dumps({"model":"local","temperature":0,"max_tokens":40,"messages":[{"role":"user","content":pr}]}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
    m=r["choices"][0]["message"]; out[i]=(m.get("reasoning_content") or "")+(m.get("content") or "")
def sane(t):
    t=t or ""; body=t.replace("<think>","")
    return len(body)>20 and sum(c.isascii() for c in body)/max(1,len(body))>0.95
for n in counts:
    out=[None]*n; ts=[threading.Thread(target=ask,args=(PROMPTS[i%8]+f" (request {i})",out,i)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    ok=sum(sane(o) for o in out)
    print(f"{n} concurrent: {ok}/{n} sane | e.g. {(out[0] or '')[:70]!r}", flush=True)
os.killpg(p.pid,signal.SIGINT)
try: p.wait(timeout=40)
except Exception: os.killpg(p.pid,signal.SIGKILL)
