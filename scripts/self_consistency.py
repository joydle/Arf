"""THE CONTROL: the SHIPPED path against ITSELF. Same binary, same flags, same 16 concurrent prompts,
two separate daemon starts. If the reference does not reproduce itself, 'differs from the reference'
means nothing about the arm under test."""
import sys, os, json, time, signal, subprocess, threading, urllib.request
sys.path.insert(0,"scripts")
PORT=8133; n=int(sys.argv[1]); env_extra=dict(kv.split("=",1) for kv in sys.argv[2:])
import importlib.util
spec=importlib.util.spec_from_file_location("e2e","scripts/mpp_e2e_parity.py")
src=open("scripts/mpp_e2e_parity.py").read()
PROMPTS=eval(src[src.index("PROMPTS=["):src.index("]\n",src.index("PROMPTS=["))+1].split("=",1)[1])
def run():
    env={**os.environ,"QUANT":"q4ks",**env_extra}
    p=subprocess.Popen(["./target/release/arf-serve","--model","models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf","--arch","qwen3.8-27b","--quant","q4ks","--port",str(PORT)],
                       env=env,stdout=open("/tmp/selfc.log","w"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=2); break
        except Exception: time.sleep(1)
    out=[None]*n
    def ask(i):
        body=json.dumps({"model":"local","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":PROMPTS[i]}]}).encode()
        r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
        m=r["choices"][0]["message"]; out[i]=(m.get("reasoning_content") or "")+(m.get("content") or "")
    ts=[threading.Thread(target=ask,args=(i,)) for i in range(n)]; [t.start() for t in ts]; [t.join() for t in ts]
    os.killpg(p.pid,signal.SIGINT)
    try: p.wait(timeout=40)
    except Exception: os.killpg(p.pid,signal.SIGKILL)
    time.sleep(4); return out
a=run(); b=run(); c=run()
eq=lambda x,y: sum(1 for p,q in zip(x,y) if p==q)
print(f"{n} streams, env {env_extra}: run1 vs run2 identical {eq(a,b)}/{n} | run1 vs run3 {eq(a,c)}/{n} | run2 vs run3 {eq(b,c)}/{n}")
