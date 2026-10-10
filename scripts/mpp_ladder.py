import os
import sys, os, re, time, signal, subprocess, urllib.request, statistics
PORT=8129
# MODEL/ARCH/EXTRA come from the environment so ONE harness serves every model. Defaults: the 27B.
MODEL=os.environ.get("MPP_MODEL","models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf")
ARCH=os.environ.get("MPP_ARCH","qwen3.8-27b")
EXTRA=os.environ.get("MPP_EXTRA","").split()
def arm(extra):
    env={**os.environ,"QUANT":"q4ks",**extra}
    log="/tmp/mpp_ladder_%s.log" % ("off" if extra else "on")
    p=subprocess.Popen(["./target/release/arf-serve","--model",MODEL,"--arch",ARCH,"--quant","q4ks",
                        "--port",str(PORT)]+EXTRA,env=env,stdout=open(log,"w"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=2); break
        except Exception: time.sleep(1)
    def bench(mode):
        o=subprocess.run(["python3","scripts/ab_bench.py",str(PORT)]+([mode] if mode else []),capture_output=True,text=True).stdout.strip().splitlines()
        m=re.search(r"([\d.]+) ms",o[-1]) if o else None
        return float(m.group(1)) if m else float("nan")
    bench("batch16")                     # discarded: builds the tile copies, warms every shape
    res={b:bench(m) for b,m in ((8,"batch8"),(12,"batch12"),(16,"batch16"))}
    mpp=[l.strip() for l in open(log,errors="replace") if "[mpp] batched" in l][-1:] 
    os.killpg(p.pid,signal.SIGINT)
    try: p.wait(timeout=40)
    except Exception: os.killpg(p.pid,signal.SIGKILL)
    time.sleep(5)
    return res, mpp
rounds=int(sys.argv[1]) if len(sys.argv)>1 else 2
acc={"OFF":{},"ON":{}}
for r in range(rounds):
    # LADDER_OFF_ENV names the variable the OFF arm sets (default: the whole MPP path), so the same
    # interleaved ladder A/Bs any single opt-out, e.g. LADDER_OFF_ENV=ARF_NO_MPP_SHARE_NARROW.
    for name,extra in (("OFF",{os.environ.get("LADDER_OFF_ENV","ARF_NO_MPP_Q4"):"1"}),("ON",{})):
        res,mpp=arm(extra)
        print(f"round {r+1} {name:3}: "+"  ".join(f"b={b} {v:6.1f} ms" for b,v in res.items())+("   "+mpp[0][:95] if mpp else ""),flush=True)
        for b,v in res.items(): acc[name].setdefault(b,[]).append(v)
print("\nstreams | OFF step ms | ON step ms | ON/OFF | aggregate tok/s OFF -> ON")
for b in (8,12,16):
    o,n=statistics.median(acc["OFF"][b]),statistics.median(acc["ON"][b])
    print(f"   {b}    |   {o:7.1f}   |  {n:7.1f}   | {n/o:5.2f}x | {b*1000/o:6.1f} -> {b*1000/n:6.1f}")
