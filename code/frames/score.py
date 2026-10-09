# Held-out DreamSim per arm and family (mean over faces of the distance to the nearest held-out
# photograph), the real floor (training photographs against the held-out ones), as dsrun.py.
# Runs in the DreamSim environment; DEV=cpu is fine.
#   python score.py <object dir> [...]     -> prints a table, writes <dir>/scores.json
import os,sys,glob,json,torch
from PIL import Image
from dreamsim import dreamsim
dev=os.environ.get("DEV","cpu"); CK=os.environ.get("DSCK",os.path.expanduser("~/project/Fruit3D_Fusion/dreamsim_ckpt"))
model,pre=dreamsim(pretrained=True,device=dev,cache_dir=CK)
def emb(p): return pre(Image.open(p).convert("RGB")).to(dev)
def ds(refs,paths):
    if not refs or not paths: return float("nan")
    R=[emb(p) for p in refs]; out=[]
    with torch.no_grad():
        for p in paths: X=emb(p); out.append(min(float(model(X,r)) for r in R))
    return sum(out)/len(out)
for D in sys.argv[1:]:
    F=f"{D}/faces"; res={}
    ref={f:sorted(glob.glob(f"{F}/_ref/{f}/*.png")) for f in ("long","trans")}
    res["real floor"]={f:ds(ref[f],sorted(glob.glob(f"{F}/_spl/{f}/*.png"))) for f in ("long","trans")}
    for a in open(f"{F}/arms.txt").read().split():
        res[a]={f:ds(ref[f],sorted(glob.glob(f"{F}/{a}/{f}/*.png"))) for f in ("long","trans")}
    for v in res.values():
        xs=[x for x in v.values() if x==x]; v["mean"]=sum(xs)/len(xs) if xs else float("nan")
    json.dump(res,open(f"{D}/scores.json","w"),indent=1)
    print(f"\n{os.path.basename(D)}   held-out DreamSim, lower is better")
    for a,v in res.items(): print(f"  {a:12s} {v['long']:.4f} {v['trans']:.4f} {v['mean']:.4f}")
