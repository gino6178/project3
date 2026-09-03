# The gate's number: DreamSim of each arm's faces to the object's own TRAINING photographs, which every object has.
import sys,os,glob,torch
from PIL import Image
from dreamsim import dreamsim
SP=os.path.dirname(os.path.abspath(__file__)); dev="cuda:1"; OD=os.environ["OBJDIR"]
model,pre=dreamsim(pretrained=True,device=dev,cache_dir=f"{SP}/dreamsim_ckpt")
sys.path.insert(0,SP); from fidelity import canon
import numpy as np
def ds(refs,paths):
    if not refs or not paths: return float("nan")
    R=torch.cat([pre(Image.open(p).convert("RGB")).to(dev) for p in refs]); out=[]
    for p in paths:
        X=pre(Image.open(p).convert("RGB")).to(dev)
        with torch.no_grad(): out.append(min(float(model(X,R[j:j+1])) for j in range(len(R))))
    return sum(out)/len(out)
os.makedirs(f"{OD}/ds_faces/_spl",exist_ok=True)
for fam in ("long","trans"):
    os.makedirs(f"{OD}/ds_faces/_spl/{fam}",exist_ok=True)
    for p in glob.glob(f"{OD}/spl_{fam}/*.png"): Image.fromarray((canon(p,512)*255).astype(np.uint8)).save(f"{OD}/ds_faces/_spl/{fam}/"+os.path.basename(p))
arms=open(f"{OD}/ds_faces/arms.txt").read().split("\n")
print(f"  {'vs TRAINING photos':16s} {'long':>8s} {'trans':>8s}")
for a in arms:
    l=ds(sorted(glob.glob(f"{OD}/ds_faces/_spl/long/*.png")),sorted(glob.glob(f"{OD}/ds_faces/{a}/long/*.png")))
    t=ds(sorted(glob.glob(f"{OD}/ds_faces/_spl/trans/*.png")),sorted(glob.glob(f"{OD}/ds_faces/{a}/trans/*.png")))
    print(f"  {a:16s} {l:8.4f} {t:8.4f}")
