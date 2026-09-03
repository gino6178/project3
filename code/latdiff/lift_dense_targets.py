"""Lift all 38 targets (both families, all planes) into one shared 3-D latent field.

Each target is a photo matched to that plane's shell and harmonised. Optimising one interior latent
to match every target at once is where 3-D consistency comes from: a colour a longitudinal plane
wants at a cell must agree with what a transverse plane wants at that same cell, because one field
renders both. Only the per-cell latent moves; a mild anchor keeps it near the fit.
"""
import os, sys, time
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr
W="/workspace/ovoxel_native"; FN="/workspace/rebuild/worktree"; OBJ="orange_sp"; dev="cuda"; S=256
STEPS=int(os.environ.get("STEPS","500")); LR=float(os.environ.get("LR","5e-3")); ANCHOR=float(os.environ.get("ANCHOR","1.0"))
ON.FDG=ON._load_ovoxel(); glctx=dr.RasterizeCudaContext(device=dev)
st=torch.load(f"{W}/state_{OBJ}.pt",map_location=dev,weights_only=False)
C=np.load(f"{W}/cams_{OBJ}_v2.npz")
p=torch.load(f"{W}/s_v2_{OBJ}/params.pt",map_location=dev)
st["dual_v"]=p["dual_v"].to(dev); st["split_w"]=p["split_w"].to(dev)
w=p["dec_i"]["stage1.0.weight"].shape[0]; nl=sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight"))-1
anchor.W_HID,anchor.N_HID=w,nl
di=anchor.ColourDecoder(len(st["interior"]),init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr=anchor.ColourDecoder(len(st["surf_rgb"]),init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad(): st["surf_rgb"]=dsr(); rgb0=di().detach().clone()
T=torch.load(f"{W}/targets_dense_{OBJ}.pt", map_location=dev)
for t in T:
    t["n"]=t["n"].to(dev).float(); t["mvp"]=t["mvp"].to(dev).float().contiguous()
    t["tgt"]=t["tgt"].to(dev).float(); t["mask"]=t["mask"].to(dev).float()
print(f"{len(T)} dense targets loaded")
def render(mvp,n,d,grad=False):
    st["interior"]=di() if grad else di().detach()
    ctx=torch.enable_grad() if grad else torch.no_grad()
    with ctx: return ON.render_section(st,glctx,mvp,n,float(d),S,exterior=True)
for pr in di.parameters(): pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt=torch.optim.AdamW([di.feat],lr=LR)
t0=time.time()
for step in range(1,STEPS+1):
    loss=0.0
    for t in T:
        img,_,_,_=render(t["mvp"],t["n"],t["d"],grad=True)
        m=t["mask"][0]
        loss=loss+(((img*2-1)-t["tgt"][0])**2*m).sum()/m.sum().clamp_min(1)
    reg=ANCHOR*((di()-rgb0)**2).mean()
    (loss/len(T)+reg).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if step%50==0 or step==1:
        print(f"  step {step:4d}  face {float(loss)/len(T):.4f}  anchor {float(reg):.4f}  {time.time()-t0:.0f}s",flush=True)
out=f"{W}/s_v2_mt3_{OBJ}"; os.makedirs(out,exist_ok=True)
q=dict(p); q["dec_i"]={k:v.clone().detach() for k,v in p["dec_i"].items()}; q["dec_i"]["feat"]=di.feat.detach().clone()
torch.save(q,f"{out}/params.pt"); open(f"{out}/run.env","w").write("CAMS_SUFFIX=_v2\n")
print(f"wrote {out}/params.pt")
# score
@torch.no_grad()
def score(planes,mvps,photos,tag):
    ref=realism._paths(photos); st["interior"]=di().detach(); paths=[]
    d=f"{W}/mt_{tag}"; os.makedirs(d,exist_ok=True)
    for k in range(len(planes)):
        n=planes[k,:3]/np.linalg.norm(planes[k,:3])
        img,_,_,_=render(torch.as_tensor(mvps[k] if mvps.ndim==3 else mvps,dtype=torch.float32,device=dev),
                         torch.as_tensor(n.astype(np.float32),device=dev),float(planes[k,3]))
        f=f"{d}/{k:02d}.png"; from PIL import Image
        Image.fromarray((img.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(f); paths.append(f)
    return realism._dreamsim(ref,paths,dev),len(ref)
HL,HH=int(C["h_lo"][0]),int(C["h_hi"][0])
hM=np.broadcast_to(C["h_mvp"][None],(HH-HL,4,4))
suites={"held long":(C["ev_planes"],C["ev_mvp"],f"{FN}/hld_orange_v"),
        "held trans":(C["eh_planes"],C["eh_mvp"],f"{FN}/hld_orange_h"),
        "supv long":(C["v_planes"],C["v_mvp"],f"{FN}/spl_orange_v"),
        "supv trans":(C["h_planes"][HL:HH],hM,f"{FN}/spl_orange_h")}
print("\n  suite         Stage1   matched   delta")
for k,(P,M,ph) in suites.items():
    di.load_state_dict(p["dec_i"]); b,nb=score(P,M,ph,k.replace(" ","_")+"_0")
    di.load_state_dict(q["dec_i"]); a,na=score(P,M,ph,k.replace(" ","_")+"_1")
    print(f"  {k:12s} {b:.4f}  {a:.4f}  {a-b:+.4f}")
