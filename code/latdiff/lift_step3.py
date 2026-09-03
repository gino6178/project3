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

import torch.nn.functional as _F
_GYCC=(0.738,0.254,0.645)   # measured orange flesh YCbCr (photo distribution); recompute per object
def _rgb2ycc(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    return torch.stack([0.299*r+0.587*g+0.114*b,-0.168736*r-0.331264*g+0.5*b+0.5,0.5*r-0.418688*g-0.081312*b+0.5],1)
def _ycc2rgb(x):
    y,cb,cr=x[:,0],x[:,1]-0.5,x[:,2]-0.5
    return torch.stack([y+1.402*cr,y-0.344136*cb-0.714136*cr,y+1.772*cb],1).clamp(0,1)
def _mb(x,m,k=31):
    p=k//2; return _F.avg_pool2d(x*m,k,1,p)/_F.avg_pool2d(m,k,1,p).clamp_min(1e-6)
def _recolour(render01,mask):
    "keep render structure, set flesh mean colour to the global photo distribution"
    r=_rgb2ycc(render01); gy,gcb,gcr=_GYCC
    def rc(ch,t):
        mn=(ch*mask).sum()/mask.sum().clamp_min(1); return ch-mn+t
    y=r[:,:1]-_mb(r[:,:1],mask)+rc(_mb(r[:,:1],mask),gy)
    return _ycc2rgb(torch.stack([y[:,0],rc(r[:,1:2],gcb)[:,0],rc(r[:,2:3],gcr)[:,0]],1))
W="/workspace/ovoxel_native"; FN="/workspace/rebuild/worktree"; OBJ="orange_sp"; dev="cuda"; S=256
STEPS=int(os.environ.get("STEPS","500"))
TAG=os.environ.get("TAG","mt21"); NORMTV=float(os.environ.get("NORMTV","0.0")); NORMDD=float(os.environ.get("NORMDD","1.5")); LNORM=float(os.environ.get("LNORM","0.0")); LAP=float(os.environ.get("LAP","0.0")); LR=float(os.environ.get("LR","5e-3")); ANCHOR=float(os.environ.get("ANCHOR","1.0"))
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
T=torch.load(f"{W}/targets_lc3_{OBJ}.pt", map_location=dev)
for t in T:
    t["n"]=t["n"].to(dev).float(); t["mvp"]=t["mvp"].to(dev).float().contiguous()
    t["tgt"]=t["tgt"].to(dev).float(); t["mask"]=t["mask"].to(dev).float()
# Supervised planes are frozen to the Stage-1 render: they already fit real photographs, so their
# target is where the fit put them, and only the held-out planes are pulled to the match+SDEdit
# target. This is the one change from the run that regressed the supervised suites by +0.024.
with torch.no_grad():
    for t in T:
        if not t["name"].endswith("_h"):    # supervised
            st["interior"]=di().detach()
            img,_,_,_=ON.render_section(st,glctx,t["mvp"],t["n"],t["d"],S,exterior=True)
            m=t["mask"][0:1].float()
            sw=_recolour(img[None].clamp(0,1),m)
            t["tgt"]=((m*sw+(1-m))*2-1).detach()
nsup=sum(1 for t in T if not t["name"].endswith("_h")); nhld=len(T)-nsup
print(f"{len(T)} targets: {nsup} supervised (frozen to render), {nhld} held-out (match+SDEdit)")
def render(mvp,n,d,grad=False):
    st["interior"]=di() if grad else di().detach()
    ctx=torch.enable_grad() if grad else torch.no_grad()
    with ctx: return ON.render_section(st,glctx,mvp,n,float(d),S,exterior=True)
for pr in di.parameters(): pr.requires_grad_(False)
di.feat.requires_grad_(True)
feat0=di.feat.detach().clone(); norm0=feat0.norm(dim=1)
# 3D Laplacian neighbour graph: undirected axis edges over the whole lattice (C1 smoothness / grid removal)
_sc=st["solid"].cpu().numpy().astype(np.int64)
_key=(_sc[:,0]*100003+_sc[:,1])*100003+_sc[:,2]
_pos={int(k):i for i,k in enumerate(_key)}
_EI=[]; _EJ=[]
for a in range(3):
    sh=_sc.copy(); sh[:,a]+=1
    shk=(sh[:,0]*100003+sh[:,1])*100003+sh[:,2]
    j=np.array([_pos.get(int(k),-1) for k in shk],dtype=np.int64)
    v=np.nonzero(j>=0)[0]; _EI.append(v); _EJ.append(j[v])
_EI=torch.as_tensor(np.concatenate(_EI),device=dev); _EJ=torch.as_tensor(np.concatenate(_EJ),device=dev)
print(f"3D Laplacian: {len(_EI)} undirected lattice edges, LAP={LAP}")
opt=torch.optim.AdamW([di.feat],lr=LR)
t0=time.time()
for step in range(1,STEPS+1):
    loss=0.0; ntv=0.0
    dd=NORMDD*float(st["hc"])
    for t in T:
        img,_,_,_=render(t["mvp"],t["n"],t["d"],grad=True)
        m=t["mask"][0]
        loss=loss+(((img*2-1)-t["tgt"][0])**2*m).sum()/m.sum().clamp_min(1)
        if NORMTV>0 and t["name"].endswith("_h"):
            img2,af2,_,_=render(t["mvp"],t["n"],t["d"]+dd,grad=True)
            m2=(af2[:1]>0).float()*m               # where both depths carry flesh
            ntv=ntv+((img-img2)**2*m2).sum()/m2.sum().clamp_min(1)
    rgb=di()
    reg=ANCHOR*((rgb-rgb0)**2).mean()
    lap=rgb.new_zeros(())
    if LAP>0:
        d=rgb[_EJ]-rgb[_EI]
        L=torch.zeros_like(rgb); L.index_add_(0,_EI,d); L.index_add_(0,_EJ,-d)   # discrete Laplacian per cell
        lap=(L**2).mean()
    (loss/len(T)+reg+NORMTV*ntv/len(T)+LAP*lap).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if LNORM>0:
        with torch.no_grad():
            cur=di.feat.norm(dim=1); scale=(LNORM*norm0.clamp_min(1e-6)/cur.clamp_min(1e-6)).clamp(max=1.0)
            di.feat.mul_(scale[:,None])
    if step%50==0 or step==1:
        print(f"  step {step:4d}  face {float(loss)/len(T):.4f}  anchor {float(reg):.4f}  lap {float(lap) if LAP>0 else 0:.4f}  ntv {float(ntv)/len(T) if NORMTV>0 else 0:.4f}  {time.time()-t0:.0f}s",flush=True)
out=f"{W}/s_v2_{TAG}_{OBJ}"; os.makedirs(out,exist_ok=True)
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
