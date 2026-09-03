# DiffusionBlend inference: at every step, RE-PARTITION the axis at a random offset, denoise each
# group of K adjacent slices together, and let the changing partition blend the groups over time.
# Nothing is averaged across a fixed seam, so no seam is baked in.
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from PIL import Image,ImageDraw
dv=os.environ.get("DEV","cuda:1")
G=torch.load(os.environ["GRID"],map_location=dv)
V0=G["V"].to(dv); CORE=G["CORE"].to(dv); SHELL=G["SHELL"].to(dv); N=G["N"]
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
T0=float(os.environ.get("T0","0.5")); STEPS=int(os.environ.get("STEPS","100"))
WLONG=float(os.environ.get("WLONG","1.0")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
class PatchNet(nn.Module):
    def __init__(s,c=64,mult=(1,2,4),K=3):
        super().__init__()
        s.net=UNet2D(c,mult); s.K=K
        s.net.inp=nn.Conv2d(3*K,c*mult[0],3,padding=1); s.net.out=nn.Conv2d(c*mult[0],3*K,3,padding=1)
        s.pos=nn.Sequential(nn.Linear(1,c),nn.SiLU(),nn.Linear(c,c*4))
    def forward(s,x,t,p):
        e=s.net.e(emb(t,s.net.c))+s.pos(p[:,None])
        h=s.net.inp(x); sk=[]
        for blk,d in zip(s.net.down,s.net.ds): h=blk(h,e); sk.append(h); h=d(h)
        for u,blk in zip(s.net.us,s.net.up):
            h=u(h); k=sk.pop()
            if h.shape[-1]!=k.shape[-1]: h=F.interpolate(h,size=k.shape[-2:],mode="nearest")
            h=blk(torch.cat([h,k],1),e)
        return s.net.out(F.silu(s.net.on(h)))
def load(p):
    d=torch.load(p,map_location=dv)
    m=PatchNet(64,tuple(d["MULT"]),d["K"]).to(dv); m.load_state_dict(d["sd"]); m.eval()
    return m,d["K"],d["AX"]
MH,KH,_=load(os.environ["CKH"]); MV,KV,_=load(os.environ["CKV"])
g=torch.Generator(device=dv).manual_seed(0)
def get(X,ax,i):
    return X[:,:,i,:] if ax==1 else (X[:,i,:,:] if ax==0 else X[:,:,:,i])
def put(E,ax,i,v):
    if ax==1: E[:,:,i,:]=v
    elif ax==0: E[:,i,:,:]=v
    else: E[:,:,:,i]=v
def eps_axis(X,ax,model,Kp,t,jump,bs=16):
    """DiffusionBlend's partition, following the reference implementation.

    Groups are NON-OVERLAPPING and each group's prediction is written straight into the score
    tensor -- there is no averaging anywhere, which is the whole point: six aggregation rules in
    this project have averaged the detail away.  The partition is re-drawn every step (random
    offset), and alternate steps use interleaved triplets instead of adjacent ones, which is what
    carries consistency further along the axis than one triplet reaches.
    """
    off=int(torch.randint(Kp,(1,),device=dv,generator=g))
    step=Kp*Kp if jump else Kp
    starts=list(range(off,N-(Kp-1)*(Kp if jump else 1),step))
    E=torch.zeros_like(X)
    batch=[]; meta=[]
    for s0 in starts:
        idx=[s0+d*(Kp if jump else 1) for d in range(Kp)]
        if idx[-1]>=N: continue
        batch.append(torch.cat([get(X,ax,i) for i in idx],0)); meta.append(idx)
    for k in range(0,len(batch),bs):
        bb=torch.stack(batch[k:k+bs])
        p=torch.tensor([meta[k+j][0]/(N-1) for j in range(len(bb))],device=dv,dtype=torch.float32)
        with torch.no_grad():
            o=model(bb,torch.full((len(bb),),t,device=dv,dtype=torch.long),p)
        for n_ in range(len(bb)):
            for d,i in enumerate(meta[k+n_]):
                put(E,ax,i,o[n_,3*d:3*d+3])        # direct assignment, never accumulated
    return E

iT=max(int(T0*(T-1)),1)
X=V0*(1-CORE)+(ab[iT].sqrt()*V0+(1-ab[iT]).sqrt()*torch.randn(3,N,N,N,device=dv,generator=g))*CORE
ts=[int(round(v)) for v in np.linspace(iT,0,STEPS+1)]
print(f"  DiffusionBlend sampling: K={KH}/{KV}, {STEPS} steps from t={iT}",flush=True)
t0=time.time()
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    # families alternate rather than being averaged -- averaging them is the aggregation that
    # cost the detail every previous time.
    jump=(i%2==1)
    E=eps_axis(X,1,MH,KH,tc,jump) if i%2==0 else eps_axis(X,0,MV,KV,tc,jump)
    x0=((X-(1-ab[tc]).sqrt()*E)/ab[tc].sqrt()).clamp(-1,1)
    x0=V0*(1-CORE)+x0*CORE
    if tn<=0: X=x0; break
    X=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*torch.randn(X.shape,device=dv,generator=g)
    X=V0*(1-CORE)+X*CORE
    if i%20==0: print(f"    step {i+1}/{STEPS}  t={tc}  {time.time()-t0:.0f}s",flush=True)
m=CORE[0]>0.5
def sharp(t):
    a=t[:,:,:,N//2].mean(0)
    return float((a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]).abs().mean())
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
for l,t in (("O-Voxel init",V0),("DiffusionBlend",X)):
    r=((t[:,m]+1)/2)
    print(f"  {l:16s} sat {float((r[0]-r[2]).mean()):+.3f}  sharpness {sharp(t):.4f}",flush=True)
torch.save({"X":X.cpu()},OUT+"/state.pt")
s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
for i,(t,lab) in enumerate(((V0,"init longitudinal"),(X,"DB longitudinal"),
                            (V0,"init transverse"),(X,"DB transverse"))):
    sl=t[:,:,:,N//2] if i<2 else t[:,:,N//2,:]
    a=((sl.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
    d.text((i*266+5,7),lab,fill=(30,30,30))
s.save(OUT+"/db.png"); print("  wrote",OUT+"/db.png",flush=True)
