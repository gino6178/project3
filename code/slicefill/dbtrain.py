# DiffusionBlend's prior: a diffusion model over a THIN 3-D PATCH -- K adjacent slices denoised
# together, with the patch's position along the axis encoded alongside the timestep.
#
# Every aggregation rule tried so far averaged the detail away, because each slice was generated
# without knowing its neighbours and the volume had to reconcile them afterwards.  Here the model
# sees the neighbours, so adjacent slices come out mutually consistent and nothing has to be
# reconciled.  K slices cost K times a single slice, far less than a full 3-D model.
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
dv=os.environ.get("DEV","cuda:1")
G=torch.load(os.environ["GRID"],map_location=dv)
V0=G["V"].to(dv); OCC=G["OCC"].to(dv); CORE=G["CORE"].to(dv); N=G["N"]
K=int(os.environ.get("K","3")); AX=int(os.environ.get("AX","1"))   # 1 = transverse (polar axis)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
STEPS=int(os.environ.get("STEPS","8000")); BS=int(os.environ.get("BS","8"))
LR=float(os.environ.get("LR","5e-4")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)

class PatchNet(nn.Module):
    """the same UNet, but 3K channels in and out, plus a position embedding beside the time one"""
    def __init__(s,c=64,mult=(1,2,4),K=3):
        super().__init__()
        s.net=UNet2D(c,mult); s.K=K
        s.net.inp=nn.Conv2d(3*K,c*mult[0],3,padding=1)
        s.net.out=nn.Conv2d(c*mult[0],3*K,3,padding=1)
        nn.init.zeros_(s.net.out.weight); nn.init.zeros_(s.net.out.bias)
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
m=PatchNet(64,MULT,K).to(dv)
print(f"  DiffusionBlend prior: K={K} slices, axis {AX}, "
      f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M params, {N}^2 slices",flush=True)
opt=torch.optim.AdamW(m.parameters(),LR); g=torch.Generator(device=dv).manual_seed(0)
def slab(z,stride=1):
    """K slices along AX as one (3K,N,N) image.  stride=1 is the adjacent triplet the blend step
    uses; stride=K is the interleaved one the jumping step uses.  Inference alternates between
    them, so training has to have seen both or the jumping step is out of distribution."""
    idx=[min(max(z+d*stride,0),N-1) for d in range(K)]
    if AX==1: sl=[V0[:,:,i,:] for i in idx]
    elif AX==0: sl=[V0[:,i,:,:] for i in idx]
    else: sl=[V0[:,:,:,i] for i in idx]
    return torch.cat(sl,0)
occ_z=[z for z in range(N-K+1) if float(OCC[0,:,z:z+K,:].sum() if AX==1 else OCC.sum())>50]
print(f"  {len(occ_z)} admissible patch positions",flush=True)
t0=time.time()
for it in range(1,STEPS+1):
    zs=[occ_z[int(torch.randint(len(occ_z),(1,),device=dv,generator=g))] for _ in range(BS)]
    st=[1 if float(torch.rand(1,device=dv,generator=g))<0.5 else K for _ in zs]
    x0=torch.stack([slab(z,s_) for z,s_ in zip(zs,st)])
    p=torch.tensor([z/(N-1) for z in zs],device=dv,dtype=torch.float32)
    t=torch.randint(T,(BS,),device=dv,generator=g)
    eps=torch.randn(x0.shape,device=dv,generator=g); a_=ab[t].view(-1,1,1,1)
    loss=F.mse_loss(m(a_.sqrt()*x0+(1-a_).sqrt()*eps,t,p),eps)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it%1000==0 or it==1:
        print(f"    {it}/{STEPS}  loss {float(loss.detach()):.4f}  {time.time()-t0:.0f}s",flush=True)
torch.save({"sd":m.state_dict(),"K":K,"AX":AX,"MULT":MULT,"N":N},f"{OUT}/model.pt")
print("DB_TRAIN_DONE",flush=True)
