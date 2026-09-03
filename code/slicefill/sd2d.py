# Two 2-D SinDiffusion priors, one per cut family, trained on the real photographs -- the only
# real data there is.  No attention, no middle block, so the receptive field stays a patch.
import os,math,glob,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image
dv=os.environ.get("DEV","cuda:1"); RES=int(os.environ.get("RES","256"))
TARGET=float(os.environ.get("TARGET","0.82"))
SRC=os.environ["SRC"]; OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
STEPS=int(os.environ.get("STEPS","12000")); BS=int(os.environ.get("BS","8"))
MCH=int(os.environ.get("MCH","64")); T=1000
ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)

def canon(p):
    """centre the section and scale it to TARGET of the frame -- the framing inference uses"""
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255
    m=a.min(2)<0.92
    ys,xs=np.where(m)
    if len(ys)<100: return None
    cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
    half=max(ys.max()-ys.min(),xs.max()-xs.min())/2/TARGET
    y0,y1=int(cy-half),int(cy+half); x0,x1=int(cx-half),int(cx+half)
    pad=max(0,-y0,-x0,y1-a.shape[0],x1-a.shape[1])
    if pad: a=np.pad(a,((pad,pad),(pad,pad),(0,0)),constant_values=1.0); y0+=pad;y1+=pad;x0+=pad;x1+=pad
    return np.asarray(Image.fromarray((a[y0:y1,x0:x1]*255).astype(np.uint8)).resize((RES,RES),Image.LANCZOS)).astype(np.float32)/255
imgs=[canon(p) for p in sorted(glob.glob(SRC+"/*.png"))]
imgs=[i for i in imgs if i is not None]
X=torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).to(dv)*2-1
print(f"  {len(imgs)} photographs from {os.path.basename(SRC)}, canonicalised to {RES}^2",flush=True)

def gn(c):
    for k in (32,16,8,4,2,1):
        if c%k==0: return nn.GroupNorm(k,c)
def emb(t,d):
    h=d//2; f=torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/h)
    a=t.float()[:,None]*f[None]; return torch.cat([a.sin(),a.cos()],1)
class RB(nn.Module):
    def __init__(s,i,o,e):
        super().__init__()
        s.n1=gn(i); s.c1=nn.Conv2d(i,o,3,padding=1); s.e=nn.Linear(e,o)
        s.n2=gn(o); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.sk=nn.Conv2d(i,o,1) if i!=o else nn.Identity()
        nn.init.zeros_(s.c2.weight); nn.init.zeros_(s.c2.bias)
    def forward(s,x,e):
        h=s.c1(F.silu(s.n1(x)))+s.e(e)[:,:,None,None]
        return s.sk(x)+s.c2(F.silu(s.n2(h)))
class UNet2D(nn.Module):
    def __init__(s,c=64,mult=(1,2,4)):
        super().__init__()
        e=c*4; s.c=c; s.e=nn.Sequential(nn.Linear(c,e),nn.SiLU(),nn.Linear(e,e))
        chs=[c*m for m in mult]; s.inp=nn.Conv2d(3,chs[0],3,padding=1)
        s.down=nn.ModuleList(); s.ds=nn.ModuleList(); prev=chs[0]
        for i,ch in enumerate(chs):
            s.down.append(RB(prev,ch,e)); prev=ch
            s.ds.append(nn.Conv2d(ch,ch,3,stride=2,padding=1) if i<len(chs)-1 else nn.Identity())
        s.up=nn.ModuleList(); s.us=nn.ModuleList()
        for i,ch in reversed(list(enumerate(chs))):
            s.us.append(nn.Identity() if i==len(chs)-1 else nn.Upsample(scale_factor=2,mode="nearest"))
            s.up.append(RB(prev+ch,ch,e)); prev=ch
        s.on=gn(prev); s.out=nn.Conv2d(prev,3,3,padding=1)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s,x,t):
        e=s.e(emb(t,s.c)); h=s.inp(x); sk=[]
        for blk,d in zip(s.down,s.ds): h=blk(h,e); sk.append(h); h=d(h)
        for u,blk in zip(s.us,s.up):
            h=u(h); k=sk.pop()
            if h.shape[-1]!=k.shape[-1]: h=F.interpolate(h,size=k.shape[-2:],mode="nearest")
            h=blk(torch.cat([h,k],1),e)
        return s.out(F.silu(s.on(h)))
m=UNet2D(MCH).to(dv)
print(f"  {sum(p.numel() for p in m.parameters())/1e6:.2f}M params",flush=True)
opt=torch.optim.AdamW(m.parameters(),2e-4)
g=torch.Generator(device=dv).manual_seed(0); t0=time.time()
for it in range(1,STEPS+1):
    i=torch.randint(len(X),(BS,),device=dv,generator=g)
    x0=X[i]
    if torch.rand(1,generator=g,device=dv)<0.5: x0=x0.flip(-1)
    t=torch.randint(T,(BS,),device=dv,generator=g)
    eps=torch.randn(x0.shape,device=dv,generator=g); a_=ab[t].view(-1,1,1,1)
    loss=F.mse_loss(m(a_.sqrt()*x0+(1-a_).sqrt()*eps,t),eps)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it%1000==0 or it==1:
        print(f"    {it}/{STEPS}  loss {float(loss.detach()):.4f}  {time.time()-t0:.0f}s",flush=True)
    if it%4000==0 or it==STEPS: torch.save(m.state_dict(),f"{OUT}/model{it:06d}.pt")
print("SD2D_DONE",flush=True)
