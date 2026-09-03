# Cross-plane CONDITIONING, the thing eight aggregation rules could not buy.
#
# The two families cover each other completely: sampled at height h, the longitudinal family is a
# full prediction of the transverse plane at h, and vice versa.  So a plane can be denoised WITH
# the other family's account of it as an extra input, and consistency becomes something the model
# produces rather than something a rule imposes afterwards.  MVDream inflates self-attention to do
# this across views; SinDiffusion has no attention by design, and conditioning gets the coupling
# without giving that up.
#
# The condition must be DEGRADED during training.  At training time the other family's view of a
# plane is the plane itself, so an undegraded condition would let the model copy it and learn
# nothing -- and at inference it would be handed an imperfect estimate and collapse.  Noise and
# blur it at a random strength, which is cascaded diffusion's conditioning augmentation.
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
dv=os.environ.get("DEV","cuda:1")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
N=vol.N; T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
FAM=os.environ.get("FAM","long")
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
STEPS=int(os.environ.get("STEPS","8000")); BS=int(os.environ.get("BS","4"))
LR=float(os.environ.get("LR","5e-4")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
PDIR=os.environ.get("PDIR","")

class CondNet(nn.Module):
    """the same UNet with three extra input channels for the other family's view"""
    def __init__(s,c=64,mult=(1,2,4)):
        super().__init__()
        s.net=UNet2D(c,mult)
        s.net.inp=nn.Conv2d(6,c*mult[0],3,padding=1)
    def forward(s,x,t,cond):
        return s.net(torch.cat([x,cond],1),t)
m=CondNet(64,MULT).to(dv)
print(f"  cross-conditioned prior ({FAM}): {sum(p.numel() for p in m.parameters())/1e6:.2f}M params",flush=True)
opt=torch.optim.AdamW(m.parameters(),LR); g=torch.Generator(device=dv).manual_seed(0)
H=vol.occupied_heights()
import glob as _glob
from PIL import Image as _Image
def _photos(d,res,target=0.82):
    out=[]
    for p in sorted(_glob.glob(d+"/*.png")):
        a=np.asarray(_Image.open(p).convert("RGB")).astype(np.float32)/255
        mm=a.min(2)<0.92; ys,xs=np.where(mm)
        if len(ys)<100: continue
        cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
        half=max(ys.max()-ys.min(),xs.max()-xs.min())/2/target
        y0,y1,x0,x1=int(cy-half),int(cy+half),int(cx-half),int(cx+half)
        pad=max(0,-y0,-x0,y1-a.shape[0],x1-a.shape[1])
        if pad:
            a=np.pad(a,((pad,pad),(pad,pad),(0,0)),constant_values=1.0); y0+=pad;y1+=pad;x0+=pad;x1+=pad
        im=np.asarray(_Image.fromarray((a[y0:y1,x0:x1]*255).astype(np.uint8))
                      .resize((res,res),_Image.LANCZOS)).astype(np.float32)/255
        out.append(torch.from_numpy(im).permute(2,0,1).to(dv)*2-1)
    return out
PHOTOS=_photos(PDIR,vol.res) if PDIR else []
print(f"  {len(PHOTOS)} photographs" if PHOTOS else "  training on volume planes",flush=True)
def degrade(x):
    """what the other family's estimate will actually look like: noisy and soft, at random strength"""
    s=float(torch.rand(1,device=dv,generator=g))
    k=int(1+2*round(3*s))
    y=F.avg_pool2d(F.pad(x,(k//2,)*4,mode="replicate"),k,1) if k>1 else x
    return y+torch.randn(y.shape,device=dv,generator=g)*(0.35*s)
def draw():
    im=[]
    for _ in range(BS):
        if PHOTOS:
            t=PHOTOS[int(torch.randint(len(PHOTOS),(1,),device=dv,generator=g))]
            if float(torch.rand(1,device=dv,generator=g))<0.5: t=t.flip(-1)
            im.append(t)
        elif FAM=="long":
            im.append(vol.long_slice(vol.V0,float(torch.rand(1,device=dv,generator=g))*math.pi))
        else:
            im.append(vol.trans_slice(vol.V0,H[int(torch.randint(len(H),(1,),device=dv,generator=g))]))
    return torch.stack(im)
t0=time.time()
for it in range(1,STEPS+1):
    x0=draw()
    cond=degrade(x0)                       # the other family's view of this same plane
    t=torch.randint(T,(BS,),device=dv,generator=g)
    eps=torch.randn(x0.shape,device=dv,generator=g); a_=ab[t].view(-1,1,1,1)
    loss=F.mse_loss(m(a_.sqrt()*x0+(1-a_).sqrt()*eps,t,cond),eps)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it%1000==0 or it==1:
        print(f"    {it}/{STEPS}  loss {float(loss.detach()):.4f}  {time.time()-t0:.0f}s",flush=True)
torch.save({"sd":m.state_dict(),"MULT":MULT},f"{OUT}/model.pt")
print("XTRAIN_DONE",flush=True)
