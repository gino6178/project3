# One single-image prior on a folder of images, taken as they are (no re-canonicalisation: the
# procedural photographs are already in the frame the sampler slices).  The recipe is the paper's
# (sd2d_net UNet, linear beta, lr 5e-4, unmasked); only the augmentation follows the image kind:
#   KIND=long    canonical longitudinal frames     mirror in s
#   KIND=polar   (r, phi) strips                   roll and mirror in phi    (= polar_train.py)
#   KIND=cart    canonical transverse frames       the eight rotations/mirrors of the square
#   PDIR=... OUT=... KIND=... STEPS=... MULT=1,2 [POLAR=1] python train_prior.py
import os,sys,glob,time,numpy as np,torch,torch.nn as nn
SPD=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill"); sys.path.insert(0,SPD)
exec(open(SPD+"/sd2d_net.py").read())
from PIL import Image
dv=os.environ.get("DEV","cuda:0"); PDIR=os.environ["PDIR"]; OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
KIND=os.environ.get("KIND","long"); RES=int(os.environ.get("RES","256"))
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2").split(",")); STEPS=int(os.environ.get("STEPS","30000"))
BS=int(os.environ.get("BS","4")); LR=float(os.environ.get("LR","5e-4"))
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
def rd(p):
    im=Image.open(p).convert("RGB")
    if KIND!="polar": im=im.resize((RES,RES),Image.LANCZOS)
    return torch.from_numpy(np.asarray(im).astype(np.float32)/127.5-1).permute(2,0,1)
imgs=torch.stack([rd(p) for p in sorted(glob.glob(PDIR+"/*.png"))]).to(dv)
print(f"  {KIND} prior on {tuple(imgs.shape)}, mult {MULT}, {STEPS} steps",flush=True)
m=UNet2D(64,MULT).to(dv); opt=torch.optim.AdamW(m.parameters(),LR); t0=time.time()
for it in range(1,STEPS+1):
    x0=imgs[torch.randint(0,len(imgs),(BS,),device=dv)]
    if KIND=="polar": x0=torch.stack([torch.roll(x,int(torch.randint(0,x.shape[-1],(1,))),dims=-1) for x in x0])
    if KIND=="cart": x0=torch.rot90(x0,int(torch.randint(0,4,(1,))),dims=(-2,-1))
    if torch.rand(1)<0.5: x0=x0.flip(-1)
    t=torch.randint(0,T,(BS,),device=dv); e=torch.randn_like(x0)
    xt=ab[t].sqrt()[:,None,None,None]*x0+(1-ab[t]).sqrt()[:,None,None,None]*e
    loss=((m(xt,t)-e)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if it%max(1,STEPS//6)==0 or it==1: print(f"    {it}/{STEPS}  loss {float(loss):.4f}  {time.time()-t0:.0f}s",flush=True)
torch.save(m.state_dict(),f"{OUT}/model.pt"); print("PRIOR_DONE",flush=True)
