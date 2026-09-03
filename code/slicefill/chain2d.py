# The reverse chain: noise added once to the interior, then each step draws one transverse plane
# at a random height and one longitudinal plane at a random azimuth, denoises those two images
# with their family's prior, and writes the result into the slab beside each plane with a
# distance weight.  That write is how a plane's answer reaches the volume around it; cells no
# plane touched this step keep their level and are picked up by a later draw.
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image,ImageDraw
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
dv=os.environ.get("DEV","cuda:1")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
N=vol.N; T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MCH=int(os.environ.get("MCH","64"))
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2").split(","))
STEPS=int(os.environ.get("STEPS","50")); T0=float(os.environ.get("T0","0.6"))
SLABW=float(os.environ.get("SLABW","2.0")); WMAX=float(os.environ.get("WMAX","0.6"))
FR=int(os.environ.get("FRAME_EVERY","5")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p):
    m=UNet2D(MCH,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(int(os.environ.get("SEED","0")))
i_T=max(int(T0*(T-1)),1)
V0,CORE,SHELL,OCC=vol.V0,vol.CORE,vol.SHELL,vol.OCC
X=V0*(1-CORE)+(ab[i_T].sqrt()*V0+(1-ab[i_T]).sqrt()*torch.randn(3,N,N,N,device=dv,generator=g))*CORE
print(f"  init: interior noised once to t={i_T}; shell ({int(SHELL.sum())} voxels) fixed",flush=True)
print(f"  {STEPS} steps, 1 transverse + 1 longitudinal per step, slab {SLABW} cells",flush=True)
WBACK=os.environ.get("WBACK","next")
def step_img(img,model,tc,tn):
    """WBACK=next writes the next-level NOISY state back; WBACK=x0 writes the clean estimate.
    Many different planes write to the same cells, and their epsilon fields do not cancel, so
    writing the noisy state injects sqrt(1-abar)~0.99 of fresh noise on every visit while the
    signal it carries is only sqrt(abar)~0.16 -- the field washes out no matter how good the
    prior is.  Writing x0 leaves the noise schedule to the chain instead of to each plane."""
    with torch.no_grad():
        e=model(img[None],torch.full((1,),tc,device=dv,dtype=torch.long))[0]
    x0=((img-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()).clamp(-1,1)
    if tn<=0 or WBACK=="x0": return x0
    return ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*e
frames=[]
def snap(i,t):
    a=((X[:,:,:,N//2].clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    b=((X[:,:,N//2,:].clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s=Image.new("RGB",(540,296),(255,255,255)); d=ImageDraw.Draw(s)
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(4,26))
    s.paste(Image.fromarray(b).resize((260,260),Image.NEAREST),(276,26))
    d.text((6,7),f"step {i}/{STEPS}   t={t}   signal {float(ab[t].sqrt()):.2f}",fill=(30,30,30))
    d.text((6,286),"longitudinal",fill=(120,120,120)); d.text((278,286),"transverse",fill=(120,120,120))
    frames.append(s)
H=vol.occupied_heights(); ts=[int(round(x)) for x in np.linspace(i_T,0,STEPS+1)]
cov=torch.zeros(N,N,N,device=dv); t0=time.time(); snap(0,ts[0]); PREVX=X.clone(); _mm=(OCC[0]>0.5)
# NH/NV planes per step per family.  0 means EVERY plane of that family: all occupied heights
# for the transverse one, and the same number of azimuths spread over [0,pi) for the longitudinal
# one.  Covering the whole family every step removes the random-draw accumulation entirely --
# no cell waits for a plane to happen to land near it.
NH=int(os.environ.get("NH","1")); NV=int(os.environ.get("NV","1"))
# The prior was trained on FULLY noised planes, so that is what it must be handed -- a plane whose
# rind is clean and whose interior is noise is a combination it never saw, and it returns flat
# brown (measured: 0.5151 against 0.1163 when tested the way it was trained).  So noise the whole
# extracted IMAGE to the current level before the call, and keep the shell fixed where it actually
# matters: in the write-back, which only ever touches the interior.
NOISE_ALL=int(os.environ.get("NOISE_ALL","0"))
def prep(img,mk,tc):
    if not NOISE_ALL: return img
    n=torch.randn(img.shape,device=dv,generator=g)
    xt=ab[tc].sqrt()*img+(1-ab[tc]).sqrt()*n
    return img*mk+xt*(1-mk)          # interior already carries the chain's noise; noise the rest
def batched(imgs,model,tc,tn,bs=16):
    out=[]
    for k in range(0,len(imgs),bs):
        b=torch.stack(imgs[k:k+bs])
        with torch.no_grad():
            e=model(b,torch.full((len(b),),tc,device=dv,dtype=torch.long))
        x0=((b-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()).clamp(-1,1)
        out.append(x0 if (tn<=0 or WBACK=="x0") else ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*e)
    return torch.cat(out)
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    hs = H if NH==0 else [H[int(torch.randint(len(H),(1,),device=dv,generator=g))] for _ in range(NH)]
    nv = len(H) if NV==0 else NV
    off=float(torch.rand(1,device=dv,generator=g))*math.pi/max(nv,1)
    thetas=[off+math.pi*k/nv for k in range(nv)] if NV==0 else \
           [float(torch.rand(1,device=dv,generator=g))*math.pi for _ in range(nv)]
    C3=vol.CORE.expand(3,-1,-1,-1).contiguous()
    outs=batched([prep(vol.trans_slice(X,j),(vol.trans_slice(C3,j)>0.5).float(),tc) for j in hs],MH,tc,tn)
    for n_,j in enumerate(hs): X=vol.write_trans(X,j,outs[n_],WMAX,SLABW)
    outs=batched([prep(vol.long_slice(X,th),(vol.long_slice(C3,th)>0.5).float(),tc) for th in thetas],MV,tc,tn)
    for n_,th in enumerate(thetas): X=vol.write_long(X,th,outs[n_],WMAX,SLABW)
    if WBACK=="chain" and tn>0:
        # Every plane wrote a CLEAN estimate, so X is now the volume's x0 -- not a sample at
        # t_next.  Feeding it back as if it were one puts the model out of distribution and the
        # chain stops converging (measured: per-step change flat at ~0.3, saturation oscillating
        # with period 2).  Re-noise the whole volume ONCE, with one epsilon, so the next call
        # sees a genuine x_{t_next}.
        X=(ab[tn].sqrt()*X+(1-ab[tn]).sqrt()*torch.randn(X.shape,device=dv,generator=g))
    X=V0*(1-CORE)+X*CORE                       # interior only, always
    for j in hs: cov+=((vol.v-(j-vol.c)).abs()<3*SLABW).float()
    if (i+1)%FR==0 or i==STEPS-1: snap(i+1,tc)
    _d=float((X-PREVX)[:,_mm].abs().mean()); PREVX=X.clone()
    _r=((X[:,_mm]+1)/2); _s=float((_r[0]-_r[2]).mean())
    if i<8 or i%max(1,STEPS//8)==0:
        print(f"    step {i+1:4d}  t={tc:4d}  |change| {_d:.4f}  sat {_s:+.3f}",flush=True)
m=OCC[0]>0.5
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
print(f"  transverse coverage: mean {float(cov[m].mean()):.1f} hits/voxel, "
      f"min {float(cov[m].min()):.0f}",flush=True)
frames[0].save(OUT+"/chain.gif",save_all=True,
    append_images=[f.convert("P",palette=Image.ADAPTIVE,colors=256,dither=Image.Dither.NONE) for f in frames[1:]],
    duration=[700]+[430]*(len(frames)-2)+[2200],loop=0,disposal=2)
torch.save({"X":X.cpu()},OUT+"/state.pt")
print("  wrote",OUT+"/chain.gif",flush=True); print("CHAIN_DONE",flush=True)
