# TPDM-style: alternate the two families directly on the volume's own axes.
#
# The volume is in the fruit's frame, so slices at fixed dim1 ARE the transverse family and slices
# at fixed dim0/dim2 ARE longitudinal ones -- at native resolution, with no resampling.  Every
# grid_sample in the plane-gather path is a low-pass filter, and memory records six independent
# aggregation schemes that all averaged away the detail they were meant to add; removing the
# resampling removes one of the two mechanisms that can do that.
import os,sys,math,time,numpy as np,torch,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from PIL import Image,ImageDraw
dv=os.environ.get("DEV","cuda:1")
G=torch.load(os.environ["GRID"],map_location=dv)
V0=G["V"].to(dv); CORE=G["CORE"].to(dv); SHELL=G["SHELL"].to(dv); OCC=G["OCC"].to(dv); N=G["N"]
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
T0=float(os.environ.get("T0","0.5")); STEPS=int(os.environ.get("STEPS","100"))
WLONG=float(os.environ.get("WLONG","1.0")); ALT=int(os.environ.get("ALT","1"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p):
    m=UNet2D(64,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(0)
def stack(X,ax):
    if ax==0: return X.permute(1,0,2,3).contiguous()
    if ax==1: return X.permute(2,0,1,3).contiguous()
    return X.permute(3,0,1,2).contiguous()
def unstack(S,ax):
    if ax==0: return S.permute(1,0,2,3).contiguous()
    if ax==1: return S.permute(1,2,0,3).contiguous()
    return S.permute(1,2,3,0).contiguous()
def eps_axis(X,ax,model,t,bs=24):
    S=stack(X,ax); out=torch.empty_like(S)
    for k in range(0,len(S),bs):
        b=S[k:k+bs]
        with torch.no_grad():
            out[k:k+bs]=model(b,torch.full((len(b),),t,device=dv,dtype=torch.long))
    return unstack(out,ax)
iT=max(int(T0*(T-1)),1)
X=V0*(1-CORE)+(ab[iT].sqrt()*V0+(1-ab[iT]).sqrt()*torch.randn(3,N,N,N,device=dv,generator=g))*CORE
ts=[int(round(v)) for v in np.linspace(iT,0,STEPS+1)]
print(f"  TPDM on {N}^3, native slices, {STEPS} steps from t={iT}, alt={ALT}",flush=True)
t0=time.time()
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    if ALT:                                  # one family per step, alternating
        E = eps_axis(X,1,MH,tc) if i%2==0 else 0.5*(eps_axis(X,0,MV,tc)+eps_axis(X,2,MV,tc))
    else:
        E = (eps_axis(X,1,MH,tc)+WLONG*0.5*(eps_axis(X,0,MV,tc)+eps_axis(X,2,MV,tc)))/(1+WLONG)
    x0=((X-(1-ab[tc]).sqrt()*E)/ab[tc].sqrt()).clamp(-1,1)
    x0=V0*(1-CORE)+x0*CORE
    if tn<=0: X=x0; break
    X=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*torch.randn(X.shape,device=dv,generator=g)
    X=V0*(1-CORE)+X*CORE
    if i%20==0: print(f"    step {i+1}/{STEPS}  t={tc}  {time.time()-t0:.0f}s",flush=True)
X=V0*(1-CORE)+X*CORE
m=CORE[0]>0.5
def sharp(t):
    a=t[:,:,:,N//2].mean(0)
    return float((a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]).abs().mean())
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
for l,t in (("O-Voxel init",V0),("TPDM",X)):
    r=((t[:,m]+1)/2)
    print(f"  {l:14s} sat {float((r[0]-r[2]).mean()):+.3f}  sharpness {sharp(t):.4f}",flush=True)
torch.save({"X":X.cpu()},OUT+"/state.pt")
s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
for i,(t,lab) in enumerate(((V0,"init longitudinal"),(X,"TPDM longitudinal"),
                            (V0,"init transverse"),(X,"TPDM transverse"))):
    sl=t[:,:,:,N//2] if i<2 else t[:,:,N//2,:]
    a=((sl.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
    d.text((i*266+5,7),lab,fill=(30,30,30))
s.save(OUT+"/tpdm.png"); print("  wrote",OUT+"/tpdm.png",flush=True)
