# No volume-level diffusion at all.  Every plane of both families is restored ONCE by its own
# 2-D prior -- a complete 2-D reverse chain per plane -- and written back into the interior.
# Each voxel therefore receives an improved value from a transverse estimate and a longitudinal
# one, and the 3-D agreement comes from those two meeting in the volume rather than from any
# third model or any outer loop.
import os,sys,math,time,numpy as np,torch
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image,ImageDraw
dv=os.environ.get("DEV","cuda:1")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
N=vol.N; T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
T0=float(os.environ.get("T0","0.5")); NSTEP=int(os.environ.get("NSTEP","200"))
NV=int(os.environ.get("NV","126")); WMAX=float(os.environ.get("WMAX","0.6"))
SLABW=float(os.environ.get("SLABW","2.0")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p):
    m=UNet2D(64,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(0)
V0,CORE,SHELL,OCC=vol.V0,vol.CORE,vol.SHELL,vol.OCC
C3=CORE.expand(3,-1,-1,-1).contiguous()
iT=max(int(T0*(T-1)),1)
ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
def restore(batch,model):
    """a full 2-D reverse chain on a batch of planes, all of them noised in full"""
    x=ab[iT].sqrt()*batch+(1-ab[iT]).sqrt()*torch.randn(batch.shape,device=dv,generator=g)
    for tc,tn in zip(ts[:-1],ts[1:]):
        with torch.no_grad():
            e=model(x,torch.full((len(x),),tc,device=dv,dtype=torch.long))
        x0=((x-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()).clamp(-1,1)
        if tn<=0: return x0
        x=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*e
    return x0
X=V0.clone(); t0=time.time(); BS=8
H=vol.occupied_heights()
print(f"  {len(H)} transverse planes + {NV} longitudinal azimuths, each restored once "
      f"({NSTEP} inner steps, T0={T0})",flush=True)
for k in range(0,len(H),BS):
    hs=H[k:k+BS]
    out=restore(torch.stack([vol.trans_slice(V0,j) for j in hs]),MH)
    for n_,j in enumerate(hs): X=vol.write_trans(X,j,out[n_],WMAX,SLABW)
print(f"  transverse done {time.time()-t0:.0f}s",flush=True)
ths=[math.pi*k/NV for k in range(NV)]
for k in range(0,len(ths),BS):
    tb=ths[k:k+BS]
    out=restore(torch.stack([vol.long_slice(V0,th) for th in tb]),MV)
    for n_,th in enumerate(tb): X=vol.write_long(X,th,out[n_],WMAX,SLABW)
print(f"  longitudinal done {time.time()-t0:.0f}s",flush=True)
X=V0*(1-CORE)+X*CORE
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
m=CORE[0]>0.5
for l,t in (("O-Voxel init",V0),("one pass",X)):
    r=((t[:,m]+1)/2)
    print(f"  {l:14s} RGB [{float(r[0].mean()):.3f} {float(r[1].mean()):.3f} {float(r[2].mean()):.3f}]"
          f"  sat {float((r[0]-r[2]).mean()):+.3f}",flush=True)
torch.save({"X":X.cpu()},OUT+"/state.pt")
def sheet(A,B):
    s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
    for i,(t,lab) in enumerate(((A,"init longitudinal"),(B,"one pass longitudinal"),
                                (A,"init transverse"),(B,"one pass transverse"))):
        sl=t[:,:,:,N//2] if i<2 else t[:,:,N//2,:]
        a=((sl.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
        s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
        d.text((i*266+5,7),lab,fill=(30,30,30))
    return s
sheet(V0,X).save(OUT+"/onepass.png"); print("  wrote",OUT+"/onepass.png",flush=True)
