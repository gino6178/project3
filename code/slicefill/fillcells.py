# Fill the interior cell by cell instead of smearing planes into slabs.
#
# Scattering a restored plane into a +-2 cell slab blurs it, and several planes overwrite the same
# cells, so the result is softer than any plane that produced it.  Every cell already belongs to
# exactly one transverse plane (its own height) and one longitudinal plane (its own azimuth), so
# restore the planes first, then GATHER: each cell reads its two planes at its own position.
# Nothing is averaged over depth, and nothing overwrites anything.
import os,sys,math,time,numpy as np,torch,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image,ImageDraw
dv=os.environ.get("DEV","cuda:1")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
N=vol.N; T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
T0=float(os.environ.get("T0","0.5")); NSTEP=int(os.environ.get("NSTEP","200"))
NAZ=int(os.environ.get("NAZ","180")); WLONG=float(os.environ.get("WLONG","1.0"))
# NOCORE=1 writes the restored planes over the WHOLE solid, shell included, instead of the
# interior only.  The shell is observed data so holding it fixed is right in principle, but if
# the fixed boundary is what limits the interior this is where it shows.
NOCORE=int(os.environ.get("NOCORE","0"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p):
    m=UNet2D(64,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(0)
V0,CORE,SHELL,OCC=vol.V0,vol.CORE,vol.SHELL,vol.OCC
iT=max(int(T0*(T-1)),1); ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
def restore(batch,model):
    x=ab[iT].sqrt()*batch+(1-ab[iT]).sqrt()*torch.randn(batch.shape,device=dv,generator=g)
    for tc,tn in zip(ts[:-1],ts[1:]):
        with torch.no_grad(): e=model(x,torch.full((len(x),),tc,device=dv,dtype=torch.long))
        x0=((x-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()).clamp(-1,1)
        if tn<=0: return x0
        x=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*e
    return x0
t0=time.time(); BS=8
TR=torch.zeros(N,3,vol.res,vol.res,device=dv)      # restored transverse plane per height
have=torch.zeros(N,dtype=torch.bool,device=dv)
H=vol.occupied_heights()
for k in range(0,len(H),BS):
    hs=H[k:k+BS]; out=restore(torch.stack([vol.trans_slice(V0,j) for j in hs]),MH)
    for n_,j in enumerate(hs): TR[j]=out[n_]; have[j]=True
print(f"  {len(H)} transverse planes restored  {time.time()-t0:.0f}s",flush=True)
LO=torch.zeros(NAZ,3,vol.res,vol.res,device=dv)    # restored longitudinal plane per azimuth bin
ths=[math.pi*k/NAZ for k in range(NAZ)]
for k in range(0,NAZ,BS):
    tb=ths[k:k+BS]; out=restore(torch.stack([vol.long_slice(V0,th) for th in tb]),MV)
    for n_,_ in enumerate(tb): LO[k+n_]=out[n_]
print(f"  {NAZ} longitudinal planes restored  {time.time()-t0:.0f}s",flush=True)

# --- gather: every cell reads its own two planes -------------------------------------------
u0,u1,v,c,E=vol.u0,vol.u1,vol.v,vol.c,vol.EXT
rad=torch.sqrt(u0**2+u1**2)
ang=torch.atan2(u1,u0)%math.pi                      # a plane at th and th+pi are the same plane
sgn=torch.where(((torch.atan2(u1,u0))%(2*math.pi))<math.pi,1.0,-1.0)
hidx=(v+c).round().long().clamp(0,N-1)              # each cell's own transverse height
# transverse: sample plane TR[h] at (u0,u1)
gt=torch.stack([u1/E,u0/E],-1) if os.environ.get("WORDER","uv")!="uv" else torch.stack([u0/E,u1/E],-1)
XT=torch.zeros(3,N,N,N,device=dv)
for j in H:
    m=(hidx==j)&((OCC[0]>0.5) if NOCORE else (CORE[0]>0.5))
    if not m.any(): continue
    XT[:,m]=F.grid_sample(TR[j][None],gt[m][None,None],mode="bilinear",
                          padding_mode="border",align_corners=True)[0,:,0,:]
print(f"  transverse gathered  {time.time()-t0:.0f}s",flush=True)
# longitudinal: bin the azimuth, sample plane LO[b] at (signed radius, axial)
b=(ang/math.pi*NAZ).long().clamp(0,NAZ-1)
gl=torch.stack([sgn*rad/E,v/E],-1) if os.environ.get("WORDER","uv")=="uv" else torch.stack([v/E,sgn*rad/E],-1)
XL=torch.zeros(3,N,N,N,device=dv)
for k in range(NAZ):
    m=(b==k)&((OCC[0]>0.5) if NOCORE else (CORE[0]>0.5))
    if not m.any(): continue
    XL[:,m]=F.grid_sample(LO[k][None],gl[m][None,None],mode="bilinear",
                          padding_mode="border",align_corners=True)[0,:,0,:]
print(f"  longitudinal gathered  {time.time()-t0:.0f}s",flush=True)
MK=(OCC if NOCORE else CORE)
X=V0*(1-MK)+((XT+WLONG*XL)/(1.0+WLONG))*MK
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
m=CORE[0]>0.5
def sharp(t):
    a=t[:,:,:,N//2].mean(0)
    return float((a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]).abs().mean())
for l,t in (("O-Voxel init",V0),("cell fill",X)):
    r=((t[:,m]+1)/2)
    print(f"  {l:14s} RGB [{float(r[0].mean()):.3f} {float(r[1].mean()):.3f} {float(r[2].mean()):.3f}]"
          f"  sat {float((r[0]-r[2]).mean()):+.3f}  sharpness {sharp(t):.4f}",flush=True)
torch.save({"X":X.cpu()},OUT+"/state.pt")
s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
for i,(t,lab) in enumerate(((V0,"init longitudinal"),(X,"cell fill longitudinal"),
                            (V0,"init transverse"),(X,"cell fill transverse"))):
    sl=t[:,:,:,N//2] if i<2 else t[:,:,N//2,:]
    a=((sl.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
    d.text((i*266+5,7),lab,fill=(30,30,30))
s.save(OUT+"/fill.png"); print("  wrote",OUT+"/fill.png",flush=True)
