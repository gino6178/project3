# The same product of experts as x3dcyl.py, on a Cartesian state A[v, u0, u1] -- the frame in
# which columnar, layered and isotropic interiors are stationary, as radial anatomy is in the
# cylinder.  Every family is again an exact axis-aligned selection of the one state:
#   longitudinal, u1 fixed  -> (v, u0) image  = vol.long_slice at azimuth 0
#   longitudinal, u0 fixed  -> (v, u1) image  = vol.long_slice at azimuth pi/2
#   transverse,   v fixed   -> (u1, u0) image = vol.trans_slice
# The longitudinal prior denoises both vertical families, the transverse prior the horizontal one,
# with equal weight per direction (no axis, so no radius rule).  Carrier: the same data terms as the
# cylinder -- shell never written, its low-pass pinned (sigma DCFIX px, here in 3-D since there is
# no preferred plane), the same t0, steps and DDIM re-noising.  Written to voxels once at the end.
import os,sys,math,time,numpy as np,torch,torch.nn.functional as F
SPD=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill"); sys.path.insert(0,SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image
dv=os.environ.get("DEV","cuda:0"); vol=Vol(os.environ["GRID"],dv,res=256); N=vol.N; E=vol.EXT; c=vol.c
M=int(os.environ.get("NC","256"))
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
T0=float(os.environ.get("T0","0.3")); NSTEP=int(os.environ.get("NSTEP","100")); DCFIX=float(os.environ.get("DCFIX","16")); BS=int(os.environ.get("BS","8"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p,mult):
    m=UNet2D(64,mult).to(dv); d=torch.load(p,map_location=dv); m.load_state_dict(d if "sd" not in d else d["sd"]); m.eval(); return m
HAVE_L=os.environ.get("CKV","none")!="none"; HAVE_T=os.environ.get("CKH","none")!="none"
MV=load(os.environ["CKV"],tuple(int(x) for x in os.environ.get("MULTV","1,2").split(","))) if HAVE_L else None
MT=load(os.environ["CKH"],tuple(int(x) for x in os.environ.get("MULTH","1,2").split(","))) if HAVE_T else None
g=torch.Generator(device=dv).manual_seed(0)
q=-E+(torch.arange(M,device=dv)+0.5)/M*2*E                        # cell centres, same span as the planes
VV,U0,U1=torch.meshgrid(q,q,q,indexing="ij")                        # state index order (v, u0, u1)
p0,p1,p2=(U0+c).reshape(1,-1),(VV+c).reshape(1,-1),(U1+c).reshape(1,-1)
A0=vol._samp(vol.V0,p0,p1,p2).reshape(3,M,M,M)
CORE=(vol._samp(vol.CORE.expand(3,-1,-1,-1).contiguous(),p0,p1,p2).reshape(3,M,M,M)[:1]>0.5).float()
print(f"  cartesian {M}^3, interior {int(CORE.sum()):,} cells",flush=True)
live=[CORE[0].flatten(1).amax(1)>0, CORE[0].permute(2,0,1).flatten(1).amax(1)>0, CORE[0].permute(1,0,2).flatten(1).amax(1)>0]  # planes with interior: v, u1, u0
# exact selections: (n_planes, 3, H, W)
def toX(x): return x.permute(3,0,1,2)                               # u1 fixed: (u1, 3, v, u0)
def fromX(P): return P.permute(1,2,3,0)
def toY(x): return x.permute(2,0,1,3)                               # u0 fixed: (u0, 3, v, u1)
def fromY(P): return P.permute(1,2,0,3)
def toT(x): return x.permute(1,0,3,2)                               # v fixed:  (v, 3, u1, u0)
def fromT(P): return P.permute(1,0,3,2)
for f,fi in ((toX,fromX),(toY,fromY),(toT,fromT)): assert torch.equal(fi(f(A0)),A0)
def blur3(x,sig):
    k=int(sig*3)*2+1; w=torch.exp(-(torch.arange(k,device=dv,dtype=x.dtype)-k//2)**2/(2*sig*sig)); w=w/w.sum()
    x=x[None]
    for d in range(3):
        sh=[1,1,1]; sh[d]=k; pad=[0]*6; pad[2*(2-d)]=pad[2*(2-d)+1]=k//2
        x=F.conv3d(F.pad(x,pad,mode="replicate"),w.view(1,1,*sh).repeat(3,1,1,1,1),groups=3)
    return x[0]
LP=blur3(A0,DCFIX) if DCFIX>0 else None
def x0of(model,P,keep,tc):
    out=P.clone(); idx=keep.nonzero().flatten()
    for k in range(0,len(idx),BS):
        j=idx[k:k+BS]
        with torch.no_grad(): e=model(P[j],torch.full((len(j),),tc,device=dv,dtype=torch.long))
        out[j]=(P[j]-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()
    return out
iT=max(int(T0*(T-1)),1); ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
efix=torch.randn(A0.shape,device=dv,generator=g)
x=ab[iT].sqrt()*A0+(1-ab[iT]).sqrt()*efix; t0=time.time()
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    ests=[]
    if HAVE_L:
        ests.append(fromX(x0of(MV,toX(x).contiguous(),live[1],tc)))
        ests.append(fromY(x0of(MV,toY(x).contiguous(),live[2],tc)))
    if HAVE_T: ests.append(fromT(x0of(MT,toT(x).contiguous(),live[0],tc)))
    x0=sum(ests)/len(ests) if ests else A0
    if LP is not None: x0=x0-blur3(x0,DCFIX)+LP
    x0=CORE*x0+(1-CORE)*A0
    if tn<=0: break
    ebar=(x-ab[tc].sqrt()*x0)/(1-ab[tc]).sqrt()
    x=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*ebar
    x=CORE*x+(1-CORE)*(ab[tn].sqrt()*A0+(1-ab[tn]).sqrt()*efix)
    if i%10==0: print(f"  step {i}/{NSTEP} t={tc} {time.time()-t0:.0f}s",flush=True)
A=(CORE*x0+(1-CORE)*A0).clamp(-1,1)
# voxels <- state, trilinear; state cell i sits at -E+(i+0.5)*2E/M, i.e. align_corners=False over [-E,E]
m=vol.CORE[0]>0.5
grid=torch.stack([vol.u1[m]/E,vol.u0[m]/E,vol.v[m]/E],-1)[None,None,None]   # grid_sample (x=W=u1, y=H=u0, z=D=v)
val=F.grid_sample(A[None],grid,mode="bilinear",padding_mode="border",align_corners=False)[0,:,0,0,:]
X=vol.V0.clone(); X[:,m]=val
torch.save({"X":X.cpu()},OUT+"/state.pt")
def im(t): t=(t.squeeze(0)+1)/2; return Image.fromarray((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
hs=vol.occupied_heights(); tiles=[im(vol.long_slice(X,th)) for th in (0.7,1.4,2.1)]+[im(vol.trans_slice(X,float(hs[len(hs)//2])))]
W=tiles[0].width; s=Image.new("RGB",(4*W,W)); [s.paste(t,(k*W,0)) for k,t in enumerate(tiles)]; s.save(OUT+"/xfill.png")
print("  wrote",OUT,f"{time.time()-t0:.0f}s",flush=True)
