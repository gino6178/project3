# 3-D SinDiffusion from two slice families.  Prior = product of the longitudinal and transverse
# single-image priors over every slice of the volume; a slice of a Gaussian-noised volume is a
# Gaussian-noised image at the same t, so the 2-D denoisers apply unchanged.  One reverse chain,
# state kept on the plane grids (never re-sliced through the voxel grid); the two families are
# coupled every step by averaging x0 along their intersections (the score-average of the product
# prior); the transverse planes sit on the longitudinal pixel rows so z needs no interpolation.
# Shell never written.  O-Voxel low-pass pinned by ILVR on the longitudinal family (soft data term).
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image
dv=os.environ.get("DEV","cuda:1"); RES=int(os.environ.get("RES","256"))
vol=Vol(os.environ["GRID"],dv,res=RES); N=vol.N; T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
T0H=float(os.environ.get("T0H","0.5")); T0V=float(os.environ.get("T0V","0.9")); NSTEP=int(os.environ.get("NSTEP","100"))
NAZ=int(os.environ.get("NAZ","90")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
SYNCW=float(os.environ.get("SYNCW","0.5"))
# Information-weighted sync: each family leads where it holds the structure the other cannot see.
# WT = how much the transverse takes from the longitudinal, WL = the reverse; with R0>0 the
# transverse takes only near the axis (columella: a strip to the longitudinal, a disc to the
# transverse) and the longitudinal takes only away from it (segment membranes: radial lines to the
# transverse, edge-on to the longitudinal). TEND: stop syncing below this fraction of the chain so
# each family finishes its own fine detail (late disagreement is ~0.01).
WT=float(os.environ.get("WT",str(SYNCW))); WL=float(os.environ.get("WL",str(SYNCW)))
R0=float(os.environ.get("R0","0")); TEND=float(os.environ.get("TEND","0"))
CONDT=os.environ.get("CONDT","sync"); DCFIX=float(os.environ.get("DCFIX","16")); BS=int(os.environ.get("BS","8"))
class CondNet(nn.Module):
    def __init__(s,c=64,mult=(1,2,4)):
        super().__init__(); s.net=UNet2D(c,mult); s.net.inp=nn.Conv2d(6,c*mult[0],3,padding=1)
    def forward(s,x,t,cond): return s.net(torch.cat([x,cond],1),t)
UNCOND=os.environ.get("UNCOND","0")=="1"   # plain SinDiffusion priors (no cond channel): the intersection sync is the only coupling
def load(p,mult=None):
    d=torch.load(p,map_location=dv)
    if UNCOND: m=UNet2D(64,mult).to(dv); m.load_state_dict(d if "sd" not in d else d["sd"]); m.eval(); return m
    m=CondNet(64,tuple(d["MULT"])).to(dv); m.load_state_dict(d["sd"]); m.eval(); return m
MV=load(os.environ["CKV"],tuple(int(x) for x in os.environ.get("MULTV","1,2").split(",")))
MH=load(os.environ["CKH"],tuple(int(x) for x in os.environ.get("MULTH","1,2,4").split(",")))
g=torch.Generator(device=dv).manual_seed(0)
V0,CORE=vol.V0,vol.CORE; E=vol.EXT; gl=torch.linspace(-E,E,RES,device=dv)
ths=[math.pi*k/NAZ for k in range(NAZ)]
occ=set(vol.occupied_heights()); rows=[i for i in range(RES) if int(round(vol.c+float(gl[i]))) in occ]   # transverse planes on pixel rows
hs=[vol.c+float(gl[i]) for i in rows]; J=len(rows); K=NAZ
print(f"  {K} longitudinal planes, {J} transverse planes on rows {rows[0]}..{rows[-1]}",flush=True)
def lowpass(x,sig):
    k=int(sig*3)*2+1; w=torch.exp(-(torch.arange(k,device=dv,dtype=x.dtype)-k//2)**2/(2*sig*sig)); w=w/w.sum()
    x=F.conv2d(F.pad(x,(k//2,k//2,0,0),mode="reflect"),w.view(1,1,1,k).repeat(3,1,1,1),groups=3)
    return F.conv2d(F.pad(x,(0,0,k//2,k//2),mode="reflect"),w.view(1,1,k,1).repeat(3,1,1,1),groups=3)
def eps(model,x,t,cond):
    out=[]
    for k in range(0,len(x),BS):
        tt=torch.full((len(x[k:k+BS]),),t,device=dv,dtype=torch.long)
        with torch.no_grad(): out.append(model(x[k:k+BS],tt) if UNCOND else model(x[k:k+BS],tt,cond[k:k+BS]))
    return torch.cat(out)
# ---- intersection geometry (built once) ----
th_t=torch.tensor(ths,device=dv); cs,sn=torch.cos(th_t),torch.sin(th_t)          # (K,)
s_=gl/E                                                                            # (RES,) signed radius, normalised
gridL=torch.stack([s_[None,:]*cs[:,None], s_[None,:]*sn[:,None]],-1)               # (K,RES,2): (x=u0,y=u1) on a transverse disc
gridL=gridL[None].expand(J,K,RES,2).contiguous()                                    # same for every transverse plane
U1,U0=torch.meshgrid(gl,gl,indexing="ij")                                          # transverse pixel (row=u1, col=u0)
ang=torch.atan2(U1,U0); binT=((ang%math.pi)/math.pi*K).long().clamp(0,K-1)          # azimuth bin of every transverse pixel
sgnT=torch.where((ang%(2*math.pi))<math.pi,1.0,-1.0); sT=sgnT*torch.sqrt(U0**2+U1**2)/E   # its signed radius
vrow=torch.tensor([float(gl[i]) for i in rows],device=dv)/E                         # (J,) z of each transverse plane, normalised
pix=[torch.nonzero(binT==k,as_tuple=False) for k in range(K)]                       # pixels per bin
rT=torch.sqrt(U0**2+U1**2)/E; rL=s_.abs()[None,:].expand(RES,RES)                # pixel radius on transverse / longitudinal planes
if R0>0: wT_map=(WT*torch.exp(-(rT/R0)**2))[None,None]; wL_map=(WL*(1-torch.exp(-(rL/R0)**2)))[None,None]
else:    wT_map=torch.full((1,1,RES,RES),WT,device=dv); wL_map=torch.full((1,1,RES,RES),WL,device=dv)
def T_onto_L(x0T):   # (J,3,RES,RES) -> (K,3,RES,RES): the transverse family's x0 on every longitudinal pixel (rows in `rows`)
    out=F.grid_sample(x0T,gridL,mode="bilinear",padding_mode="border",align_corners=True)   # (J,3,K,RES)
    L=torch.zeros(K,3,RES,RES,device=dv); L[:,:,rows,:]=out.permute(2,1,0,3); return L
def L_onto_T(x0L):   # (K,3,RES,RES) -> (J,3,RES,RES): the longitudinal family's x0 on every transverse pixel
    Tn=torch.zeros(J,3,RES,RES,device=dv)
    for k in range(K):
        p=pix[k]
        if len(p)==0: continue
        gs=torch.stack([sT[p[:,0],p[:,1]][None].expand(J,-1), vrow[:,None].expand(J,len(p))],-1)[:,None]   # (J,1,n,2)
        Tn[:,:,p[:,0],p[:,1]]=F.grid_sample(x0L[k][None].expand(J,3,RES,RES),gs,mode="bilinear",padding_mode="border",align_corners=True)[:,:,0,:]
    return Tn
# ---- the chain ----
L0=torch.stack([vol.long_slice(V0,th) for th in ths]); T0=torch.stack([vol.trans_slice(V0,h) for h in hs])
LPL=lowpass(L0,DCFIX) if DCFIX>0 else None
DCT=float(os.environ.get("DCT","0")); LPT=lowpass(T0,DCT) if DCT>0 else None   # ILVR on the transverse family too: its low-pass is z-continuous
iT=max(int(max(T0H,T0V)*(T-1)),1); iTH=int(T0H*(T-1)); iTV=int(T0V*(T-1))
ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
xL=xT=None; x0L,x0T=L0.clone(),T0.clone(); t0=time.time()
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    aL=tc<=iTV; aT=tc<=iTH
    if aL and xL is None: xL=ab[tc].sqrt()*L0+(1-ab[tc]).sqrt()*torch.randn(L0.shape,device=dv,generator=g)
    if aT and xT is None: xT=ab[tc].sqrt()*T0+(1-ab[tc]).sqrt()*torch.randn(T0.shape,device=dv,generator=g)
    condL=T_onto_L(x0T); condT=T0 if CONDT=="ov" else L_onto_T(x0L)   # CONDT=ov: the transverse is conditioned on the O-Voxel's own transverse view (the observed layout: segments), not on a family that cannot see them
    eL=eT=None
    if aL:
        eL=eps(MV,xL,tc,condL); x0L=(xL-(1-ab[tc]).sqrt()*eL)/ab[tc].sqrt()
        if LPL is not None: x0L=x0L-lowpass(x0L,DCFIX)+LPL         # ILVR: the O-Voxel's low-frequency layout
    if aT: eT=eps(MH,xT,tc,condT); x0T=(xT-(1-ab[tc]).sqrt()*eT)/ab[tc].sqrt()
    if aT and LPT is not None: x0T=x0T-lowpass(x0T,DCT)+LPT
    if aL and aT and tc>TEND*iTH:                                  # information-weighted score-average along the intersections
        cL=T_onto_L(x0T); cT=L_onto_T(x0L)
        x0L=(1-wL_map)*x0L+wL_map*cL; x0T=(1-wT_map)*x0T+wT_map*cT
    if tn<=0: break
    if aL: xL=ab[tn].sqrt()*x0L+(1-ab[tn]).sqrt()*eL
    if aT: xT=ab[tn].sqrt()*x0T+(1-ab[tn]).sqrt()*eT
    if i%10==0: print(f"  step {i}/{NSTEP} t={tc} L={aL} T={aT} {time.time()-t0:.0f}s",flush=True)
x0L=x0L.clamp(-1,1); x0T=x0T.clamp(-1,1)
# consistency between the families at the end, measured on the planes before any voxel write
dis=float((T_onto_L(x0T)-x0L)[:,:,rows,:].abs().mean()); print(f"  final intersection disagreement {dis:.4f}",flush=True)
# ---- write the asset once ----
u0,u1,v,c=vol.u0,vol.u1,vol.v,vol.c
rad=torch.sqrt(u0**2+u1**2); angv=torch.atan2(u1,u0)%math.pi
sgn=torch.where(((torch.atan2(u1,u0))%(2*math.pi))<math.pi,1.0,-1.0)
XL=torch.zeros(3,N,N,N,device=dv); XT=torch.zeros(3,N,N,N,device=dv)
b=(angv/math.pi*K).long().clamp(0,K-1); glv=torch.stack([sgn*rad/E,v/E],-1)
for k in range(K):
    m=(b==k)&(CORE[0]>0.5)
    if m.any(): XL[:,m]=F.grid_sample(x0L[k][None],glv[m][None,None],mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,:]
hidx=((v/E+1)/2*(RES-1)).round().long().clamp(0,RES-1)         # voxel -> nearest transverse plane row
rowmap=torch.full((RES,),-1,device=dv,dtype=torch.long); rowmap[torch.tensor(rows,device=dv)]=torch.arange(J,device=dv)
jidx=rowmap[hidx]; gt=torch.stack([u0/E,u1/E],-1)
for j in range(J):
    m=(jidx==j)&(CORE[0]>0.5)
    if m.any(): XT[:,m]=F.grid_sample(x0T[j][None],gt[m][None,None],mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,:]
X=V0*(1-CORE)+(0.5*(XL+XT))*CORE
torch.save({"X":X.cpu(),"x0L":x0L.cpu(),"x0T":x0T.cpu(),"rows":rows},OUT+"/state.pt")
def im(t): t=(t.squeeze(0)+1)/2; return Image.fromarray((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
tiles=[im(vol.long_slice(X,th)) for th in (0.7,1.4,2.1)]+[im(vol.trans_slice(X,float(hs[J//2])))]
W=tiles[0].width; s=Image.new("RGB",(4*W,W)); [s.paste(t,(k*W,0)) for k,t in enumerate(tiles)]; s.save(OUT+"/xfill.png")
print("  wrote",OUT,f"{time.time()-t0:.0f}s",flush=True)
