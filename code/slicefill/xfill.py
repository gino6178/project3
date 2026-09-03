# Cross-conditioned fill: the two families take turns, each denoised WITH the other's current
# account of the same plane as input.  Consistency is produced by the model, not imposed by a
# rule -- so nothing is averaged and there is nothing to blur.  The shell is never written.
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image,ImageDraw
dv=os.environ.get("DEV","cuda:1")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
N=vol.N; T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
T0=float(os.environ.get("T0","0.5")); NSTEP=int(os.environ.get("NSTEP","200"))
# The two families want different noise levels because they resolve different structures.  The
# transverse plane sees the columella as one compact central disc, and T0=0.7 erases it (tex15
# 0.107 -> 0.092 on the plane itself); the longitudinal plane needs T0=0.7 to synthesise the pith
# grain that reaches real level (0.1228 vs 0.1238).  So each family gets its own T0.
T0H=float(os.environ.get("T0H",str(T0))); T0V=float(os.environ.get("T0V",str(T0)))
NAZ=int(os.environ.get("NAZ","180")); ROUNDS=int(os.environ.get("ROUNDS","2"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
# CONDMODE=noise / zero replaces the other family's account with garbage.  If the result barely
# changes, the model is not using the condition and the "coupling" is an illusion.
CONDMODE=os.environ.get("CONDMODE","real")
# RADW>0: weight the two families by radius in the gather.  The columella lies on the axis every
# longitudinal plane shares, so 180 planes each render their own version of it and the azimuth
# bins jitter between them; a transverse plane sees it as one compact disc.  Let the transverse
# family dominate near the axis and hand over to the longitudinal one outward.
RADW=float(os.environ.get("RADW","0"))
# CONDBLUR blurs the condition at inference so it carries LAYOUT and not the O-Voxel's smoothness;
# the photograph-trained prior then has to supply the texture itself.
CONDBLUR=int(os.environ.get("CONDBLUR","0"))
def maybe_break(c):
    if CONDMODE=="noise": return torch.randn(c.shape,device=dv,generator=g)*0.5
    if CONDMODE=="zero":  return torch.zeros_like(c)
    if CONDBLUR>1:
        k=CONDBLUR|1
        return F.avg_pool2d(F.pad(c,(k//2,)*4,mode="replicate"),k,1)
    return c
class CondNet(nn.Module):
    def __init__(s,c=64,mult=(1,2,4)):
        super().__init__(); s.net=UNet2D(c,mult); s.net.inp=nn.Conv2d(6,c*mult[0],3,padding=1)
    def forward(s,x,t,cond): return s.net(torch.cat([x,cond],1),t)
def load(p):
    d=torch.load(p,map_location=dv); m=CondNet(64,tuple(d["MULT"])).to(dv)
    m.load_state_dict(d["sd"]); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(0)
V0,CORE,SHELL=vol.V0,vol.CORE,vol.SHELL
H=vol.occupied_heights(); ths=[math.pi*k/NAZ for k in range(NAZ)]

# DC swap: the prior cannot recover the mean from t~900 (small receptive field -> x0 shrinks to 0),
# but the O-Voxel's colour is right (it matches the photographs). Keep its low-pass, take the
# prior's high-pass. DCFIX = gaussian sigma in px at RES; 0 = off.
DCFIX=float(os.environ.get("DCFIX","0"))
DCMEAN=os.environ.get("DCMEAN","0")=="1"
DCMODE=os.environ.get("DCMODE","post")  # post: swap once at the end; ilvr: pin inside the chain
OWN=os.environ.get("OWN","avg")
WMODE=os.environ.get("WMODE","bilinear")  # write interpolation into the voxels
UPW=int(os.environ.get("UPW","1"))        # upsample planes before the write
DCFAM=os.environ.get("DCFAM","both")  # long: swap only the longitudinal family (the one that shrinks)
def lowpass(x,sig):
    k=int(sig*3)*2+1; g=torch.exp(-(torch.arange(k,device=x.device,dtype=x.dtype)-k//2)**2/(2*sig*sig)); g=g/g.sum()
    F=torch.nn.functional
    x=F.conv2d(F.pad(x,(k//2,k//2,0,0),mode="reflect"),g.view(1,1,1,k).repeat(3,1,1,1),groups=3)
    return F.conv2d(F.pad(x,(0,0,k//2,k//2),mode="reflect"),g.view(1,1,k,1).repeat(3,1,1,1),groups=3)
def dcfix(x0,inp):
    if DCFIX<=0: return x0
    return (lowpass(inp,DCFIX)+(x0-lowpass(x0,DCFIX))).clamp(-1,1)
def restore(batch,cond,model,t0f,bs=8,dc=True):
    iT=max(int(t0f*(T-1)),1); ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
    out=[]
    for k in range(0,len(batch),bs):
        b=batch[k:k+bs]; c=cond[k:k+bs]
        x=ab[iT].sqrt()*b+(1-ab[iT]).sqrt()*torch.randn(b.shape,device=dv,generator=g)
        for tc,tn in zip(ts[:-1],ts[1:]):
            with torch.no_grad(): e=model(x,torch.full((len(b),),tc,device=dv,dtype=torch.long),c)
            x0=(x-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()  # no clamp inside the chain: it walked the mean toward grey
            if tn<=0: x=dcfix(x0.clamp(-1,1),b) if (dc and DCMODE=="post") else x0.clamp(-1,1); break
            x=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*e
            if dc and DCFIX>0 and DCMODE=="ilvr":  # ILVR: pin the reference's low-pass at every step so layout+texture are generated consistent with the true colour
                yt=ab[tn].sqrt()*b+(1-ab[tn]).sqrt()*torch.randn(b.shape,device=dv,generator=g)
                x=x-lowpass(x,DCFIX)+lowpass(yt,DCFIX)
        out.append(x)
    return torch.cat(out)
def gather(TR,LO):
    if UPW>1:  # generate at the prior's resolution, write at a finer one: the write's tent filter then blurs at half the scale
        TR=F.interpolate(TR,scale_factor=UPW,mode="bicubic",align_corners=True); LO=F.interpolate(LO,scale_factor=UPW,mode="bicubic",align_corners=True)
    u0,u1,v,c,E=vol.u0,vol.u1,vol.v,vol.c,vol.EXT
    rad=torch.sqrt(u0**2+u1**2); ang=torch.atan2(u1,u0)%math.pi
    sgn=torch.where(((torch.atan2(u1,u0))%(2*math.pi))<math.pi,1.0,-1.0)
    hidx=(v+c).round().long().clamp(0,N-1)
    gt=torch.stack([u0/E,u1/E],-1); gl=torch.stack([sgn*rad/E,v/E],-1)
    XT=torch.zeros(3,N,N,N,device=dv); XL=torch.zeros(3,N,N,N,device=dv)
    for a,j in enumerate(H):
        m=(hidx==j)&(CORE[0]>0.5)
        if m.any(): XT[:,m]=F.grid_sample(TR[a][None],gt[m][None,None],mode=WMODE,padding_mode="border",align_corners=True)[0,:,0,:]
    b=(ang/math.pi*len(ths)).long().clamp(0,len(ths)-1)
    for k in range(len(ths)):
        m=(b==k)&(CORE[0]>0.5)
        if m.any(): XL[:,m]=F.grid_sample(LO[k][None],gl[m][None,None],mode=WMODE,padding_mode="border",align_corners=True)[0,:,0,:]
    return XT,XL
X=V0.clone(); t0=time.time()
for rd in range(ROUNDS):
    # each family is conditioned on the volume the OTHER one most recently produced
    condT=torch.stack([vol.trans_slice(X,j) for j in H])
    TR=restore(torch.stack([vol.trans_slice(V0,j) for j in H]),maybe_break(condT),MH,T0H,dc=(DCFAM!="long"))
    XT,_=gather(TR,torch.stack([vol.long_slice(X,th) for th in ths]))
    Xt=V0*(1-CORE)+XT*CORE
    condL=torch.stack([vol.long_slice(Xt,th) for th in ths])
    LO=restore(torch.stack([vol.long_slice(V0,th) for th in ths]),maybe_break(condL),MV,T0V)
    _,XL=gather(TR,LO)
    if RADW>0:
        rr=torch.sqrt(vol.u0**2+vol.u1**2)/vol.EXT
        wT=(0.5+0.5*torch.exp(-(rr/RADW)**2))[None]      # 1.0 on the axis -> 0.5 far out
        F=wT*XT+(1-wT)*XL
    else:
        F={"long":XL,"trans":XT}.get(OWN,0.5*(XT+XL))  # OWN=long|trans: one family owns the interior, no averaging
    if DCMEAN:  # the prior shrinks the mean toward 0 uniformly at t~900; put back the O-Voxel's per-channel mean, nothing else
        off=((V0-F)*CORE).sum((1,2,3))/CORE.sum(); F=F+off[:,None,None,None]; print("  DC offset",[round(float(v),4) for v in off],flush=True)
    X=V0*(1-CORE)+F*CORE
    print(f"  round {rd+1}/{ROUNDS}  {time.time()-t0:.0f}s",flush=True)
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
m=CORE[0]>0.5
def sharp(t):
    a=t[:,:,:,N//2].mean(0)
    return float((a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]).abs().mean())
for l,t in (("O-Voxel init",V0),("cross-conditioned",X)):
    r=((t[:,m]+1)/2)
    print(f"  {l:18s} sat {float((r[0]-r[2]).mean()):+.3f}  sharpness {sharp(t):.4f}",flush=True)
torch.save({"X":X.cpu()},OUT+"/state.pt")
s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
for i,(t,lab) in enumerate(((V0,"init longitudinal"),(X,"xcond longitudinal"),
                            (V0,"init transverse"),(X,"xcond transverse"))):
    sl=t[:,:,:,N//2] if i<2 else t[:,:,N//2,:]
    a=((sl.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
    d.text((i*266+5,7),lab,fill=(30,30,30))
s.save(OUT+"/xfill.png"); print("  wrote",OUT+"/xfill.png",flush=True)
