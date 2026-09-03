# Intersection sync, then cell-fill.
#
# Cell-fill loses sharpness (0.2197 -> 0.1729) because it has to reconcile planes that were
# generated without knowing each other -- where they disagree, the only thing a gather can do is
# split the difference.  Synchronising them on their shared lines DURING generation removes the
# disagreement (measured at the plane level: 0.0057, with sharpness above the initial field), so
# the fill has nothing left to average.  The shell is never written.
import os,sys,math,time,numpy as np,torch,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image,ImageDraw
dv=os.environ.get("DEV","cuda:1")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
N=vol.N; R=vol.res; T=1000
ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
T0=float(os.environ.get("T0","0.5")); NSTEP=int(os.environ.get("NSTEP","100"))
NAZ=int(os.environ.get("NAZ","32")); SYNC=float(os.environ.get("SYNC","1.0"))
WLONG=float(os.environ.get("WLONG","1.0")); OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p):
    m=UNet2D(64,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(0)
V0,CORE,SHELL=vol.V0,vol.CORE,vol.SHELL
H=vol.occupied_heights(); ths=[math.pi*k/NAZ for k in range(NAZ)]
nh,nv=len(H),NAZ
TRI=torch.stack([vol.trans_slice(V0,j) for j in H])
LOI=torch.stack([vol.long_slice(V0,th) for th in ths])
rr=torch.linspace(-1,1,R,device=dv)
COS=torch.tensor([math.cos(t) for t in ths],device=dv); SIN=torch.tensor([math.sin(t) for t in ths],device=dv)
YJ=torch.tensor([max(0,min(R-1,int(round((((h-vol.c)/vol.EXT)+1)/2*(R-1))))) for h in H],device=dv)
XI=((rr+1)/2*(R-1)).round().long().clamp(0,R-1)
def sync(t0h,t0v):
    """move both families halfway to the mean along every shared line"""
    dT=torch.zeros_like(t0h); dL=torch.zeros_like(t0v)
    for k in range(nv):
        u=rr*COS[k]; w=rr*SIN[k]
        gg=torch.stack([u,w],-1)[None,None].expand(nh,-1,-1,-1)
        lt=F.grid_sample(t0h,gg,mode="bilinear",padding_mode="border",align_corners=True)[:,:,0]  # (nh,3,R)
        yv=torch.stack([torch.full_like(rr,(H[a]-vol.c)/vol.EXT) for a in range(nh)])
        gl=torch.stack([rr[None].expand(nh,-1),yv],-1)[:,None]
        lv=F.grid_sample(t0v[k][None].expand(nh,-1,-1,-1),gl,mode="bilinear",
                         padding_mode="border",align_corners=True)[:,:,0]                        # (nh,3,R)
        mean=0.5*(lt+lv)
        xi=((u+1)/2*(R-1)).round().long().clamp(0,R-1); yi=((w+1)/2*(R-1)).round().long().clamp(0,R-1)
        dT[:,:,yi,xi]+=(mean-lt)*SYNC*0.5
        for a in range(nh): dL[k,:,YJ[a],XI]+=(mean[a]-lv[a])*SYNC*0.5/max(nh,1)*nh*0.5
    return (t0h+dT).clamp(-1,1),(t0v+dL).clamp(-1,1)
iT=max(int(T0*(T-1)),1); ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
XT=ab[iT].sqrt()*TRI+(1-ab[iT]).sqrt()*torch.randn(TRI.shape,device=dv,generator=g)
XL=ab[iT].sqrt()*LOI+(1-ab[iT]).sqrt()*torch.randn(LOI.shape,device=dv,generator=g)
def run(model,X,t,bs=12):
    o=torch.empty_like(X)
    for k in range(0,len(X),bs):
        b=X[k:k+bs]
        with torch.no_grad(): o[k:k+bs]=model(b,torch.full((len(b),),t,device=dv,dtype=torch.long))
    return o
t0=time.time()
print(f"  {nh} transverse + {nv} longitudinal, sync={SYNC}, {NSTEP} steps",flush=True)
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    et=run(MH,XT,tc); el=run(MV,XL,tc)
    t0h=((XT-(1-ab[tc]).sqrt()*et)/ab[tc].sqrt()).clamp(-1,1)
    t0v=((XL-(1-ab[tc]).sqrt()*el)/ab[tc].sqrt()).clamp(-1,1)
    if SYNC>0: t0h,t0v=sync(t0h,t0v)
    if tn<=0: XT,XL=t0h,t0v; break
    XT=ab[tn].sqrt()*t0h+(1-ab[tn]).sqrt()*et
    XL=ab[tn].sqrt()*t0v+(1-ab[tn]).sqrt()*el
    if i%25==0: print(f"    step {i+1}/{NSTEP}  t={tc}  {time.time()-t0:.0f}s",flush=True)
dis=float(np.mean([float((F.grid_sample(XT[a][None],torch.stack([rr*COS[k],rr*SIN[k]],-1)[None,None],
        mode="bilinear",padding_mode="border",align_corners=True)[0,:,0]
        -F.grid_sample(XL[k][None],torch.stack([rr,torch.full_like(rr,(H[a]-vol.c)/vol.EXT)],-1)[None,None],
        mode="bilinear",padding_mode="border",align_corners=True)[0,:,0]).abs().mean())
    for k in range(0,nv,4) for a in range(0,nh,8)]))
print(f"  line disagreement after sync: {dis:.4f}",flush=True)
# --- cell fill from the synchronised planes -------------------------------------------------
u0,u1,v,c,E=vol.u0,vol.u1,vol.v,vol.c,vol.EXT
rad=torch.sqrt(u0**2+u1**2); ang=torch.atan2(u1,u0)%math.pi
sgn=torch.where(((torch.atan2(u1,u0))%(2*math.pi))<math.pi,1.0,-1.0)
hidx=(v+c).round().long().clamp(0,N-1)
gt=torch.stack([u0/E,u1/E],-1); gl=torch.stack([sgn*rad/E,v/E],-1)
XTv=torch.zeros(3,N,N,N,device=dv); XLv=torch.zeros(3,N,N,N,device=dv)
for a,j in enumerate(H):
    m=(hidx==j)&(CORE[0]>0.5)
    if m.any(): XTv[:,m]=F.grid_sample(XT[a][None],gt[m][None,None],mode="bilinear",
                                       padding_mode="border",align_corners=True)[0,:,0,:]
b=(ang/math.pi*nv).long().clamp(0,nv-1)
for k in range(nv):
    m=(b==k)&(CORE[0]>0.5)
    if m.any(): XLv[:,m]=F.grid_sample(XL[k][None],gl[m][None,None],mode="bilinear",
                                       padding_mode="border",align_corners=True)[0,:,0,:]
X=V0*(1-CORE)+((XTv+WLONG*XLv)/(1.0+WLONG))*CORE
print(f"  shell untouched: max|d| = {float((X-V0)[:,(SHELL[0]>0.5)].abs().max()):.8f}",flush=True)
m=CORE[0]>0.5
def sharp(t):
    a=t[:,:,:,N//2].mean(0)
    return float((a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]).abs().mean())
for l,t in (("O-Voxel init",V0),("sync + cell fill",X)):
    r=((t[:,m]+1)/2)
    print(f"  {l:18s} sat {float((r[0]-r[2]).mean()):+.3f}  sharpness {sharp(t):.4f}",flush=True)
torch.save({"X":X.cpu()},OUT+"/state.pt")
s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
for i,(t,lab) in enumerate(((V0,"init longitudinal"),(X,"sync+fill longitudinal"),
                            (V0,"init transverse"),(X,"sync+fill transverse"))):
    sl=t[:,:,:,N//2] if i<2 else t[:,:,N//2,:]
    a=((sl.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
    d.text((i*266+5,7),lab,fill=(30,30,30))
s.save(OUT+"/syncfill.png"); print("  wrote",OUT+"/syncfill.png",flush=True)
