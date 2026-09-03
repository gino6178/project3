# Intersection synchronisation, the direction memory records as the only one that kept sharpness.
#
# Six aggregation rules have now averaged the detail away (the seventh, cell-fill, lost 21% of the
# sharpness).  The difference here is WHEN the families are made to agree: not after each has
# produced a plane, but at every step of generation, on the line they share.  A longitudinal plane
# at azimuth th and a transverse plane at height h intersect in one line; forcing the two to carry
# the same values along it leaves each plane free to be sharp everywhere else.
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
NAZ=int(os.environ.get("NAZ","24")); SYNC=float(os.environ.get("SYNC","1.0"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p):
    m=UNet2D(64,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
g=torch.Generator(device=dv).manual_seed(0)
V0,CORE,SHELL=vol.V0,vol.CORE,vol.SHELL
H=vol.occupied_heights(); ths=[math.pi*k/NAZ for k in range(NAZ)]
TRI=torch.stack([vol.trans_slice(V0,j) for j in H])       # (nh,3,R,R)
LOI=torch.stack([vol.long_slice(V0,th) for th in ths])    # (nv,3,R,R)
nh,nv=len(H),NAZ
print(f"  {nh} transverse + {nv} longitudinal planes, synchronised on their intersection lines",flush=True)
# where each plane's image carries the shared line ------------------------------------------
# transverse plane j, longitudinal plane th: the line is {radius r, height h_j, azimuth th}.
# On the transverse image it is the diameter at angle th; on the longitudinal image it is the
# horizontal row at height h_j.  Both are sampled at the same NR radii, signed.
NR=R
rr=torch.linspace(-1,1,NR,device=dv)
def trans_line(img,th):                    # (3,R,R) -> (3,NR) along the diameter at th
    u=rr*math.cos(th); v_=rr*math.sin(th)
    gg=torch.stack([u,v_],-1)[None,None]
    return F.grid_sample(img[None],gg,mode="bilinear",padding_mode="border",align_corners=True)[0,:,0]
def long_row(img,hj):                      # (3,R,R) -> (3,NR) at axial position of height hj
    y=torch.full_like(rr,(hj-vol.c)/vol.EXT)
    gg=torch.stack([rr,y],-1)[None,None]
    return F.grid_sample(img[None],gg,mode="bilinear",padding_mode="border",align_corners=True)[0,:,0]
iT=max(int(T0*(T-1)),1); ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
XT=ab[iT].sqrt()*TRI+(1-ab[iT]).sqrt()*torch.randn(TRI.shape,device=dv,generator=g)
XL=ab[iT].sqrt()*LOI+(1-ab[iT]).sqrt()*torch.randn(LOI.shape,device=dv,generator=g)
t0=time.time()
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    def run(model,X,t,bs=12):
        o=torch.empty_like(X)
        for k in range(0,len(X),bs):
            b=X[k:k+bs]
            with torch.no_grad():
                o[k:k+bs]=model(b,torch.full((len(b),),t,device=dv,dtype=torch.long))
        return o
    et=run(MH,XT,tc); el=run(MV,XL,tc)
    t0h=((XT-(1-ab[tc]).sqrt()*et)/ab[tc].sqrt()).clamp(-1,1)
    t0v=((XL-(1-ab[tc]).sqrt()*el)/ab[tc].sqrt()).clamp(-1,1)
    if SYNC>0:
        # every (transverse j, longitudinal k) pair shares one line; move both halfway to its mean
        dT=torch.zeros_like(t0h); dL=torch.zeros_like(t0v)
        for k,th in enumerate(ths):
            lt=torch.stack([trans_line(t0h[a],th) for a in range(nh)])      # (nh,3,NR)
            lv=torch.stack([long_row(t0v[k],H[a]) for a in range(nh)])      # (nh,3,NR)
            mean=0.5*(lt+lv)
            # write the correction back along the same line, as a thin ridge in each image
            for a in range(nh):
                c=(mean[a]-lt[a])*SYNC*0.5
                u=rr*math.cos(th); v_=rr*math.sin(th)
                xi=((u+1)/2*(R-1)).round().long().clamp(0,R-1)
                yi=((v_+1)/2*(R-1)).round().long().clamp(0,R-1)
                dT[a,:,yi,xi]+=c
                c2=(mean[a]-lv[a])*SYNC*0.5
                yj=int(round(((H[a]-vol.c)/vol.EXT+1)/2*(R-1))); yj=max(0,min(R-1,yj))
                xj=((rr+1)/2*(R-1)).round().long().clamp(0,R-1)
                dL[k,:,yj,xj]+=c2
        t0h=(t0h+dT).clamp(-1,1); t0v=(t0v+dL).clamp(-1,1)
    if tn<=0: XT,XL=t0h,t0v; break
    XT=ab[tn].sqrt()*t0h+(1-ab[tn]).sqrt()*et
    XL=ab[tn].sqrt()*t0v+(1-ab[tn]).sqrt()*el
    if i%20==0: print(f"    step {i+1}/{NSTEP}  t={tc}  {time.time()-t0:.0f}s",flush=True)
# measure the line disagreement the way memory did
dis=[]
for k,th in enumerate(ths):
    for a in range(nh):
        dis.append(float((trans_line(XT[a],th)-long_row(XL[k],H[a])).abs().mean()))
print(f"  line disagreement: {np.mean(dis):.4f}",flush=True)
def sharp2(img):
    a=img.mean(0)
    return float((a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]).abs().mean())
print(f"  sharpness: transverse {np.mean([sharp2(x) for x in XT]):.4f}  "
      f"longitudinal {np.mean([sharp2(x) for x in XL]):.4f}",flush=True)
print(f"  init      : transverse {np.mean([sharp2(x) for x in TRI]):.4f}  "
      f"longitudinal {np.mean([sharp2(x) for x in LOI]):.4f}",flush=True)
torch.save({"XT":XT.cpu(),"XL":XL.cpu(),"H":H,"ths":ths},OUT+"/planes.pt")
s=Image.new("RGB",(4*266,290),(255,255,255)); d=ImageDraw.Draw(s)
for i,(t,lab) in enumerate(((LOI[0],"init longitudinal"),(XL[0],"sync longitudinal"),
                            (TRI[nh//2],"init transverse"),(XT[nh//2],"sync transverse"))):
    a=((t.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy()
    s.paste(Image.fromarray(a).resize((260,260),Image.NEAREST),(i*266+3,26))
    d.text((i*266+5,7),lab,fill=(30,30,30))
s.save(OUT+"/sync.png"); print("  wrote",OUT+"/sync.png",flush=True)
