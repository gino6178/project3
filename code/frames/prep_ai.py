# Generated photographs -> an object the pipeline can run: a declared split, an occupancy grid,
# and every photograph in the grid's own plane frame.  CPU only.
#
#   python prep_ai.py <aiobj dir> <object>
#
# split    the accepted images of each family (REJECT.txt removed), in index order: three train,
#          the next three are held out and never seen by a prior or the carrier.
# shape    the declared kind: "revolve" turns the training longitudinal silhouettes' mean profile
#          about the axis (log, maki, onion); "box" extrudes a rectangle with the training
#          photographs' mean aspect (honeycomb, strata, terrazzo, cheese).  Shell = the outer two
#          voxels, coloured with the photographs' mean rind; the interior is the carrier's to fit.
# frame    each photograph is cut out (corner background, largest component, holes filled) and
#          mapped bbox-to-bbox onto the silhouette of the grid's own slice of its family -- the
#          carrier's similarity warp (eq. 2), per axis because specimens differ in aspect.
import os,sys,glob,math,json,numpy as np,torch,torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill"))
from planes import Vol
KIND={"log":"revolve","maki":"revolve","onion":"revolve","cable":"revolve","honeycomb":"box","strata":"box","terrazzo":"box","cheese":"box"}
root,obj=sys.argv[1],sys.argv[2]; D=f"{root}/{obj}"; N=128; C=(N-1)/2; RES=512; SIZE=50.0
rej=set()
for l in open(f"{root}/REJECT.txt"):
    if l.startswith("#") or not l.strip(): continue
    f=l.split()
    if f[0]==obj: rej|={w for w in f[1:] if w.startswith(("long_","trans_"))}
def cut(p):
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255; h,w,_=a.shape
    bg=np.median(np.concatenate([a[:16,:16].reshape(-1,3),a[:16,-16:].reshape(-1,3),a[-16:,:16].reshape(-1,3),a[-16:,-16:].reshape(-1,3)]),0)
    m=np.abs(a-bg).max(2)>0.10
    m=ndi.binary_opening(m,iterations=2); lab,n=ndi.label(m)
    if n==0: return None
    m=lab==(np.argmax(ndi.sum(m,lab,range(1,n+1)))+1); m=ndi.binary_fill_holes(m)
    return a,m
def bbox(m): ys,xs=np.where(m); return ys.min(),ys.max(),xs.min(),xs.max()
fams={}
for fam in ("long","trans"):
    ps=[p for p in sorted(glob.glob(f"{D}/raw/{fam}_*.png"),key=lambda s:int(s.rsplit("_",1)[1][:-4])) if os.path.basename(p)[:-4] not in rej]
    fams[fam]=[(p,)+cut(p) for p in ps][:6]
    assert len(fams[fam])==6, (obj,fam,len(fams[fam]))
# ---- the shape: a given grid (GRID=..., e.g. glb2grid.py from a generated exterior), or from the
# training photographs only ----
USE=os.environ.get("GRID","")
def aspect(a_m): y0,y1,x0,x1=bbox(a_m[2]); return (y1-y0)/(x1-x0)
lin=torch.arange(N,dtype=torch.float32)-C; A,B,Cc=torch.meshgrid(lin,lin,lin,indexing="ij")
if USE:
    G=torch.load(USE); torch.save(G,f"{D}/grid.pt"); core=G["CORE"][0]
elif KIND[obj]=="revolve":
    prof=[]
    for p,a,m in fams["long"][:3]:
        y0,y1,x0,x1=bbox(m); mm=m[y0:y1+1,x0:x1+1]; H_,W_=mm.shape
        rows=np.linspace(0,H_-1,N).astype(int); cx=W_/2
        half=np.array([ (np.abs(np.where(mm[r])[0]-cx).max() if mm[r].any() else 0) for r in rows])/H_
        prof.append(half)
    prof=ndi.uniform_filter1d(np.mean(prof,0),5)                                # half-width / height, per row
    Hh=SIZE/max(1.0,2*prof.max()); Hh=min(Hh,SIZE)                              # half-height
    Rmax=float(2*Hh*prof.max()); scale=SIZE/max(Hh,Rmax); Hh*=scale
    vv=((B+Hh)/(2*Hh)*(N-1)).clamp(0,N-1).long()
    rr=torch.sqrt(A**2+Cc**2); lim=torch.from_numpy(prof).float()[vv]*2*Hh
    sd=torch.maximum(rr-lim,B.abs()-Hh)
else:
    asp=np.mean([aspect(x) for x in fams["long"][:3]]); dep=np.mean([aspect(x) for x in fams["trans"][:3]])
    w=1.0; Hh=asp*w; d=dep*w; s=SIZE/max(math.hypot(w,d),Hh); w,Hh,d=w*s,Hh*s,d*s
    sd=torch.stack([A.abs()-w,Cc.abs()-d,B.abs()-Hh]).max(0).values
if not USE:
    occ=(sd<0).float(); shell=((sd<0)&(sd>-2)).float(); core=(sd<=-2).float()
    # rind colour: the photographs' outer band
    rind=[]
    for fam in ("long","trans"):
        for p,a,m in fams[fam][:3]:
            er=ndi.binary_erosion(m,iterations=max(2,int(0.02*m.shape[0]))); rind.append(a[m&~er].mean(0))
    rc=torch.tensor(np.mean(rind,0)*2-1,dtype=torch.float32)
    V=torch.ones(3,N,N,N); V[:,occ>0]=rc[:,None]
    interior=np.mean([a[ndi.binary_erosion(m,iterations=20)].mean(0) for fam in fams for p,a,m in fams[fam][:3]],0)*2-1
    V[:,core>0]=torch.tensor(interior,dtype=torch.float32)[:,None]
    G=dict(V=V,OCC=occ[None],SHELL=shell[None],CORE=core[None],N=N); torch.save(G,f"{D}/grid.pt")

# ---- every photograph into the grid's frame ----
vol=Vol(f"{D}/grid.pt","cpu",res=RES)
def sil(fam):
    if fam=="long": s=vol.long_slice(vol.OCC.expand(3,-1,-1,-1).contiguous(),0.0)
    else: hs=vol.occupied_heights(); s=vol.trans_slice(vol.OCC.expand(3,-1,-1,-1).contiguous(),float(hs[len(hs)//2]))
    return s[0].numpy()>0.5
def toframe(a,m,tgt):
    y0,y1,x0,x1=bbox(m); Y0,Y1,X0,X1=bbox(tgt)
    yy,xx=np.mgrid[0:RES,0:RES].astype(np.float32)
    sy=y0+(yy-Y0)*(y1-y0)/max(Y1-Y0,1); sx=x0+(xx-X0)*(x1-x0)/max(X1-X0,1)
    t=torch.from_numpy(np.where(m[...,None],a,1.0)).permute(2,0,1)[None].float()
    g=torch.stack([torch.from_numpy(sx/(a.shape[1]-1)*2-1),torch.from_numpy(sy/(a.shape[0]-1)*2-1)],-1)[None]
    out=F.grid_sample(t,g,mode="bilinear",padding_mode="border",align_corners=True)[0].permute(1,2,0).numpy()
    return np.where(tgt[...,None],out,1.0)
def polar(a,NR=128,NPHI=512):
    H=a.shape[0]; t=torch.from_numpy(a).permute(2,0,1)[None].float()
    r=(torch.arange(NR)+0.5)/NR*(H/2); ph=torch.arange(NPHI)/NPHI*2*np.pi; rr,pp=torch.meshgrid(r,ph,indexing="ij")
    x=(H/2+rr*torch.cos(pp))/(H-1)*2-1; y=(H/2+rr*torch.sin(pp))/(H-1)*2-1
    return F.grid_sample(t,torch.stack([x,y],-1)[None],mode="bilinear",padding_mode="border",align_corners=True)[0].permute(1,2,0).numpy()
split=[]
for fam in ("long","trans"):
    tgt=sil(fam)
    for k,(p,a,m) in enumerate(fams[fam]):
        d=f"{D}/{'spl' if k<3 else 'hld'}_{fam}"; os.makedirs(d,exist_ok=True)
        im=toframe(a,m,tgt); Image.fromarray((im*255).astype(np.uint8)).save(f"{d}/{k%3:02d}.png")
        split.append(f"{fam} {'train' if k<3 else 'held-out'} {os.path.basename(p)}")
        if fam=="trans" and k<3:
            os.makedirs(f"{D}/polar_spl_trans",exist_ok=True)
            Image.fromarray((polar(im)*255).astype(np.uint8)).save(f"{D}/polar_spl_trans/{k:02d}.png")
open(f"{D}/SPLIT.txt","w").write("\n".join(split)+"\n")
json.dump(dict(object=obj,kind=KIND[obj],core=int(core.sum()),ext=vol.EXT),open(f"{D}/meta.json","w"))
print(obj,KIND[obj],"core",int(core.sum()),"ext",round(vol.EXT,1))
