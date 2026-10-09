# The banana taken as a straight cylinder (a diagnostic): its lengthwise photographs straightened along
# their own centreline (as prep_ai.py STRAIGHTEN does), its shape the solid of revolution of those
# straightened silhouettes, its transverse photographs as they are.  Same frame as the curved object.
#   python prep_straight.py <curved object dir> <out dir>
import os,sys,glob,shutil,json,numpy as np,torch
from PIL import Image
from scipy import ndimage as ndi
src,out=sys.argv[1],sys.argv[2]; os.makedirs(out,exist_ok=True)
def straighten(p,q):
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255; m=a.min(2)<0.92; H,W=m.shape
    mid=np.array([(np.where(m[r])[0].mean() if m[r].any() else np.nan) for r in range(H)]); ok=~np.isnan(mid)
    mid=np.interp(np.arange(H),np.where(ok)[0],mid[ok]); mid=ndi.uniform_filter1d(mid,25); o=np.ones_like(a)
    for r in range(H):
        sh=int(round(W/2-mid[r])); o[r]=np.roll(a[r],sh,axis=0)
        if sh>0: o[r,:sh]=1.0
        elif sh<0: o[r,sh:]=1.0
    Image.fromarray((o*255).astype(np.uint8)).save(q)
for d in ("spl_long","hld_long"):
    os.makedirs(f"{out}/{d}",exist_ok=True)
    for p in sorted(glob.glob(f"{src}/{d}/*.png")): straighten(p,f"{out}/{d}/"+os.path.basename(p))
def centre(p,q):          # the cross-section moved to the frame centre (the straight cylinder's axis), size kept
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255; m=a.min(2)<0.92; ys,xs=np.where(m); H,W=m.shape
    cy,cx=(ys.min()+ys.max())//2,(xs.min()+xs.max())//2; o=np.roll(np.roll(a,H//2-cy,axis=0),W//2-cx,axis=1)
    Image.fromarray((o*255).astype(np.uint8)).save(q); return o
def polar(a,NR=128,NPHI=512):
    import torch.nn.functional as F
    H=a.shape[0]; t=torch.from_numpy(a).permute(2,0,1)[None].float()
    r=(torch.arange(NR)+0.5)/NR*(H/2); ph=torch.arange(NPHI)/NPHI*2*np.pi; rr,pp=torch.meshgrid(r,ph,indexing="ij")
    x=(H/2+rr*torch.cos(pp))/(H-1)*2-1; y=(H/2+rr*torch.sin(pp))/(H-1)*2-1
    return F.grid_sample(t,torch.stack([x,y],-1)[None],mode="bilinear",padding_mode="border",align_corners=True)[0].permute(1,2,0).numpy()
for d in ("spl_trans","hld_trans","polar_spl_trans"): shutil.rmtree(f"{out}/{d}",ignore_errors=True); os.makedirs(f"{out}/{d}")
for d in ("spl_trans","hld_trans"):
    for p in sorted(glob.glob(f"{src}/{d}/*.png")):
        o=centre(p,f"{out}/{d}/"+os.path.basename(p))
        if d=="spl_trans": Image.fromarray((polar(o)*255).astype(np.uint8)).save(f"{out}/polar_spl_trans/"+os.path.basename(p))
# the revolve shape, in the curved grid's own frame (same N, same extent), from the straightened silhouettes
G=torch.load(f"{src}/grid.pt"); N=G["N"]; C=(N-1)/2
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill")); from planes import Vol
vol=Vol(f"{src}/grid.pt","cpu",res=512); E=vol.EXT
prof=[]
for p in sorted(glob.glob(f"{out}/spl_long/*.png")):
    m=np.asarray(Image.open(p).convert("RGB")).min(2)<235; H,W=m.shape
    prof.append(np.array([(np.abs(np.where(m[r])[0]-W/2).max() if m[r].any() else 0) for r in range(H)]))
prof=ndi.uniform_filter1d(np.median(prof,0),9)*(2*E/512)                    # half-width per frame row, voxel units
lin=torch.arange(N).float()-C; A,B,Cc=torch.meshgrid(lin,lin,lin,indexing="ij")
row=((B+E)/(2*E)*511).round().long().clamp(0,511); lim=torch.from_numpy(prof).float()[row]
sd=torch.sqrt(A**2+Cc**2)-lim; occ=(sd<0)&(lim>0.5)
dist=torch.from_numpy(ndi.distance_transform_edt(occ.numpy())); shell=occ&(dist<=2); core=occ&(dist>2)
V=torch.ones(3,N,N,N); V[:,occ]=torch.tensor([0.80,0.70,0.25])[:,None]*2-1; V[:,core]=torch.tensor([0.93,0.88,0.75])[:,None]*2-1
torch.save(dict(V=V,OCC=occ[None].float(),SHELL=shell[None].float(),CORE=core[None].float(),N=N),f"{out}/grid.pt")
json.dump(dict(object="banana_straight",kind="revolve"),open(f"{out}/meta.json","w"))
vs=np.linspace(-64,64,129); json.dump(dict(v=vs.tolist(),cx=[0.0]*129,cz=[0.0]*129),open(f"{out}/centerline.json","w"))
print(out,"core",int(core.sum()),"ext",round(Vol(f"{out}/grid.pt","cpu",res=64).EXT,1),"(curved:",round(E,1),")")
