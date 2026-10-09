# Held-out DreamSim for a curved object cut the way its photographs were: lengthwise along surfaces that
# follow the declared centreline (azimuths k*pi/6), crosswise normal to it (parallel-transported frames,
# 30..70% of its length) -- and, for comparison, the paper's own flat protocol (dsscore.py's planes).
# Faces and held-out photographs are both canonicalised by fidelity.canon, the score is the mean distance
# to the NEAREST held-out photograph, the real floor is training photographs against held-out ones.
#   python eval_curved.py <object dir> [arm=state.pt ...]       (carrier.pt is always an arm)
import os,sys,glob,json,math,numpy as np,torch,torch.nn.functional as F
from PIL import Image
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill"))
import synth2 as S
from planes import Vol
from fidelity import canon
dv="cuda" if torch.cuda.is_available() else "cpu"; S.dv=dv
D=sys.argv[1]; R=512
vol=Vol(f"{D}/carrier.pt",dv,res=R); N=vol.N; C=(N-1)/2; ext=vol.EXT
CL=json.load(open(f"{D}/centerline.json")); cv=torch.tensor(CL["v"],device=dv); cx_=torch.tensor(CL["cx"],device=dv); cz_=torch.tensor(CL["cz"],device=dv)
def cl(v):
    i=torch.bucketize(v,cv).clamp(1,len(cv)-1); t=((v-cv[i-1])/(cv[i]-cv[i-1])).clamp(0,1); return cx_[i-1]*(1-t)+cx_[i]*t, cz_[i-1]*(1-t)+cz_[i]*t
S.CENTER=cl
zs=torch.linspace(-C,C,4*N,device=dv); px,pz=cl(zs); P3=torch.stack([px,zs,pz],1)
T=torch.zeros_like(P3); T[1:-1]=P3[2:]-P3[:-2]; T[0]=T[1]; T[-1]=T[-2]; T=T/T.norm(dim=1,keepdim=True)
Nn=torch.zeros_like(P3); n=torch.tensor([1.,0.,0.],device=dv); n=n-(n@T[0])*T[0]; Nn[0]=n/n.norm()
for i in range(1,len(zs)): n=Nn[i-1]-(Nn[i-1]@T[i])*T[i]; Nn[i]=n/n.norm()
Bb=torch.linalg.cross(T,Nn)
def nframe(v):
    i=int(torch.argmin((zs-v).abs())); return P3[i:i+1],Nn[i:i+1],T[i:i+1]
occ=[j-C for j in range(N) if float(vol.OCC[0][:,j,:].sum())>20]; h0,h1=min(occ),max(occ)
def ntrans_points(v):
    p,n,t=nframe(torch.tensor(v,device=dv)); b=torch.linalg.cross(t,n)
    gl=torch.linspace(-ext,ext,R,device=dv); GV,GU=torch.meshgrid(gl,gl,indexing="ij")
    return (p[0]+GU[...,None]*n[0]+GV[...,None]*b[0]).reshape(-1,3)
def cut(X,P):
    Q=P+C; q=torch.stack([Q[:,2],Q[:,1],Q[:,0]],-1)/(N-1)*2-1
    return F.grid_sample(X[None],q[None,None,None],mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0].T.reshape(R,R,X.shape[0])
arms={"carrier":vol.V0}
for a in sys.argv[2:]:
    n_,p=a.split("=",1)
    if os.path.exists(p): arms[n_]=torch.load(p,map_location=dv)["X"].float()
protocols={
 "curved":{"long":[S.plane_points(ext,("long",k*math.pi/6,0.0),R) for k in range(6)],
           "trans":[ntrans_points(h0+(h1-h0)*q) for q in (0.3,0.4,0.5,0.6,0.7)]},
 "flat":  {"long":[(lambda th:(lambda c_,s_:torch.stack([vol.GU*c_,vol.GV,vol.GU*s_],-1).reshape(-1,3))(math.cos(th),math.sin(th)))(k*math.pi/6) for k in range(6)],
           "trans":[torch.stack([vol.GU,torch.full_like(vol.GU,h0+(h1-h0)*q),vol.GV],-1).reshape(-1,3) for q in (0.3,0.4,0.5,0.6,0.7)]}}
S.CENTER=None
out=f"{D}/curved_faces"; os.makedirs(out,exist_ok=True)
for prot,fam in protocols.items():
    for f,Ps in fam.items():
        for a,X in arms.items():
            d=f"{out}/{prot}/{a}/{f}"; os.makedirs(d,exist_ok=True)
            for i,P in enumerate(Ps):
                im=((cut(X,P).clamp(-1,1)+1)/2).cpu().numpy(); p_=f"{d}/{i}.png"
                Image.fromarray((im*255).astype(np.uint8)).save(p_); Image.fromarray((canon(p_,R)*255).astype(np.uint8)).save(p_)
for f in ("long","trans"):
    for k,src in (("_ref",f"{D}/hld_{f}"),("_spl",f"{D}/spl_{f}")):
        d=f"{out}/{k}/{f}"; os.makedirs(d,exist_ok=True)
        for p in glob.glob(f"{src}/*.png"): Image.fromarray((canon(p,R)*255).astype(np.uint8)).save(f"{d}/"+os.path.basename(p))
from dreamsim import dreamsim; DS,DSP=dreamsim(pretrained=True,device=dv,cache_dir=os.path.expanduser("~/dreamsim_ckpt"))
def emb(p): return DSP(Image.open(p).convert("RGB")).to(dv)
def ds(refs,paths):
    R_=[emb(p) for p in refs]; o=[]
    with torch.no_grad():
        for p in paths: x=emb(p); o.append(min(float(DS(x,r)) for r in R_))
    return float(np.mean(o))
res={}
for f in ("long","trans"):
    refs=sorted(glob.glob(f"{out}/_ref/{f}/*.png")); res.setdefault("real floor",{})[f]=ds(refs,sorted(glob.glob(f"{out}/_spl/{f}/*.png")))
for prot in protocols:
    for a in arms:
        res.setdefault(f"{prot}:{a}",{})
        for f in ("long","trans"): res[f"{prot}:{a}"][f]=ds(sorted(glob.glob(f"{out}/_ref/{f}/*.png")),sorted(glob.glob(f"{out}/{prot}/{a}/{f}/*.png")))
for v in res.values(): v["mean"]=(v["long"]+v["trans"])/2
json.dump(res,open(f"{D}/curved_scores.json","w"),indent=1)
print(f"\n{os.path.basename(D)}  held-out DreamSim (long / trans / mean), lower better")
for k,v in res.items(): print(f"  {k:28s} {v['long']:.4f} {v['trans']:.4f} {v['mean']:.4f}")
