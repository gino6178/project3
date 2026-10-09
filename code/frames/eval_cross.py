# Cross-specimen evaluation (the teacher's section 12): the inputs came from six different specimens,
# so no instance truth exists.  Each arm's face on every test plane is scored by DreamSim to the
# NEAREST of three held-out specimens' faces on the same plane (the paper's held-out protocol, with
# planes the photographs never had), and the real floor is specimen-to-specimen on the same planes.
#   python eval_cross.py <object dir> [arm=state.pt ...]
import os,sys,json,glob,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import synth2 as S
from PIL import Image
dv="cuda" if torch.cuda.is_available() else "cpu"; S.dv=dv
D=sys.argv[1]; R=256
G=torch.load(f"{D}/grid.pt",map_location=dv); N=G["N"]; C=(N-1)/2; CORE=G["CORE"].float()
PJ=json.load(open(f"{D}/planes.json")); ext,h0,h1=PJ["ext"],PJ["h0"],PJ["h1"]
arms={"carrier":torch.load(f"{D}/carrier.pt",map_location=dv)["V"].float()}
for a in sys.argv[2:]:
    n,p=a.split("=",1)
    if os.path.exists(p): arms[n]=torch.load(p,map_location=dv)["X"].float()
from dreamsim import dreamsim; DS,DSP=dreamsim(pretrained=True,device=dv,cache_dir=os.path.expanduser("~/dreamsim_ckpt"))
def cut(X,spec):
    P=S.plane_points(ext,spec,R)+C; q=torch.stack([P[:,2],P[:,1],P[:,0]],-1)/(N-1)*2-1
    return F.grid_sample(X[None],q[None,None,None],mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0].T.reshape(R,R,X.shape[0])
def emb_t(t,m): t=torch.where(m[...,None],t,torch.ones_like(t)); return DSP(Image.fromarray((t.cpu().numpy()*255).astype(np.uint8))).to(dv)
def emb_p(p,size): return DSP(Image.open(p).convert("RGB").resize((size,size),Image.LANCZOS)).to(dv)
fams=[("trans",[("trans",h0+(h1-h0)*q) for q in S.TEST_HQ]),("long",[("long",th,0.0) for th in S.TEST_TH]),("obl",[("obl",)+t for t in S.TEST_OBL])]
out={a:{} for a in arms}; out["real floor"]={}
with torch.no_grad():
    for fam,specs in fams:
        acc={a:[] for a in arms}; fl=[]
        for i,spec in enumerate(specs):
            refs=[emb_p(p,R) for p in sorted(glob.glob(f"{D}/hld_{fam}/*_{i}.png"))]
            if not refs: continue
            fl.append(np.mean([min(float(DS(refs[j],refs[k])) for k in range(len(refs)) if k!=j) for j in range(len(refs))]))
            m=cut(CORE,spec)[...,0]>0.5
            for a,X in arms.items():
                e=emb_t((cut(X,spec).clamp(-1,1)+1)/2,torch.ones_like(m)); acc[a].append(min(float(DS(e,r)) for r in refs))
        for a in arms: out[a][fam]=float(np.mean(acc[a]))
        out["real floor"][fam]=float(np.mean(fl))
json.dump(out,open(f"{D}/cross_scores.json","w"),indent=1)
print(f"\n{os.path.basename(D)}  cross-specimen, DreamSim to the nearest held-out specimen on the same plane")
for a,v in out.items(): print(f"  {a:12s} " + "  ".join(f"{f} {v[f]:.4f}" for f in ("trans","long","obl")))
