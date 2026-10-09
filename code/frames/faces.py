# Cut faces of every arm of one object, in the photographs' frame, ready for the DreamSim scorer.
# CPU only.  The planes depend on the object's shape, never on the frame being scored:
#   revolve   longitudinal through the axis at 6 azimuths
#   box       longitudinal parallel to a face, both directions, at depths -12, 0, +12 voxels
#   both      transverse at 30..70% of the occupied height (5 planes)
#   both      oblique, 45 deg between the families (3 planes; no photograph exists -- for the eye)
# Arms: the carrier, and every <frame>/state.pt present.  Faces and held-out photographs are both
# canonicalised by fidelity.canon, as dsscore.py does.
#   python faces.py <object dir> [arm=path ...]
import os,sys,glob,math,json,numpy as np,torch
from PIL import Image
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill"))
from planes import Vol
from fidelity import canon
RES=512; D=sys.argv[1]
grid=f"{D}/carrier.pt" if os.path.exists(f"{D}/carrier.pt") else f"{D}/grid.pt"
vol=Vol(grid,"cpu",res=RES)
kind=json.load(open(f"{D}/meta.json"))["kind"] if os.path.exists(f"{D}/meta.json") else "revolve"
def slong(X,th,off):
    c_,s_=math.cos(th),math.sin(th)
    return vol._samp(X, vol.GU*c_-off*s_+vol.c, vol.GV+vol.c, vol.GU*s_+off*c_+vol.c)
def sobl(X,tilt,off):
    c_,s_=math.cos(tilt),math.sin(tilt)
    return vol._samp(X, vol.GU+vol.c, vol.GV*s_+off+vol.c, vol.GV*c_+vol.c)
LONG=[(k*math.pi/6,0.0) for k in range(6)] if kind=="revolve" else [(th,o) for th in (0.0,math.pi/2) for o in (-12.0,0.0,12.0)]
hs=vol.occupied_heights(); TR=[float(hs[int(len(hs)*q)]) for q in (0.3,0.4,0.5,0.6,0.7)]
OB=[(math.pi/4,o) for o in (-8.0,0.0,8.0)]
def face(t,p):
    t=(t.squeeze(0).clamp(-1,1)+1)/2; Image.fromarray((t.permute(1,2,0).numpy()*255).astype(np.uint8)).save(p)
    Image.fromarray((canon(p,RES)*255).astype(np.uint8)).save(p)
arms=[("carrier",torch.load(grid,map_location="cpu")["V"])]
for p in sorted(glob.glob(f"{D}/*/state.pt")): arms.append((os.path.basename(os.path.dirname(p)),torch.load(p,map_location="cpu")["X"]))
for a in sys.argv[2:]: n,p=a.split("=",1); arms.append((n,torch.load(p,map_location="cpu")["X"]))
for n,X in arms:
    for fam,planes,fn in (("long",LONG,slong),("trans",TR,None),("obl",OB,sobl)):
        d=f"{D}/faces/{n}/{fam}"; os.makedirs(d,exist_ok=True)
        for i,pl in enumerate(planes):
            img=vol.trans_slice(X,pl) if fam=="trans" else fn(X,*pl)
            face(img,f"{d}/{i}.png")
for fam in ("long","trans"):
    d=f"{D}/faces/_ref/{fam}"; os.makedirs(d,exist_ok=True)
    for p in glob.glob(f"{D}/hld_{fam}/*.png"): Image.fromarray((canon(p,RES)*255).astype(np.uint8)).save(f"{d}/"+os.path.basename(p))
    d=f"{D}/faces/_spl/{fam}"; os.makedirs(d,exist_ok=True)
    for p in glob.glob(f"{D}/spl_{fam}/*.png"): Image.fromarray((canon(p,RES)*255).astype(np.uint8)).save(f"{d}/"+os.path.basename(p))
open(f"{D}/faces/arms.txt","w").write("\n".join(n for n,_ in arms))
print("faces:",D,[n for n,_ in arms])
