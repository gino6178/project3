# A light carrier for the procedural objects: the interior fitted directly to the training
# photographs on the planes they were assigned to, and nothing between them but smoothness.
# What it keeps is what the paper's carrier keeps -- layout and colour, an average over specimens
# on the supervised planes, a coarse block-tiled field between them -- without the Gaussian
# exterior, the decoder or the phase solve, which the procedural objects do not need: their
# specimens share one shape and one frame.  The shell is the grid's own and is never fitted.
#
#   GRID=<synth grid.pt> OBJDIR=<dir with spl_*> OUT=<carrier.pt> python carrier.py
#
# Plane assignment: the paper's carrier's own (see below), so that what the lift inherits is what
# it would inherit from the O-Voxel fit.
import os,sys,glob,math,numpy as np,torch,torch.nn.functional as F
from PIL import Image
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill"))
from planes import Vol
dv=os.environ.get("DEV","cuda:0"); RES=256; NC=int(os.environ.get("NC","64"))
vol=Vol(os.environ["GRID"],dv,res=RES); N=vol.N; OD=os.environ["OBJDIR"]
def load(d): return [torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((RES,RES),Image.LANCZOS)).astype(np.float32)/127.5-1).permute(2,0,1).to(dv) for p in sorted(glob.glob(d+"/*.png"))]
PL,PT=load(f"{OD}/spl_long"),load(f"{OD}/spl_trans")
hs=vol.occupied_heights(); h0,h1=hs[0],hs[-1]
import json
KIND=json.load(open(f"{OD}/meta.json")).get("kind","revolve") if os.path.exists(f"{OD}/meta.json") else "revolve"
# The paper's carrier (src/train_voxel.py): N_VPLANES=10 vertical planes spread over the half turn,
# each jittered by up to JITTER=0.5 of the spacing every step, plane i drawing photograph
# i*n//N_VPLANES; sixteen transverse planes over 0.167..0.833 of the depth, the camera's roll drawn
# at random every step (upright is undetermined looking down the axis, and so is the photograph's).
NV=int(os.environ.get("N_VPLANES","10")); NH=int(os.environ.get("N_HPLANES","16")); JIT=float(os.environ.get("JITTER","0.5"))
if KIND=="box":   # a block is cut parallel to a face: the planes alternate the two face directions, spread in depth
    VP=[((i%2)*math.pi/2, -24.0+48.0*(i//2)/max(NV//2-1,1), i*len(PL)//NV) for i in range(NV)]
else:
    VP=[(math.pi/NV*i, 0.0, i*len(PL)//NV) for i in range(NV)]
HP=[(h0+(h1-h0)*(0.167+0.666*j/max(NH-1,1)), j*len(PT)//NH) for j in range(NH)]
# Same-specimen benchmark (synth2.py): the planes each input was cut on are known, so the carrier is
# fitted on exactly those planes (depth jitter of one voxel, no azimuth jitter, no roll).
EXACT=os.path.exists(f"{OD}/planes.json")
if EXACT:
    PJ=json.load(open(f"{OD}/planes.json"))
    VP=[(th,off,k) for k,(th,off) in enumerate(PJ["long"])][:len(PL)]; HP=[(h,k) for k,h in enumerate(PJ["trans"])][:len(PT)]
    NV,NH,JIT=len(VP),len(HP),0.0
def roll(img,a):
    c_,s_=math.cos(a),math.sin(a); th=torch.tensor([[c_,-s_,0],[s_,c_,0]],device=dv,dtype=img.dtype)[None]
    g=F.affine_grid(th,(1,3,RES,RES),align_corners=False)
    return F.grid_sample(img[None]+1,g,mode="bilinear",padding_mode="zeros",align_corners=False)[0]-1   # outside -> white
core=vol.CORE[0]>0.5
mean=torch.stack([p[:,(p.min(0).values<0.84)].mean(1) for p in PL+PT]).mean(0)
Z=torch.nn.Parameter(mean.view(3,1,1,1).repeat(1,NC,NC,NC).clone())
opt=torch.optim.Adam([Z],lr=float(os.environ.get("LR","0.03")))
C3=vol.CORE.expand(3,-1,-1,-1).contiguous()
def field():
    X=F.interpolate(Z[None],size=(N,N,N),mode="trilinear",align_corners=True)[0]
    return torch.where(core[None],X,vol.V0)
def slice_long(X,th,off):
    c_,s_=math.cos(th),math.sin(th)
    return vol._samp(X, vol.GU*c_-off*s_+vol.c, vol.GV+vol.c, vol.GU*s_+off*c_+vol.c)
g=torch.Generator().manual_seed(0); STEPS=int(os.environ.get("STEPS","6000")); TV=float(os.environ.get("TV","0.02"))
for it in range(1,STEPS+1):
    X=field(); loss=0
    for _ in range(4):
        if PL and (not PT or torch.rand(1,generator=g)<0.5):
            th,off,k=VP[int(torch.randint(NV,(1,),generator=g))]
            sp=(math.pi/NV) if KIND!="box" else 0.0; th=th+sp*JIT*float(torch.rand(1,generator=g)*2-1)
            off=off+float(torch.rand(1,generator=g)*6-3) if KIND=="box" and not EXACT else off+(float(torch.rand(1,generator=g)*2-1) if EXACT else 0.0)
            img,m=slice_long(X,th,off),slice_long(C3,th,off)>0.5; ref=PL[k]
        else:
            h,k=HP[int(torch.randint(NH,(1,),generator=g))]
            h=h+(float(torch.rand(1,generator=g)*2-1) if EXACT else float(torch.rand(1,generator=g)*2-1)*(h1-h0)*0.333/NH)
            img,m=vol.trans_slice(X,h),vol.trans_slice(C3,h)>0.5
            ref=PT[k] if EXACT else roll(PT[k],float(torch.rand(1,generator=g))*2*math.pi)
        loss=loss+((img-ref).abs()*m).sum()/m.sum().clamp(min=1)
    tv=sum((Z.diff(dim=d)**2).mean() for d in (1,2,3))
    (loss/4+TV*tv).backward(); opt.step(); opt.zero_grad()
    if it%500==0 or it==1: print(f"  {it}/{STEPS} L1 {float(loss)/4:.4f} tv {float(tv):.5f}",flush=True)
G=torch.load(os.environ["GRID"],map_location="cpu"); G["V"]=field().detach().clamp(-1,1).cpu(); G["GT"]=torch.load(os.environ["GRID"])["V"]
torch.save(G,os.environ["OUT"]); print("CARRIER_DONE",os.environ["OUT"])
