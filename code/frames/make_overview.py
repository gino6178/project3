# The overview figure the teacher asked for (section 22): one object per canonical coordinate system,
# its cut in the world and the same anatomy in the canonical domain the priors see -- every panel a
# real sample of the ground-truth solid, nothing drawn by hand.
#   layer cake   Cartesian      a vertical cut is already stationary: horizontal layers
#   tree trunk   cylindrical    a transverse cut unwrapped to (r, phi): rings become parallel stripes
#   curved trunk curvilinear    a cut along the bent axis, straightened along the declared centreline
#   python make_overview.py <gtbench dir> <out.png>
import os,sys,json,math,numpy as np,torch,torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import synth2 as S
S.dv="cpu"; B,OUT=sys.argv[1],sys.argv[2]; R=320
def load(o):
    G=torch.load(f"{B}/{o}/grid.pt"); PJ=json.load(open(f"{B}/{o}/planes.json")); return G,PJ
def samp(G,P):
    N=G["N"]; C=(N-1)/2; P=P+C; q=torch.stack([P[...,2],P[...,1],P[...,0]],-1)/(N-1)*2-1
    return ((F.grid_sample(G["V"].float()[None],q.reshape(1,1,1,-1,3),mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0].T.clamp(-1,1)+1)/2)
def img(t,h,w): return Image.fromarray((t.reshape(h,w,3).numpy()*255).astype(np.uint8))
panels=[]
# cake: a vertical cut (world) and the Cartesian state's own slice -- the same image, which is the point
G,PJ=load("cake"); w=img(samp(G,S.plane_points(PJ["ext"],("long",0.3,0.0),R)),R,R)
panels.append(("layer cake","Cartesian  A[x, y, z]",w,w,"layers are already stationary stripes"))
# trunk: transverse cut and its (r, phi) unwrap
G,PJ=load("wood2"); h=0.5*(PJ["h0"]+PJ["h1"]); w=img(samp(G,S.plane_points(PJ["ext"],("trans",h),R)),R,R)
E=PJ["ext"]*0.82; NR,NP=160,R; r=(torch.arange(NR)+0.5)/NR*E; ph=torch.arange(NP)/NP*2*math.pi; rr,pp=torch.meshgrid(r,ph,indexing="ij")
P=torch.stack([rr*torch.cos(pp),torch.full_like(rr,h),rr*torch.sin(pp)],-1).reshape(-1,3)
c=img(samp(G,P),NR,NP).resize((R,R))
panels.append(("tree trunk","cylindrical  A[r, φ, z]",w,c,"rings become parallel stripes in (r, φ)"))
# banana: no truth inside a generated banana, so the panel is the finished asset itself -- the curvilinear
# lift of section 4.10 -- cut through its bend, and the same cut straightened along its declared centreline
BN=os.environ.get("BANANA","")
if BN:
    sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","slicefill")); from planes import Vol
    vb=Vol(f"{BN}/carrier.pt","cpu",res=R); X=torch.load(f"{BN}/curv2/state.pt",map_location="cpu")["X"].float()
    Gb=dict(V=X,N=vb.N); E=vb.EXT; CL=json.load(open(f"{BN}/centerline.json"))
    w=img(samp(Gb,S.plane_points(E,("long",0.0,0.0),R)),R,R)
    vs=torch.linspace(-E,E,R); cv=torch.tensor(CL["v"]); cx=torch.tensor(CL["cx"]); cz=torch.tensor(CL["cz"])
    i=torch.bucketize(vs,cv).clamp(1,len(cv)-1); t=((vs-cv[i-1])/(cv[i]-cv[i-1])).clamp(0,1); Ccx=cx[i-1]*(1-t)+cx[i]*t; Ccz=cz[i-1]*(1-t)+cz[i]*t
    s_=torch.linspace(-E,E,R); P=torch.stack([Ccx[:,None]+s_[None,:],vs[:,None].expand(R,R),Ccz[:,None].expand(R,R)],-1).reshape(-1,3)
    c=img(samp(Gb,P),R,R)
    panels.append(("banana","curvilinear  A[r, φ, s]",w,c,"straightened along its centreline"))
fp="/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
F1=ImageFont.truetype(fp,24); F2=ImageFont.truetype(fp,19); F3=ImageFont.truetype(fp,17)
W=3*(2*R+70)+20; H=R+170; im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
for k,(name,coord,wi,ci,note) in enumerate(panels):
    x=20+k*(2*R+70)
    d.text((x,12),name,font=F1,fill="black"); d.text((x,46),coord,font=F2,fill=(40,80,160))
    im.paste(wi,(x,84)); im.paste(ci,(x+R+40,84))
    d.text((x+R+6,84+R//2-14),"→",font=F1,fill=(40,80,160))
    d.text((x,84+R+10),"a cut of the object",font=F3,fill=(90,90,90)); d.text((x+R+40,84+R+10),"the same, canonical",font=F3,fill=(90,90,90))
    d.text((x,84+R+40),note,font=F3,fill="black")
im.save(OUT); print(OUT,im.size)
