# Two parts of Figure 3 (v0.2) computed from the orange's own arrays rather than drawn:
#   state.png   panel (4): the lifted cylindrical state seen in 3-D -- one longitudinal plane through the axis and
#               one transverse disc, inside the cylinder, coloured by the volume itself
#   ilvr_*.png  panel (5): what "keep the carrier's layout" does to one step on one plane -- the carrier's
#               low-pass, the priors' estimate's high-pass, and their sum, the state the chain carries on with
#   python fig3_parts.py <grid.pt> <state.pt> <carrier slice.png> <estimate slice.png> <out dir>
import sys,os,math,numpy as np,torch,torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFilter
gp,sp,cp,ep,out=sys.argv[1:6]; os.makedirs(out,exist_ok=True)
G=torch.load(gp,map_location="cpu"); X=torch.load(sp,map_location="cpu")["X"].float(); N=X.shape[-1]; C=(N-1)/2
V=torch.where(G["CORE"].float()>0.5,X,G["V"].float()); OCC=G["OCC"].float()
idx=G["OCC"][0].nonzero().float(); Rr=float(torch.sqrt((idx[:,0]-C)**2+(idx[:,2]-C)**2).max()); H0,H1=float(idx[:,1].min()-C),float(idx[:,1].max()-C)
def samp(T,P): return F.grid_sample(T[None],(torch.tensor(P,dtype=torch.float32)/C).flip(-1).reshape(1,1,1,-1,3),mode="bilinear",padding_mode="zeros",align_corners=True)[0,:,0,0].T.numpy()
# ---- panel 4: orthographic view, up = -d1, the longitudinal plane d2 = 0 and the transverse disc d1 = h
RES=520; az,el=math.radians(-35),math.radians(22)
f=np.array([math.cos(el)*math.cos(az),-math.sin(el),math.cos(el)*math.sin(az)])       # toward the camera
up=np.array([0.,-1.,0.]); r=np.cross(up,f); r/=np.linalg.norm(r); u=np.cross(f,r)
E=max(Rr,(H1-H0)/2)*1.25; g=np.linspace(-E,E,RES); yy,xx=np.meshgrid(-g,g,indexing="ij")
O=(xx[...,None]*r+yy[...,None]*u+f*3*C).reshape(-1,3); D=-f
img=np.tile(np.array([248,249,250])/255.,(RES*RES,1)); best=np.full(RES*RES,np.inf); hT=0.0
for n,d0,tint in ((np.array([0.,0.,1.]),0.0,"L"),(np.array([0.,1.,0.]),hT,"T")):
    den=D@n; t=-(O@n-d0)/den; P=O+t[:,None]*D
    ins=samp(OCC,P)[:,0]>0.5; ok=ins&(t<best)
    col=((samp(V,P[ok])+1)/2).clip(0,1); img[ok]=col*(0.93 if tint=="L" else 1.0); best[ok]=t[ok]
im=Image.fromarray((img.reshape(RES,RES,3)*255).astype(np.uint8)); d=ImageDraw.Draw(im)
def pr(P): P=np.asarray(P,float); return ((P@r+E)/(2*E)*(RES-1),(E-P@u)/(2*E)*(RES-1))
def ring(y,col,w,dash=False,front_only=False):
    a=np.linspace(0,2*math.pi,241); pts=[pr([Rr*math.cos(t),y,Rr*math.sin(t)]) for t in a]
    for i in range(240):
        mid=np.array([Rr*math.cos((a[i]+a[i+1])/2),y,Rr*math.sin((a[i]+a[i+1])/2)])
        back=(mid@f)<0
        if front_only and back: continue
        if dash and back and i%4<2: continue
        d.line([pts[i],pts[i+1]],fill=col,width=w)
GREY=(150,155,162); BL=(31,95,168); OR=(200,98,26)
ring(H0,GREY,2,dash=True); ring(H1,GREY,2)
# silhouette lines of the cylinder: where the surface normal is perpendicular to the view
ph=math.atan2(r[2],r[0])
for s in (0,math.pi): p=[Rr*math.cos(ph+s),0,Rr*math.sin(ph+s)]; d.line([pr([p[0],H0,p[2]]),pr([p[0],H1,p[2]])],fill=GREY,width=2)
ring(hT,OR,3)                                                                                   # the transverse disc's rim
for a_,b_ in (([-Rr,H0,0],[Rr,H0,0]),([Rr,H0,0],[Rr,H1,0]),([Rr,H1,0],[-Rr,H1,0]),([-Rr,H1,0],[-Rr,H0,0])): d.line([pr(a_),pr(b_)],fill=BL,width=3)
d.line([pr([0,H0-8,0]),pr([0,H1+8,0])],fill=(90,90,90),width=1)                               # the axis
im.save(f"{out}/state.png")
# ---- panel 5: one ILVR step on one plane
def load(p): return np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.LANCZOS)).astype(np.float32)/255
car,est=load(cp),load(ep); mask=(car.min(2)<0.93)
def blur(a,s=16):
    m=mask.astype(np.float32); k=lambda x: np.asarray(Image.fromarray((x*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s))).astype(np.float32)/255
    num=np.stack([k(a[...,c]*m) for c in range(3)],-1); den=k(m)[...,None]; return num/np.maximum(den,1e-3)
lo=blur(car); hi=est-blur(est); res=(lo+hi).clip(0,1)
hig=np.repeat((0.5+2.2*hi.mean(2,keepdims=True)).clip(0,1),3,2)                          # detail shown as luminance about grey
for name,a in (("ilvr_low",lo),("ilvr_high",hig),("ilvr_sum",res)):
    a=np.where(mask[...,None],a,1.0); Image.fromarray((a*255).astype(np.uint8)).save(f"{out}/{name}.png")
print("wrote",out)
