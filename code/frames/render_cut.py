# The finished asset, cut: an orthographic ray-cast of the occupancy with one plane removing a half,
# the exterior shaded from the occupancy gradient, the cut face flat-lit -- every pixel's colour
# read from the volume at the point the ray stops.  Figure-1 style sequence:
#   exterior turning -> longitudinal plane turning about the axis -> transverse plane sweeping
#   -> an oblique plane tilting from transverse to longitudinal.
#   python render_cut.py <shell grid.pt (any N)> <interior state.pt or -> <out.gif> [frames.png]
# The shell grid's own V is used outside CORE; inside, the interior state's X (resampled), so a
# higher-resolution exterior can carry a 128^3 interior.
import sys,math,numpy as np,torch,torch.nn.functional as F
from PIL import Image
dv="cuda" if torch.cuda.is_available() else "cpu"
G=torch.load(sys.argv[1],map_location=dv); N=G["N"]
V=G["V"].float(); OCC=G["OCC"].float(); CORE=G["CORE"].float()
if sys.argv[2]!="-":
    X=torch.load(sys.argv[2],map_location=dv)["X"].float()
    Xi=F.interpolate(X[None],size=(N,N,N),mode="trilinear",align_corners=True)[0]
    V=torch.where(CORE>0.5,Xi,V)
import os
OUT=sys.argv[3]; RES=int(os.environ.get("RES","384")); C=(N-1)/2
def samp(T,P):  # P (...,3) in voxel units about the centre, order (u0, v, u1) = grid dims (0,1,2)
    q=(P/C).flip(-1).reshape(1,1,1,-1,3)
    return F.grid_sample(T[None],q,mode="bilinear",padding_mode="zeros",align_corners=True)[0,:,0,0].T
def cam(az,el):
    a,e=math.radians(az),math.radians(el)
    f=torch.tensor([math.cos(e)*math.cos(a),math.sin(e),math.cos(e)*math.sin(a)],device=dv)  # toward the camera
    up=torch.tensor([0.,1.,0.],device=dv); r=torch.linalg.cross(up,f); r=r/r.norm(); u=torch.linalg.cross(f,r)
    return f,r,u
LIGHT=torch.tensor([0.4,0.8,0.45],device=dv); LIGHT=LIGHT/LIGHT.norm()
def render(az,el,plane=None):
    f,r,u=cam(az,el); g=torch.linspace(-1.05,1.05,RES,device=dv)*C
    yy,xx=torch.meshgrid(-g,g,indexing="ij")
    o=xx[...,None]*r+yy[...,None]*u+f*1.8*C; d=-f
    o=o.reshape(-1,3); img=torch.ones(len(o),3,device=dv); hit=torch.zeros(len(o),dtype=torch.bool,device=dv)
    T=torch.linspace(0,3.6*C,int(3.6*C*2),device=dv)
    for k in range(0,len(o),65536):
        oo=o[k:k+65536]; P=oo[:,None,:]+T[None,:,None]*d                                    # (n,S,3)
        occ=samp(OCC,P.reshape(-1,3)).reshape(len(oo),-1)
        keep=torch.ones_like(occ,dtype=torch.bool)
        if plane is not None:
            n,dd=plane; side=(P@n+dd)
            keep=side<=0
        inside=(occ>0.5)&keep
        first=inside.float().argmax(1); any_=inside.any(1)
        Pf=P[torch.arange(len(oo)),first]
        col=(samp(V,Pf)+1)/2
        # cut face or exterior?
        if plane is not None:
            oncut=(Pf@n+dd).abs()<1.0
        else: oncut=torch.zeros(len(oo),dtype=torch.bool,device=dv)
        e=1.0; gx=torch.stack([samp(OCC,Pf+e*torch.eye(3,device=dv)[i])[:,0]-samp(OCC,Pf-e*torch.eye(3,device=dv)[i])[:,0] for i in range(3)],-1)
        nrm=-gx/(gx.norm(dim=1,keepdim=True)+1e-6)
        if plane is not None: nrm=torch.where(oncut[:,None],n.expand_as(nrm),nrm)
        sh=0.35+0.65*(nrm@LIGHT).clamp(0,1)
        sh=torch.where(oncut,0.75+0.25*(n@LIGHT).abs() if plane is not None else sh,sh)
        c=(col*sh[:,None]).clamp(0,1)
        img[k:k+65536]=torch.where(any_[:,None],c,img[k:k+65536])
    return (img.reshape(RES,RES,3).cpu().numpy()*255).astype(np.uint8)
frames=[]; stills=[]
for k in range(12): frames.append(render(30+k*30/12*3,25))                                   # exterior, a quarter turn
for k in range(24):                                                                           # longitudinal plane turning
    th=math.pi*k/24; n=torch.tensor([-math.sin(th),0.,math.cos(th)],device=dv)
    frames.append(render(60,25,(n,torch.tensor(0.,device=dv))))
    if k==6: stills.append(frames[-1])
hs=[C*0.6*(1-2*k/23) for k in range(24)]
for h in hs:                                                                                  # transverse plane sweeping down
    frames.append(render(60,35,(torch.tensor([0.,1.,0.],device=dv),torch.tensor(-h,device=dv))))
stills.append(frames[-12])
for k in range(16):                                                                           # oblique: transverse -> longitudinal
    t=math.pi/2*k/15; n=torch.tensor([0.,math.cos(t),math.sin(t)],device=dv)
    frames.append(render(60,30,(n,torch.tensor(0.,device=dv))))
    if k==8: stills.append(frames[-1])
ims=[Image.fromarray(a) for a in frames]
ims[0].save(OUT,save_all=True,append_images=ims[1:],duration=90,loop=0)
if len(sys.argv)>4:
    s=Image.new("RGB",(RES*4,RES),"white"); s.paste(ims[0],(0,0)); [s.paste(Image.fromarray(a),((i+1)*RES,0)) for i,a in enumerate(stills)]; s.save(sys.argv[4])
print("wrote",OUT,len(frames),"frames")
