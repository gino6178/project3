# The finished asset with a smooth outside: the generated mesh (TRELLIS.2, its own texture) is the exterior, the
# lifted lattice is the interior, and a cut shows the lattice wherever the plane passes through the solid.
# Ray cast, orthographic: a ray stops at the first of (a) the mesh's surface on the kept side of the plane,
# (b) the plane itself where the plane lies inside the solid.  (a) is coloured by the mesh's texture and lit by its
# smooth normal; (b) is coloured by the lifted volume, lit flat.  Nothing about the method changes: the interior
# is the lift's, the solid is the mesh's.  Figure-1 sequence: exterior turning, a lengthwise plane turning about
# the axis, a transverse plane sweeping, an oblique plane tilting.
#   python render_asset.py <object grid.pt> <mesh.glb> <interior state.pt> <out.gif> [stills.png]
import os,sys,math,numpy as np,torch,torch.nn.functional as F,trimesh
from PIL import Image
dv="cuda" if torch.cuda.is_available() else "cpu"
G=torch.load(sys.argv[1],map_location=dv); N=G["N"]; C=(N-1)/2
X=torch.load(sys.argv[3],map_location=dv)["X"].float()
# UPAXIS=2: an object whose axis lies along the grid's third dimension (the cake, the doughnut) is turned upright,
# by a rotation (d0, d1, d2) -> (d0, -d2, d1) of the grid, the interior and the mesh alike
UP=os.environ.get("UPAXIS","1")=="2"
rot=lambda T: T.permute(0,1,3,2).flip(2)
if UP:
    for k in ("V","OCC","CORE"): G[k]=rot(G[k])
    X=rot(X)
CORE=G["CORE"].float(); V=torch.where(CORE>0.5,F.interpolate(X[None],size=(N,N,N),mode="trilinear",align_corners=True)[0] if X.shape[-1]!=N else X,G["V"].float())
RES=int(os.environ.get("RES","512")); OUT=sys.argv[4]
# the mesh, placed in the grid's voxel frame exactly as glb2grid.py placed it (centre, mirror in y, scale)
if sys.argv[2].endswith(".npz"):     # a mesh already in the grid's voxel frame, vertex-coloured (grid_mesh.py: a photographed object)
    d=np.load(sys.argv[2]); v=d["v"].astype(np.float64); vcol=d["c"]; mesh=trimesh.Trimesh(vertices=v,faces=d["f"],process=False)
else:
    m=trimesh.util.concatenate([g for g in trimesh.load(sys.argv[2],force="scene").dump()])
    v=m.vertices.copy(); lo,hi=v.min(0),v.max(0); ctr=(lo+hi)/2; v-=ctr; v[:,1]*=-1
    rad=np.sqrt(v[:,0]**2+v[:,2]**2).max(); hh=(hi[1]-lo[1])/2; s=50.0*N/128/max(rad,hh); v*=s
    tex=np.asarray(m.visual.material.baseColorTexture.convert("RGB")).astype(np.float32)/255; TH,TW,_=tex.shape
    uv=m.visual.uv; vcol=tex[np.clip(((1-uv[:,1]%1)*(TH-1)).astype(int),0,TH-1),np.clip((uv[:,0]%1*(TW-1)).astype(int),0,TW-1)]
    mesh=trimesh.Trimesh(vertices=v,faces=m.faces[:,::-1],process=False)    # faces flipped back after the mirror
if UP: v=np.stack([v[:,0],-v[:,2],v[:,1]],1); mesh=trimesh.Trimesh(vertices=v,faces=mesh.faces,process=False)
vnorm=mesh.vertex_normals
import nvdiffrast.torch as dr
CTX=dr.RasterizeCudaContext()
VT=torch.tensor(v,device=dv,dtype=torch.float32); FT=torch.tensor(np.ascontiguousarray(mesh.faces),device=dv,dtype=torch.int32)
VC=torch.tensor(np.ascontiguousarray(vcol),device=dv,dtype=torch.float32).contiguous(); VN=torch.tensor(np.ascontiguousarray(vnorm),device=dv,dtype=torch.float32).contiguous()
FCEN=VT[FT.long()].mean(1)
def samp(T,P):
    q=(P/C).flip(-1).reshape(1,1,1,-1,3)
    return F.grid_sample(T[None],q,mode="bilinear",padding_mode="zeros",align_corners=True)[0,:,0,0].T
LIGHT=np.array([0.4,0.8,0.45]); LIGHT/=np.linalg.norm(LIGHT)
def cam(az,el):
    a,e=math.radians(az),math.radians(el); f=np.array([math.cos(e)*math.cos(a),-math.sin(e),math.cos(e)*math.sin(a)])  # y is down after the mirror
    up=np.array([0.,-1.,0.]); r=np.cross(up,f); r/=np.linalg.norm(r); u=np.cross(f,r); return f,r,u
def inside(P):  # inside the solid: the lattice's occupancy (the mesh's own, quantised)
    return samp(G["OCC"].float(),torch.tensor(P,device=dv,dtype=torch.float32))[:,0].cpu().numpy()>0.5
def render(az,el,plane=None,zoom=1.0):
    f,r,u=cam(az,el); E=C*1.02/zoom; g=np.linspace(-E,E,RES)
    yy,xx=np.meshgrid(-g,g,indexing="ij"); O=(xx[...,None]*r+yy[...,None]*u+f*3*C).reshape(-1,3); D=np.tile(-f,(len(O),1))
    img=np.ones((len(O),3),np.float32); tbest=np.full(len(O),np.inf)
    # (a) the mesh, rasterised (nvdiffrast) with the faces on the removed side of the plane dropped
    keepf=torch.ones(len(FT),dtype=torch.bool,device=dv) if plane is None else (FCEN@torch.tensor(plane[0],device=dv,dtype=torch.float32)+plane[1])<=0
    R3=torch.tensor(np.stack([r,u,f]),device=dv,dtype=torch.float32); pc=VT@R3.T                      # camera coords: x right, y up, z toward camera
    clip=torch.stack([pc[:,0]/E,pc[:,1]/E,-pc[:,2]/(4*C),torch.ones_like(pc[:,0])],1)[None]
    if int(keepf.sum())==0: keepf[0]=True          # a plane past the whole object keeps no face: draw one, harmless
    rast,_=dr.rasterize(CTX,clip.contiguous(),FT[keepf].contiguous(),(RES,RES))
    col,_=dr.interpolate(VC[None],rast,FT[keepf]); nrm,_=dr.interpolate(VN[None],rast,FT[keepf])
    hit=(rast[0,...,3]>0); col=col[0].flip(0); nrm=nrm[0].flip(0); hit=hit.flip(0); zc=rast[0,...,2].flip(0)
    nrm=nrm/nrm.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    sh=0.7+0.3*(nrm@torch.tensor(LIGHT,device=dv,dtype=torch.float32)).abs()
    im=torch.where(hit[...,None],(col*sh[...,None]).clamp(0,1),torch.ones_like(col))
    img=im.reshape(-1,3).cpu().numpy(); hitn=hit.reshape(-1).cpu().numpy()
    tmesh=(3*C+zc.reshape(-1).cpu().numpy()*4*C)        # ray parameter of the mesh hit (camera at 3C along f)
    tbest=np.where(hitn,tmesh,np.inf)
    # (b) the cut face: the plane where it lies inside the solid, nearer than any kept mesh hit
    if plane is not None:
        n,dd=plane; den=D@n; ok=np.abs(den)>1e-8; tp=np.where(ok,-(O@n+dd)/np.where(ok,den,1),np.inf)
        Pp=O+tp[:,None]*D; cand=np.where(ok&(tp>0)&(tp<tbest))[0]
        if len(cand):
            ins=inside(Pp[cand]); cand=cand[ins]
            vol=((samp(V,torch.tensor(Pp[cand],device=dv,dtype=torch.float32))+1)/2).clamp(0,1).cpu().numpy()
            img[cand]=vol*(0.82+0.18*abs(float(n@LIGHT)))
    return (img.reshape(RES,RES,3)*255).astype(np.uint8)
# the lengthwise still faces the viewer: its plane's normal 25 degrees off the camera's azimuth, the removed half toward it
FACE=(np.array([math.cos(math.radians(35)),0.,math.sin(math.radians(35))]),0.0)
frames=[]; stills=[]
if os.environ.get("STILLS_ONLY")!="1":
    for k in range(12): frames.append(render(30+k*7.5,22))
    for k in range(24):
        th=math.pi*k/24; frames.append(render(60,22,(np.array([-math.sin(th),0.,math.cos(th)]),0.0)))
    for k in range(24):
        h=C*0.6*(1-2*k/23); frames.append(render(60,32,(np.array([0.,-1.,0.]),h)))
    for k in range(16):
        t=math.pi/2*k/15; frames.append(render(60,28,(np.array([0.,-math.cos(t),math.sin(t)]),0.0)))
    ims=[Image.fromarray(a) for a in frames]; ims[0].save(OUT,save_all=True,append_images=ims[1:],duration=90,loop=0)
if len(sys.argv)>5:
    stills=[render(30,22),render(60,22,FACE),render(60,32,(np.array([0.,-1.,0.]),C*0.6*(1-24/23))),
            render(60,28,(np.array([0.,-math.cos(math.pi/2*8/15),math.sin(math.pi/2*8/15)]),0.0))]
    s=Image.new("RGB",(RES*4,RES),"white"); [s.paste(Image.fromarray(a),(i*RES,0)) for i,a in enumerate(stills)]; s.save(sys.argv[5])
print("wrote",OUT,len(frames),"frames")
