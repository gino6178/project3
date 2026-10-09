# Procedural objects with a known interior, for the question the seven foods cannot answer: is the
# cylinder a property of the method or of the fruit?  Each object is a family of specimens -- one
# seed, one interior -- sharing one outer shape, so a "photograph" is an exact analytic slice and a
# held-out photograph can be taken on any plane, oblique ones included.
#
#   log         axial      growth rings, rays, pith, bark            -- stationary in (r, phi, z)
#   honeycomb   columnar   hexagonal wax walls, honey, a cap         -- stationary in (x, y, z)
#   strata      layered    wavy horizontal beds of mixed rock        -- stationary in z, either frame
#   terrazzo    isotropic  coloured chips in a grey matrix           -- stationary in (x, y, z)
#
# Coordinates are the voxel units of the 128^3 grid planes.Vol reads, centred: (u0, v, u1) with v
# the polar axis (grid dim 1).  Photographs are rendered on Vol's own plane coordinates, so a
# photograph and a grid slice of the same plane are the same frame pixel for pixel.
#
#   python synth.py <object> <outdir>
#     outdir/grid.pt                     specimen 0, in the Vol format (V, OCC, SHELL, CORE, N)
#     outdir/spl_long, spl_trans         training photographs: specimens 1-3 and 4-6
#     outdir/hld_long, hld_trans, hld_obl held-out photographs: specimens 7-12, never trained on
import os,sys,math,json,numpy as np,torch,torch.nn.functional as F
from PIL import Image
dv="cuda" if torch.cuda.is_available() else "cpu"
N=128; C=(N-1)/2; RES=512

def fbm(seed,P,scale,oct=4):
    """value-noise fBm in [-1,1] at points P (M,3), feature size ~scale voxels."""
    g=torch.Generator(device=dv).manual_seed(seed); out=0; amp=1; tot=0
    for o in range(oct):
        n=min(int(math.ceil(180/(scale/2**o)))+2,384); L=torch.rand((1,1,n,n,n),generator=g,device=dv)*2-1
        q=(P/90.0).clamp(-1,1)                                       # the lattice spans +-90 voxels, one value per scale/2^o
        out=out+amp*F.grid_sample(L,q[None,None,None][...,[2,1,0]],mode="bilinear",padding_mode="border",align_corners=True).view(-1)
        tot+=amp; amp*=0.5
    return out/tot
def mix(a,b,t): t=t.clamp(0,1)[:,None]; return a*(1-t)+b*t
def col(*rgb): return torch.tensor(rgb,device=dv,dtype=torch.float32)/255
def sstep(e0,e1,x): t=((x-e0)/(e1-e0)).clamp(0,1); return t*t*(3-2*t)

class Obj:
    shape="box"; R=46.0; H=48.0; rind=3.0
    def __init__(s,seed): s.seed=seed; s.rng=np.random.default_rng(seed)
    def sdf(s,P):                                                   # negative inside, voxel units
        x,v,z=P[:,0],P[:,1],P[:,2]
        if s.shape=="cyl": return torch.maximum(torch.sqrt(x*x+z*z)-s.R, v.abs()-s.H)
        if s.shape=="sphere": return P.norm(dim=1)-s.R
        h=s.R/math.sqrt(2); return torch.stack([x.abs()-h,z.abs()-h,v.abs()-s.H],1).max(1).values

class Log(Obj):
    shape="cyl"; R=47.0; H=50.0
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]; r=torch.sqrt(x*x+z*z); ph=torch.atan2(z,x)
        w=fbm(s.seed,P*torch.tensor([1,0.15,1],device=dv),14.0)    # rings wander in plane, barely along the axis
        sp=float(s.rng.uniform(3.6,4.6)); rr=r+2.2*w; ring=(rr/sp)%1.0
        late=sstep(0.62,0.80,ring)*(1-sstep(0.93,1.0,ring))          # a sharp latewood band per ring
        early=mix(col(222,186,138),col(205,165,112),(fbm(s.seed+1,P,5.0)*0.5+0.5))
        c=mix(early,col(150,98,58),late*0.9)
        nr=int(s.rng.integers(40,60)); a0=torch.tensor(s.rng.uniform(0,2*math.pi,nr),device=dv,dtype=torch.float32)
        d=torch.remainder(ph[:,None]-a0[None]+math.pi,2*math.pi)-math.pi
        ray=(torch.exp(-(d.abs()*r[:,None]/0.6)**2)).max(1).values*(r>4).float()
        c=mix(c,col(236,206,160),ray*0.6)
        c=mix(c,col(95,60,35),1-sstep(1.5,3.0,r))                     # pith
        g=fbm(s.seed+2,P*torch.tensor([1,0.3,1],device=dv),3.0)*0.5+0.5
        bark=mix(col(92,78,66),col(58,48,40),g); return c,bark

class Honeycomb(Obj):
    shape="box"; R=50.0; H=46.0
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]
        a=float(s.rng.uniform(0,math.pi/3)); ox,oz=s.rng.uniform(0,20,2); cell=float(s.rng.uniform(8.0,9.5))
        X= math.cos(a)*x+math.sin(a)*z+ox; Z=-math.sin(a)*x+math.cos(a)*z+oz
        # distance to the nearest hexagon edge: hex lattice of circumradius cell/sqrt(3)
        q=torch.stack([X,Z],1)/cell; b1=torch.tensor([1.0,0.0],device=dv); b2=torch.tensor([0.5,math.sqrt(3)/2],device=dv)
        M=torch.stack([b1,b2],1); f=q@torch.linalg.inv(M.cpu()).to(dv).T; i=f.floor()
        best=torch.full((len(P),),1e9,device=dv); sec=best.clone()
        for di in (-1,0,1,2):
            for dj in (-1,0,1,2):
                ctr=((i+torch.tensor([di,dj],device=dv))@M.T); d=(q-ctr).norm(dim=1)
                sec=torch.minimum(sec,torch.maximum(best,d)); best=torch.minimum(best,d)
        edge=(sec-best)*cell                                          # ~0 on a wall, voxel units
        wall=1-sstep(0.5,1.3,edge)
        wob=fbm(s.seed,P,6.0)
        honey=mix(col(214,140,24),col(170,92,10),(fbm(s.seed+1,P*torch.tensor([1,0.2,1],device=dv),4.0)*0.5+0.5+0.3*wob))
        c=mix(honey,col(246,222,150),wall)
        cap=sstep(s.H-6,s.H-4,v.abs())                                 # wax caps at both ends
        c=mix(c,col(240,214,140),cap)
        return c,mix(col(200,170,110),col(170,140,90),fbm(s.seed+2,P,3.0)*0.5+0.5)

class Strata(Obj):
    shape="box"; R=50.0; H=46.0
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]
        pal=[col(196,160,118),col(120,104,92),col(222,206,176),col(166,94,62),col(90,86,84),col(204,180,140)]
        th=s.rng.uniform(3.0,9.0,40); edges=np.concatenate([[0],np.cumsum(th)])-s.rng.uniform(60,80)
        k=s.rng.integers(0,len(pal),40)
        vv=v+4.0*fbm(s.seed,P*torch.tensor([1,0.0,1],device=dv)+torch.tensor([0,1,0],device=dv),25.0)+1.0*fbm(s.seed+3,P,4.0)
        idx=torch.bucketize(vv,torch.tensor(edges,device=dv,dtype=torch.float32)).clamp(1,40)-1
        base=torch.stack(pal)[torch.tensor(k,device=dv)[idx]]
        grain=fbm(s.seed+1,P,2.0)*0.08
        return (base+grain[:,None]).clamp(0,1), col(150,140,128).expand(len(P),3)

class Terrazzo(Obj):
    shape="box"; R=50.0; H=46.0
    def color(s,P):
        n=3500; ctr=torch.tensor(s.rng.uniform(-60,60,(n,3)),device=dv,dtype=torch.float32)
        rad=torch.tensor(s.rng.uniform(1.5,4.5,n),device=dv,dtype=torch.float32)
        pal=torch.stack([col(200,60,50),col(240,240,232),col(40,40,44),col(210,170,60),col(80,130,90),col(170,170,176)])
        pk=pal[torch.tensor(s.rng.integers(0,len(pal),n),device=dv)]
        c=mix(col(150,150,146),col(126,126,124),fbm(s.seed,P,3.0)*0.5+0.5)
        for k in range(0,len(P),400000):
            Q=P[k:k+400000]; d=(torch.cdist(Q,ctr)/rad[None])+0.25*fbm(s.seed+1,Q,1.5)[:,None]
            m,j=d.min(1); t=(1-sstep(0.9,1.0,m))
            c[k:k+400000]=mix(c[k:k+400000],pk[j],t)
        return c, col(170,170,166).expand(len(P),3)

OBJS={"log":Log,"honeycomb":Honeycomb,"strata":Strata,"terrazzo":Terrazzo}

def render(o,P):
    """colour (white outside) at points P, rind from the signed distance."""
    sd=o.sdf(P); inner,rind=o.color(P)
    c=mix(inner,rind,sstep(-o.rind-0.6,-o.rind+0.6,sd))
    return torch.where((sd<0)[:,None],c,torch.ones_like(c))

def plane_points(ext,kind,param,res=RES):
    gl=torch.linspace(-ext,ext,res,device=dv); GV,GU=torch.meshgrid(gl,gl,indexing="ij")
    if kind=="long":                                                  # planes.Vol.long_slice
        c,s=math.cos(param),math.sin(param); P=torch.stack([GU*c,GV,GU*s],-1)
    elif kind=="trans":                                               # planes.Vol.trans_slice
        P=torch.stack([GU,torch.full_like(GU,param),GV],-1)
    else:                                                             # oblique: tilted param rad from transverse about u0
        c,s=math.cos(param),math.sin(param); P=torch.stack([GU,GV*s,GV*c],-1)
    return P.reshape(-1,3)

def save(t,p,res=RES):
    Image.fromarray((t.reshape(res,res,3).clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(p)

if __name__=="__main__":
    name,out=sys.argv[1],sys.argv[2]; K=OBJS[name]; os.makedirs(out,exist_ok=True)
    o0=K(0)
    lin=torch.arange(N,device=dv,dtype=torch.float32)-C
    A,B,Cc=torch.meshgrid(lin,lin,lin,indexing="ij"); P=torch.stack([A,B,Cc],-1).reshape(-1,3)
    sd=o0.sdf(P); occ=(sd<0).float()
    V=render(o0,P).T.reshape(3,N,N,N)*2-1
    shell=((sd<0)&(sd>-2.0)).float(); core=((sd<=-2.0)).float()
    G=dict(V=V.cpu(),OCC=occ.view(1,N,N,N).cpu(),SHELL=shell.view(1,N,N,N).cpu(),CORE=core.view(1,N,N,N).cpu(),N=N)
    torch.save(G,f"{out}/grid.pt")
    # the frame planes.Vol will build from this occupancy, so photographs share it exactly
    rad=torch.sqrt(A**2+Cc**2).reshape(-1); m=occ>0.5
    ext=max(float(rad[m].max()),float(B.reshape(-1)[m].abs().max()))/0.82
    hs=[float(B.reshape(-1)[m].min()),float(B.reshape(-1)[m].max())]
    rng=np.random.default_rng(1000)
    def heights(n): return [hs[0]+(hs[1]-hs[0])*q for q in rng.uniform(0.3,0.7,n)]
    plan=[("spl_long","long",[1,2,3]),("spl_trans","trans",[4,5,6]),
          ("hld_long","long",[7,8,9]),("hld_trans","trans",[10,11,12]),("hld_obl","obl",[13,14,15])]
    for d,kind,seeds in plan:
        os.makedirs(f"{out}/{d}",exist_ok=True)
        for k,sd_ in enumerate(seeds):
            o=K(sd_)
            prm=(float(rng.uniform(0,math.pi)) if kind=="long" else heights(1)[0] if kind=="trans" else math.pi/4)
            save(render(o,plane_points(ext,kind,prm)),f"{out}/{d}/{k:02d}.png")
    json.dump(dict(object=name,ext=ext,heights=hs,shape=K.shape),open(f"{out}/meta.json","w"))
    print(name,"ext",round(ext,2),"occ",int(occ.sum()),"core",int(core.sum()))
