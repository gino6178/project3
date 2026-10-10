# A synthetic benchmark with full 3-D ground truth (the teacher's protocol): ONE coherent solid is
# generated first, the 3+3 input cuts are taken from it, and every other plane -- unseen heights,
# unseen azimuths, oblique -- is ground truth.  Colours are sampled from the generated photographs
# (aiobj/log, aiobj/tiramisu), so the cuts read as material rather than as a diagram.
#
#   wood1   concentric rings about a straight axis                          (sanity)
#   wood2   rings perturbed in phi and z, uneven widths                      (natural deformation)
#   wood3   wood2 about an off-centre, curved axis c(v)                      (limit of one global axis)
#   wood4   wood2 + three branch knots that deflect the rings                (axial assumption broken)
#   cake    a layer cake: wavy horizontal sponge/cream layers, porous sponge (planar: Cartesian)
#   cable   a seven-core cable, cores twisted helically, filler and jacket   (manufactured, axial)
#   honeycomb  hexagonal wax cells running vertically, no centre            (columnar: Cartesian)
#   hose    a multilayer hose: liner, braid, ply, cover around a bore         (rolled/laminated, axial)
#   ccable  the cable bent along a curved centreline                          (curved axial: curvilinear)
#   wood2x  wood2, every input cut from a DIFFERENT specimen (seed)           (cross-specimen, stage two)
#   wood2k  wood2 + three branch knots, straight axis                         (the teacher's level 4 alone)
#   wood3r  wood2 bent along a circular arc, cross-sections turning with it   (rotation bend: parallel transport)
#   wood5   wood2 bent strongly (radius 60, ~96 deg), thinner, and SLICED NORMAL to its axis (parallel transport)
#   laminate  a block of vertical veneer sheets (layer normal along u0)          (planar, not about the axis: Cartesian)
#   roll    a sheet rolled into a spiral: kraft paper with a printed face       (rolled material, axial)
#   banana  a banana bent along an arc, its cross-sections turning with it, and SLICED as a banana
#           is -- normal to its own axis                                        (curved axial: parallel transport)
#
# The wood carries its growth-time field G (rings are its integer iso-surfaces); it is saved so the
# structural metrics can read ring centre, spacing and orientation from the truth.
#
#   python synth2.py <object> <outdir>
#     grid.pt            the GT solid at 128^3 (V = truth; SHELL/CORE as the method expects)
#     spl_long/ spl_trans/   the six input cuts, 512 px, Vol plane frames
#     planes.json        where each input cut was taken (theta, offset / height)
#     polar_spl_trans/   the transverse inputs unwrapped
import os,sys,math,json,numpy as np,torch,torch.nn.functional as F
from PIL import Image
dv="cuda" if torch.cuda.is_available() else "cpu"
N=128; C=(N-1)/2; RES=512
def col(*rgb): return torch.tensor(rgb,device=dv,dtype=torch.float32)/255
def mix(a,b,t): t=t.clamp(0,1)[:,None]; return a*(1-t)+b*t
def sstep(e0,e1,x): t=((x-e0)/(e1-e0)).clamp(0,1); return t*t*(3-2*t)
def noise(seed,P,scale,aniso=(1,1,1),oct=4):
    """value-noise fBm in [-1,1]; aniso stretches the feature size per axis (u0, v, u1)."""
    g=torch.Generator(device="cpu").manual_seed(seed); out=0; amp=1; tot=0
    Q=P/torch.tensor(aniso,device=dv,dtype=torch.float32)
    for o in range(oct):
        s=scale/2**o; n=min(int(math.ceil(200/s))+2,320)
        L=(torch.rand((1,1,n,n,n),generator=g)*2-1).to(dv)
        q=(Q/100.0).clamp(-1,1)
        out=out+amp*F.grid_sample(L,q[None,None,None][...,[2,1,0]],mode="bilinear",padding_mode="border",align_corners=True).view(-1)
        tot+=amp; amp*=0.5
    return out/tot
EARLY,LATE,MIDW,BARK=col(221,188,147),col(163,124,89),col(196,162,126),col(140,103,76)

class Wood:
    R=46.0; H=48.0
    def __init__(s,level,seed=0,knots=False,rotbend=False,Rb=110.0,R=None,H=None):
        s.level=level; s.seed=seed; rng=np.random.default_rng(seed); s.rotbend=rotbend; s.Rb=Rb
        if R is not None: s.R=R
        if H is not None: s.H=H
        w=rng.uniform(0.65,1.35,40)*3.2*np.linspace(1.15,0.85,40)        # ring widths, narrower outward
        s.Rk=torch.tensor(np.concatenate([[0],np.cumsum(w)]),device=dv,dtype=torch.float32)
        s.knots=[]
        if level>=4 or knots:
            for k in range(3):
                s.knots.append(dict(v0=float(rng.uniform(-28,28)),az=float(rng.uniform(0,2*math.pi)),up=float(rng.uniform(0.15,0.45))))
    def arc(s,P):
        """rotation bend: the axis is a circular arc of radius Rb in the (u0, v) plane, bulging to -u0;
        returns the arc-length s, and the local offsets (a along the bend normal, b = u1)"""
        x,v,z=P[:,0],P[:,1],P[:,2]; qx=x+s.Rb-6.0; th=torch.atan2(v,qx); rho=torch.sqrt(qx*qx+v*v)
        return s.Rb*th, rho-s.Rb, z
    def frame(s,sv):   # the arc's point, in-plane normal and tangent at arc length sv (rotation bend)
        th=sv/s.Rb; c=torch.stack([s.Rb*torch.cos(th)-s.Rb+6.0,s.Rb*torch.sin(th),torch.zeros_like(th)],-1)
        return c,torch.stack([torch.cos(th),torch.sin(th),torch.zeros_like(th)],-1),torch.stack([-torch.sin(th),torch.cos(th),torch.zeros_like(th)],-1)
    def axis(s,v):
        if s.rotbend:                                                           # the arc's u0 at height v (cx), and cz = 0
            th=torch.asin((v/s.Rb).clamp(-1,1)); return s.Rb*torch.cos(th)-s.Rb+6.0, torch.zeros_like(v)
        if s.level>=3: return 7.0+6.0*(v/s.H)**2, 5.0*(v/s.H)                  # off-centre and curved
        z=torch.zeros_like(v); return z,z
    def geom(s,P):
        if s.rotbend:                                                           # (r, phi) in the plane normal to the arc, v -> arc length
            sv,a,b=s.arc(P); r=torch.sqrt(a*a+b*b)+1e-6; return r,torch.atan2(b,a),sv,a,b
        x,v,z=P[:,0],P[:,1],P[:,2]; cx,cz=s.axis(v); dx,dz=x-cx,z-cz
        r=torch.sqrt(dx*dx+dz*dz)+1e-6; ph=torch.atan2(dz,dx); return r,ph,v,dx,dz
    def local(s,P):   # the solid's own (unbent) coordinates, so every texture bends with it
        if not s.rotbend: return P
        sv,a,b=s.arc(P); return torch.stack([a,sv,b],1)
    def outer(s,P):
        P=s.local(P) if s.rotbend else P
        r,ph,v,_,_=s.geom0(P) if s.rotbend else s.geom(P); Ro=s.R+1.2*noise(s.seed+9,P,6.0,(1,3,1))
        if s.level>=3: Ro=Ro-7.0                                                # keep the curved trunk inside the box
        return r,Ro
    def outer_local(s,P):
        r,ph,v,_,_=s.geom0(P); return r,s.R+1.2*noise(s.seed+9,P,6.0,(1,3,1))
    def geom0(s,P):  # straight-axis geometry in local coordinates
        x,v,z=P[:,0],P[:,1],P[:,2]; r=torch.sqrt(x*x+z*z)+1e-6; return r,torch.atan2(z,x),v,x,z
    def sdf(s,P):
        r,Ro=s.outer(P); vv=s.local(P)[:,1] if s.rotbend else P[:,1]; return torch.maximum(r-Ro,vv.abs()-s.H)
    def G(s,P):
        if s.rotbend: P=s.local(P)
        r,ph,v,dx,dz=s.geom0(P) if s.rotbend else s.geom(P); re=r
        if s.level>=2:
            re=r*(1+0.07*noise(s.seed+1,torch.stack([torch.cos(ph)*30,v*0.5,torch.sin(ph)*30],1),12.0))+1.3*noise(s.seed+2,P,9.0,(1,2.5,1))
        for kn in s.knots:                                                      # rings bow around each knot
            a=torch.tensor([math.cos(kn["az"]),kn["up"],math.sin(kn["az"])],device=dv); a=a/a.norm()
            cx,cz=s.axis(torch.tensor([kn["v0"]],device=dv)); o=torch.tensor([float(cx),kn["v0"],float(cz)],device=dv)
            q=P-o; t=(q@a).clamp(min=0); db=(q-t[:,None]*a).norm(dim=1)
            re=re-3.0*torch.exp(-(db/(2.5+0.12*t))**2)*(t>0).float()
        k=torch.bucketize(re,s.Rk).clamp(1,len(s.Rk)-1)-1
        return k.float()+(re-s.Rk[k])/(s.Rk[k+1]-s.Rk[k])
    def color(s,P):
        g=s.G(P)
        if s.rotbend: P=s.local(P)
        r,ph,v,dx,dz=s.geom0(P) if s.rotbend else s.geom(P); f=g%1.0
        late=sstep(0.62,0.86,f)*(1-sstep(0.95,1.0,f))
        c=mix(EARLY,LATE,late)
        c=mix(c,MIDW*0.92,0.35*(1-sstep(10,26,r)))                             # heartwood darker toward the pith
        pores=(noise(s.seed+3,P,0.9,(1,6,1),oct=2)>0.55).float()*(f<0.45).float()
        c=mix(c,LATE*0.8,0.5*pores)
        rays=sstep(0.72,0.9,noise(s.seed+4,torch.stack([ph*r*1.6,v*0.6,r*0.15],1),1.0,(1,1,1),oct=2))*(r>3).float()
        c=mix(c,EARLY*1.04,0.45*rays)
        grain=noise(s.seed+5,P,1.6,(1,8,1),oct=3)*0.05
        c=(c+grain[:,None]).clamp(0,1)
        c=mix(c,col(110,72,45),1-sstep(0.8,1.8,r))                              # pith
        for kn in s.knots:
            a=torch.tensor([math.cos(kn["az"]),kn["up"],math.sin(kn["az"])],device=dv); a=a/a.norm()
            cx,cz=s.axis(torch.tensor([kn["v0"]],device=dv)); o=torch.tensor([float(cx),kn["v0"],float(cz)],device=dv)
            q=P-o; t=(q@a).clamp(min=0); db=(q-t[:,None]*a).norm(dim=1); rk=1.0+0.11*t
            kc=mix(col(120,78,48),col(84,52,32),((db/0.9)%1.0>0.6).float())
            c=mix(c,kc,(1-sstep(rk-0.4,rk+0.4,db))*(t>0).float())
        rr,Ro=(s.outer(P) if not s.rotbend else s.outer_local(P)); bark=mix(BARK,BARK*0.6,noise(s.seed+6,P,1.4,(1,3,1))*0.5+0.5)
        c=mix(c,bark,sstep(Ro-3.6,Ro-2.8,rr))
        return c

class Cake:
    W=40.0; H=38.0
    def __init__(s,seed=0):
        s.seed=seed; rng=np.random.default_rng(seed); edges=[-s.H]; kinds=[]
        y=-s.H
        while y<s.H-4:
            for kind,th in (("sponge",rng.uniform(6,9)),("cream",rng.uniform(2.5,4))):
                y+=th; edges.append(min(y,s.H-4)); kinds.append(kind)
        edges.append(s.H); kinds.append("cocoa")
        s.edges=torch.tensor(edges,device=dv,dtype=torch.float32); s.kinds=kinds
    def sdf(s,P): return torch.stack([P[:,0].abs()-s.W,P[:,2].abs()-s.W,P[:,1].abs()-s.H]).max(0).values
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]
        vv=v+1.6*noise(s.seed,torch.stack([x,v*0,z],1),14.0)
        k=torch.bucketize(vv,s.edges).clamp(1,len(s.kinds))-1
        sponge=mix(col(148,89,50),col(118,66,34),noise(s.seed+1,P,2.0)*0.5+0.5)
        holes=(noise(s.seed+2,P,0.8,oct=2)>0.5).float(); sponge=mix(sponge,col(79,32,13),0.6*holes)
        cream=mix(col(237,226,203),col(222,208,182),noise(s.seed+3,P,3.0)*0.5+0.5)
        cocoa=mix(col(92,44,22),col(70,30,12),noise(s.seed+4,P,0.7)*0.5+0.5)
        isS=torch.tensor([kk=="sponge" for kk in s.kinds],device=dv)[k]; isC=torch.tensor([kk=="cream" for kk in s.kinds],device=dv)[k]
        c=torch.where(isS[:,None],sponge,torch.where(isC[:,None],cream,cocoa))
        return c

class Cable:
    """A four-core power cable as it is built: four straight stranded copper conductors (19 round wires,
    1+6+12, each lit like a wire), each in one coloured insulation, laid in a square in white fibrous
    filler, a black jacket.  Straight, so a lengthwise cut through a conductor shows one continuous copper
    band with the strands as lines along it and a strip of its insulation on each side."""
    R=44.0; H=48.0
    def __init__(s,seed=0):
        s.seed=seed; s.rc=9.0; s.ri=12.0; s.pos=17.5
        s.cols=[col(120,72,40),col(36,36,38),col(140,140,144),col(40,86,170)]          # brown, black, grey, blue
        a=[0.0]+[2*math.pi*k/6 for k in range(6)]+[2*math.pi*(k+0.5)/12 for k in range(12)]
        rr=[0.0]+[3.0]*6+[6.0]*12; s.strands=[(r*math.cos(t),r*math.sin(t)) for r,t in zip(rr,a)]; s.rs=1.55
    def sdf(s,P): return torch.maximum(torch.sqrt(P[:,0]**2+P[:,2]**2)-s.R,P[:,1].abs()-s.H)
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]; r=torch.sqrt(x*x+z*z)
        c=mix(col(232,230,224),col(204,202,196),noise(s.seed,P,0.9,(1,6,1))*0.5+0.5)      # filler, fibres along the axis
        for k in range(4):
            t=k*math.pi/2; cx,cz=s.pos*math.cos(t),s.pos*math.sin(t)          # 0, 90, 180, 270 deg: the bend plane holds two of them
            dx,dz=x-cx,z-cz; d=torch.sqrt(dx*dx+dz*dz)
            c=mix(c,s.cols[k],1-sstep(s.ri-0.3,s.ri+0.3,d))                             # insulation
            ds=torch.stack([torch.sqrt((dx-sx)**2+(dz-sz)**2) for sx,sz in s.strands],1).min(1).values
            wire=mix(col(214,140,86),col(150,84,46),(ds/s.rs).clamp(0,1)**2)              # a round wire, brighter at its crown
            cu=mix(wire,col(70,40,24),sstep(s.rs-0.25,s.rs+0.15,ds))                      # dark gaps between wires
            c=mix(c,cu,1-sstep(s.rc-0.3,s.rc+0.3,d))
        jacket=mix(col(30,30,32),col(48,48,50),noise(s.seed+1,P,1.5)*0.5+0.5)
        return mix(c,jacket,sstep(s.R-4.6,s.R-4.0,r))

class Honeycomb:
    W=40.0; H=40.0
    def __init__(s,seed=0):
        s.seed=seed; rng=np.random.default_rng(seed); s.a=float(rng.uniform(0,math.pi/3)); s.o=rng.uniform(0,20,2); s.cell=float(rng.uniform(7.5,8.5))
    def sdf(s,P): return torch.stack([P[:,0].abs()-s.W,P[:,2].abs()-s.W,P[:,1].abs()-s.H]).max(0).values
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]
        X=math.cos(s.a)*x+math.sin(s.a)*z+float(s.o[0]); Z=-math.sin(s.a)*x+math.cos(s.a)*z+float(s.o[1])
        X=X+0.8*noise(s.seed,P,10.0,(1,4,1)); Z=Z+0.8*noise(s.seed+1,P,10.0,(1,4,1))     # cells wander slightly
        q=torch.stack([X,Z],1)/s.cell; Mi=torch.tensor([[1.0,-1/math.sqrt(3)],[0.0,2/math.sqrt(3)]],device=dv)
        f=q@Mi.T; i=f.floor(); M=torch.tensor([[1.0,0.5],[0.0,math.sqrt(3)/2]],device=dv)
        best=torch.full((len(P),),1e9,device=dv); sec=best.clone()
        for di in (-1,0,1,2):
            for dj in (-1,0,1,2):
                c=(i+torch.tensor([di,dj],device=dv,dtype=torch.float32))@M.T; d=(q-c).norm(dim=1)
                sec=torch.minimum(sec,torch.maximum(best,d)); best=torch.minimum(best,d)
        wall=1-sstep(0.35,0.9,(sec-best)*s.cell)
        honey=mix(col(214,140,24),col(168,92,12),noise(s.seed+2,P,4.0,(1,5,1))*0.5+0.5)
        c=mix(honey,col(246,224,156),wall)
        return mix(c,col(236,210,140),sstep(s.H-5,s.H-3.5,v.abs()))                   # wax caps

class Hose:
    R=44.0; H=48.0
    def __init__(s,seed=0): s.seed=seed
    def sdf(s,P):
        r=torch.sqrt(P[:,0]**2+P[:,2]**2); return torch.maximum(torch.maximum(r-s.R,P[:,1].abs()-s.H),13.0-r)   # a bore
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]; r=torch.sqrt(x*x+z*z); ph=torch.atan2(z,x)
        c=mix(col(40,40,42),col(58,58,60),noise(s.seed,P,1.5)*0.5+0.5)                    # inner liner
        braid=0.5+0.5*torch.sign(torch.sin(ph*24+v*0.55))*torch.sign(torch.sin(ph*24-v*0.55))
        steel=mix(col(150,152,156),col(205,208,212),braid)
        c=mix(c,steel,sstep(18.5,19.5,r)*(1-sstep(23.5,24.5,r)))
        ply=mix(col(225,200,120),col(196,170,96),(torch.sin(r*2.4)>0).float())               # textile plies
        c=mix(c,ply,sstep(23.5,24.5,r)*(1-sstep(35.5,36.5,r)))
        c=mix(c,col(30,30,32),sstep(35.5,36.5,r)); c=mix(c,col(200,60,40),sstep(41.5,42.2,r)) # cover, red skin
        return c

class Roll:
    """A sheet rolled into a spiral about the axis -- kraft paper, one printed face -- the rolled-material
    case: radial layering that is not concentric but one continuous Archimedean spiral."""
    R=44.0; H=46.0
    def __init__(s,seed=0): s.seed=seed; s.pitch=3.2; s.core=9.0
    def sdf(s,P):
        r=torch.sqrt(P[:,0]**2+P[:,2]**2); return torch.maximum(torch.maximum(r-s.R,P[:,1].abs()-s.H),s.core-r)   # a cardboard core's bore
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]; r=torch.sqrt(x*x+z*z); ph=torch.atan2(z,x)
        u=((r-s.core)/s.pitch-ph/(2*math.pi))%1.0                               # position across one turn of the spiral
        paper=mix(col(196,160,112),col(176,138,92),noise(s.seed,P,1.2,(1,4,1))*0.5+0.5)
        ink=mix(col(46,92,150),col(30,64,112),noise(s.seed+1,P,3.0)*0.5+0.5)     # the printed face
        c=mix(paper,ink,sstep(0.70,0.74,u)*(1-sstep(0.84,0.88,u)))
        c=mix(c,col(120,92,60),(1-sstep(0.04,0.10,u))+sstep(0.94,0.98,u))       # the thin dark gap between turns
        tube=mix(col(150,120,86),col(132,104,74),noise(s.seed+2,P,2.0)*0.5+0.5)
        return mix(tube,c,sstep(s.core+2.8,s.core+3.2,r))                        # cardboard core tube

class Banana:
    """A banana bent along a circular arc of radius Rb in the (u0, v) plane; every cross-section is normal
    to the arc: a five-ridged peel (yellow, brown flecks, a pale inner pith layer), cream flesh with a fine
    radial grain, and the three-locule core -- a pale three-armed star with dark seed traces along its
    arms.  Its slices are cut normal to the arc, as a banana is sliced."""
    Rb=66.0; L=50.0; R=25.0
    def __init__(s,seed=0): s.seed=seed; s.Ox=6.0-s.Rb
    def arc(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]; qx=x-s.Ox; th=torch.atan2(v,qx); rho=torch.sqrt(qx*qx+v*v)
        return s.Rb*th, rho-s.Rb, z                                                # arc length, offset along the bend normal, u1
    def axis(s,v):
        th=torch.asin((v/s.Rb).clamp(-1,1)); return s.Ox+s.Rb*torch.cos(th), torch.zeros_like(v)
    def frame(s,sv):
        th=sv/s.Rb; c=torch.stack([s.Ox+s.Rb*torch.cos(th),s.Rb*torch.sin(th),torch.zeros_like(th)],-1)
        n=torch.stack([torch.cos(th),torch.sin(th),torch.zeros_like(th)],-1)        # in-plane normal (outward)
        t=torch.stack([-torch.sin(th),torch.cos(th),torch.zeros_like(th)],-1)        # tangent
        return c,n,t
    def radius(s,sv): u=(sv/s.L).clamp(-1,1); return s.R*(1-u**6)**0.5*(1-0.15*u*u)
    def local(s,P):
        sv,a,b=s.arc(P); r=torch.sqrt(a*a+b*b)+1e-6; return sv,a,b,r,torch.atan2(b,a)
    def sdf(s,P):
        sv,a,b,r,ph=s.local(P); Rr=s.radius(sv)*(1+0.03*torch.cos(5*ph))            # faint ridges
        return torch.maximum(r-Rr,sv.abs()-s.L)
    def color(s,P):
        """As a Cavendish banana cuts: ivory flesh with faint fibres along the axis; three broad, darker,
        translucent yellow locule lobes radiating from the centre; the shrivelled ovules as sparse tiny dark
        specks near the centre (dots, not a line); a thin peel -- yellow outside, a pale green-white layer
        inside -- on a faintly five-sided section."""
        sv,a,b,r,ph=s.local(P); Rr=s.radius(sv)*(1+0.03*torch.cos(5*ph)); q=r/Rr.clamp(min=1e-3)
        L3=torch.stack([a,sv,b],1)
        flesh=mix(col(246,238,212),col(236,226,194),noise(s.seed,L3,1.2,(1,5,1))*0.5+0.5)          # fibres along the axis
        wob=0.25*noise(s.seed+1,L3,4.0,(1,3,1))
        arm=torch.stack([(torch.remainder(ph+wob-k*2*math.pi/3+math.pi,2*math.pi)-math.pi).abs() for k in range(3)],1).min(1).values
        half=0.10+0.32*torch.sin(math.pi*(q/0.6).clamp(0,1))                                         # a petal: widest at mid-radius
        lobe=(1-sstep(half*0.6,half,arm))*(1-sstep(0.48,0.62,q))*sstep(0.02,0.07,q)
        lobe=lobe*(0.75+0.25*(noise(s.seed+5,L3,1.5)*0.5+0.5))
        c=mix(flesh,col(226,206,152),0.5*lobe)
        c=mix(c,col(240,228,190),0.35*(1-sstep(0.0,0.05,arm))*lobe)                                     # a paler midrib along each petal
        c=mix(c,col(232,214,164),0.35*(1-sstep(0.06,0.14,q)))                                          # the slightly darker centre
        ring=sstep(0.58,0.64,q)*(1-sstep(0.66,0.72,q)); c=mix(c,col(238,226,182),0.4*ring)            # faint outer locule boundary
        spk=(noise(s.seed+2,L3,0.7,(1,1.4,1),oct=2)>0.42).float()*(1-sstep(0.14,0.24,q))*sstep(0.02,0.04,q)
        c=mix(c,col(70,52,36),0.85*spk)                                                               # ovules: sparse dark specks
        c=mix(c,col(226,230,190),sstep(0.86,0.89,q))                                                  # inner peel, pale green-white
        peel=mix(col(222,198,64),col(200,182,54),noise(s.seed+3,L3,3.0,(1,4,1))*0.5+0.5)
        peel=mix(peel,col(150,150,52),0.35*sstep(0.97,1.0,q))                                         # a greener rim at the skin
        return mix(c,peel,sstep(0.915,0.94,q))

class Laminate:
    """Vertical veneer sheets stacked along u0, glued: wavy, uneven thicknesses, three wood tones, a dark glue
    line between sheets, grain along v.  Planar but NOT organised about the vertical axis."""
    W=40.0; H=38.0
    def __init__(s,seed=0):
        s.seed=seed; rng=np.random.default_rng(seed); th=rng.uniform(4.0,8.0,30)
        s.edges=torch.tensor(np.concatenate([[0],np.cumsum(th)])-rng.uniform(55,70),device=dv,dtype=torch.float32)
        s.k=torch.tensor(rng.integers(0,3,30),device=dv)
    def sdf(s,P): return torch.stack([P[:,0].abs()-s.W,P[:,2].abs()-s.W,P[:,1].abs()-s.H]).max(0).values
    def color(s,P):
        x,v,z=P[:,0],P[:,1],P[:,2]
        xx=x+1.2*noise(s.seed,torch.stack([x*0,v,z],1),18.0)
        i=torch.bucketize(xx,s.edges).clamp(1,len(s.k))-1
        pal=torch.stack([col(214,176,124),col(176,124,78),col(228,204,160)])
        base=pal[s.k[i]]; grain=noise(s.seed+1,P,1.2,(3,10,3),oct=3)*0.07
        c=(base+grain[:,None]).clamp(0,1)
        f=(xx-s.edges[i])/(s.edges[i+1]-s.edges[i]).clamp(min=1e-3)
        return mix(c,col(96,66,40),1-sstep(0.0,0.08,f))                                   # glue line at each sheet's start

class CCable(Cable):
    def bend(s,v): return 13.0*(v/s.H)**2-5.0, torch.zeros_like(v)                                 # a bow in ONE plane (u1 = 0): the lengthwise cut at theta=0 lies in it
    def axis(s,v): return s.bend(v)
    def straight(s,P):
        cx,cz=s.bend(P[:,1]); return torch.stack([P[:,0]-cx,P[:,1],P[:,2]-cz],1)
    def sdf(s,P): Q=s.straight(P); return torch.maximum(torch.sqrt(Q[:,0]**2+Q[:,2]**2)-(s.R-8),Q[:,1].abs()-s.H)
    def color(s,P):
        Q=s.straight(P); Q=torch.stack([Q[:,0]*s.R/(s.R-8),Q[:,1],Q[:,2]*s.R/(s.R-8)],1); return Cable.color(s,Q)

def make(name,seed=0):
    if name=="wood2k": return Wood(2,seed,knots=True)
    if name=="wood3r": return Wood(2,seed,rotbend=True)
    if name=="wood5": return Wood(2,seed,rotbend=True,Rb=60.0,R=28.0,H=50.0)
    if name.startswith("wood"): return Wood(int(name[4]),seed)
    return {"cake":Cake,"cable":Cable,"honeycomb":Honeycomb,"hose":Hose,"ccable":CCable,"roll":Roll,"banana":Banana,"laminate":Laminate}[name](seed)
def render(o,P):
    sd=o.sdf(P); out=torch.ones(len(P),3,device=dv); m=sd<0
    if m.any(): out[m]=o.color(P[m])
    return out
NFRAME=None   # for an object sliced normal to its centreline: s -> (point, normal, tangent)
CENTER=None   # for a curved object: v -> (cx, cz); its lengthwise cuts then follow the centreline row by row
def plane_points(ext,spec,res=RES):
    """spec: ("long",theta,off) | ("trans",h) | ("obl",tilt,az,off) -- the same frames planes.Vol uses.
    A lengthwise cut of a curved object (CENTER set) is the surface through its centreline at azimuth
    theta -- what a lengthwise cut means for a bent cable -- not a flat plane across the bend."""
    gl=torch.linspace(-ext,ext,res,device=dv); GV,GU=torch.meshgrid(gl,gl,indexing="ij")
    if spec[0]=="long":
        _,th,off=spec; c_,s_=math.cos(th),math.sin(th); P=torch.stack([GU*c_-off*s_,GV,GU*s_+off*c_],-1)
        if CENTER is not None:   # the cut surface holds the centreline; seen head-on, the bend along the cut's own direction stays in the picture
            cx,cz=CENTER(GV.reshape(-1)); cx,cz=cx.reshape(GV.shape),cz.reshape(GV.shape)
            along=cx*c_+cz*s_; P=P+torch.stack([cx-along*c_,torch.zeros_like(GV),cz-along*s_],-1)
    elif spec[0]=="trans":
        P=torch.stack([GU,torch.full_like(GU,spec[1]),GV],-1)
    elif spec[0]=="ntrans":            # the cross-section NORMAL to the centreline at arc length s (NFRAME set)
        c,n,t=NFRAME(torch.tensor([float(spec[1])],device=dv)); b=torch.tensor([0.,0.,1.],device=dv)
        P=c[0]+GU[...,None]*n[0]+GV[...,None]*b
    else:
        _,t,az,off=spec; a=torch.tensor([math.cos(az),0.,math.sin(az)],device=dv)
        b=math.cos(t)*torch.tensor([-math.sin(az),0.,math.cos(az)],device=dv)+math.sin(t)*torch.tensor([0.,1.,0.],device=dv)
        n=torch.linalg.cross(a,b); P=GU[...,None]*a+GV[...,None]*b+off*n
    return P.reshape(-1,3)
def save(t,p,res=RES): Image.fromarray((t.reshape(res,res,3).clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(p)
def polar(a,NR=128,NPHI=512):
    H=a.shape[0]; t=torch.from_numpy(a).permute(2,0,1)[None].float()
    r=(torch.arange(NR)+0.5)/NR*(H/2); ph=torch.arange(NPHI)/NPHI*2*np.pi; rr,pp=torch.meshgrid(r,ph,indexing="ij")
    x=(H/2+rr*torch.cos(pp))/(H-1)*2-1; y=(H/2+rr*torch.sin(pp))/(H-1)*2-1
    return F.grid_sample(t,torch.stack([x,y],-1)[None],mode="bilinear",padding_mode="border",align_corners=True)[0].permute(1,2,0).numpy()
# the declared protocol: inputs and the held-out test planes (fractions of the occupied height)
TRAIN_TH=[0.0,math.pi/3,2*math.pi/3]; TRAIN_HQ=[0.3,0.5,0.7]
TEST_TH=[math.radians(a) for a in (30,45,67,100,150)]; TEST_HQ=[0.2,0.4,0.6,0.8]
TEST_OBL=[(math.radians(t),math.radians(az),off) for t in (30,45,60) for az,off in ((20,0.0),(110,8.0))]
if __name__=="__main__":
    name,out=sys.argv[1],sys.argv[2]; CROSS=name.endswith("x"); base=name[:-1] if CROSS else name
    o=make(base); os.makedirs(out,exist_ok=True)
    if base in ("ccable","wood3r","banana","wood5"): CENTER=lambda v: o.axis(v)
    if base in ("banana","wood5"): NFRAME=lambda sv: o.frame(sv)
    lin=torch.arange(N,device=dv,dtype=torch.float32)-C
    A,B,Cc=torch.meshgrid(lin,lin,lin,indexing="ij"); P=torch.stack([A,B,Cc],-1).reshape(-1,3)
    sd=o.sdf(P); occ=(sd<0)
    # supersampled truth at the grid's resolution (2^3 sub-samples) so a 128^3 cell holds the cell average
    V=torch.zeros(len(P),3,device=dv)
    for dx in (-0.25,0.25):
        for dy in (-0.25,0.25):
            for dz in (-0.25,0.25):
                V+=render(o,P+torch.tensor([dx,dy,dz],device=dv))
    V=(V/8).T.reshape(3,N,N,N)*2-1
    from scipy import ndimage as ndi
    o3=occ.reshape(N,N,N).cpu().numpy(); dist=torch.from_numpy(ndi.distance_transform_edt(o3)).to(dv)
    shell=(occ.reshape(N,N,N)&(dist<=2)); core=(occ.reshape(N,N,N)&(dist>2))
    G=dict(V=V.cpu(),OCC=occ.reshape(1,N,N,N).float().cpu(),SHELL=shell[None].float().cpu(),CORE=core[None].float().cpu(),N=N)
    if isinstance(o,Wood): G["Gfield"]=o.G(P).reshape(N,N,N).cpu()
    torch.save(G,f"{out}/grid.pt")
    rad=torch.sqrt(A**2+Cc**2).reshape(-1); vv=B.reshape(-1)
    ext=max(float(rad[occ].max()),float(vv[occ].abs().max()))/0.82
    h0,h1=float(vv[occ].min()),float(vv[occ].max())
    planes=dict(ext=ext,h0=h0,h1=h1,long=[],trans=[])
    for d in ("spl_long","spl_trans","polar_spl_trans"): os.makedirs(f"{out}/{d}",exist_ok=True)
    spec=lambda k: make(base,seed=k) if CROSS else o                      # cross-specimen: input k from specimen k+1
    LONGP=[(0.0,-12.0),(0.0,0.0),(0.0,12.0)] if base=="laminate" else [(th,0.0) for th in TRAIN_TH]   # a laminate is cut across its sheets
    for k,(th,off) in enumerate(LONGP):
        save(render(spec(1+k),plane_points(ext,("long",th,off))),f"{out}/spl_long/{k:02d}.png"); planes["long"].append([th,off])
    NORMAL=base in ("banana","wood5"); HALF=getattr(o,"L",getattr(o,"H",0))
    for k,q in enumerate(TRAIN_HQ):
        h=h0+(h1-h0)*q; sp=("ntrans",(2*q-1)*HALF*0.8) if NORMAL else ("trans",h)
        im=render(spec(4+k),plane_points(ext,sp))
        save(im,f"{out}/spl_trans/{k:02d}.png"); planes["trans"].append(h); planes.setdefault("ntrans",[]).append(sp[1] if NORMAL else None)
        Image.fromarray((polar(im.reshape(RES,RES,3).cpu().numpy())*255).astype(np.uint8)).save(f"{out}/polar_spl_trans/{k:02d}.png")
    json.dump(planes,open(f"{out}/planes.json","w"))
    if CROSS:                                                             # held-out specimens 7-9, every test plane
        for d in ("hld_long","hld_trans","hld_obl"): os.makedirs(f"{out}/{d}",exist_ok=True)
        for k in range(3):
            ok=make(base,seed=7+k)
            for i,th in enumerate(TEST_TH): save(render(ok,plane_points(ext,("long",th,0.0))),f"{out}/hld_long/{k}_{i}.png")
            for i,q in enumerate(TEST_HQ): save(render(ok,plane_points(ext,("trans",h0+(h1-h0)*q))),f"{out}/hld_trans/{k}_{i}.png")
            for i,t in enumerate(TEST_OBL): save(render(ok,plane_points(ext,("obl",)+t)),f"{out}/hld_obl/{k}_{i}.png")
    json.dump(dict(object=name,kind="box" if base in ("cake","honeycomb","laminate") else "revolve",cross=CROSS),open(f"{out}/meta.json","w"))
    if hasattr(o,"axis") and base in ("wood3","wood4","ccable","wood3r","banana","wood5"):          # the declared centreline (user input for curvilinear)
        vs=np.linspace(-64,64,129); cxz=[o.axis(torch.tensor([float(v)],device=dv)) for v in vs]
        json.dump(dict(v=vs.tolist(),cx=[float(a) for a,_ in cxz],cz=[float(b) for _,b in cxz]),open(f"{out}/centerline.json","w"))
    print(name,"ext",round(ext,1),"core",int(core.sum()))
