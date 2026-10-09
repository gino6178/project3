# Ground-truth evaluation for the synthetic benchmark (synth2.py).  Every arm and the truth are cut
# on the same held-out planes -- unseen heights, unseen azimuths, oblique -- by the same trilinear
# slicer, at 256 px, and compared inside the interior (the shell is observed data in every arm).
#   appearance   PSNR, SSIM, LPIPS (alex), DreamSim (paired: the truth's own face as reference)
#   structure    orientation error: structure-tensor orientation of the face against the truth's,
#                weighted by the truth's coherence (the ring / layer / grain direction, ~ grad G)
#   wood only    read from the truth's growth-time field G (rings are its integer iso-surfaces):
#                ring alignment  -- earlywood minus latewood luminance of the face on the truth's
#                                   own early/late bands, over the truth's (1 = rings where they are)
#                ring continuity -- the fraction of 16 angular sectors whose alignment exceeds 0.5
#                gradG orient.   -- face structure orientation against the plane projection of grad G
#                ring centre     -- least-squares meeting point of the face's gradient lines (voxels)
#                ring spacing    -- radial autocorrelation period about that centre (relative error)
#                knot IoU        -- dark knot pixels against the truth's (wood4, faces through a knot)
#   volume       mean |X - V_gt| over the interior, colour in [0,1]
#   python eval_gt.py <object dir> [arm=state.pt ...]      (carrier.pt is always an arm)
import os,sys,json,math,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import synth2 as S
dv="cuda" if torch.cuda.is_available() else "cpu"; S.dv=dv
D=sys.argv[1]; R=256
G=torch.load(f"{D}/grid.pt",map_location=dv); N=G["N"]; C=(N-1)/2
PJ=json.load(open(f"{D}/planes.json")); ext,h0,h1=PJ["ext"],PJ["h0"],PJ["h1"]
name=json.load(open(f"{D}/meta.json"))["object"]
if name in ("ccable","wood3r"): _o=S.make(name); S.CENTER=lambda v: _o.axis(v)   # its lengthwise cuts follow the centreline
arms={"carrier":torch.load(f"{D}/carrier.pt",map_location=dv)["V"].float()}
for a in sys.argv[2:]:
    n,p=a.split("=",1)
    if os.path.exists(p): arms[n]=torch.load(p,map_location=dv)["X"].float()
GT=G["V"].float(); CORE=G["CORE"].float()
def cut(X,spec):
    P=S.plane_points(ext,spec,R)+C
    q=torch.stack([P[:,2],P[:,1],P[:,0]],-1)/(N-1)*2-1
    return F.grid_sample(X[None],q[None,None,None],mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0].T.reshape(R,R,X.shape[0])
planes=[("trans",("trans",h0+(h1-h0)*q)) for q in S.TEST_HQ]+[("long",("long",th,0.0)) for th in S.TEST_TH]+[("obl",("obl",)+t) for t in S.TEST_OBL]
import lpips; LP=lpips.LPIPS(net="alex",verbose=False).to(dv)
from dreamsim import dreamsim; DS,DSP=dreamsim(pretrained=True,device=dv,cache_dir=os.path.expanduser("~/dreamsim_ckpt"))
from PIL import Image
def gauss(sig):
    k=int(sig*3)*2+1; x=torch.arange(k,device=dv,dtype=torch.float32)-k//2; w=torch.exp(-x**2/(2*sig*sig)); return w/w.sum(),k
def blur(x,sig):
    w,k=gauss(sig); c=x.shape[1]
    x=F.conv2d(F.pad(x,(k//2,)*4,mode="reflect"),w.view(1,1,1,k).repeat(c,1,1,1),groups=c)
    return F.conv2d(x,w.view(1,1,k,1).repeat(c,1,1,1),groups=c)
def ssim_map(a,b):
    a=a.permute(2,0,1)[None]; b=b.permute(2,0,1)[None]; C1,C2=0.01**2,0.03**2
    ma,mb=blur(a,1.5),blur(b,1.5); va=blur(a*a,1.5)-ma**2; vb=blur(b*b,1.5)-mb**2; cv=blur(a*b,1.5)-ma*mb
    return (((2*ma*mb+C1)*(2*cv+C2))/((ma**2+mb**2+C1)*(va+vb+C2))).mean(1)[0]
def orient(img):
    g=img.mean(2)[None,None]; gx=F.conv2d(F.pad(g,(1,1,1,1),mode="replicate"),torch.tensor([[[[-1,0,1]]]],device=dv,dtype=torch.float32).expand(1,1,3,3)*torch.tensor([[1],[2],[1]],device=dv)/8)
    gy=F.conv2d(F.pad(g,(1,1,1,1),mode="replicate"),(torch.tensor([[[[-1,0,1]]]],device=dv,dtype=torch.float32).expand(1,1,3,3)*torch.tensor([[1],[2],[1]],device=dv)/8).transpose(-1,-2))
    J=blur(torch.cat([gx*gx,gy*gy,gx*gy],1),3.0)[0]; jxx,jyy,jxy=J
    th=0.5*torch.atan2(2*jxy,jxx-jyy); coh=torch.sqrt((jxx-jyy)**2+4*jxy**2)/(jxx+jyy+1e-8)
    return th,coh
def ring_center(img,m,cx0,cz0):
    """centre (in face pixels) maximising angular consistency of the luminance about it"""
    g=img.mean(2); best=(1e9,0,0); H=R
    ph=torch.linspace(0,2*math.pi,180,device=dv)[:-1]; rr=torch.linspace(4,H*0.3,60,device=dv)
    for dy in range(-16,17,2):
        for dx in range(-16,17,2):
            cy,cx=cz0+dy,cx0+dx
            ys=cy+rr[:,None]*torch.sin(ph)[None]; xs=cx+rr[:,None]*torch.cos(ph)[None]
            q=torch.stack([xs/(H-1)*2-1,ys/(H-1)*2-1],-1)[None]
            v=F.grid_sample(g[None,None],q,align_corners=True)[0,0]
            s=float(v.var(1).mean())
            if s<best[0]: best=(s,cx,cy)
    return best[1],best[2]
def ring_period(img,cx,cy):
    g=img.mean(2); H=R; ph=torch.linspace(0,2*math.pi,360,device=dv)[:-1]; rr=torch.linspace(2,H*0.32,256,device=dv)
    q=torch.stack([(cx+rr[:,None]*torch.cos(ph))/(H-1)*2-1,(cy+rr[:,None]*torch.sin(ph))/(H-1)*2-1],-1)[None]
    prof=F.grid_sample(g[None,None],q,align_corners=True)[0,0].mean(1); prof=prof-prof.mean()
    sp=torch.fft.rfft(prof*torch.hann_window(len(prof),device=dv)).abs(); sp[:3]=0
    k=int(sp.argmax()); return float((rr[-1]-rr[0])/max(k,1))
px_per_vox=R/(2*ext)
KEYS=("psnr","ssim","lpips","dreamsim","orient","ringc","rings","align","contin","gorient","knot")
res={a:{f:{k:[] for k in KEYS} for f in ("trans","long","obl")} for a in arms}
GF=G.get("Gfield"); GF=GF.float()[None].to(dv) if GF is not None else None
def plane_basis(spec):
    P=S.plane_points(ext,spec,3).reshape(3,3,3); e_col=(P[0,2]-P[0,0]); e_row=(P[2,0]-P[0,0])
    return e_col/e_col.norm(), e_row/e_row.norm()
def gradG(spec):
    P=S.plane_points(ext,spec,R)+C; out=[]
    for d in range(3):
        for sg in (1,-1):
            Q=P.clone(); Q[:,d]+=0.5*sg; q=torch.stack([Q[:,2],Q[:,1],Q[:,0]],-1)/(N-1)*2-1
            out.append(F.grid_sample(GF[None],q[None,None,None],mode="bilinear",padding_mode="border",align_corners=True).view(-1))
    return torch.stack([out[0]-out[1],out[2]-out[3],out[4]-out[5]],-1).reshape(R,R,3)
def centre_ls(img,m):
    th,coh=orient(img); w=(coh*m).flatten(); yy,xx=torch.meshgrid(torch.arange(R,device=dv).float(),torch.arange(R,device=dv).float(),indexing="ij")
    nx,ny=torch.cos(th).flatten(),torch.sin(th).flatten()                          # gradient direction; ring centre lies on these lines
    px,py=xx.flatten(),yy.flatten(); tx,ty=-ny,nx                                   # line normal = tangent
    A11=(w*tx*tx).sum(); A12=(w*tx*ty).sum(); A22=(w*ty*ty).sum(); b1=(w*tx*(tx*px+ty*py)).sum(); b2=(w*ty*(tx*px+ty*py)).sum()
    det=A11*A22-A12*A12; return float((A22*b1-A12*b2)/det), float((A11*b2-A12*b1)/det)
def radial_period(img,cx,cy):
    g=img.mean(2); ph=torch.linspace(0,2*math.pi,72,device=dv)[:-1]; rr=torch.linspace(3,R*0.3,200,device=dv)
    q=torch.stack([(cx+rr[:,None]*torch.cos(ph))/(R-1)*2-1,(cy+rr[:,None]*torch.sin(ph))/(R-1)*2-1],-1)[None]
    pr=F.grid_sample(g[None,None],q,align_corners=True)[0,0].T; pr=pr-pr.mean(1,keepdim=True)
    ac=torch.stack([(pr[:,:len(rr)-k]*pr[:,k:]).mean() for k in range(len(rr)//2)]); ac=ac/ac[0]
    k=int(torch.nonzero((ac[1:-1]>ac[:-2])&(ac[1:-1]>=ac[2:])).flatten()[0])+1 if ((ac[1:-1]>ac[:-2])&(ac[1:-1]>=ac[2:])).any() else 1
    return float(k*(rr[1]-rr[0]))
dump=f"{D}/gt_faces"; os.makedirs(dump,exist_ok=True)
for i,(fam,spec) in enumerate(planes):
    gt=(cut(GT,spec).clamp(-1,1)+1)/2; m=(cut(CORE,spec)[...,0]>0.5)
    if int(m.sum())<200: continue
    th_g,coh_g=orient(gt); w=coh_g*m
    tiles=[gt]
    for a,X in arms.items():
        im=(cut(X,spec).clamp(-1,1)+1)/2; tiles.append(im)
        mse=float(((im-gt)**2)[m].mean()); res[a][fam]["psnr"].append(10*math.log10(1/max(mse,1e-10)))
        res[a][fam]["ssim"].append(float(ssim_map(im,gt)[m].mean()))
        A_=torch.where(m[...,None],im,torch.ones_like(im)).permute(2,0,1)[None]*2-1; B_=torch.where(m[...,None],gt,torch.ones_like(gt)).permute(2,0,1)[None]*2-1
        with torch.no_grad():
            res[a][fam]["lpips"].append(float(LP(A_,B_)))
            pa=DSP(Image.fromarray((((A_[0].permute(1,2,0)+1)/2).cpu().numpy()*255).astype(np.uint8))).to(dv)
            pb=DSP(Image.fromarray((((B_[0].permute(1,2,0)+1)/2).cpu().numpy()*255).astype(np.uint8))).to(dv)
            res[a][fam]["dreamsim"].append(float(DS(pa,pb)))
        th_a,_=orient(im); d=(th_a-th_g).abs()%math.pi; d=torch.minimum(d,math.pi-d)
        res[a][fam]["orient"].append(float((d*w).sum()/w.sum().clamp(min=1e-6))*180/math.pi)
        if GF is not None:
            gf=cut(GF,spec)[...,0]; f=gf%1.0; Lg=gt.mean(2); La=im.mean(2)
            early=m&(f>0.1)&(f<0.45); late=m&(f>0.68)&(f<0.88)
            if int(early.sum())>50 and int(late.sum())>50:
                cg=float(Lg[early].mean()-Lg[late].mean()); res[a][fam]["align"].append(float(La[early].mean()-La[late].mean())/max(cg,1e-4))
                yy,xx=torch.meshgrid(torch.arange(R,device=dv).float(),torch.arange(R,device=dv).float(),indexing="ij")
                sec=((torch.atan2(yy-R/2,xx-R/2)+math.pi)/(2*math.pi)*16).long().clamp(0,15); ok=0; tot=0
                for k in range(16):
                    e_,l_=early&(sec==k),late&(sec==k)
                    if int(e_.sum())>10 and int(l_.sum())>10:
                        tot+=1; ok+=int(float(La[e_].mean()-La[l_].mean())/max(float(Lg[e_].mean()-Lg[l_].mean()),1e-4)>0.5)
                res[a][fam]["contin"].append(ok/max(tot,1))
            gg=gradG(spec); ec,er=plane_basis(spec); gx_=(gg*ec).sum(-1); gy_=(gg*er).sum(-1); gn=torch.sqrt(gx_**2+gy_**2)
            thg=torch.atan2(gy_,gx_); th_a2,_=orient(im); d=(th_a2-thg).abs()%math.pi; d=torch.minimum(d,math.pi-d); wg=gn*m
            res[a][fam]["gorient"].append(float((d*wg).sum()/wg.sum().clamp(min=1e-6))*180/math.pi)
            if fam=="trans":
                o=S.make(name[:-1] if name.endswith("x") else name); hv=torch.tensor([spec[1]],device=dv); cxv,czv=o.axis(hv)
                gxp,gyp=(float(cxv)+ext)/(2*ext)*(R-1),(float(czv)+ext)/(2*ext)*(R-1)
                cx,cy=centre_ls(im,m); res[a][fam]["ringc"].append(min(math.hypot(cx-gxp,cy-gyp)/px_per_vox,50.0))
                pg=radial_period(gt,gxp,gyp); pa_=radial_period(im,cx,cy); res[a][fam]["rings"].append(abs(pa_-pg)/pg)
            if name.startswith("wood4") or name=="wood2k":
                kn_g=m&(Lg<0.42); kn_a=m&(La<0.42)
                if int(kn_g.sum())>30: res[a][fam]["knot"].append(float((kn_g&kn_a).sum())/float((kn_g|kn_a).sum()))
    row=torch.cat(tiles,1); Image.fromarray((row.cpu().numpy()*255).astype(np.uint8)).save(f"{dump}/{i:02d}_{fam}.png")
vol={a:float(((X-GT).abs()/2).mean(0)[CORE[0]>0.5].mean()) for a,X in arms.items()}
out={a:{f:{k:(float(np.mean(v)) if v else None) for k,v in d.items()} for f,d in res[a].items()} for a in arms}
for a in arms: out[a]["volume_L1"]=vol[a]
json.dump(out,open(f"{D}/gt_scores.json","w"),indent=1)
print(f"\n{name}  arms: {list(arms)}   (faces dumped: truth | " + " | ".join(arms) + ")")
for fam in ("trans","long","obl"):
    print(f"  {fam:5s} " + "  ".join(f"{a}: psnr {out[a][fam]['psnr']:.2f} ssim {out[a][fam]['ssim']:.3f} lpips {out[a][fam]['lpips']:.3f} ds {out[a][fam]['dreamsim']:.3f} ori {out[a][fam]['orient']:.1f}"
          + "".join(f" {k} {out[a][fam][k]:.3f}" for k in ("align","contin","gorient","ringc","rings","knot") if out[a][fam][k] is not None) for a in arms))
print("  volume L1 " + "  ".join(f"{a}: {vol[a]:.4f}" for a in arms))
