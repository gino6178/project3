"""2D-supervised 3D distillation: each step, random planes, SinDiffusion one-step denoise as target.

The medical-imaging way to train a 3D prior with only 2D slices (DiffusionBlend, SNAFusion): at each
update sample random slices, run the 2D diffusion, use its score as the loss on the 3D field. Two
lessons from those papers are built in -- the score is position-aware (each plane starts from its
own render, so different planes pull differently and do not collapse to one mode), and the target is
the model's deterministic one-step denoise x0-hat rather than a raw SDS gradient (which over-smooths).
Random planes every step cover every cell, so the sweep has no bright bands.

feat only, from the fit, mild anchor. N planes per step, both families, t capped below the high-noise
regime.
"""
import os, sys, time, argparse
os.environ["CUT_DEFERRED"]="1"
import numpy as np, torch
sys.path.insert(0,"/workspace/ovoxel_native"); sys.path.insert(0,"/workspace/sindiff")
sys.path.insert(0,"/workspace/rebuild/project3/code/evaluate")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr
W="/workspace/ovoxel_native"; FN="/workspace/rebuild/worktree"; OBJ="orange_sp"; dev="cuda"; S=256
ap=argparse.ArgumentParser()
ap.add_argument("--long_ckpt",default="/workspace/sindiff/OUTPUT/sd-long00/model012000.pt")
ap.add_argument("--trans_ckpt",default="/workspace/sindiff/OUTPUT/sd-orange_h/model006000.pt")
ap.add_argument("--steps",type=int,default=1500)
ap.add_argument("--planes",type=int,default=16)
ap.add_argument("--lr",type=float,default=3e-3)
ap.add_argument("--anchor",type=float,default=0.5)
ap.add_argument("--t_lo",type=float,default=0.2)
ap.add_argument("--t_hi",type=float,default=0.6)
ap.add_argument("--tag",default="s_v2_d3")
a=ap.parse_args()
ON.FDG=ON._load_ovoxel(); glctx=dr.RasterizeCudaContext(device=dev)
st=torch.load(f"{W}/state_{OBJ}.pt",map_location=dev,weights_only=False)
C=np.load(f"{W}/cams_{OBJ}_v2.npz"); p=torch.load(f"{W}/s_v2_{OBJ}/params.pt",map_location=dev)
st["dual_v"]=p["dual_v"].to(dev); st["split_w"]=p["split_w"].to(dev)
w=p["dec_i"]["stage1.0.weight"].shape[0]; nl=sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight"))-1
anchor.W_HID,anchor.N_HID=w,nl
di=anchor.ColourDecoder(len(st["interior"]),init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr=anchor.ColourDecoder(len(st["surf_rgb"]),init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad(): st["surf_rgb"]=dsr(); rgb0=di().detach().clone()
hc=float(st["hc"]); org=torch.as_tensor(st["org"],dtype=torch.float32,device=dev)
solid=st["solid"].long(); cen=(solid.float()+0.5)*hc+org
def load(ckpt):
    d=model_and_diffusion_defaults()
    d.update(image_size=256,num_channels=64,num_head_channels=16,channel_mult="1,2,4",
             attention_resolutions="2",num_res_blocks=1,resblock_updown=False,use_fp16=False,
             use_scale_shift_norm=True,use_checkpoint=True,diffusion_steps=1000,
             noise_schedule="linear",learn_sigma=False,class_cond=False)
    m,D=create_model_and_diffusion(**d); m.load_state_dict(torch.load(ckpt,map_location="cpu"))
    m.cuda().eval()
    for pr in m.parameters(): pr.requires_grad_(False)
    return m,D
phiL,diff=load(a.long_ckpt); phiT,_=load(a.trans_ckpt)
ab=torch.as_tensor(diff.alphas_cumprod,device=dev).float()
# family sweep normals/mvps (ovcut style) + offset ranges
k=len(C["v_mvp"])//2
fams={"long":(torch.as_tensor(C["v_mvp"][k],dtype=torch.float32,device=dev).contiguous(),
              torch.as_tensor((C["v_planes"][k,:3]/np.linalg.norm(C["v_planes"][k,:3])).astype(np.float32),device=dev),phiL),
      "trans":(torch.as_tensor(C["h_mvp"],dtype=torch.float32,device=dev).contiguous(),
               torch.as_tensor((C["h_planes"][0,:3]/np.linalg.norm(C["h_planes"][0,:3])).astype(np.float32),device=dev),phiT)}
rng={}
for nm,(mvp,n,phi) in fams.items():
    pr=(cen@n); rng[nm]=(float(-pr.max())+3*hc,float(-pr.min())-3*hc)
touched=torch.zeros(len(solid),dtype=torch.bool,device=dev)
def render(mvp,n,dd,grad=False):
    st["interior"]=di() if grad else di().detach()
    ctx=torch.enable_grad() if grad else torch.no_grad()
    with ctx: return ON.render_section(st,glctx,mvp,n,float(dd),S,exterior=True)
for pr in di.parameters(): pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt=torch.optim.AdamW([di.feat],lr=a.lr)
tlo,thi=int(a.t_lo*diff.num_timesteps),int(a.t_hi*diff.num_timesteps)
t0=time.time()
print(f"  distill3d: {a.steps} steps x {a.planes} planes, t in [{tlo},{thi}]",flush=True)
for step in range(1,a.steps+1):
    loss=0.0; got=0
    for pi in range(a.planes):
        nm="long" if pi%2==0 else "trans"
        mvp,n,phi=fams[nm]; d0,d1=rng[nm]; dd=float(np.random.uniform(d0,d1))
        try:
            with torch.no_grad():
                _,af,_,_=render(mvp,n,dd)
            mask=(af[:1]>0).float()[None]
            if float(mask.sum())<100: continue
            img,_,_,_=render(mvp,n,dd,grad=True)
        except RuntimeError: continue
        x=img[None]*2-1
        t=torch.randint(tlo,thi,(1,),device=dev)
        noise=torch.randn_like(x)
        with torch.no_grad():
            x_t=diff.q_sample(x.detach(),t,noise)
            eps=phi(x_t,t)
            # deterministic one-step denoise prediction x0-hat
            x0=( (x_t - (1-ab[t]).sqrt()[:,None,None,None]*eps) / ab[t].sqrt()[:,None,None,None] ).clamp(-1,1)
        loss=loss + ((x-x0.detach())**2*mask).sum()/mask.sum().clamp_min(1)
        touched[(((cen@n)+dd).abs()<=1.5*hc)]=True
        got+=1
    if got==0: continue
    reg=a.anchor*((di()-rgb0)**2).mean()
    (loss/got+reg).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if step%100==0 or step==1:
        print(f"  step {step:5d}  loss {float(loss)/got:.4f}  anchor {float(reg):.4f}  cover {float(touched.float().mean())*100:.0f}%  {time.time()-t0:.0f}s",flush=True)
out=f"{W}/{a.tag}_{OBJ}"; os.makedirs(out,exist_ok=True)
q=dict(p); q["dec_i"]={kk:v.clone().detach() for kk,v in p["dec_i"].items()}; q["dec_i"]["feat"]=di.feat.detach().clone()
torch.save(q,f"{out}/params.pt"); open(f"{out}/run.env","w").write("CAMS_SUFFIX=_v2\n")
print(f"wrote {out}/params.pt",flush=True)
@torch.no_grad()
def score(planes,mvps,ph,tag):
    ref=realism._paths(ph); st["interior"]=di().detach(); paths=[]; from PIL import Image
    d=f"{W}/d3_{tag}"; os.makedirs(d,exist_ok=True)
    for j in range(len(planes)):
        nj=planes[j,:3]/np.linalg.norm(planes[j,:3])
        img,_,_,_=render(torch.as_tensor(mvps[j] if mvps.ndim==3 else mvps,dtype=torch.float32,device=dev),
                         torch.as_tensor(nj.astype(np.float32),device=dev),float(planes[j,3]))
        f=f"{d}/{j:02d}.png"; Image.fromarray((img.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(f); paths.append(f)
    return realism._dreamsim(ref,paths,dev),len(ref)
HL,HH=int(C["h_lo"][0]),int(C["h_hi"][0]); hM=np.broadcast_to(C["h_mvp"][None],(HH-HL,4,4))
suites={"held long":(C["ev_planes"],C["ev_mvp"],f"{FN}/hld_orange_v"),
        "held trans":(C["eh_planes"],C["eh_mvp"],f"{FN}/hld_orange_h"),
        "supv long":(C["v_planes"],C["v_mvp"],f"{FN}/spl_orange_v"),
        "supv trans":(C["h_planes"][HL:HH],hM,f"{FN}/spl_orange_h")}
print("\n  suite         Stage1   d3       delta")
for kk,(P,M,ph) in suites.items():
    di.load_state_dict(p["dec_i"]); b,_=score(P,M,ph,kk.replace(" ","_")+"0")
    di.load_state_dict(q["dec_i"]); aa,_=score(P,M,ph,kk.replace(" ","_")+"1")
    print(f"  {kk:12s} {b:.4f}  {aa:.4f}  {aa-b:+.4f}")
