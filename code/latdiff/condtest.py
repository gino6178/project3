"""Condition on IN-DISTRIBUTION (degraded photo) vs OUT-OF-DISTRIBUTION (real render). Same model."""
import os, sys
os.environ["CUT_DEFERRED"]="1"
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,"/workspace/ovoxel_native"); sys.path.insert(0,"/workspace/sindiff")
from guided_diffusion.sinddpm import UNetModel
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule)
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W="/workspace/ovoxel_native"; OBJ="orange_sp"; dev="cuda"; S=256
ON.FDG=ON._load_ovoxel(); glctx=dr.RasterizeCudaContext(device=dev)
st=torch.load(f"{W}/state_{OBJ}.pt",map_location=dev,weights_only=False)
C=np.load(f"{W}/cams_{OBJ}_v2.npz"); p=torch.load(f"{W}/s_v2_{OBJ}/params.pt",map_location=dev)
st["dual_v"]=p["dual_v"].to(dev); st["split_w"]=p["split_w"].to(dev)
w=p["dec_i"]["stage1.0.weight"].shape[0]; nl=sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight"))-1
anchor.W_HID,anchor.N_HID=w,nl
di=anchor.ColourDecoder(len(st["interior"]),init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr=anchor.ColourDecoder(len(st["surf_rgb"]),init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad(): st["interior"],st["surf_rgb"]=di(),dsr()
m=UNetModel(image_size=S,in_channels=6,model_channels=64,out_channels=3,num_res_blocks=1,
            attention_resolutions=(S//2,),channel_mult=(1,2,4),num_head_channels=16,
            use_scale_shift_norm=True,use_checkpoint=False,use_fp16=False).cuda()
m.load_state_dict(torch.load("/workspace/cond_long.pt",map_location="cpu")["model"]); m.eval()
diff=GaussianDiffusion(betas=get_named_beta_schedule("linear",1000),model_mean_type=ModelMeanType.EPSILON,
                       model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE)
def arr(t): return (t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
def gauss(x,k=9,sig=4.0):
    ax=torch.arange(k,device=dev)-k//2; g=torch.exp(-(ax**2)/(2*sig**2)); g=g/g.sum()
    ker=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
    return F.conv2d(F.pad(x,(k//2,)*4,mode="reflect"),ker,groups=3)
def degrade(x):
    b=gauss(x); gray=b.mean(1,keepdim=True); return (0.5*b+0.5*gray).clamp(0,1)
@torch.no_grad()
def samp(cond,mask):
    torch.manual_seed(0); x=torch.randn(1,3,S,S,device=dev)
    for i in reversed(range(diff.num_timesteps)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=diff.p_sample(lambda z,tt:m(torch.cat([z,cond],1),tt),x,t,clip_denoised=True,model_kwargs={})["sample"]
    white=torch.ones(1,3,S,S,device=dev); return mask*x+(1-mask)*white
def lab(t,wd):
    im=Image.new("RGB",(wd,22),(255,255,255));d=ImageDraw.Draw(im)
    try:f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:f=ImageFont.load_default()
    d.text((6,3),t,fill=(30,30,30),font=f);return np.asarray(im)
# a photo (in-distribution) and a render (OOD)
photo=torch.from_numpy(np.asarray(Image.open("/workspace/rebuild/worktree/spl_orange_v/or_long_00.png").convert("RGB").resize((S,S))).astype(np.float32)/255).permute(2,0,1).to(dev)
psil=(photo.min(0).values<0.92).float()[None,None]
cond_id=degrade(photo[None])*2-1
out_id=samp(cond_id,psil)
P,M=C["ev_planes"],C["ev_mvp"]; k=1
n=torch.as_tensor((P[k,:3]/np.linalg.norm(P[k,:3])).astype(np.float32),device=dev)
mvp=torch.as_tensor(M[k].copy(),dtype=torch.float32,device=dev).contiguous()
with torch.no_grad():
    img,_,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S)
    _,af,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S,exterior=True)
mask=(af[:1]>0).float()[None]
out_ood=samp(img[None]*2-1,mask)
r1=np.concatenate([arr(degrade(photo[None])[0]),arr(out_id[0]*0.5+0.5)],1)
r2=np.concatenate([arr(img),arr(out_ood[0]*0.5+0.5)],1)
Image.fromarray(np.concatenate([lab("IN-DIST: degraded photo cond | output",r1.shape[1]),r1,
                                lab("OOD: real render cond | output",r2.shape[1]),r2],0)).save("/workspace/condtest.png")
print("done")
