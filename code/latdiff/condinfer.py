"""Condition the trained model on several plane renders; show outputs differ by input."""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.sinddpm import UNetModel
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType,
                                                 ModelVarType, LossType, get_named_beta_schedule)
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W="/workspace/ovoxel_native"; OBJ="orange_sp"; dev="cuda"; S=256
ON.FDG=ON._load_ovoxel(); glctx=dr.RasterizeCudaContext(device=dev)
st=torch.load(f"{W}/state_{OBJ}.pt",map_location=dev,weights_only=False)
C=np.load(f"{W}/cams_{OBJ}_v2.npz")
p=torch.load(f"{W}/s_v2_{OBJ}/params.pt",map_location=dev)
st["dual_v"]=p["dual_v"].to(dev); st["split_w"]=p["split_w"].to(dev)
w=p["dec_i"]["stage1.0.weight"].shape[0]; nl=sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight"))-1
anchor.W_HID,anchor.N_HID=w,nl
di=anchor.ColourDecoder(len(st["interior"]),init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr=anchor.ColourDecoder(len(st["surf_rgb"]),init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad(): st["interior"],st["surf_rgb"]=di(),dsr()
def buildmodel(pth):
    m=UNetModel(image_size=S,in_channels=6,model_channels=64,out_channels=3,num_res_blocks=1,
                attention_resolutions=(S//2,),channel_mult=(1,2,4),num_head_channels=16,
                use_scale_shift_norm=True,use_checkpoint=False,use_fp16=False).cuda()
    m.load_state_dict(torch.load(pth,map_location="cpu")["model"]); m.eval(); return m
mL=buildmodel("/workspace/cond_long.pt"); mT=buildmodel("/workspace/cond_trans.pt")
diff=GaussianDiffusion(betas=get_named_beta_schedule("linear",1000),
                       model_mean_type=ModelMeanType.EPSILON,model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE)
def arr(t): return (t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
@torch.no_grad()
def cond_sample(m, cond, mask, seed=0):
    torch.manual_seed(seed)
    x=torch.randn(1,3,S,S,device=dev)
    for i in reversed(range(diff.num_timesteps)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=diff.p_sample(lambda z,tt: m(torch.cat([z,cond],1),tt), x, t, clip_denoised=True, model_kwargs={})["sample"]
    white=torch.ones(1,3,S,S,device=dev)
    return mask*x + (1-mask)*white
def lab(txt,wd):
    im=Image.new("RGB",(wd,22),(255,255,255));dd=ImageDraw.Draw(im)
    try:f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:f=ImageFont.load_default()
    dd.text((6,3),txt,fill=(30,30,30),font=f);return np.asarray(im)
rows=[]
for name,P,M,m in (("LONG",C["ev_planes"],C["ev_mvp"],mL),("TRANS",C["eh_planes"],C["eh_mvp"],mT)):
    for k in range(min(4,len(P))):
        n=torch.as_tensor((P[k,:3]/np.linalg.norm(P[k,:3])).astype(np.float32),device=dev)
        mvp=torch.as_tensor((M[k] if M.ndim==3 else M).copy(),dtype=torch.float32,device=dev).contiguous()
        with torch.no_grad():
            img,_,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S)
            _,af,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S,exterior=True)
        mask=(af[:1]>0).float()[None]
        cond=img[None]*2-1
        out=cond_sample(m,cond,mask)
        rows.append(np.concatenate([arr(img), arr(out[0]*0.5+0.5)],1))
    row=np.concatenate(rows[-4:],1)
    globals().setdefault("blocks",[]).append(np.concatenate([lab(f"{name} held-out: render|conditional-out (4 different planes)",row.shape[1]),row],0))
Image.fromarray(np.concatenate(blocks,0)).save("/workspace/condinfer.png")
print("done")
