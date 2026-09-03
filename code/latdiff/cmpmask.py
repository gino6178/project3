"""Where the mask breaks it: O-Voxel render | free generation | masked SDEdit, per family.

Free generation (from noise, no mask) is known to produce the columella and peel. Masked SDEdit
on the render does not. This puts the three side by side so the gap is visible: whether the mask
excludes the peel, or pins the axis to the render's weak one.
"""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W = "/workspace/ovoxel_native"; OBJ = "orange_sp"; dev = "cuda"; S = 256
STR = float(os.environ.get("STR", "0.5"))
ON.FDG = ON._load_ovoxel(); glctx = dr.RasterizeCudaContext(device=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad(): st["interior"], st["surf_rgb"] = di(), dsr()
def load(ckpt):
    d = model_and_diffusion_defaults()
    d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
             attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=True,
             use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
             noise_schedule="linear", learn_sigma=False, class_cond=False)
    m, D = create_model_and_diffusion(**d); m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.cuda().eval(); m.convert_to_fp16(); return m, D
mL, diff = load("/workspace/sindiff/OUTPUT/sd-vlong/model002000.pt")
mT, _ = load("/workspace/sindiff/OUTPUT/sd-vtrans/model002000.pt")
def arr(t): return (t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
@torch.no_grad()
def free(m):
    x = torch.randn(1,3,S,S,device=dev)
    for i in reversed(range(diff.num_timesteps)):
        t = torch.full((1,),i,device=dev,dtype=torch.long)
        x = diff.p_sample(m,x,t,clip_denoised=True,model_kwargs={})["sample"]
    return arr(x[0]*0.5+0.5)
@torch.no_grad()
def masked(m, x0, mask):
    ts=int(STR*diff.num_timesteps)-1
    x=diff.q_sample(x0,torch.full((1,),ts,device=dev,dtype=torch.long))
    for i in reversed(range(ts+1)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=mask*x+(1-mask)*diff.q_sample(x0,t)
        x=diff.p_sample(m,x,t,clip_denoised=True,model_kwargs={})["sample"]
    return arr((mask*x+(1-mask)*x0)[0]*0.5+0.5)
@torch.no_grad()
def maskimg(mask): return arr(mask[0].repeat(3,1,1))
def lab(txt,wd):
    im=Image.new("RGB",(wd,22),(255,255,255));d=ImageDraw.Draw(im)
    try:f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:f=ImageFont.load_default()
    d.text((6,3),txt,fill=(30,30,30),font=f);return np.asarray(im)
rows=[]
for name, P, M, model in (("LONG", C["v_planes"], C["v_mvp"], mL),
                          ("TRANS", C["h_planes"][int(C["h_lo"][0]):int(C["h_hi"][0])],
                           np.broadcast_to(C["h_mvp"][None],(int(C["h_hi"][0])-int(C["h_lo"][0]),4,4)), mT)):
    k = len(P)//2
    n = torch.as_tensor((P[k,:3]/np.linalg.norm(P[k,:3])).astype(np.float32),device=dev)
    mvp = torch.as_tensor((M[k] if M.ndim==3 else M).copy(),dtype=torch.float32,device=dev).contiguous()
    with torch.no_grad():
        img,_,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S)
        _,af,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S,exterior=False)
    mask=(af[:1]>0).float()[None]
    r = np.concatenate([arr(img), maskimg(mask), free(model), masked(model,img[None]*2-1,mask)],1)
    rows.append(np.concatenate([lab(f"{name}: render | flesh-mask | FREE gen | masked SDEdit(s={STR})", r.shape[1]), r],0))
Image.fromarray(np.concatenate(rows,0)).save("/workspace/cmpmask.png")
print("render | mask | free | masked, both families")
