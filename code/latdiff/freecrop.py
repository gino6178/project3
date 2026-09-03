"""Free generation, then crop to the held-out silhouette. Columella comes from free gen; the mask
only limits the outline, applied once at the end -- never during denoising."""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
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
d=model_and_diffusion_defaults()
d.update(image_size=256,num_channels=64,num_head_channels=16,channel_mult="1,2,4",
         attention_resolutions="2",num_res_blocks=1,resblock_updown=False,use_fp16=True,
         use_scale_shift_norm=True,use_checkpoint=True,diffusion_steps=1000,
         noise_schedule="linear",learn_sigma=False,class_cond=False)
model,diff=create_model_and_diffusion(**d)
model.load_state_dict(torch.load("/workspace/sindiff/OUTPUT/sd-long00/model012000.pt",map_location="cpu"))
model.cuda().eval(); model.convert_to_fp16()
def arr(t): return (t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
@torch.no_grad()
def free(seed):
    torch.manual_seed(seed); x=torch.randn(1,3,S,S,device=dev)
    for i in reversed(range(diff.num_timesteps)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=diff.p_sample(model,x,t,clip_denoised=True,model_kwargs={})["sample"]
    return x
def lab(txt,wd):
    im=Image.new("RGB",(wd,22),(255,255,255));dd=ImageDraw.Draw(im)
    try:f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:f=ImageFont.load_default()
    dd.text((6,3),txt,fill=(30,30,30),font=f);return np.asarray(im)
# held-out silhouette
P,M=C["ev_planes"],C["ev_mvp"]; k=0
n=torch.as_tensor((P[k,:3]/np.linalg.norm(P[k,:3])).astype(np.float32),device=dev)
mvp=torch.as_tensor(M[k].copy(),dtype=torch.float32,device=dev).contiguous()
with torch.no_grad():
    img,_,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S)
    _,af,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S,exterior=True)
mask=(af[:1]>0).float()[None]
white=torch.ones(1,3,S,S,device=dev)
cells=[arr(img)]
for s in range(3):
    f=free(s)
    crop = mask*(f*0.5+0.5) + (1-mask)*white
    cells.append(arr(crop[0]))
row=np.concatenate(cells,1)
Image.fromarray(np.concatenate([lab("render | free-gen cropped to held-out silhouette x3",row.shape[1]),row],0)).save("/workspace/freecrop.png")
print("done")
