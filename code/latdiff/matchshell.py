"""Match the reference photo into the O-Voxel shell, then SDEdit.

The photo has real columella, membranes and peel but its own shape; the render has the correct
shell for this plane but a weak interior. Warp the photo's silhouette onto the render's silhouette
so the real interior lands inside the correct shell, then SDEdit lightly to harmonise. The
columella and peel now come from a real photo placed in the right shape, not generated from noise.
"""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torch.nn.functional as F
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

# reference photo and its silhouette
photo = torch.from_numpy(np.asarray(Image.open("/workspace/sindiff/data/long00.png").convert("RGB").resize((S,S))).astype(np.float32)/255).permute(2,0,1).to(dev)
psil = (photo.min(0).values < 0.92)   # non-white
def bbox(m):
    ys,xs=torch.where(m); return xs.min().item(),ys.min().item(),xs.max().item(),ys.max().item()

@torch.no_grad()
def warp_to(rmask):
    """Affine-warp the photo so its silhouette bbox matches the render silhouette bbox."""
    px0,py0,px1,py1=bbox(psil); rx0,ry0,rx1,ry1=bbox(rmask[0,0]>0.5)
    sx=(rx1-rx0)/max(px1-px0,1); sy=(ry1-ry0)/max(py1-py0,1)
    # build affine grid: map render coords -> photo coords
    ys,xs=torch.meshgrid(torch.arange(S,device=dev),torch.arange(S,device=dev),indexing="ij")
    src_x=(xs-(rx0+rx1)/2)/sx+(px0+px1)/2; src_y=(ys-(ry0+ry1)/2)/sy+(py0+py1)/2
    grid=torch.stack([src_x/(S-1)*2-1, src_y/(S-1)*2-1],-1)[None]
    wp=F.grid_sample(photo[None],grid,align_corners=True,padding_mode="border")[0]
    return wp

@torch.no_grad()
def sdedit(x0,mask,strength):
    ts=int(strength*diff.num_timesteps)-1
    x=diff.q_sample(x0,torch.full((1,),ts,device=dev,dtype=torch.long))
    for i in reversed(range(ts+1)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=mask*x+(1-mask)*diff.q_sample(x0,t)
        x=diff.p_sample(model,x,t,clip_denoised=True,model_kwargs={})["sample"]
    return mask*x+(1-mask)*x0
def lab(txt,wd):
    im=Image.new("RGB",(wd,22),(255,255,255));dd=ImageDraw.Draw(im)
    try:f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:f=ImageFont.load_default()
    dd.text((6,3),txt,fill=(30,30,30),font=f);return np.asarray(im)

P,M=C["ev_planes"],C["ev_mvp"]
rows=[]
for k in range(3):
    n=torch.as_tensor((P[k,:3]/np.linalg.norm(P[k,:3])).astype(np.float32),device=dev)
    mvp=torch.as_tensor(M[k].copy(),dtype=torch.float32,device=dev).contiguous()
    with torch.no_grad():
        img,_,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S)
        _,af,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S,exterior=True)
    mask=(af[:1]>0).float()[None]
    warped=warp_to(mask)
    white=torch.ones(1,3,S,S,device=dev)
    matched=mask*warped[None]+(1-mask)*white           # photo interior placed in the shell
    out=sdedit(matched*2-1,mask,0.3)
    row=np.concatenate([arr(img), arr(matched[0]), arr(out[0]*0.5+0.5)],1)
    rows.append(row)
grid=np.concatenate(rows,0)
Image.fromarray(np.concatenate([lab("render | photo matched to shell | +SDEdit s=0.3  (3 held-out planes)",grid.shape[1]),grid],0)).save("/workspace/matchshell.png")
print("done")
