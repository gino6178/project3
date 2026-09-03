"""Generate a target for every plane (both families): photo matched to the O-Voxel shell, then SDEdit.

For each plane the O-Voxel shell silhouette is rendered, the family's reference photo is affine-warped
so its silhouette fills that shell, composited, and SDEdit (s=0.3) harmonises it. The columella,
membranes and peel come from a real photo placed in the correct shape. Saved for the 3-D lift.
"""
import os, sys, time
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W="/workspace/ovoxel_native"; OBJ="orange_sp"; dev="cuda"; S=256
STR=float(os.environ.get("STR","0.3"))
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
def load(ckpt):
    d=model_and_diffusion_defaults()
    d.update(image_size=256,num_channels=64,num_head_channels=16,channel_mult="1,2,4",
             attention_resolutions="2",num_res_blocks=1,resblock_updown=False,use_fp16=True,
             use_scale_shift_norm=True,use_checkpoint=True,diffusion_steps=1000,
             noise_schedule="linear",learn_sigma=False,class_cond=False)
    m,D=create_model_and_diffusion(**d); m.load_state_dict(torch.load(ckpt,map_location="cpu"))
    m.cuda().eval(); m.convert_to_fp16(); return m,D
mL,diff=load("/workspace/sindiff/OUTPUT/sd-long00/model012000.pt")
mT,_=load("/workspace/sindiff/OUTPUT/sd-orange_h/model006000.pt")
def loadphoto(pth):
    a=np.asarray(Image.open(pth).convert("RGB").resize((S,S))).astype(np.float32)/255
    return torch.from_numpy(a).permute(2,0,1).to(dev)
photoL=loadphoto("/workspace/rebuild/worktree/spl_orange_v/or_long_00.png")
photoT=loadphoto("/workspace/rebuild/worktree/spl_orange_h/or_trans_00.png")
def bbox(m):
    ys,xs=torch.where(m); return xs.min().item(),ys.min().item(),xs.max().item(),ys.max().item()
def warp(photo, psil, rmask):
    px0,py0,px1,py1=bbox(psil); rx0,ry0,rx1,ry1=bbox(rmask[0,0]>0.5)
    sx=(rx1-rx0)/max(px1-px0,1); sy=(ry1-ry0)/max(py1-py0,1)
    ys,xs=torch.meshgrid(torch.arange(S,device=dev),torch.arange(S,device=dev),indexing="ij")
    src_x=(xs-(rx0+rx1)/2)/sx+(px0+px1)/2; src_y=(ys-(ry0+ry1)/2)/sy+(py0+py1)/2
    grid=torch.stack([src_x/(S-1)*2-1,src_y/(S-1)*2-1],-1)[None]
    return F.grid_sample(photo[None],grid,align_corners=True,padding_mode="border")[0]
psilL=(photoL.min(0).values<0.92); psilT=(photoT.min(0).values<0.92)
@torch.no_grad()
def sdedit(x0,mask,strength):
    ts=int(strength*diff.num_timesteps)-1
    x=diff.q_sample(x0,torch.full((1,),ts,device=dev,dtype=torch.long))
    for i in reversed(range(ts+1)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=mask*x+(1-mask)*diff.q_sample(x0,t)
        x=diff.p_sample(model_cur,x,t,clip_denoised=True,model_kwargs={})["sample"]
    return mask*x+(1-mask)*x0
white=torch.ones(1,3,S,S,device=dev)

HL,HH=int(C["h_lo"][0]),int(C["h_hi"][0])
groups=[("long", C["v_planes"], C["v_mvp"], mL, photoL, psilL),
        ("long_h", C["ev_planes"], C["ev_mvp"], mL, photoL, psilL),
        ("trans", C["h_planes"][HL:HH], np.broadcast_to(C["h_mvp"][None],(HH-HL,4,4)), mT, photoT, psilT),
        ("trans_h", C["eh_planes"], C["eh_mvp"], mT, photoT, psilT)]
targets=[]
t0=time.time()
for name,P,M,model_cur,photo,psil in groups:
    for k in range(len(P)):
        n=torch.as_tensor((P[k,:3]/np.linalg.norm(P[k,:3])).astype(np.float32),device=dev)
        mvp=torch.as_tensor((M[k] if M.ndim==3 else M).copy(),dtype=torch.float32,device=dev).contiguous()
        with torch.no_grad():
            _,af,_,_=ON.render_section(st,glctx,mvp,n,float(P[k,3]),S,exterior=True)
        mask=(af[:1]>0).float()[None]
        if float(mask.sum())<100: continue
        warped=warp(photo,psil,mask)
        matched=mask*warped[None]+(1-mask)*white
        out=sdedit(matched*2-1,mask,STR)
        targets.append(dict(name=name,k=k,n=n.cpu(),d=float(P[k,3]),
                            mvp=mvp.cpu(),tgt=out.cpu().half(),mask=mask.cpu().bool()))
    print(f"  {name}: done ({time.time()-t0:.0f}s)",flush=True)
torch.save(targets, f"{W}/targets_{OBJ}.pt")
print(f"saved {len(targets)} targets, {time.time()-t0:.0f}s")
