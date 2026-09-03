"""Every cross-section (supervised + held-out, both families) through the 2-D SinDiffusion.

Each plane is rendered from the O-Voxel field, its own silhouette taken as the mask (the growth
region), and SDEdit run under that family's model with everything outside the mask pinned to the
render at each step. Shows O-Voxel render | SinDiffusion output, per plane, grouped by family and
by supervised/held-out. This is the 2-D capability that a 3-D lift would then have to make
consistent across families.
"""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W = "/workspace/ovoxel_native"; OBJ = "orange_sp"; dev = "cuda"; S = 200
STR = float(os.environ.get("STR", "0.5"))
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad():
    st["interior"], st["surf_rgb"] = di(), dsr()

def load(ckpt):
    d = model_and_diffusion_defaults()
    d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
             attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=True,
             use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
             noise_schedule="linear", learn_sigma=False, class_cond=False)
    m, D = create_model_and_diffusion(**d); m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.cuda().eval(); m.convert_to_fp16(); return m, D
mL, diff = load("/workspace/sindiff/OUTPUT/sd-vlong/model006000.pt")
mT, _ = load("/workspace/sindiff/OUTPUT/sd-vtrans/model006000.pt")

@torch.no_grad()
def sdedit(m, x0, mask):
    ts = int(STR*diff.num_timesteps)-1
    x = diff.q_sample(x0, torch.full((len(x0),), ts, device=dev, dtype=torch.long))
    for i in reversed(range(ts+1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask*x + (1-mask)*diff.q_sample(x0, t)
        x = diff.p_sample(m, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return (mask*x + (1-mask)*x0).clamp(-1,1)
def arr(t): return (t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
def label(txt, wd):
    im=Image.new("RGB",(wd,22),(255,255,255)); d=ImageDraw.Draw(im)
    try:f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:f=ImageFont.load_default()
    d.text((6,3),txt,fill=(30,30,30),font=f); return np.asarray(im)

@torch.no_grad()
def pair(pl, mv, model):
    n = torch.as_tensor((pl[:3]/np.linalg.norm(pl[:3])).astype(np.float32), device=dev)
    mvp = torch.as_tensor((mv if mv.ndim==2 else mv).copy(), dtype=torch.float32, device=dev).contiguous()
    img,_,_,_ = ON.render_section(st, glctx, mvp, n, float(pl[3]), S)
    _,af,_,_ = ON.render_section(st, glctx, mvp, n, float(pl[3]), S, exterior=False)
    mask = (af[:1]>0).float()[None]
    out = sdedit(model, img[None]*2-1, mask)
    return np.concatenate([arr(img), np.full((S,4,3),255,np.uint8), arr(out[0]*0.5+0.5)],1)

HL,HH=int(C["h_lo"][0]),int(C["h_hi"][0])
groups=[("LONGITUDINAL supervised", C["v_planes"], C["v_mvp"], mL, True),
        ("LONGITUDINAL held-out",    C["ev_planes"], C["ev_mvp"], mL, False),
        ("TRANSVERSE supervised",    C["h_planes"][HL:HH], np.broadcast_to(C["h_mvp"][None],(HH-HL,4,4)), mT, True),
        ("TRANSVERSE held-out",      C["eh_planes"], C["eh_mvp"], mT, False)]
blocks=[]
for name, P, M, model, sup in groups:
    ks = np.linspace(0,len(P)-1,min(6,len(P))).astype(int)
    row = np.concatenate([pair(P[k], M[k] if M.ndim==3 else M, model) for k in ks],1)
    blocks += [label(f"{name}  (render | SinDiffusion, s={STR})", row.shape[1]), row,
               np.full((12,row.shape[1],3),255,np.uint8)]
Image.fromarray(np.concatenate(blocks,0)).save("/workspace/allsd.png")
print("all slices through SinDiffusion")
