"""Supervised vs held-out cross-section faces, O-Voxel Stage-1, per family.

The held-out planes are the ones with no photograph. This lays the supervised faces beside the
held-out faces of the same family so the quality gap the diffusion is meant to close is visible.
"""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/workspace/ovoxel_native")
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W = "/workspace/ovoxel_native"; OBJ = "orange_sp"; dev = "cuda"; S = 200
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
def arr(t): return (t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
@torch.no_grad()
def face(pl, mv):
    n = torch.as_tensor((pl[:3]/np.linalg.norm(pl[:3])).astype(np.float32), device=dev)
    mvp = torch.as_tensor(mv.copy(), dtype=torch.float32, device=dev).contiguous()
    img,_,_,_ = ON.render_section(st, glctx, mvp, n, float(pl[3]), S, exterior=True)
    return arr(img)
def label(txt, wd):
    im = Image.new("RGB",(wd,22),(255,255,255)); d=ImageDraw.Draw(im)
    try: f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",15)
    except: f=ImageFont.load_default()
    d.text((6,3),txt,fill=(30,30,30),font=f); return np.asarray(im)
blocks=[]
for name, sup_P, sup_M, hld_P, hld_M in (
    ("LONGITUDINAL", C["v_planes"], C["v_mvp"], C["ev_planes"], C["ev_mvp"]),
    ("TRANSVERSE", C["h_planes"][int(C["h_lo"][0]):int(C["h_hi"][0])],
     np.broadcast_to(C["h_mvp"][None], (int(C["h_hi"][0])-int(C["h_lo"][0]),4,4)),
     C["eh_planes"], C["eh_mvp"])):
    sk = np.linspace(0,len(sup_P)-1,min(6,len(sup_P))).astype(int)
    sup = np.concatenate([face(sup_P[k], sup_M[k] if sup_M.ndim==3 else sup_M) for k in sk],1)
    hld = np.concatenate([face(hld_P[k], hld_M[k]) for k in range(len(hld_P))],1)
    wd = max(sup.shape[1], hld.shape[1])
    def pad(a): return np.pad(a,((0,0),(0,wd-a.shape[1]),(0,0)),constant_values=255)
    blocks += [label(f"{name}  supervised (has photo)", wd), pad(sup),
               label(f"{name}  held-out (no photo)", wd), pad(hld),
               np.full((14,wd,3),255,np.uint8)]
Image.fromarray(np.concatenate(blocks,0)).save("/workspace/hocmp.png")
print("supervised vs held-out, both families")
