"""Clean O-Voxel Stage-1 cross-section faces across depth, both families -- just the O-Voxel."""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
from PIL import Image
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
def face(planes, mvps, k):
    n = torch.as_tensor((planes[k,:3]/np.linalg.norm(planes[k,:3])).astype(np.float32), device=dev)
    mvp = torch.as_tensor((mvps[k] if mvps.ndim==3 else mvps).copy(), dtype=torch.float32, device=dev).contiguous()
    img,_,_,_ = ON.render_section(st, glctx, mvp, n, float(planes[k,3]), S, exterior=True)
    return arr(img)
HL,HH = int(C["h_lo"][0]), int(C["h_hi"][0])
vP,vM = C["v_planes"], C["v_mvp"]
hP = C["h_planes"][HL:HH]; hM = np.broadcast_to(C["h_mvp"][None],(HH-HL,4,4))
# 6 longitudinal across full range, 6 transverse
vk = np.linspace(0, len(vP)-1, 6).astype(int)
hk = np.linspace(0, len(hP)-1, 6).astype(int)
long_row = np.concatenate([face(vP,vM,int(k)) for k in vk], 1)
trans_row = np.concatenate([face(hP,hM,int(k)) for k in hk], 1)
gap = np.full((10, long_row.shape[1], 3), 255, np.uint8)
Image.fromarray(np.concatenate([long_row, gap, trans_row], 0)).save("/workspace/ovslices.png")
print("top=6 longitudinal depths, bottom=6 transverse depths")
