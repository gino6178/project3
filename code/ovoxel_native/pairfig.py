"""The 2x2 under each object's loop: supervised cuts above, held-out cuts below.

    OBJ=orange_sp RUN=s_v2_orange_sp python pairfig.py

The four planes are taken from the same lists `scorefull.py` scores, and by the same names, so the
picture and the table are statements about the same cuts. Top row is a plane a photograph existed
for, bottom row is one from the held-out half that nothing looked at during training; transverse
left, longitudinal right. The middle of each list is used, so nothing is chosen by hand per object.

The generator that made the first version of these was never committed, and when the baseline moved
there was nothing to re-run. This one is.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJDIR = "/workspace/rebuild/project3/code/objects"
OBJ = os.environ.get("OBJ", "orange_sp")
RUN = os.environ.get("RUN", f"s_v2_{OBJ}")
CS = os.environ.get("CAMS_SUFFIX", "_v2")
RES = int(os.environ.get("RES", "512"))
OUT = os.environ.get("OUT", f"{W}/pair_{OBJ}.jpg")
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}{CS}.npz")
p = torch.load(f"{W}/{RUN}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
if "dec_i" in p:
    w = p["dec_i"]["stage1.0.weight"].shape[0]
    n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
    anchor.W_HID, anchor.N_HID = w, n
    di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
    di.load_state_dict(p["dec_i"])
    ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()

ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])


def mid(planes):
    return planes[len(planes) // 2]


sup_h = [(T(C["h_mvp"]), T(C["h_planes"][0, :3]), C["h_planes"][H_LO + i, 3])
         for i in range(H_HI - H_LO)]
sup_v = [(T(C["v_mvp"][j]), T(C["v_planes"][j, :3]), C["v_planes"][j, 3])
         for j in range(len(C["v_planes"]))]
hld_h = [(T(C["eh_mvp"][i]), T(C["eh_planes"][i, :3]), C["eh_planes"][i, 3])
         for i in range(len(C["eh_planes"]))]
hld_v = [(T(C["ev_mvp"][i]), T(C["ev_planes"][i, :3]), C["ev_planes"][i, 3])
         for i in range(len(C["ev_planes"]))]

quad = []
for name, pl in (("shown h", sup_h), ("shown v", sup_v), ("unseen h", hld_h), ("unseen v", hld_v)):
    if not pl:
        quad.append(np.ones((RES, RES, 3), np.uint8) * 255)
        print(f"  {name}: none")
        continue
    mvp, nn, dd = mid(pl)
    with torch.no_grad():
        img, _, _, _ = ON.render_section(st, glctx, mvp, nn, float(dd), RES)
    quad.append((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    print(f"  {name}: {len(pl)} planes, drew the middle one")

sheet = np.concatenate([np.concatenate(quad[:2], 1), np.concatenate(quad[2:], 1)], 0)
Image.fromarray(sheet).save(OUT, quality=93)
print(f"{OBJ}: {OUT}  {sheet.shape[1]}x{sheet.shape[0]}  from {RUN} on cams{CS}")
