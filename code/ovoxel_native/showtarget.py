"""What the loss actually compares the render with.

The target is not the photograph. `section_match.section_target` starts from the render, finds the
connected components of the cut face, and warps the reference into each of them -- so what the
pixel loss sees is the photograph after it has been fitted to the shape the renderer produced. If
that fitting distorts the reference, every conclusion about the loss is about the distorted version
rather than the photograph.

Three columns per plane: the render, the photograph as it arrives, and the target the loss is
given.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import ovnative as ON
import anchor
import refsel
import section_match as sm
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "watermelon_sp")
RUN = os.environ.get("ST_RUN", f"s_rs_{OBJ}")
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
RES = int(os.environ.get("RES", "384"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
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

conf = open(f"{OBJDIR}/{OBJ}.conf").read()


def spec(k):
    return [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)][0]


vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
NV = len(vp)
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH = H_HI - H_LO

rows = []
for fam in ("v", "h"):
    for k in (0, NV // 3 if fam == "v" else NH // 3):
        if fam == "v":
            mv, n, d = vmvp[k], torch.as_tensor(vp[k, :3], dtype=torch.float32, device=dev), vp[k, 3]
            ref = refsel.as_array(refsel.photo(f"{FN}/{spec('REF_V=')}", k, NV), RES)
        else:
            mv, n, d = hmvp, hn, hd[H_LO + k]
            ref = refsel.as_array(refsel.solved_photo(f"{FN}/{spec('REF_H=')}", k, NH), RES)
        with torch.no_grad():
            img, al, _, _ = ON.render_section(st, glctx, mv, n, float(d), RES)
            os.environ["SEC_MAP"] = "ray"
            t_ray = sm.section_target(img, ref, alpha=al)
            os.environ["SEC_MAP"] = "affine"
            t_aff = sm.section_target(img, ref, alpha=al)
            os.environ["SEC_MAP"] = "similarity"
            t_sim = sm.section_target(img, ref, alpha=al)
            os.environ["SEC_MAP"] = "ray"
        trio = [torch.as_tensor(ref, device=dev).permute(2, 0, 1), t_ray, t_aff, t_sim, img]
        rows.append((fam, torch.cat([t.clamp(0, 1) for t in trio], -1)))

sheet = torch.cat([r for _, r in rows], -2).permute(1, 2, 0)
a = (sheet.cpu().numpy() * 255).astype(np.uint8)
img = Image.fromarray(a)
dr_ = ImageDraw.Draw(img)
for i, (fam, _) in enumerate(rows):
    for j, tag in enumerate(("the PHOTOGRAPH", "TARGET per-ray (current)", "TARGET affine (new)",
                             "TARGET similarity", "our RENDER")):
        x, y = j * RES + 8, i * RES + 6
        t = f"{tag}  ({'longitudinal' if fam == 'v' else 'transverse'})"
        dr_.rectangle([x - 4, y - 2, x + 6 * len(t), y + 16], fill=(255, 255, 255))
        dr_.text((x, y), t, fill=(150, 30, 30) if j == 2 else (20, 20, 120))
img.save(f"{W}/target_{OBJ}.jpg", quality=95)
print(f"SHEET target_{OBJ}.jpg  (render | mask used | target | mask of the cut alone | the "
      f"target that would give)")
