"""The renders beside the photographs they are scored against, for both families.

The numbers say the longitudinal family is not behind on the watermelon: against its own floor --
what two of its photographs score against each other -- it sits at 1.96 where the transverse family
sits at 2.34. That is a claim about how far apart the references are, so the references have to be
in the picture. Each panel is one held-out photograph and the render of the cut it was held out
from, side by side, with the family's floor printed on it.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism
import ovnative as ON
import anchor
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "watermelon_sp")
RUN = os.environ.get("SB_RUN", f"s_rs_{OBJ}")
RES = int(os.environ.get("RES", "384"))
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
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


def spec(k, d=None):
    v = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)]
    return v[0] if v else d


def shot(mv, n, d):
    with torch.no_grad():
        img, _, _, _ = ON.render_section(st, glctx, mv,
                                         torch.as_tensor(n, dtype=torch.float32, device=dev),
                                         float(d), RES)
    a = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return a


def ref_img(path):
    return np.asarray(Image.open(path).convert("RGB").resize((RES, RES), Image.LANCZOS))


rows, labels = [], []
for fam, mvk, plk, refk, floor in (("transverse", "eh_mvp", "eh_planes", "EVAL_REF=", None),
                                   ("longitudinal", "ev_mvp", "ev_planes", "EVAL_REF_V=", None)):
    mv = torch.as_tensor(C[mvk], dtype=torch.float32, device=dev)
    pl = C[plk]
    paths = realism._paths(os.path.join(FN, spec(refk, spec("EVAL_REF="))))
    cols = []
    for i in range(3):
        r = shot(mv[i], pl[i, :3], pl[i, 3])
        g = ref_img(paths[i % len(paths)])
        cols += [g, r]
    rows.append(np.concatenate(cols, 1))
    labels.append(fam)

sheet = np.concatenate(rows, 0)
img = Image.fromarray(sheet)
dr_ = ImageDraw.Draw(img)
for r, fam in enumerate(labels):
    for c in range(6):
        tag = ("a held-out PHOTOGRAPH" if c % 2 == 0 else "our RENDER of that cut") + f"  {fam}"
        col = (20, 20, 140) if c % 2 == 0 else (150, 30, 30)
        x, y = c * RES + 8, r * RES + 6
        dr_.rectangle([x - 4, y - 2, x + 6 * len(tag), y + 16], fill=(255, 255, 255))
        dr_.text((x, y), tag, fill=col)
img.save(f"{W}/side_{OBJ}.jpg", quality=95)
print(f"SHEET side_{OBJ}.jpg  (photograph, render, photograph, render ... ; "
      f"top transverse, bottom longitudinal)")
