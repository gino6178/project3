"""Every object's held-out cuts beside the photographs they are scored against.

The planes here are the ones no photograph reached during training, taken from the same lists
`scorefull.py` scores, and the photograph beside each is the held-out image the number in Table 2
is a distance to. Four columns: the unseen transverse cut, the photograph it is scored against, the
unseen longitudinal cut, and its photograph. Nothing is chosen by hand -- the middle of each list.

This replaces the figure inherited from the splatted pipeline, which showed the same planes drawn
by a different renderer.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import ovnative as ON
import anchor
import refsel
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

W = os.path.dirname(os.path.abspath(__file__))
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
OBJS = os.environ.get("H7_OBJS", "watermelon_sp,orange_sp,apple1_sp,bread_sp,cake2_sp,"
                                 "pomegranate2_sp,doughnut").split(",")
RES = int(os.environ.get("RES", "300"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
rows, names = [], []

for OBJ in OBJS:
    try:
        st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
        conf = open(f"{OBJDIR}/{OBJ}.conf").read()
        C = np.load(f"{W}/cams_{OBJ}_v2.npz")
        p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
    except Exception as e:
        print(f"{OBJ}: skipped ({type(e).__name__}: {e})"); continue
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

    def spec(k):
        # an object with one photograph in a family has no held-out half and names no eval set
        v = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)]
        return v[0] if v else None

    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    panels = []
    for fam, key in (("h", "EVAL_REF="), ("v", "EVAL_REF_V=")):
        mvp = C["eh_mvp"] if fam == "h" else C["ev_mvp"]
        pl = C["eh_planes"] if fam == "h" else C["ev_planes"]
        if len(pl) == 0:
            panels += [torch.ones(3, RES, RES, device=dev)] * 2
            continue
        k = len(pl) // 2
        with torch.no_grad():
            img, _, _, _ = ON.render_section(st, glctx, T(mvp[k]), T(pl[k, :3]), float(pl[k, 3]),
                                             RES)
        panels.append(img.clamp(0, 1))
        sp = spec(key)
        files = sorted(refsel.photos_in(f"{FN}/{sp}")) if sp else []
        ref = (refsel.as_array(refsel.photo(f"{FN}/{sp}", k, max(len(pl), 1)), RES) if files
               else np.ones((RES, RES, 3), np.float32))
        panels.append(torch.as_tensor(np.asarray(ref), device=dev).permute(2, 0, 1).float())
    def fit(t, frac=0.86):
        """Render and photograph at one size. The render arrives at whatever fraction of the frame
        the object's camera gives it and the photograph fills its own; comparing them at those two
        sizes compares framings."""
        m = t.mean(0) < 0.985
        ys, xs = m.nonzero(as_tuple=True)
        if ys.numel() < 16:
            return t
        cy, cx = float(ys.float().mean()), float(xs.float().mean())
        h = max(float(ys.max() - ys.min()), float(xs.max() - xs.min())) / 2 + 1
        k = frac * t.shape[-1] / 2 / h
        g = torch.nn.functional.affine_grid(
            torch.tensor([[[1 / k, 0, (2 * cx / (t.shape[-1] - 1) - 1)],
                           [0, 1 / k, (2 * cy / (t.shape[-2] - 1) - 1)]]], device=t.device),
            (1, 3, t.shape[-2], t.shape[-1]), align_corners=True)
        return torch.nn.functional.grid_sample(t[None] - 1.0, g, align_corners=True)[0] + 1.0

    rows.append(torch.cat([fit(q).clamp(0, 1) for q in panels], -1))
    names.append(OBJ)
    print(f"  {OBJ}: {len(C['eh_planes'])} held-out transverse, {len(C['ev_planes'])} longitudinal")

sheet = torch.cat(rows, -2).permute(1, 2, 0)
im = Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8))
d = ImageDraw.Draw(im)
for i, nm in enumerate(names):
    for j, t in enumerate(("held-out transverse cut", "the photograph it is scored against",
                           "held-out longitudinal cut", "the photograph it is scored against")):
        x, y = j * RES + 5, i * RES + 4
        lab = f"{nm} - {t}" if j == 0 else t
        d.rectangle([x - 3, y - 2, x + 6 * len(lab), y + 13], fill=(255, 255, 255))
        d.text((x, y), lab, fill=(20, 20, 120))
im.save(f"{W}/heldout7.jpg", quality=93)
print(f"SHEET heldout7.jpg  {im.size[0]}x{im.size[1]}  ({len(names)} objects)")
