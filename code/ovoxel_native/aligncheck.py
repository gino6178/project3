"""Does the target's structure land where the object's own shell says it should?

    TAG=ov2 python aligncheck.py OUT.png

`section_target` maps a reference onto the render's silhouette by a polar map -- per-ray inner and
outer radius on both sides, then a monotone reparametrisation by the colour path so the layers of
one land on the layers of the other. Whether that succeeds is not visible in the loss, which is
computed after the map and therefore grades the map's own output.

This draws the target's material boundaries over the render. The render's shell is not in question:
the skin cells are pinned from the released model, so its rind is where the object's rind is. If
the target's boundaries sit on the render's own, the reference is aligned to the shell; if they sit
inside or outside it, everything the reference says about a layer is being written at the wrong
radius.
"""
import os
import re
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr
from scipy import ndimage

sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/rebuild/project3/code/src")
import ovcut
import ovnative as ON
import refsel
import section_match as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
TAG = os.environ.get("TAG", "ov2")
RES = 512
OBJS = [("orange_sp", "orange"), ("watermelon_sp", "watermelon"), ("apple1_sp", "apple"),
        ("bread_sp", "loaf"), ("cake2_sp", "cake"), ("pomegranate2_sp", "pomegranate"),
        ("doughnut", "doughnut")]


def refdir(obj, which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    return os.path.join(FN, m.group(1))


def edges(a, mask, q=88):
    """The strongest colour boundaries inside the mask, as a thin binary map."""
    g = np.zeros(a.shape[:2])
    for k in range(3):
        gy, gx = np.gradient(ndimage.gaussian_filter(a[:, :, k], 1.2))
        g += np.hypot(gy, gx)
    inner = ndimage.binary_erosion(mask, np.ones((5, 5)))
    if inner.sum() < 200:
        return np.zeros_like(mask)
    return (g > np.percentile(g[inner], q)) & inner


ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device="cuda")
fig, ax = plt.subplots(2, len(OBJS), figsize=(2.15 * len(OBJS), 5.0))
for c, (obj, label) in enumerate(OBJS):
    st = ovcut.load(obj, TAG)
    C = np.load(f"{W}/cams_{obj}.npz")
    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    NH, NV = H_HI - H_LO, len(C["v_mvp"])
    for r, (fam, which, mvp, pl, idx, n) in enumerate((
            ("rh", "REF_H", C["h_mvp"], C["h_planes"][H_LO + NH // 2], NH // 2, NH),
            ("rv", "REF_V", C["v_mvp"][NV // 2], C["v_planes"][NV // 2], NV // 2, NV))):
        ref = refsel.as_array(
            (refsel.solved_photo if fam == "rh" else refsel.photo)(refdir(obj, which), idx, n),
            RES)
        nn = torch.as_tensor(pl[:3], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            img, al, _, _ = ON.render_section(
                st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"),
                nn, float(pl[3]), RES)
            tgt = sm.section_target(img, ref, alpha=al)
        R = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        T = tgt.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        m = R.min(2) < 0.97
        over = R.copy()
        e = edges(T, m)
        over[e] = [0.05, 0.35, 0.85]                      # the target's boundaries, in blue
        ax[r, c].imshow(np.clip(over, 0, 1))
        ax[r, c].set_axis_off()
    ax[0, c].set_title(label, fontsize=10.5)

for r, name in enumerate(("transverse", "longitudinal")):
    ax[r, 0].text(-0.05, 0.5, name, rotation=90, va="center", ha="right", fontsize=9,
                  transform=ax[r, 0].transAxes)
fig.suptitle("the target's material boundaries (blue) over the render they supervise",
             fontsize=11)
fig.subplots_adjust(left=0.032, right=0.996, top=0.9, bottom=0.005, wspace=0.03, hspace=0.03)
fig.savefig(sys.argv[1], dpi=170, facecolor="white")
print("  ->", sys.argv[1])
