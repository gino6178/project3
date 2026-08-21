"""Each object's exterior stood on each of the six axes, to see which one is really its top.

    TAG=ov2 python sixup.py OUT.png

The turntable's arrows and the section renders have to be in the same frame or the answers read off
one do not apply to the other, and two of the answers disagree with what the sections show. This
does not ask which arrow points where; it stands the object on each axis in turn and lets the
object say. Whichever column looks upright is the top, and it is measured in the same lattice frame
`h_planes` is stated in, so it can be compared with the axis the training cuts along.
"""
import os
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr

sys.path.insert(0, "/workspace/ovoxel_native")
import axisviews as AV
import ovcut
import ovnative as ON

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
TAG = os.environ.get("TAG", "ov2")
RES = 320
UPS = [("+x", [1, 0, 0]), ("-x", [-1, 0, 0]), ("+y", [0, 1, 0]),
       ("-y", [0, -1, 0]), ("+z", [0, 0, 1]), ("-z", [0, 0, -1])]
OBJS = [("orange_sp", "orange"), ("watermelon_sp", "watermelon"), ("apple1_sp", "apple"),
        ("bread_sp", "loaf"), ("cake2_sp", "cake"), ("pomegranate2_sp", "pomegranate"),
        ("doughnut", "doughnut")]


def crop(a, pad=0.05):
    a = np.asarray(a, np.float32)
    m = a.min(2) < 0.97
    if m.sum() < 100:
        return a
    ys, xs = np.where(m)
    s = int(pad * max(ys.max() - ys.min(), xs.max() - xs.min()))
    return a[max(ys.min() - s, 0):ys.max() + s, max(xs.min() - s, 0):xs.max() + s]


ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device="cuda")
fig, ax = plt.subplots(len(OBJS), len(UPS), figsize=(2.0 * len(UPS), 2.0 * len(OBJS)))
for r, (obj, label) in enumerate(OBJS):
    st = ovcut.load(obj, TAG)
    hc = float(st["hc"])
    org = np.asarray(st["org"], np.float64)
    lo = st["solid"].min(0).values.cpu().numpy() * hc + org
    hi = (st["solid"].max(0).values.cpu().numpy() + 1) * hc + org
    cen, rad = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
    for c, (nm, u) in enumerate(UPS):
        up = np.asarray(u, float)
        side = np.array([1.0, 0.0, 0.0])
        if abs(side @ up) > 0.9:
            side = np.array([0.0, 1.0, 0.0])
        side = side - (side @ up) * up
        side /= np.linalg.norm(side)
        mvp = AV.look_at(cen + side * rad * 3.2, cen, up, 38.0)
        with torch.no_grad():
            img, _ = ON.render_exterior(
                st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"), RES)[:2]
        ax[r, c].imshow(np.clip(crop(img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()), 0, 1))
        ax[r, c].set_axis_off()
        if r == 0:
            ax[r, c].set_title(f"{nm} is up", fontsize=10.5)
    ax[r, 0].text(-0.05, 0.5, label, rotation=90, va="center", ha="right", fontsize=9.5,
                  transform=ax[r, 0].transAxes)
fig.subplots_adjust(left=0.032, right=0.996, top=0.965, bottom=0.004, wspace=0.03, hspace=0.03)
fig.savefig(sys.argv[1], dpi=160, facecolor="white")
print("  ->", sys.argv[1])
