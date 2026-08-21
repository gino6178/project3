"""Each object stood up the way Gino says it stands, beside the two cuts it is being given.

    TAG=ov2 python upright.py OUT.png

The exterior is rendered with the stated top as screen up, from an equatorial direction, so every
object is the right way up and the cuts beneath it can be read against it. The two cuts are the
ones the training actually supervises, unchanged.

A transverse cut should be perpendicular to the object's own axis and a longitudinal one should
contain it. Where the axis the pipeline cuts along is not the object's, neither is, and the panel
says so rather than leaving it to be noticed.
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
RES = 384
AXV = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
       "+z": [0, 0, 1], "-z": [0, 0, -1]}
SAID = {"orange_sp": "-y", "watermelon_sp": "+y", "apple1_sp": "+z", "bread_sp": "-y",
        "cake2_sp": "-z", "pomegranate2_sp": "-y", "doughnut": "+z"}
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
fig, ax = plt.subplots(3, len(OBJS), figsize=(2.3 * len(OBJS), 7.6))
for c, (obj, label) in enumerate(OBJS):
    st = ovcut.load(obj, TAG)
    C = np.load(f"{W}/cams_{obj}.npz")
    hc = float(st["hc"])
    org = np.asarray(st["org"], np.float64)
    lo = st["solid"].min(0).values.cpu().numpy() * hc + org
    hi = (st["solid"].max(0).values.cpu().numpy() + 1) * hc + org
    cen, rad = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2

    up = np.asarray(AXV[SAID[obj]], float)
    side = np.array([1.0, 0.0, 0.0])
    if abs(side @ up) > 0.9:
        side = np.array([0.0, 1.0, 0.0])
    side = side - (side @ up) * up
    side /= np.linalg.norm(side)
    mvp = AV.look_at(cen + side * rad * 3.2, cen, up, 38.0)
    with torch.no_grad():
        img, _ = ON.render_exterior(
            st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"), RES)[:2]
    ext = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    NH, NV = H_HI - H_LO, len(C["v_mvp"])
    cuts = []
    for mvp2, pl in ((C["h_mvp"], C["h_planes"][H_LO + NH // 2]),
                     (C["v_mvp"][NV // 2], C["v_planes"][NV // 2])):
        n = torch.as_tensor(pl[:3], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            im, _, _, _ = ON.render_section(
                st, glctx, torch.as_tensor(mvp2, dtype=torch.float32, device="cuda"),
                n, float(pl[3]), RES)
        cuts.append(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy())

    nrm = np.asarray(C["h_planes"][0, :3], float)
    ang = np.degrees(np.arccos(np.clip(abs(float(nrm @ up / np.linalg.norm(nrm))), 0, 1)))
    for r, a in enumerate((ext, cuts[0], cuts[1])):
        ax[r, c].imshow(np.clip(crop(a), 0, 1))
        ax[r, c].set_axis_off()
    ax[0, c].set_title(f"{label}\ntop {SAID[obj]}", fontsize=10)
    if ang > 60:
        for r in (1, 2):
            ax[r, c].set_title(f"cut along {'-y' if abs(nrm[1]) > 0.5 else '?'}, "
                               f"{ang:.0f}° from the axis", fontsize=8.4, color="#a8412a")

for r, name in enumerate(("stood up as stated", "called transverse", "called longitudinal")):
    ax[r, 0].text(-0.05, 0.5, name, rotation=90, va="center", ha="right", fontsize=9,
                  transform=ax[r, 0].transAxes)
fig.subplots_adjust(left=0.036, right=0.996, top=0.9, bottom=0.005, wspace=0.03, hspace=0.09)
fig.savefig(sys.argv[1], dpi=170, facecolor="white")
print("  ->", sys.argv[1])
