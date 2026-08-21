"""For one family: the outside seen from the cut's own camera, the cut, and the photograph.

    TAG=ov2 python famcorr.py rh|rv OUT.png

Three rows that answer one question each. The first is what the training's camera for this family
sees of the object with no plane at all -- so the direction the cut is taken from is visible as a
view of the object rather than as an axis name. The second is the cut that camera takes. The third
is the photograph that supervises it.

If the first row and the third row are of the same aspect of the object, the family is pointed the
right way; if the outside is a side view and the photograph is a top-down section, it is not.
"""
import os
import re
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr

sys.path.insert(0, "/workspace/ovoxel_native")
import ovcut
import ovnative as ON
import refsel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
TAG = os.environ.get("TAG", "ov2")
RES = 384
fam, out = sys.argv[1], sys.argv[2]
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


def refdir(obj, which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    return os.path.join(FN, m.group(1))


ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device="cuda")
fig, ax = plt.subplots(3, len(OBJS), figsize=(2.3 * len(OBJS), 7.4))
for c, (obj, label) in enumerate(OBJS):
    st = ovcut.load(obj, TAG)
    C = np.load(f"{W}/cams_{obj}.npz")
    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    NH, NV = H_HI - H_LO, len(C["v_mvp"])
    if fam == "rh":
        which, mvp, pl, idx, n = "REF_H", C["h_mvp"], C["h_planes"][H_LO + NH // 2], NH // 2, NH
        ref = refsel.as_array(refsel.solved_photo(refdir(obj, which), idx, n), RES)
    else:
        k = NV // 2
        which, mvp, pl, idx, n = "REF_V", C["v_mvp"][k], C["v_planes"][k], k, NV
        ref = refsel.as_array(refsel.photo(refdir(obj, which), idx, n), RES)
    mt = torch.as_tensor(mvp, dtype=torch.float32, device="cuda")
    nt = torch.as_tensor(pl[:3], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        ex, _ = ON.render_exterior(st, glctx, mt, RES)[:2]
        cut, _, _, _ = ON.render_section(st, glctx, mt, nt, float(pl[3]), RES)
    rows = [ex.permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
            cut.permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
            np.asarray(ref, np.float32)]
    for r, a in enumerate(rows):
        ax[r, c].imshow(np.clip(crop(a), 0, 1))
        ax[r, c].set_axis_off()
    ax[0, c].set_title(label, fontsize=10.5)

for r, name in enumerate(("the outside, from this cut's camera",
                          "the cut that camera takes",
                          "the photograph supervising it")):
    ax[r, 0].text(-0.05, 0.5, name, rotation=90, va="center", ha="right", fontsize=8.2,
                  transform=ax[r, 0].transAxes)
fig.suptitle("transverse family" if fam == "rh" else "longitudinal family", fontsize=11.5)
fig.subplots_adjust(left=0.036, right=0.996, top=0.93, bottom=0.005, wspace=0.03, hspace=0.03)
fig.savefig(out, dpi=170, facecolor="white")
print("  ->", out)
