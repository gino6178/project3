"""A sheet to choose each family's orientation from, by eye.

    TAG=ov2 python flippick.py rh|rv OUT.png

The automatic search can only decide this where the reference and the shell already agree in
shape, which is not where it is needed, so the answer is set by hand in objects/<obj>.conf as
REF_H_FLIP / REF_V_FLIP. This is what to look at while setting it.

The left column is the object's own cut at the middle supervised plane -- its shell is pinned, so
that outline and the rind inside it are the object's, not something being learned. The four to the
right are the reference under each setting. Pick the one whose top is the object's top.
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
import refalign
import refsel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
TAG = os.environ.get("TAG", "ov2")
RES = 512
fam, out = sys.argv[1], sys.argv[2]
OBJS = [("orange_sp", "orange"), ("watermelon_sp", "watermelon"), ("apple1_sp", "apple"),
        ("bread_sp", "loaf"), ("cake2_sp", "cake"), ("pomegranate2_sp", "pomegranate"),
        ("doughnut", "doughnut")]
NAMES = ["none", "ud", "lr", "rot180"]


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
fig, ax = plt.subplots(len(OBJS), 5, figsize=(11.0, 2.05 * len(OBJS)))
for r, (obj, label) in enumerate(OBJS):
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
    nn = torch.as_tensor(pl[:3], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        img, _, _, _ = ON.render_section(
            st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"),
            nn, float(pl[3]), RES)
    ax[r, 0].imshow(np.clip(crop(img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()), 0, 1))
    ax[r, 0].set_ylabel(label, fontsize=10)
    ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    for sp in ax[r, 0].spines.values():
        sp.set_visible(False)
    for c, nm in enumerate(NAMES):
        ax[r, c + 1].imshow(np.clip(crop(refalign.BY_NAME[nm](np.asarray(ref, np.float32))), 0, 1))
        ax[r, c + 1].set_axis_off()
        if r == 0:
            ax[r, c + 1].set_title(nm, fontsize=11)
ax[0, 0].set_title("the object's own cut", fontsize=11)
fig.suptitle(f"{'transverse (REF_H_FLIP)' if fam == 'rh' else 'longitudinal (REF_V_FLIP)'}"
             f" — pick the column whose top is the object's top", fontsize=11.5)
fig.subplots_adjust(left=0.055, right=0.996, top=0.955, bottom=0.005, wspace=0.03, hspace=0.05)
fig.savefig(out, dpi=160, facecolor="white")
print("  ->", out)
