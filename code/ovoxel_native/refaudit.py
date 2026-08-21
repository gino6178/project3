"""Is each object's reference set a section OF that object?

    python refaudit.py OUT.png

`section_target` maps the whole of a reference's disc onto the render's silhouette, so anything in
the photograph that is not the cut face -- a frosting side, a crown, the outside of the fruit seen
past the cut -- is written into the interior. That is invisible in the loss and visible only by
putting the object's own exterior next to the photographs that are supposed to be sections of it.

Row 1 is the object's exterior, rendered from the dual grid with no plane at all. Rows 2 and 3 are
the first raw file of each reference family, as it sits on disk, before any blending or mapping.
"""
import glob
import os
import re
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image

sys.path.insert(0, "/workspace/ovoxel_native")
import ovcut
import ovnative as ON

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
TAG = os.environ.get("TAG", "ov2")
SKIP = ("_depth", "_mask", "_alpha", "_normal")
OBJS = [("orange_sp", "orange"), ("watermelon_sp", "watermelon"), ("apple1_sp", "apple"),
        ("bread_sp", "loaf"), ("cake2_sp", "cake"), ("pomegranate2_sp", "pomegranate"),
        ("doughnut", "doughnut")]


def crop(a, pad=0.05):
    m = a.min(2) < 0.97
    if m.sum() < 100:
        return a
    ys, xs = np.where(m)
    s = int(pad * max(ys.max() - ys.min(), xs.max() - xs.min()))
    return a[max(ys.min() - s, 0):ys.max() + s, max(xs.min() - s, 0):xs.max() + s]


def first_ref(obj, which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    fs = [f for f in sorted(glob.glob(os.path.join(FN, m.group(1), "*")))
          if f.lower().endswith((".png", ".jpg", ".jpeg"))
          and not any(t in os.path.basename(f) for t in SKIP)]
    if not fs:
        return None, m.group(1)
    return np.asarray(Image.open(fs[0]).convert("RGB"), np.float32) / 255., m.group(1)


ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device="cuda")
fig, ax = plt.subplots(3, len(OBJS), figsize=(2.15 * len(OBJS), 7.2))
for c, (obj, label) in enumerate(OBJS):
    st = ovcut.load(obj, TAG)
    C = np.load(f"{W}/cams_{obj}.npz")
    with torch.no_grad():
        img, _ = ON.render_exterior(
            st, glctx, torch.as_tensor(C["e_mvp"][0], dtype=torch.float32, device="cuda"), 512)[:2]
    ext = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    rh, dh = first_ref(obj, "REF_H")
    rv, dv = first_ref(obj, "REF_V")
    for r, a in enumerate((ext, rh, rv)):
        if a is None:
            ax[r, c].text(0.5, 0.5, "none", ha="center", fontsize=9)
        else:
            ax[r, c].imshow(np.clip(crop(a), 0, 1))
        ax[r, c].set_axis_off()
    ax[0, c].set_title(label, fontsize=10.5)
    ax[1, c].set_title(os.path.basename(dh), fontsize=7.5, color="#5c5c5c")
    ax[2, c].set_title(os.path.basename(dv), fontsize=7.5, color="#5c5c5c")

for r, name in enumerate(("the object's own exterior", "REF_H, first file",
                          "REF_V, first file")):
    ax[r, 0].text(-0.05, 0.5, name, rotation=90, va="center", ha="right", fontsize=8.6,
                  transform=ax[r, 0].transAxes)
fig.subplots_adjust(left=0.036, right=0.996, top=0.93, bottom=0.005, wspace=0.03, hspace=0.10)
fig.savefig(sys.argv[1], dpi=170, facecolor="white")
print("  ->", sys.argv[1])
