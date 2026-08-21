"""Every file in one object's two reference families, as it sits on disk.

    python refall.py OBJ OUT.png

The audit showed one file each. A family is not one file, and an object whose transverse and
longitudinal references look like the same kind of photograph is worth seeing in full before
anything is claimed about it.
"""
import glob
import os
import re
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
SKIP = ("_depth", "_mask", "_alpha", "_normal")
obj, out = sys.argv[1], sys.argv[2]


def files(which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    d = os.path.join(FN, m.group(1))
    return m.group(1), [f for f in sorted(glob.glob(os.path.join(d, "*")))
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                        and not any(t in os.path.basename(f) for t in SKIP)]


fams = [("REF_H (transverse)",) + files("REF_H"), ("REF_V (longitudinal)",) + files("REF_V")]
n = max(len(f[2]) for f in fams)
fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5.4), squeeze=False)
for r, (name, d, fs) in enumerate(fams):
    for c in range(n):
        ax[r][c].set_axis_off()
        if c < len(fs):
            a = np.asarray(Image.open(fs[c]).convert("RGB"), np.float32) / 255.
            ax[r][c].imshow(np.clip(a, 0, 1))
            ax[r][c].set_title(os.path.basename(fs[c]), fontsize=7.5, color="#5c5c5c")
    ax[r][0].text(-0.06, 0.5, f"{name}\n{d}  ({len(fs)})", rotation=90, va="center",
                  ha="right", fontsize=8.4, transform=ax[r][0].transAxes)
fig.suptitle(obj, fontsize=11)
fig.subplots_adjust(left=0.05, right=0.996, top=0.88, bottom=0.01, wspace=0.03, hspace=0.14)
fig.savefig(out, dpi=160, facecolor="white")
print("  ->", out, [len(f[2]) for f in fams])
