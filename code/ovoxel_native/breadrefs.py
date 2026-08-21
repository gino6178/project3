"""Every photograph FruitNinja released for the loaf, in one place.

    python breadrefs.py OUT.png

The two reference directories hold the same five files by md5, so the question is not which family
each belongs to but whether the five contain two kinds of section at all. A loaf sliced across its
length gives one kind of face; sliced along it, another. If all five are the same kind, no split of
them makes two families and the loaf's longitudinal supervision does not exist.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

FN = "/workspace/rebuild/worktree"
fs = []
for d in ("spl_bread_h", "hld_bread_h"):
    for f in sorted(glob.glob(os.path.join(FN, d, "*.png"))):
        fs.append((os.path.basename(f), "supervised" if d.startswith("spl") else "held out", f))
fs.sort(key=lambda t: t[0])
fig, ax = plt.subplots(1, len(fs), figsize=(2.6 * len(fs), 3.2))
for c, (nm, role, f) in enumerate(fs):
    a = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.
    ax[c].imshow(np.clip(a, 0, 1))
    ax[c].set_axis_off()
    ax[c].set_title(f"{nm}\n{role}", fontsize=10)
fig.suptitle("every photograph released for the loaf", fontsize=11.5)
fig.subplots_adjust(left=0.004, right=0.996, top=0.82, bottom=0.01, wspace=0.03)
fig.savefig(sys.argv[1], dpi=160, facecolor="white")
print("  ->", sys.argv[1], len(fs), "files")
