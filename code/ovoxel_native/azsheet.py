"""The four arms, both families, on held-out planes.

    python azsheet.py OUT.png

Same object, same budget, same held-out planes. What differs is only whether the longitudinal
planes turn about the axis and whether each plane's reference is chosen where it was jittered to.
Three held-out cuts per family so a single lucky plane cannot carry the impression.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
ARMS = [("ov2_orange_sp", "control"),
        ("rf_orange_sp", "reference follows"),
        ("az_orange_sp", "planes turn"),
        ("azrf_orange_sp", "both")]
SHOTS = [("rv", 0), ("rv", 2), ("rv", 4), ("rh", 0), ("rh", 3)]


def crop(a, pad=0.05):
    a = np.asarray(a, np.float32)
    m = a.min(2) < 0.97
    if m.sum() < 100:
        return a
    ys, xs = np.where(m)
    s = int(pad * max(ys.max() - ys.min(), xs.max() - xs.min()))
    return a[max(ys.min() - s, 0):ys.max() + s, max(xs.min() - s, 0):xs.max() + s]


fig, ax = plt.subplots(len(SHOTS), len(ARMS), figsize=(2.4 * len(ARMS), 2.5 * len(SHOTS)))
for c, (tag, label) in enumerate(ARMS):
    for r, (fam, k) in enumerate(SHOTS):
        f = sorted(glob.glob(f"{W}/{tag}/eval_final/{fam}{k}_*.png"))
        if f:
            ax[r, c].imshow(np.clip(crop(np.asarray(Image.open(f[0]).convert("RGB"),
                                                    np.float32) / 255.), 0, 1))
        ax[r, c].set_axis_off()
    ax[0, c].set_title(label, fontsize=11)
for r, (fam, k) in enumerate(SHOTS):
    ax[r, 0].text(-0.05, 0.5, f"{'longitudinal' if fam == 'rv' else 'transverse'} {k}",
                  rotation=90, va="center", ha="right", fontsize=9,
                  transform=ax[r, 0].transAxes)
fig.subplots_adjust(left=0.045, right=0.996, top=0.965, bottom=0.005, wspace=0.03, hspace=0.03)
fig.savefig(sys.argv[1], dpi=165, facecolor="white")
print("  ->", sys.argv[1])
