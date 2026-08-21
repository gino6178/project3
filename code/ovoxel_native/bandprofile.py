"""The axial luminance profile of a longitudinal cut, drawn rather than summarised.

Four summary statistics in a row failed to separate arms the eye separates immediately -- a period
that is not there, hard edges that are soft, a spectral band the pipeline occupies just as much,
and a turning-point count that counts noise. The profile itself is the evidence: the mean
luminance of each row across the middle of the face, with the object's own falloff removed.

A hard switch between supervising photographs shows as a step. Structure that belongs to the fruit
shows as something that is not a step, and the same photographs supervise the pipeline, so
whatever both share is the fruit and whatever only one has is its own.
"""
import glob
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402


def profile(path, n=256):
    L = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255.
    ys, xs = np.where(L < 0.97)
    if len(ys) < 500:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    w = x1 - x0
    c = L[y0 + (y1 - y0) // 12:y1 - (y1 - y0) // 12, x0 + w // 3:x1 - w // 3]
    p = c.mean(1)
    p = p - ndimage.gaussian_filter1d(p, len(p) / 5.0)               # drop the falloff
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)


arms = [("existing pipeline (route 1)", "baseline/eval_orange_b/rv*.png", "0.35"),
        ("O-Voxel, free per-cell RGB", "r1_free/eval_final/rv*.png", "tab:green"),
        ("O-Voxel, decoder, full parity", "r1_pin_full/eval_final/rv*.png", "tab:blue")]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=True)
for ax, (label, pat, col) in zip(axes, arms):
    fs = sorted(glob.glob(pat))[:6]
    ps = [p for p in (profile(f) for f in fs) if p is not None]
    for p in ps:
        ax.plot(np.linspace(0, 1, len(p)), p, color=col, lw=0.9, alpha=0.55)
    if ps:
        m = np.mean(ps, 0)
        ax.plot(np.linspace(0, 1, len(m)), m, color="k", lw=1.6)
        # a step's signature is a large jump between neighbouring rows relative to the profile's
        # own scale; report the largest few so the plot carries a number as well as a shape
        j = np.sort(np.abs(np.diff(m)))[-5:].mean()
        ax.set_title(f"{label}\n{len(ps)} sections, mean of the five largest row-to-row "
                     f"jumps {j:.4f}", fontsize=9.5)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("position down the polar axis")
axes[0].set_ylabel("luminance, object falloff removed")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150, facecolor="white")
print("  ->", sys.argv[1])
