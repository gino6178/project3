"""Both families, control against the two blend arms, whole and magnified.

The claim under test is narrow: the transverse family's reference assignment was a step function,
each step drew a line across the polar axis, and a longitudinal cut crossed every one. If that is
right the horizontal banding leaves the rv row and the rh row is left alone -- a transverse cut is
perpendicular to the axis, so it never crosses a switch and had nothing to lose. An rh row that
changes as much as the rv row would mean the diagnosis was about something else.

    python tbsheet.py OUT.png ARM [ARM ...]
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

LABEL = {"r1_pin_full": "control\n(block rule)",
         "r1_tb2": "on the common disc,\nnearest photograph",
         "r1_tb1": "on the common disc,\nblended across depth",
         "r1_sw": "sliced Wasserstein\non the patches"}


def load(arm, fam, k):
    for pat in (f"{arm}/eval_final/{fam}{k}_*.png", f"{arm}/{fam}{k}_*.png"):
        f = sorted(glob.glob(pat))
        if f:
            return np.asarray(Image.open(f[0]).convert("RGB"), np.float32) / 255.
    return None


def main(out, arms, k=0):
    rows = [("rv", "longitudinal (crosses every switch)"),
            ("rh", "transverse (crosses none)")]
    fig, axes = plt.subplots(4, len(arms), figsize=(3.5 * len(arms), 13.6))
    for c, arm in enumerate(arms):
        for r, (fam, _) in enumerate(rows):
            a = load(arm, fam, k)
            if a is None:
                axes[2 * r, c].text(0.5, 0.5, f"no {fam} for {arm}", ha="center")
                axes[2 * r, c].set_axis_off(); axes[2 * r + 1, c].set_axis_off()
                continue
            axes[2 * r, c].imshow(a)
            h = a.shape[0] // 8
            cy = cx = a.shape[0] // 2
            axes[2 * r, c].add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                                   fill=False, lw=0.9, ec="k"))
            axes[2 * r + 1, c].imshow(a[cy - h:cy + h, cx - h:cx + h],
                                      interpolation="nearest")
            for ax in (axes[2 * r, c], axes[2 * r + 1, c]):
                ax.set_axis_off()
        axes[0, c].set_title(LABEL.get(arm, arm), fontsize=10)
    for r, (_, name) in enumerate(rows):
        axes[2 * r, 0].set_ylabel(name, fontsize=9)
        axes[2 * r, 0].axis("on")
        axes[2 * r, 0].set_xticks([]); axes[2 * r, 0].set_yticks([])
        for s in axes[2 * r, 0].spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.955, bottom=0.005,
                        wspace=0.02, hspace=0.03)
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
