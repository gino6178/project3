"""Figure 3: the two pipelines stage by stage, on the same photographs.

FruitNinja's models are not released and its training could not be reproduced here, so this
figure compares what each pipeline *requires*, taken from its paper, rather than intermediate
results we could not obtain. The stages are aligned in columns so the comparison is per stage
and not a count of boxes: the only column where the two differ in kind is the third, where the
prior work fits a generative model to the object before any interior exists, and where this
work has nothing at all.

Both rows begin from the same cross-section photographs -- drawn once, between the rows, with
an arrow into each -- because the difference is not what is photographed but what the
photographs are used for: there they are a fine-tuning set for a diffusion model that is then
sampled, here they are the target a rendered section is compared against.

    python pipecmp.py OUT.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch         # noqa: E402

RED, BLUE, INK = "#c0392b", "#4a7ba7", "#222"

HEAD = ["input", "exterior", "interior prior", "interior supervision",
        "stored as", "a cut is"]

TOP = [("multi-view\nimages", BLUE),
       ("3D Gaussians", BLUE),
       ("latent diffusion,\nfine-tuned per object", RED),
       ("SDS through\nthat model", RED),
       ("opaque atomic\nGaussians", BLUE),
       ("the Gaussians\nnear the plane", BLUE)]

BOT = [("multi-view images,\nor a released shape", BLUE),
       ("two-level lattice", BLUE),
       (None, None),
       ("the photographs\nthemselves", BLUE),
       ("8-d per cell,\none shared MLP", BLUE),
       ("closed form\non the lattice", BLUE)]

L, R, GAP = 0.075, 0.997, 0.020
W = (R - L - 5 * GAP) / 6
XS = [L + i * (W + GAP) + W / 2 for i in range(6)]
H = 0.205
Y_TOP, Y_BOT, Y_PIL = 0.795, 0.135, 0.465


def box(ax, x, y, text, col, dashed=False):
    p = FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                       boxstyle="round,pad=0,rounding_size=0.012",
                       linewidth=1.5 if not dashed else 1.1,
                       edgecolor=col if not dashed else "#b8b8b8",
                       facecolor=(col + "14") if not dashed else "none",
                       linestyle="-" if not dashed else (0, (3, 3)),
                       transform=ax.transAxes, zorder=3)
    ax.add_patch(p)
    if text is None:                      # the column this work does not have
        ax.text(x, y, "none", transform=ax.transAxes, ha="center", va="center",
                fontsize=9.4, style="italic", color="#a8a8a8", zorder=4)
    else:
        ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=9.4, color=INK, linespacing=1.45, zorder=4)


def arrow(ax, a, b, col="#8a8a8a", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, color=col, lw=lw,
                                 arrowstyle="-|>", mutation_scale=11,
                                 shrinkA=0, shrinkB=0, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def main(out):
    fig = plt.figure(figsize=(13.4, 3.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for x, h in zip(XS, HEAD):
        ax.text(x, 0.968, " ".join(h.upper()).replace("   ", "\u2002\u2002"), transform=ax.transAxes, ha="center", va="center",
                fontsize=8.2, color="#8a8a8a")

    for i, ((t, c), (tb, cb)) in enumerate(zip(TOP, BOT)):
        box(ax, XS[i], Y_TOP, t, c)
        box(ax, XS[i], Y_BOT, tb, cb or "#b8b8b8", dashed=tb is None)
        if i < 5:
            for y in (Y_TOP, Y_BOT):
                arrow(ax, (XS[i] + W / 2, y), (XS[i + 1] - W / 2, y))

    # The same photographs, once, feeding the stage that consumes them in each row.
    px = (XS[2] + XS[3]) / 2
    pw, ph = 0.215, 0.098
    ax.add_patch(FancyBboxPatch((px - pw / 2, Y_PIL - ph / 2), pw, ph,
                                boxstyle="round,pad=0,rounding_size=0.045",
                                linewidth=1.3, edgecolor="#6a6a6a", facecolor="#f2f2f2",
                                transform=ax.transAxes, zorder=3))
    ax.text(px, Y_PIL, "cross-section photographs", transform=ax.transAxes,
            ha="center", va="center", fontsize=9.4, color=INK, zorder=4)
    arrow(ax, (px - pw / 2, Y_PIL + 0.012), (XS[2], Y_TOP - H / 2), "#6a6a6a", 1.2, 0.14)
    arrow(ax, (px + pw / 2, Y_PIL - 0.012), (XS[3], Y_BOT + H / 2), "#6a6a6a", 1.2, 0.14)

    for y, lab in ((Y_TOP, "FruitNinja"), (Y_BOT, "this work")):
        ax.text(0.062, y, lab, transform=ax.transAxes, ha="right", va="center",
                fontsize=10.5, style="italic", color=INK, rotation=90)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pipecmp.png")
