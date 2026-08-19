"""How many cross-sections the interior is worth, measured two ways.

    python slicefig.py OUT.png

Left: a perceptual distance from each arm's held-out sections to the ten photographs no arm
below twenty was trained on. Right: how much consecutive sections differ along a ninety-six
plane sweep -- not the mean of that difference, which is flat, but its spread, which is what a
volume with structure along depth has and a volume that repeats one section does not.

The twenty-photograph arm is drawn hollow because the ten references are held out from every
other arm and not from it.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

N = np.array([1, 3, 5, 10, 20])
DS = np.array([0.1798, 0.1793, 0.1819, 0.1873, 0.1926])
SD = np.array([0.00015, 0.00030, 0.00033, 0.00054, 0.00044])
MEAN = np.array([0.00423, 0.00431, 0.00453, 0.00447, 0.00406])
RED, BLUE = "#c0392b", "#4a7ba7"


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.7))
    for a in ax:
        a.set_xscale("log"); a.set_xticks(N)
        a.set_xticklabels([str(v) for v in N], fontsize=9)
        a.set_xlabel("transverse photographs used in training", fontsize=9.5)
        a.tick_params(labelsize=8.5)
        a.grid(alpha=0.18, lw=0.6)

    ax[0].plot(N[:4], DS[:4], "-o", color=RED, lw=1.5, ms=5)
    ax[0].plot(N[4:], DS[4:], "o", color=RED, ms=6, mfc="white", mew=1.5)
    ax[0].plot([N[3], N[4]], [DS[3], DS[4]], ":", color=RED, lw=1.2)
    ax[0].set_ylabel("DreamSim to the held-out ten", fontsize=9.5)
    ax[0].set_ylim(0.170, 0.200)
    ax[0].set_title("(a)  what a section looks like", fontsize=10, loc="left")

    ax[1].plot(N[:4], SD[:4] * 1e3, "-o", color=BLUE, lw=1.5, ms=5)
    ax[1].plot(N[4:], SD[4:] * 1e3, "o", color=BLUE, ms=6, mfc="white", mew=1.5)
    ax[1].plot([N[3], N[4]], [SD[3] * 1e3, SD[4] * 1e3], ":", color=BLUE, lw=1.2)
    # The mean of that difference is flat -- 4.06 to 4.53 e-3 across the whole sweep -- and on
    # a shared axis it would compress the quantity that does move into the bottom tenth of the
    # panel. It is stated in the caption instead of drawn.
    ax[1].set_ylabel("spread of the section-to-section\ndifference, e$^{-3}$", fontsize=9.5)
    ax[1].set_ylim(0, 0.62)
    ax[1].set_title("(b)  how it changes with depth", fontsize=10, loc="left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "slicefig.png")
