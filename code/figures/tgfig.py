"""What the two rules ask each supervised plane for.

    python tgfig.py tg.npz OUT.png

One column per supervised plane. The block rule repeats a photograph for as many planes as the
integer division gives it and then changes all at once; equation (7) mixes each photograph with
the next at the fractional part, so no two adjacent planes are asked for the same thing.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402


def crop(a):
    m = a.min(2) < 245
    ys, xs = np.where(m)
    if len(ys) < 50:
        return a
    p = 4
    return a[max(ys.min() - p, 0):ys.max() + p, max(xs.min() - p, 0):xs.max() + p]


def main(npz, out):
    z = np.load(npz)
    blk, con, which = z["block"], z["cont"], z["which"]
    off = int(z["off"]) if "off" in z else 0   # this strip may be a window on a longer one
    n = len(which)
    edges = [i for i in range(1, n) if which[i] != which[i - 1]]

    fig = plt.figure(figsize=(1.16 * n, 3.35))
    gs = fig.add_gridspec(2, n, wspace=0.035, hspace=0.06,
                          left=0.052, right=0.995, top=0.855, bottom=0.03)
    for r, (arr, lab) in enumerate(((blk, "block assignment"), (con, "equation (7)"))):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(crop(arr[c]))
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"{c + off}", fontsize=9, color="#8a8a8a")
        fig.text(0.046, 0.63 - 0.42 * r, lab, fontsize=10.5, style="italic",
                 rotation=90, va="center", ha="right")

    # Where the block rule changes photograph, in figure coordinates.
    x0, x1 = 0.052, 0.995
    for e in edges:
        x = x0 + (x1 - x0) * (e / n) - 0.0018
        fig.add_artist(plt.Line2D([x, x], [0.03, 0.855], color="#c0392b", lw=1.4))
    fig.text((x0 + x1) / 2, 0.965, "supervised plane", fontsize=9.5, color="#8a8a8a",
             ha="center")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out, "changeovers at", edges)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
