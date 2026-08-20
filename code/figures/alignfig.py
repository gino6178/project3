"""What happens when the alignment is made differentiable and optimised with the volume.

    python alignfig.py drift.txt sections.png OUT.png

Left: three held-out sections from each arm, the same planes under the same seed. Right: how far
the alignment parameters travel from the moment fit that initialised them, against the distance a
random walk of the same optimiser and step count would cover.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from PIL import Image                                                 # noqa: E402

RED, BLUE = "#c0392b", "#4a7ba7"
LR = 0.01


def read(p):
    j, ds, dt = [], [], []
    for line in open(p):
        m = re.search(r"align j=(\d+).*?\|dlog s\| mean ([\d.]+) max ([\d.]+), "
                      r"\|dt\| mean ([\d.]+)", line)
        if m:
            j.append(int(m.group(1))); ds.append(float(m.group(2))); dt.append(float(m.group(4)))
    return np.array(j), np.array(ds), np.array(dt)


def main(drift, sections, out):
    j, ds, dt = read(drift)
    img = np.asarray(Image.open(sections).convert("RGB"))
    h = img.shape[0] // 2

    fig = plt.figure(figsize=(12.4, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], wspace=0.13, hspace=0.10,
                          left=0.035, right=0.985, top=0.90, bottom=0.115)
    for r, (lab, sl) in enumerate((("alignment fixed by moments", slice(0, h)),
                                   ("alignment optimised jointly", slice(h, 2 * h)))):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(img[sl])
        ax.set_axis_off()
        ax.set_title(f"({'ab'[r]})  {lab}", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[:, 1])
    ax.plot(j, ds, "-", color=RED, lw=1.6, label=r"$|\Delta \log s|$")
    ax.plot(j, dt, "-", color=BLUE, lw=1.6, label=r"$|\Delta t|$")
    ax.plot(j, LR * np.sqrt(j), "--", color="0.6", lw=1.2,
            label="a random walk of the same steps")
    ax.set_xlabel("gradient steps taken on the alignment", fontsize=9.5)
    ax.set_ylabel("distance from the moment fit", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(c)  the parameters do not stay put", fontsize=10, loc="left")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out, f"  {len(j)} drift samples, final |dlog s| {ds[-1]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
