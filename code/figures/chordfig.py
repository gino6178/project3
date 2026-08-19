"""Why the phases are solved on the chords, shown on the planes that pay for it.

A transverse plane and a longitudinal plane meet in a one-dimensional chord, and on that chord
the two families describe the same physical line. Equation (4) aligns each family to itself and
leaves the two families free to disagree there; equation (5) chooses the angles and the
assignment together so they do not. The disagreement is invisible on the planes either family
supervised -- those are fits -- so this figure is drawn on 45 degree cuts, which neither family
supervised and which inherit whatever the two families could not agree about.

    FN_ROOT=... python chordfig.py ALIGN_RUN SOLVE_RUN OUT.png

  (a) the geometry: two planes, their chord, and the two one-dimensional signals that must agree
      along it.
  (b) 45 degree cuts from a model whose phases came from equation (4), the greedy per-family
      alignment.
  (c) the same cuts from a model whose phases came from equation (5), solved on the chords.

Both models are trained by the same program on the same lattice with the same references; the
only difference is REF_PHASE_MODE.
"""
import glob
import os
import sys

import numpy as np

FN = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FN)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import cv2                                                             # noqa: E402


def chord_panel(ax):
    """Two planes, the chord they share, and the two signals that must agree on it."""
    th = np.linspace(0, 2 * np.pi, 200)
    # the transverse plane, as a disc seen obliquely
    ax.plot(np.cos(th), 0.34 * np.sin(th) + 0.15, color="#4a7ba7", lw=1.6)
    ax.fill(np.cos(th), 0.34 * np.sin(th) + 0.15, color="#4a7ba7", alpha=0.10)
    # the longitudinal plane, edge-on and upright
    ax.plot([-0.30, 0.30, 0.30, -0.30, -0.30], [-0.92, -0.62, 1.22, 0.92, -0.92],
            color="#c0392b", lw=1.6)
    ax.fill([-0.30, 0.30, 0.30, -0.30], [-0.92, -0.62, 1.22, 0.92], color="#c0392b", alpha=0.10)
    # the chord: where they meet
    ax.plot([-0.30, 0.30], [0.02, 0.32], color="#111", lw=3.0, solid_capstyle="round", zorder=5)
    ax.scatter([-0.30, 0.30], [0.02, 0.32], s=26, c="#111", zorder=6)
    ax.text(0.0, 0.44, "the chord", fontsize=9, ha="center", color="#111")
    ax.text(-1.06, 0.16, r"$\Pi_j$", fontsize=11, color="#4a7ba7")
    ax.text(0.36, 1.10, r"$\Pi_{j'}$", fontsize=11, color="#c0392b")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.5, 1.45)
    ax.set_aspect("equal"); ax.set_axis_off()


def strip(ax, paths, label, sub):
    ims = [cv2.imread(p)[:, :, ::-1] for p in paths[:4]]
    if not ims:
        ax.set_axis_off(); return
    h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims)
    ax.imshow(np.hstack([cv2.resize(i, (w, h)) for i in ims]))
    ax.set_axis_off()
    ax.set_title(label, fontsize=10, loc="left")



def main(a_run, b_run, out):
    pa = sorted(glob.glob(os.path.join(FN, f"eval45_{a_run}", "rh*_init_0.png")))
    pb = sorted(glob.glob(os.path.join(FN, f"eval45_{b_run}", "rh*_init_0.png")))
    # How far apart the two models are on the same plane, and how far apart two planes of one
    # model are, so the reader has a scale rather than a claim. At this size a mean difference
    # of 0.05 is not a visible difference, and a figure that implies one would be arguing.
    dif, scale = [], []
    for qa, qb in zip(pa, pb):
        ia = cv2.imread(qa).astype(np.float32) / 255.
        ib = cv2.imread(qb).astype(np.float32) / 255.
        fg = (ia.min(2) < 0.96) | (ib.min(2) < 0.96)
        dif.append((np.abs(ia - ib), float(np.abs(ia - ib)[fg].mean())))
    for i in range(len(pa) - 1):
        ia = cv2.imread(pa[i]).astype(np.float32) / 255.
        ib = cv2.imread(pa[i + 1]).astype(np.float32) / 255.
        fg = (ia.min(2) < 0.96) | (ib.min(2) < 0.96)
        scale.append(float(np.abs(ia - ib)[fg].mean()))
    md, ms = float(np.mean([d for _, d in dif])), float(np.mean(scale))

    fig = plt.figure(figsize=(13.4, 6.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.62, 1.38], hspace=0.46, wspace=0.10,
                          left=0.02, right=0.985, top=0.89, bottom=0.09)
    axg = fig.add_subplot(gs[:, 0]); chord_panel(axg)
    axg.set_title("(a)  the chord the two families share", fontsize=10, loc="left")
    strip(fig.add_subplot(gs[0, 1]), pa, "(b)  phases from equation (4)",
          "45° cuts, supervised by neither family")
    strip(fig.add_subplot(gs[1, 1]), pb, "(c)  phases from equation (5)",
          "the same cuts, the same lattice, the same references")
    axd = fig.add_subplot(gs[2, 1])
    ims = [np.clip(a * 4.0, 0, 1)[:, :, ::-1] for a, _ in dif]
    h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims)
    axd.imshow(np.hstack([cv2.resize(i, (w, h)) for i in ims]))
    axd.set_axis_off()
    axd.set_title(f"(d)  their difference at 4×:  {md:.4f}, against {ms:.4f} between two cuts",
                  fontsize=10, loc="left")
    fig.suptitle("Chordal consistency, on 45° cuts neither family supervised",
                 fontsize=12, y=0.975)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  -> {out}  ({len(pa)} and {len(pb)} cuts)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
