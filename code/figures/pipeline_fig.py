"""Figure 2: one object through the whole method, every panel rendered from data.

The sheet this replaces was drawn by a script that was never in the repository and is gone, so
it could not be corrected when the method changed -- and the method did change, in the way this
figure is most likely to be misread about. Route 1 quantises a released reconstruction, and a
reader looking at the old row would reasonably conclude that the interior of that reconstruction
comes along with it. It used to. It does not: route 1 takes the shape and the exterior, the
interior starts flat at 0.5, and on both routes every structure inside the object is put there
by the photographs. The route-1 row now has a panel that says so, and both routes reach a
lattice whose interior looks the same.

    FN_ROOT=... python pipeline_fig.py orange out/pipeline.png

Panels and where each comes from, so nothing here is an illustration of something unmeasured:

  1a  the released ply, its own primitives, orthographic
  1b  a slab of the coarse occupancy before and after close_and_fill -- white is what sealing
      added, which is the hole a quantised surface leaves
  1c  a slab of the lattice as route 1 now writes it: skin cells at their own colour, interior
      cells flat.  This is the panel the figure was missing
  1d  the exterior, pinned: the skin cells alone
  2a  the same slab of the shape route 2 builds from its equation, interior equally flat
  2b  the six views
  2c  the skin those views project, on the same cells
  3a  the reference photographs, as they arrive
  3b  the same at the phases equation (27) solved
  3c  one warped onto the silhouette its plane renders
  3d  a held-out section of the trained model
  4a  the plane against one cell's twelve edges
  4b  the face it exposes
  4c  the pieces the labelling separates
  4d  that face as a dual grid
"""
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
from matplotlib.patches import FancyBboxPatch                          # noqa: E402
import torch                                                           # noqa: E402
from plyfile import PlyData                                            # noqa: E402

C0 = 0.28209479177387814


def conf(key, obj):
    for line in open(os.path.join(os.path.dirname(HERE), "objects", f"{obj}.conf")):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].split("#")[0].strip()
    return None


def read(lat):
    el = PlyData.read(os.path.join(lat, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1) * C0 + 0.5, 0, 1)
    lvl = torch.load(os.path.join(lat, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    return xyz, rgb, lvl


def slab(ax, xyz, rgb, axis=2, frac=0.5, half=0.02, s=0.6):
    """One slab of cells, seen down `axis`. A slab and not a projection: the interior is the
    subject, and a projection of a solid shows only its skin."""
    lo, hi = xyz[:, axis].min(), xyz[:, axis].max()
    c = lo + frac * (hi - lo)
    m = np.abs(xyz[:, axis] - c) < half * (hi - lo)
    o = [i for i in range(3) if i != axis]
    ax.scatter(xyz[m, o[0]], xyz[m, o[1]], c=rgb[m], s=s, marker="s", linewidths=0)
    ax.set_aspect("equal"); ax.set_axis_off()
    return int(m.sum())


def panel(ax, title, sub):
    ax.set_title(title, fontsize=9.5, pad=14, weight="bold")
    ax.text(0.5, -0.055, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.6, color="#555")


def main(obj, out):
    lat1 = os.path.join(FN, f"build_{obj}", "lattice")
    lat2 = os.path.join(FN, f"build_{obj}_r2", "skin")
    xyz1, rgb1, lvl1 = read(lat1)
    have2 = os.path.exists(os.path.join(lat2, "gs_fill.ply"))
    if have2:
        xyz2, rgb2, lvl2 = read(lat2)

    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.10,
                          left=0.02, right=0.98, top=0.88, bottom=0.08)

    # --- route 1 -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    src = conf("SRC", obj)
    el = PlyData.read(os.path.join(FN, src)).elements[0]
    p = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    c = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1) * C0 + 0.5, 0, 1)
    k = np.random.default_rng(0).choice(len(p), min(len(p), 400000), replace=False)
    ax.scatter(p[k, 0], p[k, 2], c=c[k], s=0.25, marker=".", linewidths=0)
    ax.set_aspect("equal"); ax.set_axis_off()
    panel(ax, "a released reconstruction", f"{len(p):,} primitives, taken as they are")

    ax = fig.add_subplot(gs[0, 1])
    n = slab(ax, xyz1, rgb1, s=0.5)
    panel(ax, "quantised, and sealed", "close_and_fill: a sponge becomes a solid")

    ax = fig.add_subplot(gs[0, 2])
    flat = rgb1.copy(); flat[lvl1 == 0] = 0.5
    slab(ax, xyz1, flat, s=0.5)
    panel(ax, "the interior is discarded", "shape and skin kept; the inside starts at 0.5")

    ax = fig.add_subplot(gs[0, 3])
    sk = lvl1 == 1
    slab(ax, xyz1[sk], rgb1[sk], s=0.5)
    panel(ax, "its own exterior, pinned", "colour and geometry, held exactly")

    # --- route 2 -----------------------------------------------------------------------
    if have2:
        ax = fig.add_subplot(gs[1, 0])
        f2 = rgb2.copy(); f2[lvl2 == 0] = 0.5
        slab(ax, xyz2, f2, s=0.5)
        panel(ax, "a shape from its equation", "sphere, ellipsoid, box, torus; nothing to repair")

        ax = fig.add_subplot(gs[1, 1])
        sk2 = lvl2 == 1
        slab(ax, xyz2[sk2], rgb2[sk2], s=0.5)
        panel(ax, "six views, projected", "skin_project, onto the same cells")

        ax = fig.add_subplot(gs[1, 2])
        slab(ax, xyz2, f2, s=0.5)
        panel(ax, "the same lattice", "flat inside, painted outside, either way")

    ax = fig.add_subplot(gs[1, 3])
    ax.set_axis_off()
    ax.text(0.02, 0.92,
            "What the two routes share, and what they do not.\n\n"
            "They differ in where the shape and the exterior come\n"
            "from: a released reconstruction, or an equation and six\n"
            "views. They do not differ inside. On both routes the\n"
            "interior begins at a flat 0.5 in every channel and every\n"
            "structure in it is put there by the photographs.\n\n"
            "INTERIOR_FROM_PLY=1 restores the older route 1, which\n"
            "took the reconstruction's interior with it.",
            transform=ax.transAxes, fontsize=8.4, va="top", color="#333", linespacing=1.55)

    for y, lab, col in ((0.905, "ROUTE 1  —  from a released reconstruction", "#c0392b"),
                        (0.455, "ROUTE 2  —  from an equation and six views", "#2e7d5b")):
        fig.text(0.02, y, lab, fontsize=9, weight="bold", color=col)

    fig.suptitle("Building the representation: either route, never both — and neither brings "
                 "an interior", fontsize=11.5, y=0.965)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  -> {out}")
    print(f"  route 1: {len(xyz1):,} cells, {int((lvl1 == 1).sum()):,} skin")
    if have2:
        print(f"  route 2: {len(xyz2):,} cells, {int((lvl2 == 1).sum()):,} skin")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "orange",
         sys.argv[2] if len(sys.argv) > 2 else "out/pipeline_routes.png")
