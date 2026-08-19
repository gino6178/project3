"""The closed-form cut, in the three states the operator passes through.

Section 3.3 states that the exposed face is exact and that its polygons meet edge to edge. Both
are properties of a construction rather than of a fit, so the figure is built by running the
construction on a real lattice and drawing what comes out, not by illustrating the idea.

    FN_ROOT=... python cutgeom_fig.py orange out/cutgeom.png

  (a) one slab of the lattice, one cell thick, with the plane's line across it. Cells carry the
      sign of their centre; the crossed band is the set the separating-axis test of equation (10)
      returns, and it is drawn from that test rather than from a threshold on distance.
  (b) one crossed cell, enlarged. Its eight corners carry the sign of the plane equation, the
      edges whose endpoints disagree are the ones that contribute, and the intersection points
      are placed at the t_e of equation (11). Ordering them by angle about their own centroid
      gives the polygon directly, with no case table.
  (c) the same polygons at the skin, where the cut face meets the exterior. What is being shown
      is that the two are one surface: every cut polygon's rim vertex is a point on a cell edge,
      and the cell that owns that edge is the one the exterior is drawn on.
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
from matplotlib.patches import Polygon as MplPolygon                   # noqa: E402
import torch                                                           # noqa: E402
from plyfile import PlyData                                            # noqa: E402

C0 = 0.28209479177387814
CORN = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
EDGE = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def load(obj):
    d = os.path.join(FN, f"build_{obj}", "lattice")
    el = PlyData.read(os.path.join(d, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1) * C0 + 0.5, 0, 1)
    lat = torch.load(os.path.join(d, "lattice.pt"))
    lvl = torch.load(os.path.join(d, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    return xyz, rgb, lvl, float(lat["coarse_dx"])


def main(obj="orange", out="out/cutgeom.png"):
    xyz, rgb, lvl, hc = load(obj)
    coarse = lvl == 0
    P, C = xyz[coarse], rgb[coarse]

    # an oblique, off-grid plane through the middle -- the case the construction is for
    # All three components non-zero. With n_y = 0 the plane is parallel to an axis and every
    # cell it crosses is cut through a pair of opposite faces -- four edges, every time, which
    # makes the general case look like the only case. Obliquely it is a hexagon.
    n = np.array([0.55, 0.48, 0.68]); n /= np.linalg.norm(n)
    ctr = P.mean(0)
    d = -float(n @ ctr)

    # one slab, one cell thick, seen down y
    m = np.abs(P[:, 1] - ctr[1]) < 0.5 * hc
    S, Sc = P[m], C[m]
    sgn = np.sign(S @ n + d)
    band = np.abs(S @ n + d) <= 0.5 * hc * np.abs(n).sum()      # equation (10)

    fig = plt.figure(figsize=(13.2, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.16, left=0.03, right=0.985, top=0.86, bottom=0.05)

    # --- (a) the slab, the signs, the crossed band --------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for s, col in ((1, "#cfe0f0"), (-1, "#f6ddd0")):
        q = S[(sgn == s) & ~band]
        ax.scatter(q[:, 0], q[:, 2], s=3.0, c=col, marker="s", linewidths=0)
    q = S[band]
    ax.scatter(q[:, 0], q[:, 2], s=3.4, c="#c0392b", marker="s", linewidths=0)
    lo, hi = S[:, 0].min(), S[:, 0].max()
    # the plane's trace in *this* slab: the n_y y term is not optional once n_y is non-zero,
    # and dropping it draws the line beside the band it is supposed to mark rather than on it
    zs = (-d - n[0] * np.array([lo, hi]) - n[1] * ctr[1]) / n[2]
    ax.plot([lo, hi], zs, color="#222", lw=1.3)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("(a)  the cells the plane crosses", fontsize=10, loc="left")

    # --- (b) one crossed cell ------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    # The cell with the most contributing edges, not the one nearest the plane. A cell clipped
    # at a corner gives a triangle and a sliver on the page; the general case is a hexagon and
    # that is what the construction has to handle without a case table.
    def nedges(cell):
        Vv = (cell - 0.5 * hc)[None] + CORN * hc
        ss = Vv @ n + d
        return sum(1 for a, b in EDGE if (ss[a] < 0) != (ss[b] < 0))
    cand = S[band]
    c0 = cand[int(np.argmax([nedges(c) for c in cand]))] - 0.5 * hc
    V = c0[None] + CORN * hc
    sv = V @ n + d
    # Isometric, not the plane's own basis. Projecting the cell onto the basis of the plane
    # that cuts it puts the eye in the plane, the cube collapses to a rectangle and the polygon
    # to a segment -- which is what the first version of this figure drew.
    def proj(Q):
        R = (Q - c0 - 0.5 * hc) / hc
        cs, sn = np.cos(np.radians(30)), np.sin(np.radians(30))
        return np.stack([(R[:, 0] - R[:, 1]) * cs, R[:, 2] + (R[:, 0] + R[:, 1]) * sn], 1)

    for a, b in EDGE:
        pa, pb = proj(V[[a, b]])
        cross = (sv[a] < 0) != (sv[b] < 0)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color="#c0392b" if cross else "#c9c9c9", lw=2.2 if cross else 0.9,
                zorder=3 if cross else 1)
    pv = proj(V)
    for i in range(8):
        ax.scatter(*pv[i], s=42, c="#4a7ba7" if sv[i] > 0 else "#e0a58a", zorder=4,
                   edgecolors="#444", linewidths=0.5)
    pts = []
    for a, b in EDGE:
        if (sv[a] < 0) != (sv[b] < 0):
            t = sv[a] / (sv[a] - sv[b])
            pts.append(V[a] + t * (V[b] - V[a]))
    pts = np.array(pts)
    pp = proj(pts)
    ang = np.arctan2(pp[:, 1] - pp[:, 1].mean(), pp[:, 0] - pp[:, 0].mean())
    o = np.argsort(ang)
    ax.add_patch(MplPolygon(pp[o], closed=True, facecolor="#c0392b", alpha=0.16,
                            edgecolor="#c0392b", lw=1.8, zorder=2))
    ax.scatter(pp[:, 0], pp[:, 1], s=34, c="#c0392b", zorder=5)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"(b)  one cell: {len(pts)} edges disagree", fontsize=10, loc="left")

    # --- (c) the polygons at the skin ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    # Seen along the plane normal, over the whole object: the cut face is a face from here and
    # a line from anywhere in the plane, which is why this panel does not reuse (a)'s view.
    e1 = np.array([-n[2], 0.0, n[0]]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    flat = lambda Q: np.stack([(Q - ctr) @ e1, (Q - ctr) @ e2], 1)
    allband = np.abs(P @ n + d) <= 0.5 * hc * np.abs(n).sum()
    sk = lvl == 1
    Q, Qc = xyz[sk], rgb[sk]
    skb = np.abs(Q @ n + d) <= 1.2 * hc
    fq = flat(Q[skb])
    ax.scatter(fq[:, 0], fq[:, 1], s=2.0, c=Qc[skb], marker="s", linewidths=0, zorder=1)
    npoly = 0
    for cell in P[allband]:
        cc = cell - 0.5 * hc
        Vv = cc[None] + CORN * hc
        ss = Vv @ n + d
        pl = [Vv[a] + (ss[a] / (ss[a] - ss[b])) * (Vv[b] - Vv[a])
              for a, b in EDGE if (ss[a] < 0) != (ss[b] < 0)]
        if len(pl) < 3:
            continue
        pl = flat(np.array(pl))
        an = np.arctan2(pl[:, 1] - pl[:, 1].mean(), pl[:, 0] - pl[:, 0].mean())
        ax.add_patch(MplPolygon(pl[np.argsort(an)], closed=True, facecolor="#c0392b",
                                alpha=0.5, edgecolor="#8e2b1f", lw=0.35, zorder=3))
        npoly += 1
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("(c)  the exposed face", fontsize=10, loc="left")

    fig.suptitle("The closed-form cut", fontsize=12, y=0.965)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  -> {out}")
    print(f"  slab: {int(band.sum()):,} crossed of {len(S):,}; the drawn cell has "
          f"{len(pts)} contributing edges; {npoly:,} polygons over the whole face")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "orange",
         sys.argv[2] if len(sys.argv) > 2 else "out/cutgeom.png")
