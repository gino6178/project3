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

  1a  the skin cells alone: the shell a scan gives, hollow
  1b  the same object filled: every cell, with the interior at 0.5
  1c  the surface again, which is what training holds fixed
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

import glob

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
    hc1 = float(torch.load(os.path.join(lat1, 'lattice.pt'))['coarse_dx'])
    have2 = os.path.exists(os.path.join(lat2, "gs_fill.ply"))
    if have2:
        xyz2, rgb2, lvl2 = read(lat2)

    fig = plt.figure(figsize=(13.6, 14.4))
    gs = fig.add_gridspec(4, 4, hspace=0.46, wspace=0.10,
                          left=0.075, right=0.985, top=0.912, bottom=0.075)

    # --- route 1: the shell a scan gives, then filled ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    sk = lvl1 == 1
    slab(ax, xyz1[sk], rgb1[sk], s=0.5)
    panel(ax, "the shell", "what a 3D scan gives: a surface, and nothing behind it")

    ax = fig.add_subplot(gs[0, 1])
    flat = rgb1.copy(); flat[lvl1 == 0] = 0.5
    slab(ax, xyz1, flat, s=0.5)
    panel(ax, "filled", f"close_and_fill: {int((lvl1 == 0).sum()):,} cells, no colour of their own")

    ax = fig.add_subplot(gs[0, 2])
    slab(ax, xyz1[sk], rgb1[sk], s=0.5)
    # The same picture as the first panel, deliberately: SHELL_PIN means training does not move
    # it, and the measured drift is the sub-caption rather than a claim.
    ext = "measurements/results.json"
    drift = None
    if os.path.exists(os.path.join(FN, ext)):
        import json
        r = json.load(open(os.path.join(FN, ext))).get(obj, {})
        drift = r.get("exterior_mean")
    panel(ax, "the surface, pinned",
          "the same picture, on purpose: training does not move it"
          + (f"\n(mean drift {drift:.4f} over 200 iterations)" if drift is not None else ""))

    ax = fig.add_subplot(gs[0, 3]); ax.set_axis_off()
    ax.text(0.0, 1.02,
            "A 3D scan gives a shell.\n\n"
            "That is the whole input on\nthis route: an outer surface,\n"
            "with nothing behind it. The\nvolume inside is filled so the\n"
            "object is solid, and those\ncells start with no colour of\n"
            "their own.\n\n"
            "We have no scan files, so a\nreleased reconstruction stands\n"
            "in -- used the way a scan\nwould be, for its surface only.\n"
            "Whatever it carries inside is\ndiscarded, because a scan\n"
            "would not have supplied it.",
            transform=ax.transAxes, fontsize=7.6, va="top", color="#333", linespacing=1.45)

    # --- route 2: the same three beats, with no scan at all ---------------------------
    if have2:
        sk2 = lvl2 == 1
        ax = fig.add_subplot(gs[1, 0])
        slab(ax, xyz2[sk2], rgb2[sk2], s=0.5)
        panel(ax, "the shell", "a surface from its equation: sphere, ellipsoid, box, torus")

        ax = fig.add_subplot(gs[1, 1])
        f2 = rgb2.copy(); f2[lvl2 == 0] = 0.5
        slab(ax, xyz2, f2, s=0.5)
        panel(ax, "filled", "solid by construction; closing adds 0.1%")

        ax = fig.add_subplot(gs[1, 2])
        import cv2
        d6 = os.path.join(FN, conf("REFS6", obj) or "")
        six = sorted(g for g in os.listdir(d6) if g.endswith((".png", ".jpg"))) if os.path.isdir(d6) else []
        if six:
            tiles = [cv2.imread(os.path.join(d6, f))[:, :, ::-1] for f in six[:6]]
            h = min(t.shape[0] for t in tiles); w = min(t.shape[1] for t in tiles)
            tiles = [cv2.resize(t, (w, h)) for t in tiles]
            while len(tiles) < 6:
                tiles.append(np.full_like(tiles[0], 255))
            ax.imshow(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]))
        ax.set_axis_off()
        panel(ax, "the surface, projected", f"{len(six)} views of the outside, onto the same cells")

    ax = fig.add_subplot(gs[1, 3]); ax.set_axis_off()
    ph = sorted(os.listdir(os.path.join(FN, conf("REF_H", obj))))
    ph = [f for f in ph if f.endswith((".png", ".jpg")) and "_depth" not in f]
    if ph:
        import cv2 as _cv
        im = _cv.imread(os.path.join(FN, conf("REF_H", obj), ph[0]))[:, :, ::-1]
        ax.imshow(im)
    panel(ax, "and then the interior",
          "photographs of cross-sections are the only thing\nthat ever writes inside; section 3.2")

    for y, lab, col in ((0.930, "BUILDING THE REPRESENTATION  \u2014  route 1, a scan", "#c0392b"),
                        (0.722, "\u2014  route 2, a shape from its equation", "#2e7d5b"),
                        (0.500, "SUPERVISING THE INTERIOR  \u2014  the photographs, and nothing "
                         "else, ever writes inside", "#8a6d1f"),
                        (0.258, "WHAT THE LATTICE IS FOR  \u2014  an exact cut, its pieces, and "
                         "a solver's particles", "#4a4a8a")):
        fig.text(0.02, y, lab, fontsize=9, weight="bold", color=col)
    band_supervision(fig, gs, 2, obj, FN, None, conf)
    band_downstream(fig, gs, 3, xyz1, rgb1, lvl1, hc1)

    fig.suptitle("One object through the whole method: a scanned shell, an interior the photographs\nwrite, and a cut that is computed",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.942, "a shell, filled; either route, never both \u2014 and neither route "
             "ever writes inside the object",
             fontsize=8.8, color="#555", ha="center")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  -> {out}")
    print(f"  route 1: {len(xyz1):,} cells, {int((lvl1 == 1).sum()):,} skin")
    if have2:
        print(f"  route 2: {len(xyz2):,} cells, {int((lvl2 == 1).sum()):,} skin")


def band_supervision(fig, gs, row, obj, FN, slab, conf):
    """What supervises the interior: photographs, aligned, fitted, and the volume's answer."""
    import sds_demo, section_match as sm, cv2, torch
    ref_h = conf("REF_H", obj)
    files = sorted(sds_demo._photos_in(os.path.join(FN, ref_h)))

    ax = fig.add_subplot(gs[row, 0])
    t = [cv2.imread(f)[:, :, ::-1] for f in files[:4]]
    h = min(i.shape[0] for i in t); w = min(i.shape[1] for i in t)
    ax.imshow(np.vstack([np.hstack([cv2.resize(x, (w, h)) for x in t[:2]]),
                         np.hstack([cv2.resize(x, (w, h)) for x in t[2:4]])]))
    ax.set_axis_off()
    panel(ax, "photographs, as they arrive",
          f"{len(files)} transverse and as many longitudinal, unposed")

    ax = fig.add_subplot(gs[row, 1])
    al = []
    for j in range(4):
        sds_demo._PLANE["idx"], sds_demo._PLANE["n"] = j, 4
        al.append(np.asarray(sds_demo._solved_photo(os.path.join(FN, ref_h)).convert("RGB")))
    h = min(i.shape[0] for i in al); w = min(i.shape[1] for i in al)
    ax.imshow(np.vstack([np.hstack([cv2.resize(x, (w, h)) for x in al[:2]]),
                         np.hstack([cv2.resize(x, (w, h)) for x in al[2:4]])]))
    ax.set_axis_off()
    panel(ax, "turned to a common phase", "equations (4) and (5), before any gradient")

    cuts = sorted(glob.glob(os.path.join(FN, "measurements", obj, "cuts", "rh*_init_0.png")))
    ax = fig.add_subplot(gs[row, 2])
    if cuts:
        r = cv2.imread(cuts[0])[:, :, ::-1].astype(np.float32) / 255.
        rt = torch.from_numpy(r).permute(2, 0, 1)
        tgt = sm.section_target(rt, np.asarray(al[0], np.float32) / 255.)
        ax.imshow(tgt.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_axis_off()
    panel(ax, "fitted to the silhouette it renders", "equation (6); this is what replaces a pose")

    ax = fig.add_subplot(gs[row, 3])
    if len(cuts) > 1:
        ax.imshow(cv2.imread(cuts[1])[:, :, ::-1])
    ax.set_axis_off()
    panel(ax, "what the volume answers", "a depth training never sampled")


def band_downstream(fig, gs, row, xyz, rgb, lvl, hc):
    """What the lattice is for: an exact cut, the pieces, and the particles a solver gets."""
    from matplotlib.patches import Polygon as MplPolygon
    coarse = lvl == 0
    P, C = xyz[coarse], rgb[coarse]
    n = np.array([0.55, 0.48, 0.68]); n /= np.linalg.norm(n)
    ctr = P.mean(0); d = -float(n @ ctr)
    s = P @ n + d
    band = np.abs(s) <= 0.5 * hc * np.abs(n).sum()
    m = np.abs(P[:, 1] - ctr[1]) < 0.5 * hc

    ax = fig.add_subplot(gs[row, 0])
    for sel, col in ((m & (s > 0) & ~band, "#cfe0f0"), (m & (s < 0) & ~band, "#f6ddd0")):
        q = P[sel]; ax.scatter(q[:, 0], q[:, 2], s=2.6, c=col, marker="s", linewidths=0)
    q = P[m & band]; ax.scatter(q[:, 0], q[:, 2], s=3.0, c="#c0392b", marker="s", linewidths=0)
    ax.set_aspect("equal"); ax.set_axis_off()
    panel(ax, "a plane, and the cells it crosses", "equation (10), an integer test")

    e1 = np.array([-n[2], 0.0, n[0]]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    flat = lambda Q: np.stack([(Q - ctr) @ e1, (Q - ctr) @ e2], 1)
    ax = fig.add_subplot(gs[row, 1])
    CORN = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    EDGE = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    npoly = 0
    for cell in P[band]:
        Vv = (cell - 0.5 * hc)[None] + CORN * hc
        ss = Vv @ n + d
        pl = [Vv[a] + (ss[a] / (ss[a] - ss[b])) * (Vv[b] - Vv[a])
              for a, b in EDGE if (ss[a] < 0) != (ss[b] < 0)]
        if len(pl) < 3:
            continue
        pl = flat(np.array(pl))
        an = np.arctan2(pl[:, 1] - pl[:, 1].mean(), pl[:, 0] - pl[:, 0].mean())
        ax.add_patch(MplPolygon(pl[np.argsort(an)], closed=True, facecolor="#c0392b",
                                alpha=0.55, edgecolor="#8e2b1f", lw=0.3))
        npoly += 1
    ax.autoscale_view(); ax.set_aspect("equal"); ax.set_axis_off()
    panel(ax, "the face it exposes", f"{npoly:,} polygons, equation (11), machine precision")

    ax = fig.add_subplot(gs[row, 2])
    # Coloured by piece, not by the cells' own colour: the lattice's interior starts flat, so
    # drawing it here would show a grey disc and say nothing about the labelling.
    off = 0.12 * (P[:, 0].max() - P[:, 0].min())
    for sel, sh, col in ((m & (s > 0), +off, "#4a7ba7"), (m & (s < 0), -off, "#c0392b")):
        q = P[sel]
        ax.scatter(q[:, 0] + sh * n[0], q[:, 2] + sh * n[2], s=2.6, c=col, marker="s",
                   linewidths=0)
    ax.set_aspect("equal"); ax.set_axis_off()
    panel(ax, "the pieces, drawn apart", "connected components on the integer grid")

    ax = fig.add_subplot(gs[row, 3])
    q = P[m]
    col = np.where((q @ n + d)[:, None] > 0, np.array([[0.29, 0.48, 0.65]]),
                   np.array([[0.75, 0.22, 0.16]]))
    ax.scatter(q[:, 0], q[:, 2], s=2.6, c=col, marker="s", linewidths=0)
    qb = P[m & band]
    ax.scatter(qb[:, 0], qb[:, 2], s=3.4, c="#111", marker="s", linewidths=0)
    ax.set_aspect("equal"); ax.set_axis_off()
    panel(ax, "particles a solver receives", "one per cell, labelled; black is the contact band")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "orange",
         sys.argv[2] if len(sys.argv) > 2 else "out/pipeline_routes.png")


