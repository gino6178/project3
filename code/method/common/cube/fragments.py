"""Exact cut-cell fragments: what the voxel approximation is actually losing.

Section 1.2's first omission. A leaf the plane passes through is handed whole to whichever side
its centre fell on, so the material on the other side is misassigned, and the only remedy the
first version has is to make the leaves smaller. Subdivision shrinks the error; clipping removes
it, by computing the volume that is really there.

A cube and a set of half-spaces both being convex, their intersection is a convex polyhedron and
its volume is a determinant sum. So for each cut leaf and each side code, clip the cube's vertex
set successively -- keep the vertices inside, add the crossing point on every edge that leaves --
and take the volume of the convex hull of what remains. Every code with a non-zero volume is a
fragment of the piece that code belongs to.

Emitting only the leaf's own side is the trap, and it is a quiet one: the total then *under*-counts
by exactly the material the voxel approximation was misassigning, which is the same error in a
different hat. Every code has to be emitted, and the fragments of one code accumulate into that
code's piece.

The criterion is a shape whose answer is known. A sphere of radius R cut by a plane at distance a
from its centre has a cap of volume pi (R-a)^2 (2R+a) / 3, so the relative error of a piece's
volume is a number and not an impression.

    python method/common/cube/fragments.py            # the self-test
    python method/common/cube/fragments.py LATTICE    # on a real one
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT]

from method.common.cube import subdivide as sd                      # noqa: E402

CUBE = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.float64)
EDGE = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                 if int(np.abs(CUBE[a] - CUBE[b]).sum()) == 1], np.int64)


def clip_half(pts, n, d, keep_positive):
    """The convex body's vertices after cutting away one half-space.

    Sutherland-Hodgman in three dimensions, on the vertex set rather than the faces: a convex
    body's clipped vertices are the ones it keeps plus the crossings on the edges of its hull.
    Running it on the *cube's* edges is exact for the first cut and an approximation afterwards,
    so the hull is recomputed between cuts instead.
    """
    s = pts @ n + d
    s = s if keep_positive else -s
    inside = s >= 0
    if inside.all():
        return pts
    if not inside.any():
        return pts[:0]
    out = [pts[inside]]
    if len(pts) == 8:
        pairs = EDGE
    else:
        from scipy.spatial import ConvexHull
        try:
            h = ConvexHull(pts)
            e = set()
            for simp in h.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        e.add((min(simp[a], simp[b]), max(simp[a], simp[b])))
            pairs = np.array(sorted(e), np.int64)
        except Exception:
            return pts[inside]
    a, b = pairs[:, 0], pairs[:, 1]
    cross = (s[a] > 0) != (s[b] > 0)
    if cross.any():
        t = s[a[cross]] / (s[a[cross]] - s[b[cross]])
        out.append(pts[a[cross]] + t[:, None] * (pts[b[cross]] - pts[a[cross]]))
    return np.concatenate(out)


def hull_volume(pts, tol=1e-12):
    if len(pts) < 4:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def cell_fragments(lo, h, planes):
    """Volume per side code for one cell. Codes with no volume are absent."""
    out = {}
    for code in range(1 << len(planes)):
        pts = lo[None, :] + CUBE * h
        for q, (n, d) in enumerate(planes):
            pts = clip_half(pts, np.asarray(n, np.float64), d,
                            bool(code >> (len(planes) - 1 - q) & 1))
            if len(pts) < 4:
                break
        v = hull_volume(pts)
        if v > tolerance(h):
            out[code] = v
    return out


def tolerance(h):
    return 1e-9 * h ** 3


def piece_volumes(state, h, planes, exact=True, limit=None):
    """Volume per piece, either by counting cells or by clipping the cut ones.

    The uncut leaves are whole and cost nothing to add up; only the leaves a plane crosses need
    clipping, which is the same O(N^(2/3)) band the refinement already pays for.
    """
    leaf, lvl, piece = state["leaf"], state["level"], state["piece"]
    code = state["code"]
    hl = h / (2.0 ** lvl.astype(np.float64))
    vol = np.zeros(state["K"])

    crossed = np.zeros(len(leaf), bool)
    for n, d in planes:
        crossed |= sd.crossed((leaf + 0.5) * hl[:, None], hl, np.asarray(n, np.float64), d)
    if not exact:
        np.add.at(vol, piece, hl ** 3)
        return vol, int(crossed.sum())

    whole = ~crossed
    np.add.at(vol, piece[whole], hl[whole] ** 3)

    # which piece each side code belongs to, read off the leaves that carry it
    by_code = {}
    for c in np.unique(code):
        m = code == c
        by_code[int(c)] = int(np.bincount(piece[m]).argmax())

    idx = np.nonzero(crossed)[0]
    if limit:
        idx = idx[np.linspace(0, len(idx) - 1, min(limit, len(idx))).astype(int)]
        scale = float(int(crossed.sum())) / len(idx)
    else:
        scale = 1.0
    for i in idx:
        for c, v in cell_fragments(leaf[i] * hl[i], hl[i], planes).items():
            k = by_code.get(c)
            if k is not None:
                vol[k] += v * scale
    return vol, len(idx)


def _cap_volume(R, a):
    """The smaller part of a sphere of radius R cut at distance a from its centre."""
    return np.pi * (R - a) ** 2 * (2 * R + a) / 3.0


def _selftest():
    bad = 0
    h = 1.0
    R = 12.0
    ball = sd._ball(int(R))
    off = 0.37                      # of a cell, so the plane never lands on a boundary
    n = np.array([0.0, 0.0, 1.0])

    c = (ball + 0.5) * h
    ctr = c.mean(0)
    d = float(-ctr @ n - off * h)
    a = off * h
    want_small = _cap_volume(R, a)
    want_big = 4.0 / 3.0 * np.pi * R ** 3 - want_small
    print(f"  ball R = {R}, plane {off} of a cell off centre; the closed form gives "
          f"{want_small:.2f} and {want_big:.2f}")

    from method.common.cube import multicut as mc
    print(f"  {'h_target':>10}{'leaves':>12}{'voxel-approx err':>19}{'exact-fragment err':>21}")
    prev = None
    for name, ht in (("none", h), ("h/2", h / 2), ("h/4", h / 4), ("h/8", h / 8)):
        st = mc.cut(ball, h, [(n, d)], ht)
        va, ncut = piece_volumes(st, h, [(n, d)], exact=False)
        ex, nc2 = piece_volumes(st, h, [(n, d)], exact=True)
        va = np.sort(va); ex = np.sort(ex)
        e_v = abs(va[0] - want_small) / want_small
        e_e = abs(ex[0] - want_small) / want_small
        print(f"  {name:>10}{len(st['leaf']):>12,}{e_v:>18.5f}{e_e:>20.5f}")
        if prev is not None:
            bad += not (e_e <= prev + 1e-6)
        prev = e_e

    st = mc.cut(ball, h, [(n, d)], h / 4)
    ex, _ = piece_volumes(st, h, [(n, d)], exact=True)
    va, _ = piece_volumes(st, h, [(n, d)], exact=False)
    tot_e, tot_v = ex.sum(), va.sum()
    want_tot = 4.0 / 3.0 * np.pi * R ** 3
    ok = abs(tot_e - tot_v) / tot_v < 1e-6
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the fragments conserve the volume: exact {tot_e:.2f} "
          f"against counted {tot_v:.2f}, sphere {want_tot:.2f}")

    # every code has to be emitted, or the total under-counts by what the voxel test misassigns
    st = mc.cut(ball, h, [(n, d)], h)
    one_side = 0.0
    for i in np.nonzero(sd.crossed((st["leaf"] + 0.5) * h, h, n, d))[0]:
        fr = cell_fragments(st["leaf"][i] * h, h, [(n, d)])
        own = st["code"][i]
        one_side += fr.get(int(own), 0.0)
    both = sum(sum(cell_fragments(st["leaf"][i] * h, h, [(n, d)]).values())
               for i in np.nonzero(sd.crossed((st["leaf"] + 0.5) * h, h, n, d))[0])
    ok = one_side < 0.75 * both
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} emitting only the leaf's own side loses "
          f"{100 * (1 - one_side / both):.1f}% of the band's volume")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    import torch
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid
    from method.common.cube import multicut as mc

    ld = sys.argv[1]
    lat = torch.load(_os.path.join(ld, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(ld, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(ld, "cell_level.pt")).reshape(-1)[:len(xyz)].numpy()
    # from a corner, not from a centre: floor of a centre sitting exactly on a cell
    # boundary lets floating point choose the side, and on a lattice whose cells are at
    # (i + 1/2)h that discards 49% of them. Offset by half the finest spacing used here.
    org = xyz[lvl == 0].min(0) - 0.5 * hf
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1
    c = ((solid + 0.5) * hc).mean(0)
    n = np.array([0.13, 0.97, -0.21]); n /= np.linalg.norm(n)
    d = float(-c @ n) + 0.37 * hc

    st = mc.cut(solid, hc, [(n, d)], hf)
    va, ncut = piece_volumes(st, hc, [(n, d)], exact=False)
    ex, nsub = piece_volumes(st, hc, [(n, d)], exact=True, limit=20000)
    print(f"  {len(solid):,} solid cells, {ncut:,} crossed, {st['K']} pieces "
          f"(clipping {nsub:,} of them)")
    print(f"  counted cells : {np.sort(va).round(6).tolist()}   total {va.sum():.6f}")
    print(f"  exact fragments: {np.sort(ex).round(6).tolist()}   total {ex.sum():.6f}")
    print(f"  the split moves by {100 * abs(np.sort(va)[0] / va.sum() - np.sort(ex)[0] / ex.sum()):.3f}"
          f" percentage points of the object")
