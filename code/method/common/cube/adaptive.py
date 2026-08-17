"""Refine a cut cell by how wrong it is, not by how deep the band is.

Section 1.2's second omission. A fixed block spends the deepest level any cut cell needed on
every cut cell, and what actually varies across a band is how badly a leaf is *misclassified*.
That has an exact form now that the fragments exist: a leaf is handed whole to one side, so the
error is everything on the other sides,

    e = h^3 - max_c vol_c,

the leaf's volume minus the largest exactly-clipped side. Refine only while e exceeds a
tolerance, and `level` stops being a depth and becomes a ceiling.

The criterion has to be cheap or it costs more than it saves, and for one plane it is closed
form. Clip the unit cube by a half-space and the volume is an inclusion-exclusion over the
corners,

    V = 1/(6 n1 n2 n3) [ s+^3 - sum_i (s-ni)+^3 + sum_i<j (s-ni-nj)+^3 - (s-n1-n2-n3)+^3 ],

with n taken componentwise positive by symmetry and s the plane's value at the low corner. No
hull, no tetrahedra, and it vectorises over every cell in the band at once. `_selftest` checks it
against the convex hull on random cubes and planes rather than trusting the algebra.

    python method/common/cube/adaptive.py            # the self-test
    python method/common/cube/adaptive.py LATTICE    # on a real one
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT]

from method.common.cube import subdivide as sd                      # noqa: E402


def cube_plane_volume(lo, h, n, d):
    """Volume of the part of each cube with n.x + d >= 0, in closed form.

    The identity is the inclusion-exclusion over corners, in as many dimensions as the normal
    actually has. For k axes with a non-zero component, the volume of the cube below a threshold
    t along a = |n| is

        V_k(t) = 1/(k! prod a_i) sum_{S subset [k]} (-1)^|S| (t - sum_{i in S} a_i h)_+^k

    times h^(3-k) for the axes the plane does not vary along. Two mistakes are worth marking
    because each produced a plausible table before it produced a wrong answer. The formula gives
    the volume *below* the threshold, not above, so returning it directly reports the wrong side
    of every cut. And a degenerate axis cannot be nudged: setting a_k to 1e-12 makes the bracket
    a third difference of numbers of order one, so float64 cancellation destroys it entirely and
    an axis-aligned plane reports zero volume, which reads as "nothing to refine" rather than as
    a failure. Dropping to k dimensions is exact and has no epsilon in it.
    """
    n = np.asarray(n, np.float64)
    lo = np.atleast_2d(np.asarray(lo, np.float64))
    a = np.abs(n)
    big = a > 1e-9 * max(a.max(), 1e-300)
    k = int(big.sum())
    if k == 0:
        return np.full(len(lo), h ** 3 if d >= 0 else 0.0)

    # y = x - lo in [0,h]^3; reflect the axes with a negative component so every one is positive
    t = -(lo @ n + d) + float((a[(n < 0)]).sum()) * h
    aa = a[big]

    def below(t):
        v = np.zeros_like(t)
        for mask in range(1 << k):
            sub = sum(aa[i] for i in range(k) if mask >> i & 1)
            sgn = -1.0 if bin(mask).count("1") % 2 else 1.0
            v = v + sgn * np.maximum(t - sub * h, 0.0) ** k
        fact = {1: 1.0, 2: 2.0, 3: 6.0}[k]
        return v / (fact * float(np.prod(aa))) * (h ** (3 - k))

    return np.clip(h ** 3 - below(t), 0.0, h ** 3)


def misclassified(lo, h, planes):
    """h^3 minus the largest exactly-clipped side: what handing the cell to one side costs."""
    if len(planes) == 1:
        n, d = planes[0]
        v = cube_plane_volume(lo, h, n, d)
        return np.minimum(v, h ** 3 - v)
    from method.common.cube.fragments import cell_fragments
    out = np.zeros(len(lo))
    for i in range(len(lo)):
        fr = cell_fragments(lo[i], h, planes)
        out[i] = h ** 3 - (max(fr.values()) if fr else h ** 3)
    return out


def refine_adaptive(coords, h, planes, tol, levels_max=8, h_floor=None):
    """Subdivide a crossed cell only while it is still getting more than `tol` wrong.

    `tol` is a fraction of a *coarse* cell's volume, so it means the same thing at every level
    and on every object. `h_floor` is the ceiling in disguise -- the finest spacing allowed --
    and the point of the criterion is that most cells stop well above it.
    """
    h_floor = h_floor or h / (2.0 ** levels_max)
    thresh = tol * h ** 3

    leaf = [np.asarray(coords, np.int64)]
    lvl = [np.zeros(len(coords), np.int8)]
    keep = np.ones(len(coords), bool)
    cur, hl = np.asarray(coords, np.int64), h

    for L in range(1, levels_max + 1):
        if hl <= h_floor:
            break
        c = (cur + 0.5) * hl
        crossed = np.zeros(len(cur), bool)
        for n, d in planes:
            crossed |= sd.crossed(c, hl, np.asarray(n, np.float64), d)
        m = crossed.copy()
        if m.any():
            e = np.zeros(len(cur))
            e[m] = misclassified(cur[m] * hl, hl, planes)
            m &= e > thresh
        if not m.any():
            break
        if L == 1:
            keep = ~m
        else:
            lvl[-1] = lvl[-1][~m]; leaf[-1] = leaf[-1][~m]
        off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
        cur = ((cur[m] * 2)[:, None, :] + off[None]).reshape(-1, 3)
        hl = hl / 2.0
        leaf.append(cur); lvl.append(np.full(len(cur), L, np.int8))

    leaf[0] = leaf[0][keep]; lvl[0] = lvl[0][keep]
    return (np.concatenate(leaf), np.concatenate(lvl),
            int(max(int(v.max()) if len(v) else 0 for v in lvl)))


def cut(coords, h, planes, tol, levels_max=8, h_floor=None, min_cells=0):
    """The adaptive band, then the same coding, joining and labelling as everywhere else."""
    from method.common.cube import multicut as mc
    leaf, lvl, top = refine_adaptive(coords, h, planes, tol, levels_max, h_floor)
    e = sd.adjacency(leaf, lvl, top)
    code, sgn, centre = mc.side_codes(leaf, lvl, h, planes)
    e_all = e
    if len(e):
        e = e[code[e[:, 0]] == code[e[:, 1]]]
    piece, K = sd.components(len(leaf), e)
    raw_K = K
    if min_cells > 0:
        piece, K = mc.merge_slivers(piece, K, leaf, lvl, e_all, min_cells)
    return dict(leaf=leaf, level=lvl, top=top, centre=centre, code=code, side=sgn,
                edges=e, edges_all=e_all, piece=piece, K=K, raw_K=raw_K, planes=list(planes))


def band_error(state, h, planes):
    """Total misclassified volume left in the band, which is what both schemes are trading."""
    hl = h / (2.0 ** state["level"].astype(np.float64))
    tot = 0.0
    for L in sorted({int(x) for x in state["level"]}):
        m = state["level"] == L
        hh = h / (2.0 ** L)
        c = (state["leaf"][m] + 0.5) * hh
        cr = np.zeros(int(m.sum()), bool)
        for n, d in planes:
            cr |= sd.crossed(c, hh, np.asarray(n, np.float64), d)
        if cr.any():
            tot += float(misclassified(state["leaf"][m][cr] * hh, hh, planes).sum())
    return tot


def _selftest():
    bad = 0
    rng = np.random.default_rng(0)

    # the closed form against the hull, on random cubes and planes
    from method.common.cube.fragments import CUBE, clip_half, hull_volume
    err = 0.0
    for _ in range(300):
        lo = rng.normal(size=3)
        h = float(rng.uniform(0.3, 3.0))
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        d = float(-n @ (lo + 0.5 * h) + rng.uniform(-0.9, 0.9) * h)
        v1 = float(cube_plane_volume(lo[None], h, n, d)[0])
        v2 = hull_volume(clip_half(lo[None] + CUBE * h, n, d, True))
        err = max(err, abs(v1 - v2) / h ** 3)
    ok = err < 1e-9
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the closed form matches the hull on 300 random cubes "
          f"(worst {err:.2e} of a cell)")

    h = 1.0
    from method.common.cube import multicut as mc
    for name, coords, planes in [
            ("ball, oblique", sd._ball(12),
             [(np.array([0.13, -0.21, 0.97]) / np.linalg.norm([0.13, -0.21, 0.97]), -0.37)]),
            ("ball, axis-aligned", sd._ball(12), [(np.array([0., 0., 1.]), -0.37)]),
            ("dumbbell, through one ball", sd._dumbbell(), [(np.array([0., 0., 1.]), -11.37)])]:
        print(f"  {name}")
        for L in (1, 2, 3):
            u = mc.cut(coords, h, planes, h / 2 ** L)
            eu = band_error(u, h, planes)
            best = None
            for tol in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                        5e-4, 2e-4, 1e-4, 1e-5):
                a = cut(coords, h, planes, tol, h_floor=h / 2 ** L)
                if band_error(a, h, planes) <= eu * 1.02:
                    best = (tol, a, band_error(a, h, planes))
                    break
            if best is None:
                print(f"      uniform level {L}: {len(u['leaf']):>9,} leaves, "
                      f"error {eu:.5f} -- adaptive did not reach it")
                continue
            tol, a, ea = best
            same = a["K"] == u["K"]
            bad += not same
            print(f"      uniform level {L}: {len(u['leaf']):>9,} leaves, error {eu:.5f}"
                  f"   adaptive tol {tol:<6}: {len(a['leaf']):>9,} leaves, error {ea:.5f}"
                  f"   saved {100 * (1 - len(a['leaf']) / len(u['leaf'])):>5.1f}%"
                  f"   pieces {u['K']}/{a['K']} {'ok' if same else 'DIFFER'}")
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
    planes = [(n, float(-c @ n) + 0.37 * hc)]

    print(f"  {len(solid):,} solid cells, h_c {hc:.5f}")
    for L in (1, 2):
        u = mc.cut(solid, hc, planes, hc / 2 ** L)
        eu = band_error(u, hc, planes)
        print(f"  uniform to h_c/{2 ** L}: {len(u['leaf']):>10,} leaves, "
              f"misclassified {eu:.6f}, {u['K']} pieces")
        for tol in (0.2, 0.1, 0.05, 0.02):
            a = cut(solid, hc, planes, tol, h_floor=hc / 2 ** L)
            ea = band_error(a, hc, planes)
            print(f"      tol {tol:<5}: {len(a['leaf']):>10,} leaves, misclassified {ea:.6f}"
                  f"  ({100 * (1 - len(a['leaf']) / len(u['leaf'])):>5.1f}% fewer), "
                  f"{a['K']} pieces")
