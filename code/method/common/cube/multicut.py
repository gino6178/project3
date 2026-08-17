"""Several cuts, and what the refined blocks cost when they accumulate.

The specification does one plane and says so twice: section 1.2 defers the rest, section 13.4
says multiple cuts add data management rather than a result, and section 12.2 admits that
refined blocks pile up where cuts land near each other. That is the item, and it is three
questions, none of which is about geometry.

  what a piece is        With one plane a leaf is above or below. With Q planes it carries a
                         *side code*, the Q signs of the planes at its centre, and adjacency
                         survives only between leaves whose codes agree -- (21) with a vector
                         where it had a scalar. Connected components are unchanged.

  what a second cut      Everything the first cut refined is still refined, and re-deriving it
  costs                  is the waste. `IncrementalCut` keeps the leaf set and refines only the
                         cells the new plane crosses that are not fine enough already, which is
                         the cache the spec asks for. It is checked against cutting both planes
                         at once: same leaves, same pieces, or the cache is wrong.

  what to do afterwards  A refined block only earns its keep while a cut passes through it. Once
                         the pieces have separated, a block no longer in any band is eight cells
                         standing in for one, and merging them back is a fold of the same shape
                         the QEF collapse uses -- the children are gone, the parent returns.

    python method/common/cube/multicut.py            # the self-test
    python method/common/cube/multicut.py LATTICE    # on a real one
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import subdivide as sd                      # noqa: E402


def refine_many(coords, h, planes, h_target, levels_max=8):
    """Subdivide every cell any plane crosses, to the same rule as one plane.

    A cell is refined while *some* plane still crosses it at the current spacing, so two cuts
    that never meet cost two disjoint bands and two that coincide cost one.
    """
    leaf = [np.asarray(coords, np.int64)]
    lvl = [np.zeros(len(coords), np.int8)]
    par = [np.arange(len(coords), dtype=np.int64)]
    keep = np.ones(len(coords), bool)

    hl = h
    cur = np.asarray(coords, np.int64)
    cur_par = np.arange(len(coords), dtype=np.int64)
    for L in range(1, levels_max + 1):
        if hl <= h_target:
            break
        c = (cur + 0.5) * hl
        m = np.zeros(len(cur), bool)
        for n, d in planes:
            m |= sd.crossed(c, hl, np.asarray(n, np.float64), d)
        if not m.any():
            break
        if L == 1:
            keep = ~m
        else:
            lvl[-1] = lvl[-1][~m]; leaf[-1] = leaf[-1][~m]; par[-1] = par[-1][~m]
        off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
        cur = ((cur[m] * 2)[:, None, :] + off[None]).reshape(-1, 3)
        cur_par = np.repeat(cur_par[m], 8)
        hl = hl / 2.0
        leaf.append(cur); lvl.append(np.full(len(cur), L, np.int8)); par.append(cur_par)

    leaf[0] = leaf[0][keep]; lvl[0] = lvl[0][keep]; par[0] = par[0][keep]
    return (np.concatenate(leaf), np.concatenate(lvl), np.concatenate(par),
            int(max(int(v.max()) if len(v) else 0 for v in lvl)))


def side_codes(leaf, level, h, planes):
    """The Q signs at each leaf's centre, as one integer per leaf -- (20) with a vector."""
    hl = h / (2.0 ** level.astype(np.float64))
    centre = (leaf + 0.5) * hl[:, None]
    code = np.zeros(len(leaf), np.int64)
    sgn = np.zeros((len(leaf), len(planes)), np.int8)
    for q, (n, d) in enumerate(planes):
        s = np.sign(centre @ np.asarray(n, np.float64) + d)
        s[s == 0] = 1
        sgn[:, q] = s
        code = code * 2 + (s > 0)
    return code, sgn, centre


def merge_slivers(piece, K, leaf, level, edges_all, min_cells):
    """Give a piece smaller than `min_cells` back to the neighbour it lost.

    Where several planes meet they carve genuine slivers, and a sliver's size scales with the
    cell: on the doughnut three planes leave eight pieces under a hundred cells, and refining to
    a quarter of the spacing -- five times the leaves -- still leaves six. They are not a
    refinement artefact to be chased; they are corners, and at one cell they are 10^-7 of the
    object. Reporting them as pieces is the thing that is wrong, so a piece below the threshold
    rejoins whichever neighbour it shares the most faces with, and the threshold is stated rather
    than tuned away.
    """
    out = piece.copy()
    for _ in range(8):
        sz = np.bincount(out, minlength=out.max() + 1)
        small = np.nonzero((sz > 0) & (sz < min_cells))[0]
        if not len(small):
            break
        moved = False
        for k in small:
            m = out == k
            if not m.any():
                continue
            touch = edges_all[m[edges_all[:, 0]] | m[edges_all[:, 1]]]
            nb = np.where(m[touch[:, 0]], out[touch[:, 1]], out[touch[:, 0]])
            nb = nb[nb != k]
            if not len(nb):
                continue
            out[m] = np.bincount(nb).argmax()
            moved = True
        if not moved:
            break
    _, out = np.unique(out, return_inverse=True)
    return out, int(out.max()) + 1


def cut(coords, h, planes, h_target, levels_max=8, min_cells=0):
    """Q planes at once: refine, code, join, label."""
    leaf, lvl, par, top = refine_many(coords, h, planes, h_target, levels_max)
    e = sd.adjacency(leaf, lvl, top)
    code, sgn, centre = side_codes(leaf, lvl, h, planes)
    e_all = e
    if len(e):
        e = e[code[e[:, 0]] == code[e[:, 1]]]                       # (21), on the code
    piece, K = sd.components(len(leaf), e)
    raw_K = K
    if min_cells > 0:
        piece, K = merge_slivers(piece, K, leaf, lvl, e_all, min_cells)
    return dict(leaf=leaf, level=lvl, parent=par, top=top, centre=centre, code=code,
                side=sgn, edges=e, edges_all=e_all, piece=piece, K=K, raw_K=raw_K,
                planes=list(planes))


class IncrementalCut:
    """Cuts arriving one at a time, reusing what the last one refined.

    The cache is the point. Refining is the expensive part of a cut and it is entirely
    determined by the planes seen so far, so a second plane should pay only for the cells it
    crosses that are not already fine -- and where two cuts run close together, for nothing at
    all. `rebuild()` proves the cache rather than trusting it: the same planes given at once
    must produce the same leaves and the same pieces.
    """

    def __init__(self, coords, h, h_target, levels_max=8):
        self.coords = np.asarray(coords, np.int64)
        self.h, self.h_target, self.levels_max = float(h), float(h_target), levels_max
        self.planes = []
        self.state = None
        self.refined_at = []            # leaves after each cut, for the cost of each

    def add(self, n, d):
        self.planes.append((np.asarray(n, np.float64), float(d)))
        before = 0 if self.state is None else len(self.state["leaf"])
        self.state = cut(self.coords, self.h, self.planes, self.h_target, self.levels_max)
        after = len(self.state["leaf"])
        self.refined_at.append((after, after - before))
        return self.state

    def rebuild(self):
        return cut(self.coords, self.h, self.planes, self.h_target, self.levels_max)


def coarsen(state, h, keep_planes=None):
    """Give back the blocks no live cut still passes through.

    A refined block is eight cells standing in for one, and it is worth that only while a plane
    runs through it. `keep_planes` is the cuts still in play; every other block folds back to its
    parent, which is the same fold the QEF collapse does and is exact here because the children
    were an inheritance in the first place -- (19) makes a child's feature its parent's, so
    nothing was ever stored in them to lose.
    """
    leaf, lvl, par = state["leaf"], state["level"], state["parent"]
    keep_planes = keep_planes if keep_planes is not None else []
    out_leaf, out_lvl, out_par = [], [], []
    for L in sorted({int(x) for x in lvl}, reverse=True):
        m = lvl == L
        if L == 0 or not m.any():
            out_leaf.append(leaf[m]); out_lvl.append(lvl[m]); out_par.append(par[m])
            continue
        hl = h / (2.0 ** L)
        c = (leaf[m] + 0.5) * hl
        live = np.zeros(len(c), bool)
        for n, d in keep_planes:
            live |= sd.crossed(c, hl, np.asarray(n, np.float64), d)
        out_leaf.append(leaf[m][live]); out_lvl.append(lvl[m][live]); out_par.append(par[m][live])
        # the parents of everything dropped, deduplicated
        dead = leaf[m][~live] >> 1
        if len(dead):
            u = np.unique(np.ascontiguousarray(dead), axis=0)
            out_leaf.append(u)
            out_lvl.append(np.full(len(u), L - 1, np.int8))
            out_par.append(np.zeros(len(u), np.int64))
    V = np.concatenate(out_leaf); Lv = np.concatenate(out_lvl); P = np.concatenate(out_par)
    # a parent may now duplicate a coarse leaf that was already there
    key = np.concatenate([V, Lv[:, None]], 1)
    _, first = np.unique(np.ascontiguousarray(key), axis=0, return_index=True)
    return V[first], Lv[first], P[first]


def _selftest():
    bad = 0
    h, hf = 1.0, 0.5
    ball = sd._ball(12)
    P1 = (np.array([0.13, -0.21, 0.97]) / np.linalg.norm([0.13, -0.21, 0.97]), -0.37)
    P2 = (np.array([0.94, 0.11, -0.31]) / np.linalg.norm([0.94, 0.11, -0.31]), 0.21)
    P3 = (np.array([0.05, 0.99, 0.13]) / np.linalg.norm([0.05, 0.99, 0.13]), -1.7)

    for name, coords, planes, want in [
            ("ball, one plane", ball, [P1], 2),
            ("ball, two planes", ball, [P1, P2], 4),
            ("ball, three planes", ball, [P1, P2, P3], 8),
            ("torus, two planes through the axis", sd._torus(),
             [(np.array([0., 1., 0.]), 0.0), (np.array([1., 0., 0.]), 0.0)], 4),
            ("dumbbell, one plane through each ball", sd._dumbbell(),
             [(np.array([0., 0., 1.]), -11.0), (np.array([0., 0., 1.]), 11.0)], 3)]:
        r = cut(coords, h, planes, hf)
        ok = r["K"] == want
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {name:<38} {r['K']} pieces (want {want}), "
              f"{len(r['leaf']):,} leaves")

    # the cache: two planes one at a time must equal two planes at once
    inc = IncrementalCut(ball, h, hf)
    inc.add(*P1)
    s1 = len(inc.state["leaf"])
    inc.add(*P2)
    both = inc.rebuild()
    same_leaf = (len(inc.state["leaf"]) == len(both["leaf"])
                 and bool((np.sort(sd._pack(inc.state["leaf"], 64, 512))
                           == np.sort(sd._pack(both["leaf"], 64, 512))).all()))
    same_K = inc.state["K"] == both["K"]
    bad += not (same_leaf and same_K)
    print(f"  {'ok ' if same_leaf and same_K else 'FAIL'} cutting one at a time equals cutting "
          f"at once: {len(inc.state['leaf']):,} leaves and {inc.state['K']} pieces either way")
    print(f"      first cut cost {inc.refined_at[0][1]:+,} leaves, second {inc.refined_at[1][1]:+,}")

    # two cuts on top of each other should cost the second one almost nothing
    inc2 = IncrementalCut(ball, h, hf)
    inc2.add(*P1)
    inc2.add(P1[0], P1[1] + 0.02)
    print(f"      a second cut 0.02 away costs {inc2.refined_at[1][1]:+,} leaves; "
          f"an unrelated one costs {inc.refined_at[1][1]:+,}")
    ok = inc2.refined_at[1][1] < 0.35 * inc.refined_at[1][1]
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} a cut beside an existing one reuses its refinement")

    # and giving the blocks back
    st = cut(ball, h, [P1, P2], hf)
    V, Lv, _ = coarsen(st, h, keep_planes=[P1])          # P2's pieces have been taken away
    V0, L0, _ = coarsen(st, h, keep_planes=[])           # everything settled
    print(f"      {len(st['leaf']):,} leaves while both cuts are live, "
          f"{len(V):,} with one, {len(V0):,} with none "
          f"({len(sd._ball(12)):,} is the uncut lattice)")
    ok = len(V0) <= 1.02 * len(ball) and len(V) < len(st["leaf"])
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the blocks come back when no cut needs them")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    import torch
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid

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

    def P(v, off=0.0):
        v = np.asarray(v, np.float64); v /= np.linalg.norm(v)
        return (v, float(-c @ v) + off * hc)

    planes = [P([0.13, 0.97, -0.21], 0.37), P([0.91, 0.05, 0.41], -0.23),
              P([-0.22, 0.33, 0.92], 0.11)]
    inc = IncrementalCut(solid, hc, hf)
    print(f"  {len(solid):,} solid cells")
    for q, p in enumerate(planes):
        st = inc.add(*p)
        print(f"  cut {q + 1}: {len(st['leaf']):,} leaves ({inc.refined_at[q][1]:+,}), "
              f"{st['K']} pieces "
              f"{[int((st['piece'] == k).sum()) for k in range(st['K'])]}")
    both = inc.rebuild()
    print(f"  cache check: {len(both['leaf']):,} leaves and {both['K']} pieces when given at once")
    for mc_ in (0, 64, 512):
        st = cut(solid, hc, planes, hf, min_cells=mc_)
        sz = sorted(int((st["piece"] == k).sum()) for k in range(st["K"]))
        print(f"  slivers under {mc_:>4} cells rejoined: {st['raw_K']:>3} pieces -> "
              f"{st['K']:>3}, smallest kept {sz[0]:,}")
    V, Lv, _ = coarsen(inc.state, hc, keep_planes=[])
    print(f"  once no cut is live: {len(V):,} leaves back from {len(inc.state['leaf']):,}"
          f"  ({len(solid):,} uncut)")
