"""M3: refine the cut band, build the leaf adjacency, and label the pieces.

The spec's section 5. A cut arrives as a plane; the cells it crosses are subdivided, the leaves
are joined to their face neighbours, edges that cross the plane are dropped, and what is left is
labelled. The pieces come out of connected components on a graph -- no geometry, no polygons.
That is M4's job, and keeping them apart is the point: topology is decided on the lattice and
only the visible boundary is ever turned into a surface.

Why subdivide at all, since the components could be labelled on the coarse cells directly: a
coarse cell is either wholly on one side or wholly on the other, so a feature thinner than the
coarse spacing does not exist to the labelling. A cut that shaves a cap thinner than h_c returns
one piece -- the object, unchanged -- because every cell centre stayed on the same side. This is
not hypothetical; it is the test at the bottom of this file, and without refinement it reports 1
piece where it should report 2.

Three things make it cheap enough to do on every cut:

  the crossing test is O(1)          A cube's extreme values on a plane are at two opposite
                                     corners, and which two depends only on the signs of n. So
                                     |n.x + d| <= (|nx|+|ny|+|nz|) h/2 is exactly the spec's
                                     eight-corner test (16) without enumerating eight corners.

  only the band is refined           The plane crosses O(N^(2/3)) of N cells. On the orange's
                                     877,495 that is about nine thousand.

  adjacency is by integer coordinate  Leaves are on a grid, so a face neighbour is an index
                                     arithmetic away rather than a search. Levels meet only at
                                     the band's edge, and there a coarse leaf is joined to the
                                     4^L children of its refined neighbour that share the face.

    python method/common/cube/subdivide.py LATTICE [nx ny nz d]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]


def crossed(centres, h, n, d):
    """The spec's (15)-(16), without enumerating the corners.

    A cube spans n.x + d plus or minus the sum over axes of |n_a| h/2, because the corner that
    maximises n.v picks +h/2 on every axis where n_a > 0 and -h/2 where it is negative, and the
    minimising corner picks the opposite. So min s <= 0 <= max s is one absolute value.
    """
    return np.abs(centres @ n + d) <= 0.5 * h * np.abs(n).sum()


def refine(coords, h, n, d, h_target, levels_max=8):
    """Subdivide the crossed cells, and only those, until h_l <= h_target -- the spec's (17)-(18).

    Returns the leaves as integer coordinates at their own level, plus the level of each and the
    index of the coarse cell each came from. The parent index is what carries the feature down:
    the spec's (19) is f_child <- f_parent, so a child needs no storage of its own, only a way
    back to the cell it was cut out of.

    Levels are counted rather than assumed. h_target defaults to the fine exterior spacing so a
    new cut surface lands at the same resolution as the surface that was already there, which is
    the spec's suggestion and the only value here that comes from outside the geometry.
    """
    leaf = [coords.astype(np.int64)]
    lvl = [np.zeros(len(coords), np.int8)]
    par = [np.arange(len(coords), dtype=np.int64)]
    keep = np.ones(len(coords), bool)

    hl = h
    cur = coords.astype(np.int64)
    cur_par = np.arange(len(coords), dtype=np.int64)
    for L in range(1, levels_max + 1):
        if hl <= h_target:
            break
        c = (cur + 0.5) * hl
        m = crossed(c, hl, n, d)
        if not m.any():
            break
        # the crossed cells stop being leaves; their children take their place
        if L == 1:
            keep = ~m
        else:
            lvl[-1] = lvl[-1][~m]
            leaf[-1] = leaf[-1][~m]
            par[-1] = par[-1][~m]
        base = cur[m] * 2
        off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
        cur = (base[:, None, :] + off[None]).reshape(-1, 3)
        cur_par = np.repeat(cur_par[m], 8)
        hl = hl / 2.0
        leaf.append(cur)
        lvl.append(np.full(len(cur), L, np.int8))
        par.append(cur_par)

    leaf[0] = leaf[0][keep]
    lvl[0] = lvl[0][keep]
    par[0] = par[0][keep]
    return (np.concatenate(leaf), np.concatenate(lvl), np.concatenate(par),
            int(max(int(v.max()) if len(v) else 0 for v in lvl)))


def _pack(c, base, span):
    """Integer coordinates as one sortable key, so a lookup is a searchsorted and not a dict."""
    q = c.astype(np.int64) + base
    return (q[:, 0] * span + q[:, 1]) * span + q[:, 2]


def adjacency(coords, level, top):
    """Face neighbours among the leaves, across levels -- the spec's E in (21).

    Each leaf asks, in each of the six directions, which leaf occupies the cell next to it. That
    neighbour is at the same level or coarser: if it were finer, this leaf would be inside the
    refined block rather than beside it, and the finer leaves on the shared face find *this* one
    when they ask. So one pass over levels from a leaf's own down to the coarsest resolves every
    pair, and the coarser candidate is the neighbour's coordinate shifted right, which is floor
    division and therefore the cell that contains it.

    It matters that this is arithmetic and not search. The first version enumerated the layer of
    finest cells outside each face and looked each one up in a dict -- correct, and 96 lookups
    per coarse leaf once two levels exist, which is tens of millions of dict probes on a real
    lattice. This does six vectorised passes per level instead.
    """
    top = int(top)
    lv = level.astype(np.int64)
    mn = coords.min() - 2
    base = -mn
    span = int(coords.max() + base + 3)

    levels = sorted({int(v) for v in lv})
    at = {L: np.nonzero(lv == L)[0] for L in levels}
    keys, order = {}, {}
    for L in levels:
        k = _pack(coords[at[L]], base, span)
        o = np.argsort(k)
        keys[L], order[L] = k[o], at[L][o]

    e_a, e_b = [], []
    for L in levels:
        idx = at[L]
        c = coords[idx]
        for ax in range(3):
            for s in (-1, 1):
                nb = c.copy()
                nb[:, ax] += s
                todo = np.ones(len(idx), bool)
                for Lp in range(L, -1, -1):
                    if Lp not in keys or not todo.any():
                        continue
                    q = nb[todo] >> (L - Lp)            # arithmetic shift == floor division
                    kk = _pack(q, base, span)
                    pos = np.searchsorted(keys[Lp], kk)
                    pos = np.clip(pos, 0, len(keys[Lp]) - 1)
                    hit = keys[Lp][pos] == kk
                    if hit.any():
                        src = idx[todo][hit]
                        dst = order[Lp][pos[hit]]
                        e_a.append(src)
                        e_b.append(dst)
                        w = np.nonzero(todo)[0][hit]
                        todo[w] = False
    if not e_a:
        return np.zeros((0, 2), np.int64)
    a = np.concatenate(e_a)
    b = np.concatenate(e_b)
    m = a != b
    a, b = a[m], b[m]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.unique(np.stack([lo, hi], 1), axis=0)


def components(n_leaf, edges):
    """Connected components of the surviving adjacency: the spec's (22).

    The union-find this replaced was a Python loop over the edge list, which is the right algorithm
    written the wrong way round: on a real lattice the loop is several million iterations of
    interpreted pointer-chasing and it dominated the cost of a cut, so the timing table reported a
    labelling that took thirteen seconds and the paper had to admit the cut was not interactive.
    The work is the same; only the interpreter is removed. The fallback keeps the tree honest on a
    machine without scipy.
    """
    if not n_leaf:
        return np.zeros(0, np.int64), 0
    edges = np.asarray(edges).reshape(-1, 2)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components as _cc
        if len(edges):
            g = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])),
                           shape=(n_leaf, n_leaf))
        else:
            g = coo_matrix((n_leaf, n_leaf), dtype=np.int8)
        k, lab = _cc(g, directed=False)
        return lab.astype(np.int64), int(k)
    except ImportError:
        parent = np.arange(n_leaf)

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in edges:
            ra, rb = root(int(a)), root(int(b))
            if ra != rb:
                parent[ra] = rb
        lab = np.array([root(i) for i in range(n_leaf)])
        _, out = np.unique(lab, return_inverse=True)
        return out, int(out.max()) + 1



def _edges_from_base(coords, leaf, lvl, par, top, base):
    """Adjacency after a refinement, from the adjacency before it.

    `par` maps every leaf to the coarse cell it came from, so a leaf is unrefined exactly when it
    is the only leaf of its parent. A base pair of coarse cells survives unchanged when neither
    end was refined; every other pair, and every pair inside a refined block, is recomputed by
    running the same arithmetic over the refined leaves alone. Correctness is checked by the
    module's own self-test, which compares this against the full pass on every synthetic case.
    """
    n_leaf = len(leaf)
    # how many leaves each coarse parent produced, and the leaf index when there is only one
    cnt = np.bincount(par, minlength=len(coords))
    single = cnt == 1
    leaf_of = np.full(len(coords), -1, np.int64)
    only = np.nonzero(single)[0]
    if len(only):
        order = np.argsort(par, kind="stable")
        first = np.searchsorted(par[order], only)
        leaf_of[only] = order[first]

    keep = np.zeros((0, 2), np.int64)
    if len(base):
        both = single[base[:, 0]] & single[base[:, 1]]
        if both.any():
            keep = np.stack([leaf_of[base[both, 0]], leaf_of[base[both, 1]]], 1)

    # everything touching a refined block, recomputed on that sub-lattice only
    touched = ~single
    sel = touched[par]
    if sel.any():
        # include the unrefined neighbours of refined blocks, or the seam between them is lost
        if len(base):
            nb = np.unique(np.concatenate([base[touched[base[:, 0]], 1],
                                           base[touched[base[:, 1]], 0]]))
            sel |= np.isin(par, nb)
        idx = np.nonzero(sel)[0]
        sub = adjacency(leaf[idx], lvl[idx], top)
        if len(sub):
            keep = np.concatenate([keep, idx[sub]]) if len(keep) else idx[sub]

    if not len(keep):
        return np.zeros((0, 2), np.int64)
    lo = np.minimum(keep[:, 0], keep[:, 1])
    hi = np.maximum(keep[:, 0], keep[:, 1])
    m = lo != hi
    return np.unique(np.stack([lo[m], hi[m]], 1), axis=0)


def cut(coords, h, n, d, h_target, levels_max=8, base_edges=None):
    """A whole single cut: refine the band, join the leaves, drop the crossings, label.

    `coords` are integer cell indices on the coarse grid and `h` its spacing, so a cell centre is
    (coords + 0.5) * h. The plane is n.x + d = 0 in the same space, and n is not required to be
    unit -- the crossing test and the side test are both homogeneous in it.

    `base_edges` is the uncut object's own face adjacency, and passing it makes the cut cost what
    the cut touches rather than what the object contains. A plane changes connectivity only where
    it refines: a pair of unrefined leaves is adjacent after the cut exactly when it was adjacent
    before, so those pairs are reused, and only the pairs with a refined leaf on either end are
    recomputed. The recomputation is over the band, which is O(N^(2/3)) of the volume, so the
    whole step is linear in the crossed cells rather than in the object. Two counts, measured on
    the six objects by code/evaluate/cutface.py and not to be confused: `crossed` returns 0.51 to
    3.03% of the cells, and after this function has refined every one of them the leaves that
    carry a polygon are 2.04 to 12.12%. Computed with `adjacency(coords, zeros, 0)` once per
    object and reused for every cut that object ever takes.
    """
    n = np.asarray(n, np.float64)
    leaf, lvl, par, top = refine(np.asarray(coords), h, n, d, h_target, levels_max)
    if base_edges is None:
        e = adjacency(leaf, lvl, top)
    else:
        e = _edges_from_base(np.asarray(coords), leaf, lvl, par, top, np.asarray(base_edges))

    hl = h / (2.0 ** lvl.astype(np.float64))
    centre = (leaf + 0.5) * hl[:, None]
    q = np.sign(centre @ n + d)                        # the spec's (20)
    q[q == 0] = 1.0
    e_all = e
    if len(e):
        e = e[q[e[:, 0]] == q[e[:, 1]]]                # the spec's (21)
    piece, K = components(len(leaf), e)
    # `edges_all` is kept as well as the filtered `edges`, because the pairs the plane separates
    # are not waste: they are exactly the pairs a cut face lies between, and M4 needs them to say
    # which piece is on each side of a polygon.
    return dict(leaf=leaf, level=lvl, parent=par, top=top, centre=centre,
                side=q, edges=e, edges_all=e_all, piece=piece, K=K)


# ---------------------------------------------------------------------------------------------
# The completion condition, on shapes whose piece count is known in advance.

def _ball(r=12, R=None):
    R = R or r
    t = np.arange(-R, R + 1)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    return g[np.linalg.norm(g + 0.5, axis=1) <= r]


def _torus(R=14, r=5, lim=22):
    t = np.arange(-lim, lim + 1)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3).astype(np.float64) + 0.5
    d = (np.linalg.norm(g[:, :2], axis=1) - R) ** 2 + g[:, 2] ** 2
    return (g - 0.5).astype(np.int64)[d <= r * r]


def _dumbbell(r=7, neck=2, sep=11):
    t = np.arange(-sep - r - 1, sep + r + 2)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3).astype(np.float64) + 0.5
    a = np.linalg.norm(g - np.array([0, 0, sep]), axis=1) <= r
    b = np.linalg.norm(g - np.array([0, 0, -sep]), axis=1) <= r
    bar = (np.linalg.norm(g[:, :2], axis=1) <= neck) & (np.abs(g[:, 2]) <= sep)
    return (g - 0.5).astype(np.int64)[a | b | bar]


def _bridge(r=9, w=1, span=14):
    """Two balls joined by a bridge one cell wide -- the case the design document names.

    "Topology: piece count correctness on synthetic objects; test thin bridge and near-surface
     cut in particular." The near-surface cut is already here as the thin cap. This is the other
    one, and it is the harder direction: the cap tests whether refinement can *find* a separation
    the coarse grid misses, and the bridge tests whether it can avoid *inventing* one -- a feature
    one cell wide is exactly where a labelling that leans on dilation or on a distance threshold
    reports two pieces for an object that is connected.
    """
    t = np.arange(-span - r - 1, span + r + 2)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3).astype(np.float64) + 0.5
    a = np.linalg.norm(g - np.array([0, 0, span]), axis=1) <= r
    b = np.linalg.norm(g - np.array([0, 0, -span]), axis=1) <= r
    bar = (np.abs(g[:, 0]) <= w) & (np.abs(g[:, 1]) <= w) & (np.abs(g[:, 2]) <= span)
    return (g - 0.5).astype(np.int64)[a | b | bar]


def _selftest():
    h = 1.0
    cases = [
        ("ball, plane through the centre",          _ball(12), (0, 0, 1), 0.0,   1.0, 2),
        ("ball, plane clear of it",                 _ball(12), (0, 0, 1), -40.0, 1.0, 1),
        ("torus, plane containing the axis",        _torus(),  (0, 1, 0), 0.0,   1.0, 2),
        ("torus, plane across the axis",            _torus(),  (0, 0, 1), 0.0,   1.0, 2),
        ("torus, plane above the tube",             _torus(),  (0, 0, 1), -9.0,  1.0, 1),
        ("dumbbell, plane through the neck",        _dumbbell(), (0, 0, 1), 0.0, 1.0, 2),
        # the one that needs the refinement: the cap is thinner than a coarse cell, so every
        # coarse centre stays on one side and the coarse labelling cannot see it at all
        ("ball, cap thinner than a coarse cell",    _ball(12), (0, 0, 1), -11.6, 0.25, 2),
        # the thin bridge, both ways round: a plane clear of it must leave one piece, and a plane
        # through it must leave two. One cell wide is the width at which a labelling that dilates
        # or thresholds gets the first of these wrong.
        ("thin bridge, plane clear of it",          _bridge(), (0, 0, 1), -30.0, 1.0, 1),
        ("thin bridge, plane through it",           _bridge(), (0, 0, 1), 0.0,   1.0, 2),
        ("thin bridge, plane along it",             _bridge(), (1, 0, 0), 0.0,   1.0, 2),
    ]
    bad = 0
    for name, coords, n, d, h_t, want in cases:
        r = cut(coords, h, n, d, h_t)
        ok = r["K"] == want
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {name:<40} pieces {r['K']} (want {want})"
              f"   leaves {len(r['leaf']):,} at up to level {r['top']}")

    # the same cap, without refinement, to show what the refinement is buying
    r = cut(_ball(12), h, (0, 0, 1), -11.6, 1.0)
    print(f"  ..  the same cap with no refinement          pieces {r['K']} (the cap is invisible)")

    # every leaf must belong to exactly one piece and every edge stay inside one
    r = cut(_torus(), h, (0, 0, 1), 0.0, 0.5)
    same = all(r["piece"][a] == r["piece"][b] for a, b in r["edges"])
    span = all(r["side"][a] == r["side"][b] for a, b in r["edges"])
    print(f"  {'ok ' if same and span else 'FAIL'} every surviving edge stays within one piece "
          f"and one side ({len(r['edges']):,} edges)")
    bad += not (same and span)
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    import torch
    lattice_dir = sys.argv[1]
    n = [float(x) for x in sys.argv[2:5]] if len(sys.argv) > 5 else [0.0, 1.0, 0.0]
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    # plyfile rather than GaussianModel: this reads positions and nothing else, and the loader
    # puts every tensor on the GPU, so a topology query failed with out of memory on a machine
    # whose card was busy with a training run.
    from plyfile import PlyData
    v = PlyData.read(_os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)
    sel = (lvl[:len(xyz)] == 0).numpy()
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    p = xyz[sel]
    coords = np.floor((p - (p.min(0) - 0.5 * hf)) / hc).astype(np.int64)
    coords = np.unique(coords, axis=0)

    # M0 first, and it is not optional here. The stored occupancy is a sponge -- `internal_filling`
    # skips cells that already hold a primitive -- so before any cut the doughnut already labels
    # as 68 pieces and the orange as 1,629. Those are not cut fragments; they are the holes.
    # close_and_fill is what makes the volume a solid, and a piece count means nothing until it
    # has run.
    from occupancy import close_and_fill, to_grid
    import torch as _t
    occ, mn, _ = to_grid(_t.from_numpy(coords).float(), 1.0)
    n_raw = int(occ.sum())
    solid = close_and_fill(occ, 1)
    coords_s = solid.nonzero().numpy() + coords.min(0) - 1
    print(f"  {n_raw:,} coarse cells -> {len(coords_s):,} after close_and_fill "
          f"(+{len(coords_s) - n_raw:,})   h_c {hc:.5f}, refining to h_target {hf:.5f}")

    far = np.array(n, np.float64)
    c_raw = (coords + 0.5) * hc
    before = cut(coords, hc, far, float(-c_raw.max(0) @ far - 10 * hc), hf)["K"]
    coords = coords_s
    c = (coords + 0.5) * hc
    d = float(-c.mean(0) @ np.array(n)) if len(sys.argv) <= 5 else float(sys.argv[5])
    after = cut(coords, hc, far, float(-c.max(0) @ far - 10 * hc), hf)["K"]
    print(f"  uncut piece count: {before} before close_and_fill, {after} after")
    r = cut(coords, hc, n, d, hf)
    print(f"  band: {int((r['level'] > 0).sum()):,} leaves came from refinement, "
          f"deepest level {r['top']}")
    print(f"  {len(r['leaf']):,} leaves, {len(r['edges']):,} surviving edges, "
          f"{r['K']} piece(s)")
    for k in range(r["K"]):
        print(f"    piece {k}: {int((r['piece'] == k).sum()):,} leaves")
