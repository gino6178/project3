"""Adaptive O-Voxel: collapse the dual grid where the surface does not need it.

A Flexible Dual Grid stores one node per active surface voxel, so its cost is O(A/h^2) -- the
surface area over the voxel area -- and that is set by the resolution rather than by the surface.
A plane costs as much as a crumpled sheet of the same extent, which is the wrong thing to pay
for.

The dual vertex is already the minimiser of a quadric, and that is what makes this fixable
without approximating anything twice. For a set of supporting planes {(n_i, d_i)},

    Q(x) = sum_i (n_i . x - d_i)^2 = x^T A x - 2 b^T x + c,
    A = sum_i n_i n_i^T,   b = sum_i d_i n_i,   c = sum_i d_i^2,

so a quadric is ten numbers, and -- the property everything here rests on -- it is *additive over
the set of planes*. The quadric of a merged block is the sum of its children's, exactly, with no
reference to the mesh they came from:

    Q_parent = sum_children Q_child.

So the grid can be collapsed bottom-up. Merge a 2x2x2 block of nodes, add their quadrics, solve
A x = b for the merged vertex, and read off the residual

    r = c - b^T x        (the value of Q at its own minimum, so r >= 0)

which is the sum of squared distances from that one vertex to every plane the block covers. If
r <= eps the block is representable by a single node to that tolerance and is collapsed; if the
surface bends inside the block, r rises and it is kept. Flat regions coarsen, curvature does not,
and nothing decides that but the geometry.

A is rank-deficient wherever the block is locally planar -- every normal points the same way, so
A has rank one and the vertex is only determined along that normal. Solving it with a plain
inverse puts the vertex anywhere on the plane, usually far outside its own cell. The pseudo-
inverse with small singular values clamped is the standard fix and is what dual contouring does:
it takes the minimum-norm solution relative to the block centre, which leaves the vertex on the
surface and inside the cell.

    python method/common/cube/qef.py OVOX.npz [eps ...]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT]

SVD_CLAMP = float(_os.environ.get("QEF_CLAMP", "1e-3"))


def quadrics_from_mesh(V, F, pos, voxel, h, origin):
    """One quadric per active voxel, accumulated from the triangles inside it.

    A triangle contributes its own supporting plane once per sample point it owns; using the
    triangle centroid and its unit normal makes each face one plane weighted by nothing, which
    is the plain form of the QEF. Weighting by area is the usual refinement and is left out so
    that the residual has the units the tolerance is expressed in.
    """
    tv = V[F]
    ctr = tv.mean(1)
    nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ok = ln[:, 0] > 1e-20
    ctr, nrm = ctr[ok], nrm[ok] / ln[ok]
    d = (nrm * ctr).sum(1)

    key = np.floor((ctr - origin) / h).astype(np.int64)
    # map each triangle to the index of the voxel it sits in
    base = int(min(key.min(), voxel.min())) - 2
    span = int(max(key.max(), voxel.max()) + (-base) + 3)

    def pack(c):
        q = c - base
        return (q[:, 0] * span + q[:, 1]) * span + q[:, 2]

    kv = pack(voxel)
    o = np.argsort(kv)
    ks, idx = kv[o], np.arange(len(voxel))[o]
    kt = pack(key)
    p = np.clip(np.searchsorted(ks, kt), 0, len(ks) - 1)
    hit = ks[p] == kt
    vid = np.full(len(kt), -1, np.int64)
    vid[hit] = idx[p[hit]]

    n = len(voxel)
    A = np.zeros((n, 6))
    b = np.zeros((n, 3))
    c = np.zeros(n)
    w = np.zeros(n)
    g = np.zeros((n, 3))
    m = vid >= 0
    v, nn, dd = vid[m], nrm[m], d[m]
    cols = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for j, (a1, a2) in enumerate(cols):
        np.add.at(A, (v, j), nn[:, a1] * nn[:, a2])
    for a1 in range(3):
        np.add.at(b, (v, a1), dd * nn[:, a1])
    np.add.at(c, v, dd * dd)
    np.add.at(w, v, 1.0)
    # The oriented normal, summed. A alone cannot give one -- it is n n^T, so its eigenvector has
    # no sign -- and the boundary mesh's winding does. Summing is additive like the rest, so a
    # merged node's orientation is its children's without recomputing anything.
    for a1 in range(3):
        np.add.at(g, (v, a1), nn[:, a1])
    return A, b, c, w, g, int(m.sum())


def _solve(A6, b, centre):
    """Minimise the quadric, robustly, relative to a point inside the cell.

    Substituting x = centre + y makes the minimum-norm solution the one closest to the cell's
    own centre, which is what keeps a vertex inside its cell where A is rank deficient.
    """
    n = len(b)
    A = np.zeros((n, 3, 3))
    cols = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for j, (a1, a2) in enumerate(cols):
        A[:, a1, a2] = A6[:, j]
        A[:, a2, a1] = A6[:, j]
    rhs = b - np.einsum("nij,nj->ni", A, centre)
    U, S, Vt = np.linalg.svd(A)
    Si = np.where(S > SVD_CLAMP * S[:, :1].clip(1e-30), 1.0 / np.maximum(S, 1e-30), 0.0)
    y = np.einsum("nij,nj->ni", np.transpose(Vt, (0, 2, 1)),
                  Si * np.einsum("nji,nj->ni", U, rhs))
    return centre + y


def residual(A6, b, c, x):
    """Q(x) at the solved vertex. Non-negative up to rounding, and zero on a plane."""
    cols = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    quad = np.zeros(len(x))
    for j, (a1, a2) in enumerate(cols):
        w = 1.0 if a1 == a2 else 2.0
        quad += w * A6[:, j] * x[:, a1] * x[:, a2]
    return np.maximum(quad - 2.0 * (b * x).sum(1) + c, 0.0)


def collapse(voxel, A6, b, c, w, h, origin, tau, max_levels=8, alpha=0.5, g=None,
             pos0=None):
    """Merge 2x2x2 blocks bottom-up while the merged quadric stays within tau * h.

    The test is on the root-mean-square distance, sqrt(r / w) where w counts the planes the
    block covers, not on the raw residual. That makes the knob a length: tau = 0.25 means the
    merged vertex is within a quarter of a voxel of the planes it replaces, on average, and it
    means the same thing at every resolution and on every object. A raw residual does not -- it
    grows with how many planes fall in the block, so the same number coarsens a dense region and
    refuses a sparse one for no geometric reason.

    A parent is active when *any* child is, which is the rule for a surface and not for a
    volume. Requiring all eight collapses nothing: a 2-manifold crossing a 2x2x2 block activates
    four to six of its cells, never all eight, so the first version of this refused every merge
    and reported 100.0% of the uniform grid at every tolerance.
    """
    cur = voxel.copy()
    curA, curb, curc, curw = A6.copy(), b.copy(), c.copy(), w.copy()
    curg = np.zeros((len(cur), 3)) if g is None else g.copy()
    keptg = []
    lvl = np.zeros(len(cur), np.int8)
    hl = h
    keptV, keptA, keptb, keptc, keptL, kepth = [], [], [], [], [], []
    # Whether `cur` still holds nodes nobody has taken. When a level merges nothing the loop
    # stops, and at that point `cur` has *already* been kept in full as `cur[~merge]` -- keeping
    # it again after the loop counts the whole surviving set twice. It inflated every node count
    # in proportion to how early the collapse ran out: an identity collapse, which must return
    # its input untouched, returned 617,722 nodes for 442,253 inputs.
    pending = False

    for L in range(1, max_levels + 1):
        uniq, inv = np.unique(np.ascontiguousarray(cur >> 1), axis=0, return_inverse=True)

        # accumulate the children's quadrics into their parent -- the additivity is the whole
        # algorithm, and it is exact
        pA = np.zeros((len(uniq), 6)); pb = np.zeros((len(uniq), 3))
        pc = np.zeros(len(uniq)); pw = np.zeros(len(uniq)); pg = np.zeros((len(uniq), 3))
        np.add.at(pA, inv, curA)
        np.add.at(pb, inv, curb)
        np.add.at(pc, inv, curc)
        np.add.at(pw, inv, curw)
        np.add.at(pg, inv, curg)

        pcentre = (uniq + 0.5) * (hl * 2) + origin
        px = _solve(pA, pb, pcentre)
        pr = residual(pA, pb, pc, px)
        rms = np.sqrt(pr / np.maximum(pw, 1.0))
        inside = (np.abs(px - pcentre) <= hl).all(1)

        # The residual measures distance to planes and says nothing about coverage, and on its
        # own that is exploitable: a block holding one or two triangles has an rms of almost
        # zero, so it merges all the way up, and a node 256 cells wide then represents its whole
        # cell on the evidence of a single plane. Its vertex ends up wherever the cell centre
        # projects, which is how the worst error reached 82 cells while the mean stayed at 0.35.
        #
        # A genuine surface patch of side 2^L h contains on the order of (2^L)^2 fine cells, so
        # requiring the plane count to keep up with the area is the condition the residual is
        # missing. It is a statement about evidence, not a tuning knob: a node may only claim an
        # area it has seen.
        covered = pw >= alpha * (4.0 ** L)
        ok = (rms <= tau * h) & inside & covered
        merge = ok[inv]

        keptV.append(cur[~merge]); keptA.append(curA[~merge])
        keptb.append(curb[~merge]); keptc.append(curc[~merge])
        keptL.append(lvl[~merge]); kepth.append(np.full(int((~merge).sum()), hl))
        keptg.append(curg[~merge])

        take = np.nonzero(ok)[0]
        if not len(take):
            pending = False
            break
        cur = uniq[take]
        curA, curb, curc, curw, curg = pA[take], pb[take], pc[take], pw[take], pg[take]
        pending = True
        lvl = np.full(len(cur), L, np.int8)
        hl = hl * 2

    if pending:
        keptV.append(cur); keptA.append(curA); keptb.append(curb); keptc.append(curc)
        keptL.append(lvl); kepth.append(np.full(len(cur), hl)); keptg.append(curg)

    V = np.concatenate(keptV); A = np.concatenate(keptA)
    B = np.concatenate(keptb); C = np.concatenate(keptc)
    Lv = np.concatenate(keptL); H = np.concatenate(kepth)
    centre = (V + 0.5) * H[:, None] + origin
    X = _solve(A, B, centre)
    X = np.clip(X, centre - 0.5 * H[:, None], centre + 0.5 * H[:, None])

    # A node that was never merged already has a vertex, and it is a better one. The library
    # solves its QEF from Hermite data on the grid edges -- the exact crossing points and the
    # normals there -- while the quadric here is rebuilt from the boundary mesh's triangles,
    # which is a coarser description of the same surface. Re-solving an untouched node therefore
    # moves it for no reason: with every vertex re-solved, an identity collapse produced the
    # library's own connectivity and triangle count exactly, and 117% of its area.
    if pos0 is not None:
        keep0 = Lv == 0
        if keep0.any():
            idx = _index_of(V[keep0], voxel)
            has = idx >= 0
            sel = np.nonzero(keep0)[0][has]
            X[sel] = pos0[idx[has]]
    G = np.concatenate(keptg)
    G = G / np.linalg.norm(G, axis=1, keepdims=True).clip(1e-30)
    return dict(voxel=V, level=Lv, h=H, pos=X, normal=G, resid=residual(A, B, C, X))


def _index_of(query, table):
    """Row of `table` matching each row of `query`, or -1."""
    mn = int(min(query.min(), table.min())) - 2
    span = int(max(query.max(), table.max()) + (-mn) + 3)
    k = _pack_i(table, -mn, span)
    o = np.argsort(k)
    ks, idx = k[o], np.arange(len(table))[o]
    kk = _pack_i(query, -mn, span)
    pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
    out = np.full(len(query), -1, np.int64)
    hit = ks[pos] == kk
    out[hit] = idx[pos[hit]]
    return out


def _pack_i(c, base, span):
    q = c.astype(np.int64) + base
    return (q[:, 0] * span + q[:, 1]) * span + q[:, 2]


def bits_per_node(levels):
    """What a node costs once the two free wins are taken.

    The dual vertex lies in its own cell by construction, so storing it as a fraction quantised
    to b bits has error at most sqrt(3)/2 * h * 2^-b -- at b = 8 that is h/512 in the worst case,
    which is nothing against h. The index is delta-coded along a space-filling curve, which the
    library already provides in o_voxel.serialize. Neither is a contribution; both are exact.
    """
    return dict(vertex_bits=3 * 8, index_bits=int(np.ceil(np.log2(max(len(levels), 2)))) + 3,
                flags_bits=3, colour_bits=3 * 8)


def main(npz, epss):
    z = np.load(npz)
    voxel, pos = z["voxel"].astype(np.int64), z["pos"].astype(np.float64)
    h, origin = float(z["voxel_size"]), z["origin"].astype(np.float64)
    if "mesh_v" not in z.files:
        raise SystemExit(f"{npz} has no mesh; run globalovox with a CUDA device first")
    V, F = z["mesh_v"].astype(np.float64), z["mesh_f"].astype(np.int64)
    A6, b, c, w, gn, used = quadrics_from_mesh(V, F, pos, voxel, h, origin)
    print(f"  {len(voxel):,} active voxels, {len(F):,} triangles, "
          f"{used:,} of them landed in an active voxel")

    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    base = bits_per_node(np.zeros(len(voxel)))
    per = sum(base.values())
    print(f"  uniform grid: {len(voxel):,} nodes, {len(voxel) * per / 8 / 2 ** 20:.2f} MiB "
          f"at {per} bits a node")
    # Both directions, because one is not enough and the shortfall is not subtle. Measuring only
    # from each collapsed node to the nearest original vertex asks "is this node on the surface"
    # and never "is the surface still covered": a node that stands in for a large patch scores
    # perfectly as long as it sits on it. Read that way the error is not even monotone in tau --
    # it peaks near 0.75 and falls again at 1.0 while the node count keeps dropping, which is the
    # measure running out of things to say rather than the surface improving. The reverse
    # direction, from each original vertex to the nearest surviving node, is what sees coverage,
    # and the symmetric mean of the two is what should be quoted.
    print(f"  {'tau':>8}{'nodes':>12}{'of uniform':>11}{'MiB':>8}"
          f"{'fwd':>9}{'rev':>9}{'sym':>9}{'sym 95th':>11}{'sym max':>10}"
          f"   (in units of h)")
    for tau in epss:
        r = collapse(voxel, A6, b, c, w, h, origin, tau, g=gn, pos0=pos)
        fwd, _ = tree.query(r["pos"], k=1)                  # node -> original
        rev, _ = cKDTree(r["pos"]).query(pos, k=1)          # original -> node
        both = np.concatenate([fwd, rev])
        n = len(r["voxel"])
        print(f"  {tau:>8.3f}{n:>12,}{100 * n / len(voxel):>10.1f}%"
              f"{n * per / 8 / 2 ** 20:>8.2f}"
              f"{fwd.mean() / h:>9.3f}{rev.mean() / h:>9.3f}{both.mean() / h:>9.3f}"
              f"{np.percentile(both, 95) / h:>11.3f}{both.max() / h:>10.3f}")


if __name__ == "__main__":
    e = [float(x) for x in sys.argv[2:]] or [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    main(sys.argv[1], e)
