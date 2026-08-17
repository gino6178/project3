"""The collapsed grid as an actual triangle mesh.

A uniform dual grid meshes trivially: every grid edge the surface crosses is shared by four
voxels, and their four dual vertices are a quad. After a collapse the four are at different
levels, which is the case Ju et al. solve by walking the octree with mutually recursive cell,
face and edge procedures. The same faces come out of a lookup, and a lookup vectorises:

  * an edge that the surface crosses is a *fine* edge, and it stays one -- collapsing cells does
    not move where the surface is, only which node speaks for it;
  * each of the edge's four fine voxels is resolved to the node that survived the collapse and
    contains it, which is the coordinate shifted right until a node is found, finest first;
  * the four resolved nodes are the quad. Where two of them are the same node -- which is what a
    collapse across an edge means -- the quad degenerates to a triangle, and where three are, it
    degenerates to nothing and is dropped.

That is adaptive dual contouring's output without its recursion, and it is what makes the crack
problem not arise: neighbouring cells of different sizes share the *same* resolved node along
their shared edge, so there is no T-junction to seal.

Winding is decided by the collapse's own oriented normal rather than by the edge's sign, since
the collapse already carries one and a sign convention would be a second source of truth.

    python method/common/cube/qef_mesh.py OVOX.npz [tau]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT]

from method.common.cube import qef                                  # noqa: E402

# the four voxels around each axis-aligned grid edge, as the library orders them
EDGE_NB = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],      # x
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],      # y
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],      # z
], np.int64)


def resolve(fine, r):
    """The surviving node containing each fine voxel: shift right until one is found."""
    out = np.full(len(fine), -1, np.int64)
    for L in sorted({int(x) for x in r["level"]}):
        m = r["level"] == L
        v = r["voxel"][m]
        mn = int(min(v.min(), (fine >> L).min())) - 2
        span = int(max(v.max(), (fine >> L).max()) + (-mn) + 3)
        k = _pack(v, -mn, span)
        o = np.argsort(k)
        ks, idx = k[o], np.nonzero(m)[0][o]
        q = fine >> L
        kk = _pack(q, -mn, span)
        inr = ((q - mn) >= 0).all(1) & ((q - mn) < span).all(1)
        pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
        got = inr & (ks[pos] == kk) & (out < 0)
        out[got] = idx[pos[got]]
    return out


def _pack(c, base, span):
    q = c.astype(np.int64) + base
    return (q[:, 0] * span + q[:, 1]) * span + q[:, 2]


def mesh(voxel, inter, r):
    """Quads from the crossed fine edges, resolved through the collapse, then triangles."""
    tri = []
    for ax in range(3):
        sel = np.nonzero(inter[:, ax])[0]
        if not len(sel):
            continue
        base = voxel[sel]
        corner = [resolve(base + EDGE_NB[ax, k][None, :], r) for k in range(4)]
        q = np.stack(corner, 1)                                     # (E, 4)
        good = (q >= 0).all(1)
        q = q[good]
        if not len(q):
            continue
        # a collapse across the edge makes two corners the same node; the quad is then a
        # triangle, and if it collapses further there is no face left to draw
        a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        t1 = np.stack([a, b, c], 1)
        t2 = np.stack([a, c, d], 1)
        for t in (t1, t2):
            ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
            tri.append(t[ok])
    if not tri:
        return np.zeros((0, 3), np.int64)
    F = np.concatenate(tri)

    # one winding, taken from the collapse's own normals rather than from the edge's sign
    V, N = r["pos"], r["normal"]
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ref = N[F].mean(1)
    flip = (fn * ref).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    return F


def edge_manifold(F):
    """How many triangles each undirected edge carries. Two is a closed surface."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return np.bincount(cnt.clip(0, 5))


def area(V, F):
    return float(0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())


def main(npz, taus):
    z = np.load(npz)
    voxel, pos0 = z["voxel"].astype(np.int64), z["pos"].astype(np.float64)
    inter = z["inter"].astype(bool)
    h, origin = float(z["voxel_size"]), z["origin"].astype(np.float64)
    V0, F0 = z["mesh_v"].astype(np.float64), z["mesh_f"].astype(np.int64)
    A6, b, c, w, gn, _ = qef.quadrics_from_mesh(V0, F0, pos0, voxel, h, origin)

    a0 = area(V0, F0)
    m0 = edge_manifold(F0)
    print(f"  the library's own mesh: {len(F0):,} triangles, area {a0:.5f}, "
          f"edges carrying 1/2/3+ triangles: {m0[1] if len(m0)>1 else 0:,} / "
          f"{m0[2] if len(m0)>2 else 0:,} / {int(m0[3:].sum()) if len(m0)>3 else 0:,}")
    print(f"  {'tau':>8}{'nodes':>11}{'triangles':>12}{'area':>11}{'vs uniform':>12}"
          f"{'boundary edges':>16}{'non-manifold':>14}")
    for tau in taus:
        r = qef.collapse(voxel, A6, b, c, w, h, origin, tau, g=gn, pos0=pos0)
        F = mesh(voxel, inter, r)
        a = area(r["pos"], F)
        mm = edge_manifold(F)
        e1 = int(mm[1]) if len(mm) > 1 else 0
        e3 = int(mm[3:].sum()) if len(mm) > 3 else 0
        print(f"  {tau:>8.2f}{len(r['pos']):>11,}{len(F):>12,}{a:>11.5f}"
              f"{100 * a / a0:>11.1f}%{e1:>16,}{e3:>14,}")
    return


if __name__ == "__main__":
    t = [float(x) for x in sys.argv[2:]] or [0.0, 0.35, 0.5, 1.0]
    main(sys.argv[1], t)
