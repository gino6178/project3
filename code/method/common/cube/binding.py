"""The surface follows the deformation, by binding its vertices to the particles.

Section 12.2's fourth limitation: if the solver deforms the material, an extracted surface has to
move with it, and a rigid transform per piece is only correct while nothing bends. Section 8.3
offers two ways and asks for the first in a first version -- bind each surface vertex to the k
nearest MPM particles with fixed weights and carry it along, rather than sampling the background
grid's velocity, because the binding is computed once and the advection is then a weighted sum.

    x_v(t) = sum_j w_vj * x_j(t),        w_vj fixed at bind time, sum_j w_vj = 1

The weights are the thing to get right, and there is a property worth insisting on: if every
particle moves by the same affine map, every vertex must follow it *exactly*. That is what makes
a rigid motion and a uniform stretch free of error rather than merely small, and it holds for any
weights that sum to one -- but only if the vertex is written in terms of the particles' positions
and not their displacements plus a stale offset. So the binding stores no offset. A vertex that
does not coincide with its particles' weighted centre is *moved* to it at bind time, and the
residual is reported rather than hidden: it is the price of the representation, paid once, and it
is what the surface's own resolution against the particle spacing buys down.

The weights themselves are the usual smooth kernel over the k nearest, normalised. Fixed at bind
time, per section 8.3, so a vertex cannot change which particles it belongs to halfway through a
simulation and jump.

    python method/common/cube/binding.py            # the self-test
    python method/common/cube/binding.py OVOX.npz LATTICE
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT]

K_DEFAULT = int(_os.environ.get("BIND_K", "8"))


class SurfaceBinding:
    """Vertices expressed as fixed convex combinations of particles.

    `bind` is O(V log P) once; `apply` is a gather and a weighted sum per step, which is what
    makes this cheap enough to run every frame on a surface with a million vertices.
    """

    def __init__(self, verts, parts, k=K_DEFAULT, eps=1e-12):
        from scipy.spatial import cKDTree
        k = min(k, len(parts))
        d, j = cKDTree(parts).query(verts, k=k)
        d = np.atleast_2d(d.T).T if k > 1 else d[:, None]
        j = np.atleast_2d(j.T).T if k > 1 else j[:, None]

        # A smooth, compactly supported weight over the k found, normalised. The support is the
        # k-th distance, so a vertex in a dense region binds tightly and one in a sparse region
        # reaches further, without a length scale being chosen anywhere.
        r = d / np.maximum(d[:, -1:], eps)
        w = np.clip(1.0 - r ** 2, 0.0, 1.0) ** 3 + eps
        w = w / w.sum(1, keepdims=True)

        self.idx, self.w, self.k = j, w, k
        self.rest = (parts[j] * w[..., None]).sum(1)
        self.residual = np.linalg.norm(verts - self.rest, axis=1)

    def apply(self, parts_t):
        """Where the vertices are now, given where the particles are now."""
        return (parts_t[self.idx] * self.w[..., None]).sum(1)


def _affine(p, A, t):
    return p @ np.asarray(A, np.float64).T + np.asarray(t, np.float64)


def _selftest():
    bad = 0
    rng = np.random.default_rng(0)

    # a particle set and a surface on it: a sphere of particles, vertices on its shell
    p = rng.normal(size=(60000, 3))
    p = p / np.linalg.norm(p, axis=1, keepdims=True) * rng.uniform(0, 1, (60000, 1)) ** (1 / 3)
    v = rng.normal(size=(20000, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * 0.98

    b = SurfaceBinding(v, p)
    print(f"  {len(v):,} vertices bound to {len(p):,} particles, k = {b.k}")
    print(f"      bind-time residual: mean {b.residual.mean():.5f}, 95th "
          f"{np.percentile(b.residual, 95):.5f}, max {b.residual.max():.5f}")

    # a rigid motion has to be followed exactly, since the weights sum to one
    th = np.radians(37)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    got = b.apply(_affine(p, R, [0.3, -0.2, 0.9]))
    want = _affine(b.rest, R, [0.3, -0.2, 0.9])
    err = np.abs(got - want).max()
    ok = err < 1e-9
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} a rigid motion is followed exactly (worst {err:.2e})")

    # and so does any affine map, including a stretch and a shear
    A = np.array([[1.4, 0.3, 0.0], [0.0, 0.7, 0.2], [0.1, 0.0, 1.1]])
    got = b.apply(_affine(p, A, [0, 0, 0]))
    err = np.abs(got - _affine(b.rest, A, [0, 0, 0])).max()
    ok = err < 1e-9
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} a stretch and a shear are followed exactly "
          f"(worst {err:.2e})")

    # a non-affine deformation is followed to the accuracy the binding can offer, and the
    # measurement is against where the surface *should* be under the same field
    def bend(q):
        out = q.copy()
        out[:, 0] = q[:, 0] + 0.35 * q[:, 1] ** 2
        out[:, 2] = q[:, 2] * (1 + 0.25 * q[:, 1])
        return out

    got = b.apply(bend(p))
    want = bend(b.rest)
    e = np.linalg.norm(got - want, axis=1)
    ok = e.mean() < 0.01
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} a bend is followed to {e.mean():.5f} mean, "
          f"{np.percentile(e, 95):.5f} at the 95th, on an object of radius 1")

    # k is a trade and the trade should be visible
    for k in (1, 4, 8, 16, 32):
        bb = SurfaceBinding(v, p, k=k)
        e = np.linalg.norm(bb.apply(bend(p)) - bend(bb.rest), axis=1)
        print(f"      k = {k:>2}: bind residual {bb.residual.mean():.5f}, "
              f"bend error {e.mean():.5f}")

    # binding must not stitch a cut together: vertices of one piece may only use its particles
    lo, hi = p[:, 2] < 0, p[:, 2] >= 0
    vv = v[v[:, 2] < 0]
    b_all = SurfaceBinding(vv, p)
    b_own = SurfaceBinding(vv, p[lo])
    crossed = float((hi[b_all.idx]).any(1).mean())
    crossed_own = float((hi[np.nonzero(lo)[0][b_own.idx]]).any(1).mean())
    ok = crossed > 0.01 and crossed_own == 0.0
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} binding per piece is necessary and sufficient: "
          f"{100 * crossed:.1f}% of a piece's vertices reach across the cut when bound to every "
          f"particle, {100 * crossed_own:.1f}% when bound to its own")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    import torch
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid
    from method.common.cube import subdivide as sd
    from method.common.cube.physics import CollisionIndex, particles_to_pieces

    npz, ld = sys.argv[1], sys.argv[2]
    z = np.load(npz)
    V = z["mesh_v"].astype(np.float64) if "mesh_v" in z.files else z["pos"].astype(np.float64)
    lat = torch.load(_os.path.join(ld, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(ld, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(ld, "cell_level.pt")).reshape(-1)[:len(xyz)].numpy()
    org = xyz[lvl == 0].min(0) - 0.5 * hc
    # From a corner, not from a centre. `floor((p - min)/h)` puts every cell centre exactly on a
    # cell boundary and lets floating point decide which side it falls on: measured on the
    # generated lattices, whose cells sit exactly at (i + 1/2)h, that loses 49% of them and the
    # physics then runs on half the volume. Offsetting the origin by half a cell puts each centre
    # in the middle of its own cell, half a cell from either boundary, which is the largest margin
    # there is. It is also the right question -- which cell does this point fall in -- for the
    # quantised lattices, whose cells are not on any grid at all (median distance to the nearest
    # lattice point 0.23 to 0.26 of a cell), and where the old form was splitting one cell's worth
    # of material across two addresses.
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1
    n = np.array([0.13, 0.97, -0.21]); n /= np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n)
    r = sd.cut(solid, hc, n, d, hf)
    ix = CollisionIndex(r, hc, org=org, plane=(n, d))
    pid, _ = particles_to_pieces(xyz, ix)

    print(f"  {len(V):,} surface vertices, {len(xyz):,} particles, {r['K']} pieces")
    for k in range(r["K"]):
        pk = xyz[pid == k]
        side = np.sign(np.median(r["side"][r["piece"] == k]))
        vk = V[np.sign((V - org) @ n + d) == side]
        if not len(vk) or not len(pk):
            continue
        b = SurfaceBinding(vk, pk)
        print(f"    piece {k}: {len(vk):,} vertices on {len(pk):,} particles, "
              f"bind residual mean {b.residual.mean():.5f} = {b.residual.mean() / hf:.2f} h_f, "
              f"95th {np.percentile(b.residual, 95) / hf:.2f} h_f")
