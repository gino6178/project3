"""M4: the cut surface as an exact planar mesh, one polygon per cut leaf.

The spec's section 6.1. Topology was decided on the lattice and is voxel-approximate by
construction; the visible cut face must not be, or it reads as a staircase. So the boundary is
analytic even though the classification is discrete -- the spec's (24), "discrete voxel topology
+ analytic planar rendering boundary".

Per leaf the construction is the spec's (23): intersect the plane with the cube's twelve edges,
take the points where an edge changes sign, order them around their centroid in the plane's own
basis, and fan them. The result is a convex polygon of three to six vertices lying exactly on
the plane, so the union over cells is exactly plane-intersect-solid at the leaf resolution, with
no staircase anywhere.

Two properties fall out of M3 rather than being arranged here:

  the mesh is conforming      Every cell the plane crosses is refined until h <= h_target, so
                              every cut cell is at the deepest level. Cut polygons therefore
                              meet edge to edge and there are no T-junctions between them. This
                              is asserted, not assumed -- see `stats["levels_at_cut"]`.

  both sides come for free    A cut polygon is the boundary of the material on either side of
                              it, so it is emitted twice with opposite winding, once for the
                              piece above and once for the piece below (the spec's 6.2). Which
                              pieces those are is already in M3: one side is the cut cell's own
                              piece, and the other is the face neighbour that (21) removed.

Appearance (6.2) is a lookup rather than a computation: (19) makes a child's feature its
parent's, so a face carries the index of the coarse cell it was cut out of and the decoder is
applied by whoever renders it.

    python method/common/cube/cutmesh.py            # the self-test
    python method/common/cube/cutmesh.py LATTICE    # on a real one
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import subdivide as sd                      # noqa: E402

# the cube's 8 corners as 0/1 offsets, and its 12 edges as corner index pairs
CORNER = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.float64)
EDGE = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                 if int(np.abs(CORNER[a] - CORNER[b]).sum()) == 1], np.int64)


def plane_basis(n):
    """Two axes spanning the plane, chosen off whichever axis n is least aligned with."""
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return n, u, np.cross(n, u)


def polygons(leaf, level, h, n, d):
    """One convex polygon per cut leaf, as vertices ordered around the polygon.

    Returns (verts, counts, cell) where verts is (M, 6, 3) padded, counts is how many of the six
    slots are real, and cell indexes back into `leaf`. Cells the plane misses are dropped.
    """
    n, u, v = plane_basis(n)
    hl = h / (2.0 ** level.astype(np.float64))
    lo = leaf * hl[:, None]
    corners = lo[:, None, :] + CORNER[None] * hl[:, None, None]     # (N, 8, 3)
    s = corners @ n + d                                             # (N, 8)

    sa, sb = s[:, EDGE[:, 0]], s[:, EDGE[:, 1]]
    hit = (sa > 0) != (sb > 0)                                      # strict, so a corner on the
    denom = np.where(np.abs(sa - sb) < 1e-30, 1.0, sa - sb)         # plane never doubles a point
    t = np.where(hit, sa / denom, 0.0)[..., None]
    ca, cb = corners[:, EDGE[:, 0]], corners[:, EDGE[:, 1]]
    p = ca + t * (cb - ca)                                          # (N, 12, 3), the spec's (23)

    cnt = hit.sum(1)
    keep = cnt >= 3
    p, hit, cnt = p[keep], hit[keep], cnt[keep]
    cell = np.nonzero(keep)[0]
    if not len(cell):
        return np.zeros((0, 6, 3)), np.zeros(0, np.int64), cell

    # order each cell's points around their centroid, in the plane's basis. A convex polygon is
    # star-shaped about its centroid, so the angle is a total order on its vertices.
    ctr = (p * hit[..., None]).sum(1) / cnt[:, None]
    q = p - ctr[:, None, :]
    ang = np.arctan2(q @ v, q @ u)
    ang = np.where(hit, ang, np.inf)                                # unused slots sort last
    order = np.argsort(ang, axis=1)
    p = np.take_along_axis(p, order[..., None], axis=1)[:, :6]
    return p, cnt, cell


def triangulate(verts, cnt):
    """Fan each polygon from its first vertex. A convex polygon needs nothing cleverer."""
    tri, owner = [], []
    for k in range(3, 7):
        m = np.nonzero(cnt == k)[0]
        if not len(m):
            continue
        for j in range(1, k - 1):
            tri.append(np.stack([verts[m, 0], verts[m, j], verts[m, j + 1]], 1))
            owner.append(m)
    if not tri:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int64)
    return np.concatenate(tri), np.concatenate(owner)


def merge(tri, tol):
    """Shared vertices become one. Neighbouring cells cut the same edge, so the points they
    produce are the same point and should not be two."""
    flat = tri.reshape(-1, 3)
    key = np.round(flat / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return flat[first], inv.reshape(-1, 3)


def cut_mesh(coords, h, n, d, h_target, levels_max=8):
    """The whole of M4 on top of M3: polygons, triangles, merged vertices, two sides.

    Every triangle is emitted twice, with opposite winding and therefore opposite normal, and
    each copy is tagged with the piece it bounds -- the spec's 6.2.
    """
    r = sd.cut(coords, h, n, d, h_target, levels_max)
    nn, _, _ = plane_basis(n)
    leaf, level = r["leaf"], r["level"]

    verts, cnt, cell = polygons(leaf, level, h, nn, d)
    tri, owner = triangulate(verts, cnt)
    V, F = merge(tri, tol=h * 1e-6)

    # Which piece is on each side of a polygon, from M3's own labelling rather than by probing.
    #
    # In the discrete model a leaf belongs wholly to one piece -- its centre decides -- so the
    # material on the side of the polygon where the cell's centre lies *is* that cell's piece,
    # and the other side is the face neighbour the plane separated it from. That neighbour is in
    # `edges_all` and nowhere else, since (21) is exactly the rule that removed it.
    #
    # Probing instead, by stepping a fraction of a leaf along +-n from the face, is wrong and
    # quietly so: the polygon sits inside the cell, so a short step stays in the same cell and
    # both sides report the same piece. On the doughnut that gave one of the two halves no cut
    # surface at all while the orange, whose polygons happen to sit nearer their cell faces,
    # looked correct.
    hl = h / (2.0 ** level.astype(np.float64))
    face_cell = cell[owner]
    q = r["side"]
    across = np.full(len(leaf), -1, np.int64)
    ea = r["edges_all"]
    if len(ea):
        opp = q[ea[:, 0]] != q[ea[:, 1]]
        a, b = ea[opp, 0], ea[opp, 1]
        across[a] = r["piece"][b]
        across[b] = r["piece"][a]
    own = r["piece"][face_cell]
    oth = across[face_cell]
    up = q[face_cell] > 0
    top = np.where(up, own, oth)
    bot = np.where(up, oth, own)

    # Two copies, and the pairing is the one that makes a normal mean something: a boundary's
    # normal points away from its own material. The cut face of the piece *below* the plane is
    # that piece's upper boundary, so it is the copy wound to +n; the piece above gets -n. The
    # other pairing type-checks and is wrong, and the test that catches it asks whether stepping
    # along a face's normal leaves the piece it belongs to -- with the pairing swapped, 0% did.
    F2 = np.concatenate([F, F[:, ::-1]])
    side = np.concatenate([np.ones(len(F), np.int8), -np.ones(len(F), np.int8)])
    piece = np.concatenate([bot, top])
    parent = np.concatenate([r["parent"][face_cell], r["parent"][face_cell]])

    stats = dict(cut_cells=len(cell), levels_at_cut=sorted({int(x) for x in level[cell]}),
                 tris=len(F), verts=len(V), pieces=r["K"],
                 faces_with_no_piece_across=int((oth < 0).sum()))
    return dict(V=V, F=F2, side=side, piece=piece, parent=parent, cut=r, stats=stats)


def _piece_at(pts, r, coords, h):
    """Which piece occupies these points, or -1 for empty space.

    A point is resolved from the deepest level down: the leaf that contains it is the finest one
    whose block it falls in, which is the same rule M3's adjacency uses to find a neighbour.
    """
    out = np.full(len(pts), -1, np.int64)
    for L in sorted({int(x) for x in r["level"]}, reverse=True):
        m = r["level"] == L
        hl = h / (2.0 ** L)
        c = np.floor(pts / hl).astype(np.int64)
        # the bounds have to come from this level's own coordinates. Taking them from the coarse
        # grid made the key space too small for level 1, whose indices are twice as large, so the
        # packed keys wrapped and collided: on the doughnut every face then resolved to the same
        # piece and one of the two halves came back with no cut surface at all.
        mn = int(min(r["leaf"][m].min(), c.min())) - 2
        span = int(max(r["leaf"][m].max(), c.max()) + (-mn) + 3)
        keys = sd._pack(r["leaf"][m], -mn, span)
        o = np.argsort(keys)
        ks, idx = keys[o], np.nonzero(m)[0][o]
        kk = sd._pack(c, -mn, span)
        pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
        got = (ks[pos] == kk) & (out < 0)
        out[got] = r["piece"][idx[pos[got]]]
    return out


def mesh_area(V, F):
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def _selftest():
    bad = 0
    h = 1.0

    # a disc: the cut face of a ball is a circle, and its area is known
    r_ball = 12.0
    m = cut_mesh(sd._ball(int(r_ball)), h, (0, 0, 1), 0.0, 0.5)
    one = m["F"][m["side"] == 1]
    area = mesh_area(m["V"], one)
    want = np.pi * r_ball ** 2
    err = abs(area - want) / want
    ok = err < 0.03
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} ball cut through the centre: area {area:.1f} vs "
          f"pi r^2 {want:.1f}  ({err * 100:.1f}%)")

    # an annulus: the cut face of a torus across its axis
    R, rr = 14.0, 5.0
    m = cut_mesh(sd._torus(int(R), int(rr)), h, (0, 0, 1), 0.0, 0.5)
    one = m["F"][m["side"] == 1]
    area = mesh_area(m["V"], one)
    want = np.pi * ((R + rr) ** 2 - (R - rr) ** 2)
    err = abs(area - want) / want
    ok = err < 0.05
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} torus cut across the axis: area {area:.1f} vs "
          f"4 pi R r {want:.1f}  ({err * 100:.1f}%)")

    # every vertex is on the plane, exactly
    off = np.abs(m["V"] @ np.array([0.0, 0, 1]) + 0.0).max()
    ok = off < 1e-9
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} every vertex lies on the plane (worst {off:.2e})")

    # the cut cells are all at one level, so the polygons conform
    ok = len(m["stats"]["levels_at_cut"]) == 1
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the cut cells are all at one level "
          f"{m['stats']['levels_at_cut']} -- no T-junctions between polygons")

    # normals: +n for one copy, -n for the other, and nothing in between
    V, F, side = m["V"], m["F"], m["side"]
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    nrm = np.cross(a, b)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-30)
    dot = nrm @ np.array([0.0, 0, 1])
    ok = np.allclose(dot, side, atol=1e-9)
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} every face normal is +n or -n and matches its side "
          f"(worst {np.abs(dot - side).max():.2e})")

    # a normal points away from its own material: stepping along it must leave the piece
    ctr = V[F].mean(1)
    step = ctr + 0.30 * nrm
    got = _piece_at(step, m["cut"], sd._torus(int(R), int(rr)), h)
    away = float((got != m["piece"]).mean())
    ok = away > 0.98
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} {away * 100:.1f}% of faces point out of their own piece")

    # both pieces are represented, and each face bounds two different ones
    two = float((m["piece"][:len(F) // 2] != m["piece"][len(F) // 2:]).mean())
    ok = two > 0.98
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} {two * 100:.1f}% of faces have a different piece on "
          f"each side")

    # merging did something: a conforming mesh shares most of its vertices
    per_tri = len(F) // 2 * 3
    print(f"  ..  {m['stats']['tris']:,} triangles from {m['stats']['cut_cells']:,} cut cells, "
          f"{m['stats']['verts']:,} vertices after merging (from {per_tri:,} corners)")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    import torch
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid

    ld = sys.argv[1]
    lat = torch.load(_os.path.join(ld, "lattice.pt"))
    v = PlyData.read(_os.path.join(ld, "gs_fill.ply")).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(ld, "cell_level.pt")).reshape(-1)
    p = xyz[(lvl[:len(xyz)] == 0).numpy()]
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    coords = np.unique(np.floor((p - p.min(0)) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    coords = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1

    n = [0.0, 1.0, 0.0]
    c = (coords + 0.5) * hc
    d = float(-c.mean(0) @ np.array(n))
    m = cut_mesh(coords, hc, n, d, hf)
    s = m["stats"]
    print(f"  {len(coords):,} solid cells, plane through the centroid")
    print(f"  {s['cut_cells']:,} cut cells, all at level {s['levels_at_cut']}")
    print(f"  {s['tris']:,} triangles, {s['verts']:,} merged vertices, {s['pieces']} pieces")
    print(f"  {s['faces_with_no_piece_across']:,} faces have empty space across them "
          f"({100 * s['faces_with_no_piece_across'] / max(s['tris'], 1):.2f}% -- the cut meeting "
          f"the outer surface)")
    one = m["F"][m["side"] == 1]
    print(f"  cut face area {mesh_area(m['V'], one):.5f}   "
          f"(a {2 * (np.abs((m['V'] - m['V'].mean(0)) @ np.array([1.0, 0, 0])).max()):.5f} "
          f"wide section)")
    for k in range(s["pieces"]):
        f = m["F"][m["piece"] == k]
        print(f"    piece {k}: {len(f):,} faces on the cut, area {mesh_area(m['V'], f):.5f}")
