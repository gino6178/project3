"""M7: piece identity for the existing particles, and collision that stays a table lookup.

The spec's section 8. Two responsibilities are kept apart on purpose (28): the O-Voxel patch is
the visible boundary and takes no part in the physics, while the cube hierarchy is the physical
volume and answers every question about where the material is. Replacing the interior Gaussians
with cubes therefore costs nothing that the occupancy-based collision was already giving.

Three things, and none of them rebuilds the solver.

  particles keep their identity   (31) is a relabelling, not a resampling: the existing MPM
                                  particle set stays as it is and each particle takes the piece
                                  of the leaf it falls in. Subdividing the cut band multiplies
                                  leaves, not particles, which is the point of doing it there.

  collision stays integer         (29)-(30) is a floor division and an occupancy test, then one
                                  more of each if the coarse cell was refined. No point-in-
                                  polyhedron query appears anywhere, which is what the cube
                                  representation is for.

  the cut face is analytic        A leaf the plane passes through is occupied as a whole, so the
                                  voxel test claims material up to half a leaf beyond where the
                                  cut actually is. Near the band the spec adds a plane test, and
                                  it is one comparison: the point must be on the piece's own
                                  side of Pi. Measured below, that is the difference between two
                                  freshly separated halves reporting 109 penetrating particles
                                  and reporting none.

Pieces move independently because each carries its own rigid transform and every query is asked
in the frame of the body being tested, so two bodies need no shared frame and no re-voxelisation
when they move.

    python method/common/cube/physics.py            # the self-test
    python method/common/cube/physics.py LATTICE    # on a real one
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

import subdivide as sd                      # noqa: E402


class Body:
    """One piece: its cells, its side of the cut, and where it has moved to.

    The transform is rigid and per body, which is all "the two halves move independently" needs.
    A query in world space is carried into this body's own frame and answered there, so nothing
    is re-voxelised when a body moves and two bodies never need a common grid.
    """

    def __init__(self, index, piece, R=None, t=None):
        self.ix = index
        self.piece = int(piece)
        self.R = np.eye(3) if R is None else np.asarray(R, np.float64)
        self.t = np.zeros(3) if t is None else np.asarray(t, np.float64)
        sel = index.piece == self.piece
        self.side = float(np.sign(np.median(index.side[sel]))) or 1.0

    def to_local(self, x):
        return (np.asarray(x, np.float64) - self.t) @ self.R

    def to_world(self, x):
        return np.asarray(x, np.float64) @ self.R.T + self.t

    def move(self, R=None, t=None):
        return Body(self.ix, self.piece, self.R if R is None else R,
                    self.t if t is None else t)


class CollisionIndex:
    """The cube hierarchy, as the only thing the physics asks about material.

    Built once per cut. `occupied` is the spec's (29)-(30); `penetration` adds the near-cut
    plane test from 8.2 and reports how deep each hit is, which is what a contact solver wants.
    """

    def __init__(self, cut, h, org=None, plane=None):
        self.leaf = cut["leaf"]
        self.level = cut["level"]
        self.piece = cut["piece"]
        self.side = cut["side"]
        self.h = float(h)
        self.org = np.zeros(3) if org is None else np.asarray(org, np.float64)
        self.K = int(cut["K"])
        self.n, self.d = (None, None) if plane is None else (
            np.asarray(plane[0], np.float64) / np.linalg.norm(plane[0]), float(plane[1]))
        self._tab = {}
        for L in sorted({int(v) for v in self.level}):
            m = self.level == L
            mn = int(self.leaf[m].min()) - 2
            span = int(self.leaf[m].max() + (-mn) + 3)
            k = sd._pack(self.leaf[m], -mn, span)
            o = np.argsort(k)
            self._tab[L] = (k[o], np.nonzero(m)[0][o], -mn, span)

    def leaf_at(self, x_local):
        """Which leaf holds each point -- (29), finest level first, or -1."""
        p = np.asarray(x_local, np.float64) - self.org
        out = np.full(len(p), -1, np.int64)
        for L in sorted(self._tab, reverse=True):
            keys, idx, base, span = self._tab[L]
            c = np.floor(p / (self.h / (2.0 ** L))).astype(np.int64)
            # The key is a polynomial in the coordinates, so a point far outside the range the
            # table was built over aliases onto a valid key and reports material that is not
            # there. It showed up as two halves pulled 6 cells apart reporting 205 penetrating
            # particles while 0.5 and 2.0 apart reported none -- a false positive that grows
            # with distance, which is the opposite of what a contact test should do.
            inr = ((c + base) >= 0).all(1) & ((c + base) < span).all(1)
            kk = sd._pack(np.where(inr[:, None], c, 0), base, span)
            pos = np.clip(np.searchsorted(keys, kk), 0, len(keys) - 1)
            got = inr & (keys[pos] == kk) & (out < 0)
            out[got] = idx[pos[got]]
        return out

    def occupied(self, x_local, piece=None):
        """(30): does material sit here, and if `piece` is given, this piece's material."""
        lf = self.leaf_at(x_local)
        ok = lf >= 0
        if piece is not None:
            ok &= np.where(lf >= 0, self.piece[np.clip(lf, 0, None)] == piece, False)
        return ok, lf

    def penetration(self, x_local, piece, side):
        """Occupancy, corrected at the cut face, with a depth.

        The voxel test is right everywhere except within half a leaf of the plane, where a cell
        the cut passes through is occupied as a whole and so claims material the cut removed.
        Section 8.2's narrow phase is one comparison there: the point has to be on the piece's
        own side. Depth is the distance to the nearest face of the containing leaf, which is
        what it costs to push the point out.
        """
        hit, lf = self.occupied(x_local, piece)
        p = np.asarray(x_local, np.float64) - self.org
        if self.n is not None and hit.any():
            hl = self.h / (2.0 ** self.level[np.clip(lf, 0, None)].astype(np.float64))
            s = p @ self.n + self.d
            band = np.abs(s) <= 0.5 * hl * np.abs(self.n).sum()
            hit &= ~(band & (np.sign(s) != side))
        depth = np.zeros(len(p))
        if hit.any():
            hl = self.h / (2.0 ** self.level[lf[hit]].astype(np.float64))
            lo = self.leaf[lf[hit]] * hl[:, None]
            q = p[hit] - lo
            depth[hit] = np.minimum(q, hl[:, None] - q).min(1)
        return hit, depth


def particles_to_pieces(parts, index, nearest=True):
    """(31): the existing particle set keeps its particles and takes piece labels.

    A particle just outside the grid -- the skin sits half a cell beyond the outermost cell
    centre -- takes the nearest leaf rather than being dropped, since a particle with no piece
    is a particle no solver will move.
    """
    lf = index.leaf_at(parts)
    out = np.where(lf >= 0, index.piece[np.clip(lf, 0, None)], -1)
    stray = out < 0
    if nearest and stray.any():
        from scipy.spatial import cKDTree
        hl = index.h / (2.0 ** index.level.astype(np.float64))
        lc = (index.leaf + 0.5) * hl[:, None] + index.org
        _, j = cKDTree(lc).query(parts[stray], k=1)
        out[stray] = index.piece[j]
    return out, int(stray.sum())


def contact(a, b, parts_a):
    """Which of body a's particles are inside body b, and how deep.

    The particles are carried from a's frame to the world and into b's, which is the whole of
    what independent motion requires: no shared grid, no re-voxelisation, one rigid map each.
    """
    xb = b.to_local(a.to_world(parts_a))
    return b.ix.penetration(xb, b.piece, b.side)


def _selftest():
    bad = 0
    h, hf = 1.0, 0.5
    ball = sd._ball(12)

    # A plane that is oblique and off the grid, and particles that are not at cell centres.
    # Both matter, and the first version of this test had neither. With the cut on a lattice
    # boundary no leaf is ever straddled, so section 8.2's discrepancy cannot occur and the
    # narrow phase looks unnecessary; with particles at cell centres a push of half a cell lands
    # them exactly on cell faces and penetration reads as zero. Neither is what a solver sees.
    n = np.array([0.13, -0.21, 0.97])
    n /= np.linalg.norm(n)
    d = -0.37
    r = sd.cut(ball, h, n, d, hf)
    ix = CollisionIndex(r, h, plane=(n, d))
    A, B = Body(ix, 0), Body(ix, 1)
    print(f"  {r['K']} pieces on an oblique off-grid cut, sides {A.side:+.0f} and {B.side:+.0f}")

    rng = np.random.default_rng(0)
    p = rng.uniform(-12.5, 12.5, (60000, 3))
    parts = p[np.linalg.norm(p, axis=1) <= 12.0]
    pid, stray = particles_to_pieces(parts, ix)
    ok = (pid >= 0).all()
    bad += not ok
    counts = [int((pid == k).sum()) for k in range(r["K"])]
    print(f"  {'ok ' if ok else 'FAIL'} every particle carries a piece: {counts} of "
          f"{len(parts):,}, {stray} took a nearest leaf")

    ok = sum(counts) == len(parts) and min(counts) > 0
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the pieces partition the particle set "
          f"({counts[0]:,} + {counts[1]:,} = {sum(counts):,})")

    # every particle should be on its own piece's side of the plane, which is the labelling
    # agreeing with the geometry rather than with itself
    sgn = np.sign(parts @ n + d)
    agree = float(np.mean(sgn == np.where(pid == A.piece, A.side, B.side)))
    ok = agree > 0.98
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} {100 * agree:.2f}% of labels agree with which side of "
          f"the cut the particle is on")

    # 8.2, and the truth has to come from the geometry rather than from the labelling. Asking
    # whether a particle labelled by the leaf it sits in is inside some *other* leaf is circular
    # -- it never is, and the first version of this test read 0 against 0 and looked like a pass.
    # The plane is the ground truth: a particle on A's side belongs to A whatever cell it is in,
    # and the ones the voxel test gets wrong are exactly those in a leaf the cut straddles.
    truth = np.where(np.sign(parts @ n + d) == A.side, A.piece, B.piece)
    pa = parts[truth == A.piece]
    raw, _ = ix.occupied(pa, B.piece)
    hit, dep = contact(A, B, pa)
    ok = int(hit.sum()) == 0 and int(raw.sum()) > 0
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} at rest: voxel occupancy alone reports "
          f"{int(raw.sum()):,} of {len(pa):,} particles inside the other piece; with the cut "
          f"plane, {int(hit.sum()):,}")

    for gap in (0.5, 2.0, 6.0):
        Am = A.move(t=n * A.side * gap)
        hit, _ = contact(Am, B, pa)
        print(f"      pulled apart by {gap:>4.1f}: {int(hit.sum()):,} penetrating")

    got = []
    for push in (0.25, 0.5, 1.0, 2.0, 4.0):
        Am = A.move(t=n * -A.side * push)
        hit, dep = contact(Am, B, pa)
        got.append(int(hit.sum()))
        print(f"      pushed in by {push:>4.2f}: {int(hit.sum()):>7,} penetrating, "
              f"deepest {dep.max():.3f}")
    ok = all(got[i] < got[i + 1] for i in range(len(got) - 1)) and got[0] > 0
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} penetration is detected from the first quarter cell "
          f"and grows monotonically with the push")

    th = np.radians(35)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    Am = A.move(R=Rz, t=n * -A.side * 2.0)
    hit, _ = contact(Am, B, pa)
    ok = int(hit.sum()) > 0
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} a body carrying a rotation as well as a translation is "
          f"still tested in its own frame ({int(hit.sum()):,} penetrating)")

    inside_grid = ix.leaf_at(pa) >= 0
    hit_self, _ = contact(A, A, pa[inside_grid])
    ok = float(hit_self.mean()) > 0.98
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} a body still contains its own particles "
          f"({100 * hit_self.mean():.2f}% of the {int(inside_grid.sum()):,} that are inside "
          f"the grid at all)")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    import torch
    from plyfile import PlyData
    from occupancy import close_and_fill, to_grid

    ld = sys.argv[1]
    lat = torch.load(_os.path.join(ld, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(ld, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(ld, "cell_level.pt")).reshape(-1)
    keep = (lvl[:len(xyz)] == 0).numpy()

    org = xyz[keep].min(0) - 0.5 * hc
    # From a corner, not from a centre. `floor((p - min)/h)` puts every cell centre exactly on a
    # cell boundary and lets floating point decide which side it falls on: measured on the
    # generated lattices, whose cells sit exactly at (i + 1/2)h, that loses 49% of them and the
    # physics then runs on half the volume. Offsetting the origin by half a cell puts each centre
    # in the middle of its own cell, half a cell from either boundary, which is the largest margin
    # there is. It is also the right question -- which cell does this point fall in -- for the
    # quantised lattices, whose cells are not on any grid at all (median distance to the nearest
    # lattice point 0.23 to 0.26 of a cell), and where the old form was splitting one cell's worth
    # of material across two addresses.
    coords = np.unique(np.floor((xyz[keep] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1

    # An oblique cut, off the grid. An axis-aligned plane through the centroid can land on a
    # cell boundary and straddle nothing, and then the narrow phase has nothing to correct and
    # looks unnecessary; a user's cut is not aligned to anyone's lattice.
    n = np.array([0.13, 0.97, -0.21])
    n /= np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n) + 0.37 * hc
    r = sd.cut(solid, hc, n, d, hf)
    ix = CollisionIndex(r, hc, org=org, plane=(n, d))
    A, B = Body(ix, 0), Body(ix, 1)

    # the particle set the solver already has: every primitive of the model
    pid, stray = particles_to_pieces(xyz, ix)
    print(f"  {len(solid):,} solid cells, {r['K']} pieces")
    print(f"  {len(xyz):,} particles labelled: "
          f"{[int((pid == k).sum()) for k in range(r['K'])]}, {stray:,} took a nearest leaf")

    # the plane is the truth, not the labelling -- see the self-test
    truth = np.where(np.sign((xyz - org) @ n + d) == A.side, A.piece, B.piece)
    pa = xyz[truth == A.piece]
    raw, _ = ix.occupied(pa, B.piece)
    hit, _ = contact(A, B, pa)
    print(f"  at rest: voxel occupancy alone reports {int(raw.sum()):,} of "
          f"{len(pa):,} particles inside the other piece, the cut plane leaves "
          f"{int(hit.sum()):,}")
    for push in (0.25, 0.5, 1.0):
        Am = A.move(t=n * -A.side * push * hc * 4)
        hit, dep = contact(Am, B, pa)
        print(f"  pushed {push * 4:.0f} coarse cells in: {int(hit.sum()):,} penetrating, "
              f"deepest {dep.max():.5f}")
