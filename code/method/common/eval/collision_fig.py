"""What the collision test does, drawn.

Collision here is three integer operations: a floor division to find the cell, a table lookup to
ask whether it is occupied by this piece, and one sign comparison against the cut plane. The last
is the one worth showing. A leaf the plane passes through is occupied as a whole, so occupancy
alone claims material on both sides of the cut and two freshly separated pieces read as
interpenetrating along their whole shared face. The sign test removes exactly that band and
nothing else.

The figure is a slab through the contact region, with each particle drawn as what the test says
about it:

    grey    outside the other piece by both tests
    orange  claimed by occupancy alone -- the band the cut passes through
    red     still claimed once the plane test is applied

A correct narrow phase makes the third colour empty at rest and fills it in proportion to how far
the pieces are actually pushed together, which is the panel sequence.

    python method/common/eval/collision_fig.py LATTICE OUT.png
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

SIZE = int(_os.environ.get("COLL_SIZE", "420"))


def main(lattice_dir, out_png):
    import cv2
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid
    from method.common.cube import subdivide as sd
    from method.common.cube.physics import Body, CollisionIndex, contact, particles_to_pieces

    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]

    # Every constant below is the one method/common/cube/physics.py measures with, so the panels
    # and the numbers in the text are the same experiment: origin at a cell corner, the plane
    # oblique and pushed 0.37 of a cell off the grid so that it straddles cells rather than
    # landing on their boundaries, and the push measured in coarse cells.
    org = xyz[lvl == 0].min(0) - 0.5 * hc
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1
    n = np.array([0.13, 0.97, -0.21])
    n /= np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n) + 0.37 * hc
    r = sd.cut(solid, hc, n, d, hf)
    ix = CollisionIndex(r, hc, org=org, plane=(n, d))
    A, B = Body(ix, 0), Body(ix, 1)
    pid, _ = particles_to_pieces(xyz, ix)
    print(f"  {len(xyz):,} particles, {r['K']} pieces {[int((pid == k).sum()) for k in range(r['K'])]}")

    # Which piece a particle belongs to comes from the plane, not from the leaf it sits in.
    # Asking whether a particle labelled by its own leaf falls inside some *other* leaf is
    # circular: it never does, and the test reads zero against zero and looks like a pass. The
    # plane is the ground truth -- a particle on A's side belongs to A whatever cell it is in --
    # and the ones occupancy gets wrong are exactly those in a leaf the cut straddles.
    truth = np.where(np.sign((xyz - org) @ n + d) == A.side, A.piece, B.piece)
    a = xyz[truth == A.piece]
    print(f"  by the plane: {len(a):,} particles on piece 0's side")
    # the slab to draw: a band about the plane, seen along the plane's own normal-perpendicular
    e1 = np.eye(3)[int(np.argmin(np.abs(n)))]
    u1 = np.cross(n, e1); u1 /= np.linalg.norm(u1)
    u2 = np.cross(n, u1)
    band = np.abs(np.dot(a - a.mean(0), u2)) < 1.5 * hc      # a thin slice of the contact region
    tiles = []
    for push in (0.0, 1.0, 2.0, 4.0):                        # in coarse cells
        Am = A.move(t=n * -A.side * push * hc)
        q = a - n * A.side * push * hc
        occ_hit, _ = ix.occupied(q, B.piece)
        pen, _ = contact(Am, B, a)
        sel = band
        proj = np.stack([np.dot(a[sel] - a[sel].mean(0), u1),
                         np.dot(a[sel] - a[sel].mean(0), n)], 1)
        half = 1.05 * float(np.abs(proj).max())
        u = ((proj[:, 0] / half) * 0.5 + 0.5) * (SIZE - 1)
        v = (0.5 - (proj[:, 1] / half) * 0.5) * (SIZE - 1)
        img = np.full((SIZE, SIZE, 3), 255, np.uint8)
        col = np.tile(np.array([[205, 205, 205]], np.uint8), (int(sel.sum()), 1))
        col[occ_hit[sel]] = (60, 150, 245)                   # orange in BGR
        col[pen[sel]] = (40, 40, 210)                        # red in BGR
        uu = np.clip(u, 0, SIZE - 1).astype(np.int64)
        vv = np.clip(v, 0, SIZE - 1).astype(np.int64)
        order = np.argsort(occ_hit[sel].astype(int) + 2 * pen[sel].astype(int))
        img[vv[order], uu[order]] = col[order]
        head = np.full((30, SIZE, 3), 255, np.uint8)
        cv2.putText(head, f"pushed in {push:g} coarse cells", (6, 21), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (60, 60, 60), 1, cv2.LINE_AA)
        sub = np.full((26, SIZE, 3), 255, np.uint8)
        cv2.putText(sub, f"occupancy {int(occ_hit.sum()):,}   with the plane test "
                         f"{int(pen.sum()):,}", (6, 18), cv2.FONT_HERSHEY_DUPLEX, 0.38,
                    (110, 110, 110), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, img, sub]))
        print(f"  pushed {push:g} coarse cells: occupancy alone {int(occ_hit.sum()):,}, "
              f"with the plane test {int(pen.sum()):,}")
    cv2.imwrite(out_png, np.hstack(tiles))
    print(f"  -> {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
