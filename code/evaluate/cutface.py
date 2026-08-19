"""The cut face of every object, counted rather than asserted: band, levels, edges, planarity.

Section 4 claims four things about a planar cut and until now three of them had been counted on
one object by a script that was not in this repository. This is that script, on every object and
on four planes each, driving `figures/cutmesh.py` itself so the numbers and the geometry cannot
come apart.

    K and K_leaf       the cells equation (10) returns on the lattice as it is stored, and what
                       is left of that band after the operator refines every one of them to the
                       fine spacing. The second is about four times the first, and it is the one
                       a cut's work is proportional to, so quoting either without saying which
                       makes the two look like a disagreement.
    levels at the cut  how many distinct leaf levels carry a polygon. The claim that the
                       polygons meet edge to edge is exactly the claim that this is one, and it
                       is one by construction: `subdivide.refine` refines every crossed cell
                       until h <= h_target, so the band comes out uniform. An adaptive criterion
                       would not, and there is no adaptive criterion in this implementation.
    edge use           on the single-winding face set, how many edges are used by one triangle
                       (the rim), by two (the interior), and by more than two (which would be a
                       non-manifold edge). The emitted mesh is double-sided, so counting on it
                       doubles every edge and hides the rim; the reduction is done here.
    planarity          max |n.v + d| over the merged vertices, which is the residual of the
                       plane equation the vertices were solved from.

    python code/evaluate/cutface.py orange=build_orange/lattice ...
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path += [_os.path.join(_os.path.dirname(_HERE), "figures"),
             _os.path.join(_os.path.dirname(_HERE), "src"), _FN_ROOT,
             _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]

from cutmesh import cut_mesh                                     # the operator itself
from subdivide import crossed                                     # equation (10), the only copy

# Three axes and one oblique off-grid plane. The oblique one is the general case: an axis-aligned
# plane cuts four edges of a cell and lands on coordinates exactly, and a claim tested only there
# is a claim about the easy case.
PLANES = [("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0)),
          ("oblique", (0.3717, 0.7431, 0.5560))]


def solid_cells(lat):
    """The coarse cells a cut is taken on, built as `cutmesh.py`'s own main builds them."""
    import torch
    from plyfile import PlyData
    from occupancy import close_and_fill, to_grid

    meta = torch.load(_os.path.join(lat, "lattice.pt"), map_location="cpu")
    hc, hf = float(meta["coarse_dx"]), float(meta["fine_dx"])
    el = PlyData.read(_os.path.join(lat, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(lat, "cell_level.pt"), map_location="cpu").reshape(-1)
    p = xyz[(lvl[:len(xyz)] == 0).numpy()]
    coords = np.unique(np.floor((p - (p.min(0) - 0.5 * hf)) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    return close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1, hc, hf


def edge_use(F):
    """How many triangles each undirected edge belongs to, on one winding only."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c == 2).sum()), int((c > 2).sum())


def main(*specs):
    print(f"{'object':<11} {'plane':<8} {'N':>9} {'K':>7} {'K/N':>6} {'K_leaf':>8} "
          f"{'/N':>6} {'levels':>7} {'tris':>9} {'once':>7} {'twice':>9} {'>2':>3} "
          f"{'|n.v+d|':>10}")
    worst = 0.0
    for spec in specs:
        obj, _, lat = spec.partition("=")
        coords, hc, hf = solid_cells(lat)
        c = (coords + 0.5) * hc
        mid = c.mean(0)
        for name, n in PLANES:
            n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
            # the oblique plane is moved off the grid by a fraction of a cell, so that no
            # intersection lands on a cell corner and the residual is not flattered
            d = float(-mid @ n) - (0.137 * hc if name == "oblique" else 0.0)
            kc = int(crossed(c, hc, n, d).sum())
            m = cut_mesh(coords, hc, n, d, hf)
            s = m["stats"]
            F = m["F"][m["side"] == 1]
            one, two, many = edge_use(F)
            res = float(np.abs(m["V"] @ n + d).max())
            worst = max(worst, res)
            print(f"{obj:<11} {name:<8} {len(coords):>9,} {kc:>7,} "
                  f"{100 * kc / len(coords):>5.2f}% {s['cut_cells']:>8,} "
                  f"{100 * s['cut_cells'] / len(coords):>5.2f}% "
                  f"{str(s['levels_at_cut']):>7} {len(F):>9,} {one:>7,} {two:>9,} {many:>3} "
                  f"{res:>10.2e}")
    print(f"\n  worst plane-equation residual over every vertex of every cut: {worst:.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:])
