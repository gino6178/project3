"""Does the dual grid round the staircase off, and at what voxel size?

    python ovoxsweep.py

A ball of cells is the case the demo feeds it. Its blocky boundary has every vertex on a lattice
corner, so if one dual voxel contains exactly one such corner the quadric fit has a single plane
to satisfy and returns the corner unchanged -- which is what a voxel_size of one cell produces.
Enlarging the dual voxel is what gives the fit several faces to reconcile.

The number to watch is the distance from each surface vertex to the sphere it should lie on. A
staircase has a spread of about a third of a cell; a fitted surface should be under a tenth.
"""
import numpy as np

import globalovox as gov
import ovoxel as ov

R = 16
z, y, x = np.mgrid[-R:R + 1, -R:R + 1, -R:R + 1]
cells = np.stack(np.where(x * x + y * y + z * z <= R * R), 1).astype(np.int64)
V, F = gov.boundary_mesh(cells, 1.0)
c = (cells.min(0) + cells.max(0)) / 2.0 + 0.5


def err(P):
    d = np.linalg.norm(np.asarray(P, np.float64) - c, axis=1)
    return np.abs(d - np.median(d))


e0 = err(V)
print(f"  ball r={R}: {len(cells):,} cells, blocky surface {len(V):,} verts, {len(F):,} tris")
print(f"    blocky        |r - r0|  mean {e0.mean():.4f}  95th {np.percentile(e0,95):.4f}")
for vs in (1.0, 2.0, 3.0, 4.0, 6.0):
    try:
        patch = ov.patch_to_o_voxel(V, F, vs, colour=None, device="cpu")
        Vd, Fd = ov.dual_to_mesh(patch, device="cpu")
    except Exception as ex:
        print(f"    voxel_size {vs:<4}  failed: {str(ex)[:60]}")
        continue
    e = err(Vd)
    print(f"    voxel_size {vs:<4}  |r - r0|  mean {e.mean():.4f}  95th {np.percentile(e,95):.4f}"
          f"   {len(Vd):>7,} verts {len(Fd):>7,} tris")
