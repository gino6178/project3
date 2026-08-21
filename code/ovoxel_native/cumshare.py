"""Over the whole schedule, how many cells does each family reach, and how many do both?

    python cumshare.py OBJ

The 1.0% measured earlier was one transverse plane against one longitudinal plane in one step, and
they share a line. Gino's point is that this is not the quantity that matters: a cell is crossed by
SOME transverse plane at one step and SOME longitudinal plane at another, and those two also
disagree about it even though they never appear together. If most cells are reached by both
families over the run, the conflict is everywhere and confining a term to the intersection line
addresses almost none of it.

Pure geometry, no training. A plane crosses a cell when the cell's corners are not all on one side
of it. The transverse family is the 16 supervised depths, widened by the jitter that is actually
applied; the longitudinal family is its 10 azimuths.
"""
import os
import sys

import numpy as np
import torch

W = "/workspace/ovoxel_native"
obj = sys.argv[1] if len(sys.argv) > 1 else "orange_sp"
JIT = float(os.environ.get("JITTER", "0.5"))

st = torch.load(f"{W}/state_{obj}.pt", map_location="cpu", weights_only=False)
C = np.load(f"{W}/cams_{obj}.npz")
hc = float(st["hc"])
org = np.asarray(st["org"], np.float64)
cells = st["solid"].numpy().astype(np.float64)
n_cell = len(cells)

# the eight corners of every cell, as offsets
off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
corners = (cells[:, None, :] + off[None]) * hc + org          # (N, 8, 3)

H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hd = C["h_planes"][:, 3]
h_step = float(hd[1] - hd[0])
hn = np.asarray(C["h_planes"][0, :3], float)


def crossed(n, ds):
    """Cells any plane with normal n and one of the offsets ds passes through."""
    p = corners @ np.asarray(n, float)                        # (N, 8)
    lo, hi = p.min(1), p.max(1)
    hit = np.zeros(n_cell, bool)
    for d in ds:
        hit |= (lo <= -d) & (hi >= -d)
    return hit


# the transverse depths the trainer actually visits: each supervised depth, jittered up to half a
# step either way, so the reachable set is the interval and not the 16 points
hds = []
for j in range(H_LO, H_HI):
    for f in np.linspace(-JIT, JIT, 9):
        hds.append(float(hd[j]) + h_step * f)
th = crossed(hn, hds)

sys.path.insert(0, "/workspace/ovoxel_native")
import azjitter

AZ = float(os.environ.get("JITTER_AZ", "0"))
axis = np.asarray(C["h_planes"][0, :3], float)
cen = ((st["solid"].float().mean(0) + 0.5) * hc).numpy() + org
NVp = len(C["v_planes"])
spacing = np.radians(180.0 / max(NVp, 1))

tv = np.zeros(n_cell, bool)
for i in range(NVp):
    n_i, d_i = C["v_planes"][i, :3], float(C["v_planes"][i, 3])
    if AZ <= 0:
        tv |= crossed(n_i, [d_i])
        continue
    # the plane turns about the axis, so the set it can reach is the wedge it sweeps
    for f in np.linspace(-AZ, AZ, 25):
        _, n2, d2 = azjitter.turn(np.eye(4), n_i, d_i, axis, cen, f * spacing)
        tv |= crossed(n2, [d2])

both = th & tv
either = th | tv
print(f"  {obj}: {n_cell:,} solid cells, JITTER_AZ={AZ}")
print(f"    reached by the transverse family    {100 * th.mean():5.1f}%")
print(f"    reached by the longitudinal family  {100 * tv.mean():5.1f}%")
print(f"    reached by both                     {100 * both.mean():5.1f}%")
print(f"    of the cells either family reaches, both reach "
      f"{100 * both.sum() / max(either.sum(), 1):5.1f}%")
print(f"    reached by neither                  {100 * (~either).mean():5.1f}%")
