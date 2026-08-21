"""Coverage against turning magnitude, so the sweep's cost can be read against what buys it.

    python covsweep.py OBJ

The training arms measure what turning costs the transverse family. This measures what it buys:
the fraction of cells the longitudinal family can reach at each magnitude, by geometry alone. If
coverage saturates well before 0.5 while the conflict keeps growing, the best setting is smaller
than the one that exactly fills the gap between azimuths.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/ovoxel_native")
import azjitter

W = "/workspace/ovoxel_native"
obj = sys.argv[1] if len(sys.argv) > 1 else "orange_sp"
st = torch.load(f"{W}/state_{obj}.pt", map_location="cpu", weights_only=False)
C = np.load(f"{W}/cams_{obj}.npz")
hc = float(st["hc"])
org = np.asarray(st["org"], np.float64)
cells = st["solid"].numpy().astype(np.float64)
off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
corners = (cells[:, None, :] + off[None]) * hc + org
axis = np.asarray(C["h_planes"][0, :3], float)
cen = ((st["solid"].float().mean(0) + 0.5) * hc).numpy() + org
NV = len(C["v_planes"])
spacing = np.radians(180.0 / max(NV, 1))

H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hd = C["h_planes"][:, 3]
h_step = float(hd[1] - hd[0])
hn = np.asarray(C["h_planes"][0, :3], float)


def crossed(n, ds):
    p = corners @ np.asarray(n, float)
    lo, hi = p.min(1), p.max(1)
    out = np.zeros(len(cells), bool)
    for d in ds:
        out |= (lo <= -d) & (hi >= -d)
    return out


th = crossed(hn, [float(hd[j]) + h_step * f
                  for j in range(H_LO, H_HI) for f in np.linspace(-0.5, 0.5, 9)])
print(f"  {obj}: transverse reaches {100 * th.mean():.1f}% at every setting")
print(f"  {'turn':>6} {'longitudinal':>13} {'both':>8} {'neither':>8}")
for az in (0.0, 0.25, 0.35, 0.5, 0.75, 1.0):
    tv = np.zeros(len(cells), bool)
    for i in range(NV):
        n_i, d_i = C["v_planes"][i, :3], float(C["v_planes"][i, 3])
        if az <= 0:
            tv |= crossed(n_i, [d_i])
            continue
        for f in np.linspace(-az, az, 31):
            _, n2, d2 = azjitter.turn(np.eye(4), n_i, d_i, axis, cen, f * spacing)
            tv |= crossed(n2, [d2])
    print(f"  {az:6.2f} {100 * tv.mean():12.1f}% {100 * (th & tv).mean():7.1f}% "
          f"{100 * (~(th | tv)).mean():7.1f}%")
