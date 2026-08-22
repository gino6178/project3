"""What fraction of the interior do the photographs actually touch?

A cell only ever moves if some supervised cut face's render depends on it. The transverse family
alone reaches a third of the orange; the longitudinal family cuts through the axis and reaches a
different part, and what matters for the paper is the union: the share of the interior that the
photographs determine at all, against the share that is whatever the initialisation and the priors
left there.

Reach is measured, not derived: one backward pass per plane, and a cell counts as reached if it
receives any gradient at all.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import refsel
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
STATE = os.environ.get("STATE", f"{W}/state_r1.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_mv.npz")
RES = int(os.environ.get("RES", "512"))
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]

st["interior"] = st["interior"].detach().clone().requires_grad_(True)
N = st["interior"].shape[0]
hit = {f: torch.zeros(N, dtype=torch.bool, device=dev) for f in ("h", "v")}


def touch(mvp, n, d):
    st["interior"].grad = None
    img, al, _, _ = ON.render_section(st, glctx, mvp, n, float(d), RES)
    img.sum().backward()
    g = st["interior"].grad
    return torch.zeros(N, dtype=torch.bool, device=dev) if g is None else (g.abs().sum(1) > 0)


for i in range(H_LO, H_HI):
    hit["h"] |= touch(hmvp, hn, hd[i])
for k in range(len(vp)):
    nv = torch.as_tensor(vp[k, :3], dtype=torch.float32, device=dev)
    hit["v"] |= touch(vmvp[k], nv, vp[k, 3])

both = hit["h"] | hit["v"]
print(f"{OBJ}: {N:,} interior cells")
print(f"  transverse   {H_HI - H_LO:3d} photographed planes reach {100 * hit['h'].float().mean():5.1f}%")
print(f"  longitudinal {len(vp):3d} photographed planes reach {100 * hit['v'].float().mean():5.1f}%")
print(f"  both         {H_HI - H_LO + len(vp):3d} planes reach {100 * both.float().mean():5.1f}%"
      f"   overlap {100 * (hit['h'] & hit['v']).float().mean():.1f}%")
print(f"  so {100 * (~both).float().mean():.1f}% of the interior is never touched by a photograph")
