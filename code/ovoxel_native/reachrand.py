"""If the planes are drawn afresh each step instead of sitting at fixed depths, how far do they get?

The fixed schedule supervises the same 26 cut faces for the whole run, and measured, those faces
touch about half the interior; the rest of the cells never receive a gradient and keep whatever
they were initialised to. Drawing the planes at random spends the same number of renders per step
but spreads them, so the union grows with the number of steps rather than staying put.

This measures the union alone -- pure geometry, no fitting -- because that is what decides whether
the idea is worth training on. What it cannot say is whether a photograph is a fair target for a
plane it was not taken at; that is the next question, and a separate one.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
STATE = os.environ.get("STATE", f"{W}/state_{OBJ}.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_{OBJ}_bal.npz")
RES = int(os.environ.get("RES", "512"))
DRAWS = int(os.environ.get("RR_DRAWS", "400"))
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
NH, NV = H_HI - H_LO, len(vp)

st["interior"] = st["interior"].detach().clone().requires_grad_(True)
N = st["interior"].shape[0]


def touch(mvp, n, d):
    st["interior"].grad = None
    img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(d), RES)
    img.sum().backward()
    g = st["interior"].grad
    return torch.zeros(N, dtype=torch.bool, device=dev) if g is None else (g.abs().sum(1) > 0)


rng = np.random.default_rng(0)
lo, hi = float(hd[H_LO]), float(hd[H_HI - 1])
step = (hi - lo) / max(NH - 1, 1)
# the band the fixed depths tile, plus the half step each end that they stand in the middle of
lo, hi = lo - step / 2, hi + step / 2

hit = torch.zeros(N, dtype=torch.bool, device=dev)
marks = sorted({NH + NV, 50, 100, 200, DRAWS})
print(f"{OBJ}: {N:,} interior cells; fixed schedule is {NH} transverse and {NV} longitudinal")
t0 = time.time()
for k in range(DRAWS):
    if rng.random() < NH / (NH + NV):
        hit |= touch(hmvp, hn, rng.uniform(lo, hi))            # a depth anywhere in the band
    else:
        j = int(rng.integers(NV))                              # an azimuth between two cameras
        a = rng.random()
        nv = (1 - a) * vp[j, :3] + a * vp[(j + 1) % NV, :3]
        nv = nv / np.linalg.norm(nv)
        d = float(np.dot(nv, vp[j, :3] * vp[j, 3]))
        hit |= touch(torch.as_tensor(vmvp[j]), torch.as_tensor(nv, dtype=torch.float32,
                                                               device=dev), d)
    if (k + 1) in marks:
        print(f"  {k + 1:4d} drawn planes reach {100 * hit.float().mean():5.1f}%"
              f"   {time.time() - t0:.0f}s", flush=True)
