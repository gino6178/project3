"""What the schedule actually lays down, drawn on the object.

The tables say how much of the interior each way of choosing planes reaches. This draws the planes
themselves: every cell is coloured by how many of the first N planes touched it, so a cell no plane
reached stays black and one that several planes crossed is bright. Three schedules, the same N.

Rendered as a cut through the middle of that field rather than as a surface, because what is being
shown is coverage inside the object.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RES = int(os.environ.get("RES", "640"))
NS = [int(x) for x in os.environ.get("SCHED_N", "26,100,400").split(",")]
dev = "cuda"
PHI = (1 + 5 ** 0.5) / 2

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH, NV = H_HI - H_LO, len(C["v_planes"])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vp = C["v_planes"]
step_h = float(hd[H_LO + 1] - hd[H_LO])
lo, hi = float(hd[H_LO]) - step_h / 2, float(hd[H_HI - 1]) + step_h / 2
c_all = ((st["solid"].float() + 0.5) * st["hc"]
         + torch.as_tensor(st["org"], dtype=torch.float32, device=dev))
ctr = c_all.mean(0)
N_CELL = len(st["interior"])


def radical(k, base=2):
    f, r = 1.0, 0.0
    while k:
        f /= base
        r += f * (k % base)
        k //= base
    return r


def touched(n, d):
    st["interior"].grad = None
    _, _, k = ON.cut_polygons(st, n, float(d), device=dev)
    if k == 0:
        return torch.zeros(N_CELL, device=dev)
    img, _, _, _ = ON.render_section(st, glctx, hmvp, n, float(d), RES, exterior=False)
    img.sum().backward()
    g = st["interior"].grad
    return (g.abs().sum(1) > 0).float() if g is not None else torch.zeros(N_CELL, device=dev)


def planes(kind, n):
    """The first n planes each schedule lays down."""
    out, rng, kh, kv = [], np.random.default_rng(0), 0, 0
    for k in range(n):
        if kind == "fixed":
            if k % (NH + NV) < NH:
                out.append((hn, float(hd[H_LO + (k % NH)])))
            else:
                j = k % NV
                out.append((torch.as_tensor(vp[j, :3], dtype=torch.float32, device=dev),
                            float(vp[j, 3])))
            continue
        transverse = (rng.random() < NH / (NH + NV)) if kind == "random" else \
            ((k * NH) % (NH + NV) < NH)
        if transverse:
            kh += 1
            u = rng.random() if kind == "random" else \
                (radical(kh) + (PHI * (kh // 64)) % 1.0) % 1.0
            out.append((hn, lo + u * (hi - lo)))
        else:
            kv += 1
            f = rng.random() if kind == "random" else \
                ((kv * PHI) + PHI * PHI * (kv // 64)) % 1.0
            j = int(f * NV) % NV
            a = f * NV - j
            nv = (1 - a) * vp[j, :3] + a * vp[(j + 1) % NV, :3]
            nv = nv / np.linalg.norm(nv)
            out.append((torch.as_tensor(nv, dtype=torch.float32, device=dev),
                        float(np.dot(nv, vp[j, :3] * vp[j, 3]))))
    return out


st["interior"] = st["interior"].detach().clone().requires_grad_(True)
keep = st["interior"].detach().clone()
rows = []
for kind in ("fixed", "random", "cycle"):
    row = []
    for n in NS:
        cnt = torch.zeros(N_CELL, device=dev)
        for nn, dd in planes(kind, n):
            cnt += touched(nn, dd)
        frac = float((cnt > 0).float().mean())
        with torch.no_grad():
            v = (cnt / cnt.max().clamp(min=1)).clamp(0, 1) ** 0.4
            st["interior"] = torch.stack([v, v * 0.75 + 0.1, v * 0.35], 1)
            img, _, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[(H_LO + H_HI) // 2]),
                                             RES, exterior=False, bg=0.0)
        row.append(img)
        print(f"  {kind:<7} {n:4d} planes: {100 * frac:5.1f}% of cells touched, "
              f"{float(cnt.max()):.0f} times at most", flush=True)
        st["interior"] = keep.clone().requires_grad_(True)
    rows.append(torch.cat(row, -1))
sheet = torch.cat(rows, -2).clamp(0, 1).permute(1, 2, 0)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/demo_sched_{OBJ}.jpg",
                                                                   quality=94)
print(f"SHEET demo_sched_{OBJ}.jpg  (rows: the fixed schedule, independent draws, the formula; "
      f"columns: {NS} planes)")
