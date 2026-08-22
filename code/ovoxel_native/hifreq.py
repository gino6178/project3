"""How much of the photographs' texture survives, on planes that had one and on planes that did not.

The error a fit reaches says how close the colours are; it says nothing about whether the structure
is there. A supervised cut face keeps 91% of its photograph's gradient, measured. This asks the same
of the faces nobody photographed, which is the only place the answer was ever in doubt -- and it is
the number that decides whether a frequency-aware loss is needed or whether coverage already did it.
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
STATE = os.environ.get("STATE", f"{W}/state_{OBJ}.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_{OBJ}_bal.npz")
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
RES = int(os.environ.get("RES", "512"))
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH = H_HI - H_LO
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
step_h = float(hd[H_LO + 1] - hd[H_LO]) if NH > 1 else 1.0
lo, hi = float(hd[H_LO]) - step_h / 2, float(hd[H_HI - 1]) + step_h / 2

conf = open(f"{OBJDIR}/{OBJ}.conf").read()
spec = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith("REF_H=")][0]
refs = [torch.as_tensor(refsel.as_array(refsel.solved_photo(f"{FN}/{spec}", i, NH), RES),
                        device=dev).permute(2, 0, 1) for i in range(NH)]


def grad_on(a, m):
    gx = (a[..., 1:] - a[..., :-1]).abs() * m[..., 1:] * m[..., :-1]
    gy = (a[..., 1:, :] - a[..., :-1, :]).abs() * m[..., 1:, :] * m[..., :-1, :]
    n = (m[..., 1:] * m[..., :-1]).sum().clamp(min=1) + (m[..., 1:, :] * m[..., :-1, :]).sum().clamp(min=1)
    return float((gx.sum() + gy.sum()) / n / 3)


gp = float(np.mean([grad_on(refs[i], torch.ones_like(refs[i][:1])) for i in range(NH)]))
print(f"{OBJ}: the photographs' own gradient {gp:.4f}")

for name in [x for x in os.environ.get("HF_RUNS", "difftrain_orange_sp,difftrain_orange_sp_fixed").split(",") if x]:
    p = f"{W}/{name}.pt"
    if not os.path.exists(p):
        print(f"  {name}: no such run"); continue
    st["interior"] = torch.load(p, map_location=dev)["interior"].to(dev)
    with torch.no_grad():
        sup, uns = [], []
        for i in range(NH):
            img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[H_LO + i]), RES)
            sup.append(grad_on(img, (al > 0.5).float()))
        for f in np.linspace(0.1, 0.9, 9):          # depths between the photographed ones
            d = lo + float(f) * (hi - lo)
            if min(abs(d - float(hd[H_LO + i])) for i in range(NH)) < step_h * 0.25:
                continue
            img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
            uns.append(grad_on(img, (al > 0.5).float()))
    print(f"  {name}: photographed planes {np.mean(sup):.4f} ({100 * np.mean(sup) / gp:.0f}% of the "
          f"photographs), unphotographed depths {np.mean(uns):.4f} "
          f"({100 * np.mean(uns) / gp:.0f}%), {len(uns)} of them")
