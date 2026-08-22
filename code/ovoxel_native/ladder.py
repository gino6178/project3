"""How many planes can one interior satisfy at once?

One plane fits its photograph to 0.0129 from noise, so neither the representation nor the fitting
is the difficulty. This asks where that stops. N transverse planes are supervised together, all
from the same noise, and two numbers are reported: the error on the planes that were given a
photograph, and the error on the planes of the same band that were not.

The first says whether the interior can hold N photographs at once. The second says whether holding
them puts anything sensible between them, which is the only reason any of this is being built.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import refsel
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
STATE = os.environ.get("STATE", f"{W}/state_r1.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_mv.npz")
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
RES = int(os.environ.get("RES", "512"))
ITERS = int(os.environ.get("LD_ITERS", "400"))
LR = float(os.environ.get("LD_LR", "0.05"))
COUNTS = [int(x) for x in os.environ.get("LD_N", "1,2,4,8,16").split(",")]
dev = "cuda"

st0 = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH = H_HI - H_LO
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]

conf = open(f"{OBJDIR}/{OBJ}.conf").read()
spec = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith("REF_H=")][0]
tgt = {}
for i in range(H_LO, H_HI):
    a = refsel.as_array(refsel.solved_photo(f"{FN}/{spec}", i - H_LO, NH), RES)
    tgt[i] = torch.as_tensor(a, device=dev).permute(2, 0, 1)
print(f"{OBJ}: {NH} supervised depths {H_LO}..{H_HI - 1}, references at {RES}px")


def err_on(st, i):
    img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[i]), RES)
    m = (al > 0.5).float()
    return (((img - tgt[i]).abs() * m).sum() / m.sum().clamp(min=1) / 3), img


for N in COUNTS:
    # centres of N equal parts of the band, so that N=1 is the middle plane and not the pole
    pick = sorted({H_LO + int(round((k + 0.5) * NH / N - 0.5)) for k in range(N)})
    rest = [i for i in range(H_LO, H_HI) if i not in pick]
    st = dict(st0)
    g = torch.Generator(dev).manual_seed(0)
    st["interior"] = torch.rand(st0["interior"].shape, device=dev,
                                generator=g).requires_grad_(True)   # noise, always
    opt = torch.optim.Adam([st["interior"]], lr=LR)
    t0 = time.time()
    for j in range(ITERS):
        loss = sum(err_on(st, i)[0] for i in pick) / len(pick)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            st["interior"].clamp_(0, 1)
    # how much of the interior these planes can even reach: a cell no plane crosses gets no
    # gradient and keeps whatever it was initialised to, for ever
    st["interior"].grad = None
    (sum(err_on(st, i)[0] for i in pick) / len(pick)).backward()
    reach = float((st["interior"].grad.abs().sum(1) > 0).float().mean())
    st["interior"].grad = None
    with torch.no_grad():
        fit = float(np.mean([float(err_on(st, i)[0]) for i in pick]))
        held = float(np.mean([float(err_on(st, i)[0]) for i in rest])) if rest else float("nan")
    print(f"  {N:3d} planes: fitted {fit:.4f}   the {len(rest)} planes between them {held:.4f}"
          f"   they reach {100 * reach:.1f}% of the interior   {time.time() - t0:.0f}s", flush=True)
    if N == COUNTS[-1]:
        with torch.no_grad():
            cols = []
            for i in (pick[:3] + rest[:3]):
                _, img = err_on(st, i)
                cols.append(torch.cat([tgt[i], img.clamp(0, 1)], -2))
            sheet = torch.cat(cols, -1).permute(1, 2, 0)
        Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)) \
            .save(f"{W}/ladder_{OBJ}.jpg", quality=92)
        print(f"SHEET ladder_{OBJ}.jpg  (photograph above, render below; three supervised planes "
              f"then three that were not)")
