"""One plane, one photograph. Can the representation hold it at all?

Everything else in this line -- priors, critics, volumes reconciled across two families -- assumes
that a cut face CAN be made to look like its photograph when the photograph is handed to it. This
asks only that. One transverse plane, its own photograph as the target, the interior cells as the
only thing that moves, and no prior of any kind in the loss.

Three numbers come out of it. The error the optimiser reaches; the error of the best a per-cell
field could possibly reach on this plane, which is the photograph itself passed through the
lattice's resolution and back; and how much of the photograph's gradient each of them keeps. If the
optimiser lands on the second number, the representation is doing everything it can and the limit
is the cell size. If it lands well short, the limit is the fitting and everything downstream of it
was built on sand.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
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
ITERS = int(os.environ.get("OP_ITERS", "300"))
LR = float(os.environ.get("OP_LR", "0.05"))
PLANE = int(os.environ.get("OP_PLANE", "-1"))       # -1 -> the middle supervised depth
# What the interior starts as. "released" is whatever `ovnative.build` seeded from the captured
# model's own colours, which is not an honest starting point for this question: the interior is
# supposed to come from the photographs and nothing else, and a run that starts already looking
# like an orange has been given the answer. "flat" is the neutral 0.5 the pipeline uses when the
# interior is required to be earned, and "noise" is where a diffusion chain would start.
INIT = os.environ.get("OP_INIT", "flat")
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
i = (H_LO + (H_HI - H_LO) // 2) if PLANE < 0 else PLANE
d = float(hd[i])

conf = open(f"{OBJDIR}/{OBJ}.conf").read()
spec = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith("REF_H=")][0]
NH = H_HI - H_LO
photo = refsel.as_array(refsel.solved_photo(f"{FN}/{spec}", i - H_LO, NH), RES)
tgt = torch.as_tensor(photo, device=dev).permute(2, 0, 1)   # the renderer is CHW
print(f"{OBJ}: plane {i} of {len(hd)} (supervised {H_LO}..{H_HI - 1}), d {d:+.4f}, "
      f"reference {tgt.shape[-2]}x{tgt.shape[-1]}")


def grad(a):
    return float((a[..., 1:] - a[..., :-1]).abs().mean() +
                 (a[..., 1:, :] - a[..., :-1, :]).abs().mean()) / 2


# what a per-cell field can hold on this plane: the photograph at the lattice's own resolution
cells = float(np.ptp(st["solid"].cpu().numpy(), 0).max()) + 1
k = max(int(round(RES / cells)), 1)
low = F.avg_pool2d(tgt[None], k)
ceil_img = F.interpolate(low, (RES, RES), mode="nearest")[0]
print(f"  the lattice is {cells:.0f} cells across, so {RES}px of photograph is {RES / k:.0f} "
      f"samples: ceiling error {float((ceil_img - tgt).abs().mean()):.4f}, "
      f"gradient {grad(ceil_img):.4f} of the photograph's {grad(tgt):.4f} "
      f"({100 * grad(ceil_img) / grad(tgt):.0f}%)")

base = st["interior"].detach()
if INIT == "flat":
    base = torch.full_like(base, 0.5)
elif INIT == "noise":
    g = torch.Generator(dev).manual_seed(0)
    base = torch.rand(base.shape, device=dev, generator=g)
print(f"  interior starts from: {INIT}")
st["interior"] = base.clone().requires_grad_(True)
opt = torch.optim.Adam([st["interior"]], lr=LR)
img0 = None
t0 = time.time()
for j in range(ITERS + 1):
    img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
    m = (al > 0.5).float()
    loss = ((img - tgt).abs() * m).sum() / m.sum().clamp(min=1) / 3
    if j == 0:
        img0 = img.detach().clone()
        print(f"  before: error {float(loss):.4f}, gradient {grad(img0):.4f} "
              f"({100 * grad(img0) / grad(tgt):.0f}% of the photograph)")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        st["interior"].clamp_(0, 1)
    if j % max(ITERS // 6, 1) == 0:
        print(f"    {j:4d}  error {float(loss):.4f}   {time.time() - t0:.0f}s", flush=True)

with torch.no_grad():
    img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
    m = (al > 0.5).float()
    # on the cut face only, the same as the loss: off it the render is background and the
    # photograph is whatever the photographer's table was, and comparing those says nothing
    err = float((((img - tgt).abs() * m).sum() / m.sum().clamp(min=1) / 3))
    gi, gc, gt = grad(img * m), grad(ceil_img * m), grad(tgt * m)
print(f"  after {ITERS}: error {err:.4f} on the cut face, gradient {gi:.4f} "
      f"({100 * gi / gt:.0f}% of the photograph's {gt:.4f}; the ceiling keeps "
      f"{100 * gc / gt:.0f}%)")

sheet = torch.cat([tgt, ceil_img, img0, img], -1).clamp(0, 1).permute(1, 2, 0)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)) \
    .save(f"{W}/oneplane_{OBJ}.jpg", quality=92)
print(f"SHEET oneplane_{OBJ}.jpg  (photograph | lattice ceiling | before | after)")
