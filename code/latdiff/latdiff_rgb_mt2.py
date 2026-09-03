"""Stage 2, part one: a diffusion model over the lattice stage 1 fitted.

One object is one training example, so the model is kept at patch scale and trained on random 32^3
crops. Three channels of the input are not diffused and are always known: the occupancy, and the
radius and axial coordinate of each cell about the object's own polar axis. The axis is measured,
not labelled -- it is the same one the plane families are built on -- and it is what carries the
organisation the cross-sections have. Asking a patch model to invent radial structure with no
notion of where the centre is was the failure that made this conditioning worth its three channels.
"""
import os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/sindiff")
from unet3d import UNet3D
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType,
                                                 ModelVarType, LossType,
                                                 get_named_beta_schedule)

W = "/workspace/ovoxel_native"
OBJ = os.environ.get("OBJ", "orange_sp")
CROP = int(os.environ.get("CROP", "32"))
MULTS = tuple(int(x) for x in os.environ.get("MULTS", "1,2,4").split(","))
BATCH = int(os.environ.get("BATCH", "8"))
STEPS = int(os.environ.get("STEPS", "8000"))
LR = float(os.environ.get("LR", "2e-4"))
dev = "cuda"

D = np.load(f"{W}/lat_rgb_mt2_{OBJ}.npz")
vol = torch.from_numpy(D["vol"]).to(dev)                       # (X, Y, Z, 8)
occ = torch.from_numpy(D["occ"]).to(dev)
mean = torch.from_numpy(D["mean"]).to(dev)
std = torch.from_numpy(D["std"]).to(dev)
hc, org = float(D["hc"]), torch.from_numpy(D["org"]).to(dev)
X, Y, Z, C = vol.shape

x = ((vol - mean) / std) * occ[..., None].float()              # unit scale, empty cells at zero
x = x.permute(3, 0, 1, 2).contiguous()

cam = np.load(f"{W}/cams_{OBJ}_v2.npz")
axis = torch.from_numpy(cam["h_planes"][0, :3].astype(np.float32)).to(dev)
axis = axis / axis.norm()
g = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in (X, Y, Z)],
                               indexing="ij"), -1).float()
pos = (g + 0.5) * hc + org
mid = pos[occ].mean(0)
rel = pos - mid
ax = rel @ axis
rad = (rel - ax[..., None] * axis).norm(dim=-1)
R = float(rad[occ].max())
cond = torch.stack([occ.float(), rad / R, ax / R], 0).contiguous()
print(f"{OBJ}: volume {X}x{Y}x{Z}, {int(occ.sum()):,} solid, radius {R/hc:.0f} cells; "
      f"conditioning = occupancy, radius, axial")

# crops that contain the object: sampled from the cells themselves, so a crop is never all empty
idx = torch.nonzero(occ)
diff = GaussianDiffusion(betas=get_named_beta_schedule("cosine", 1000),
                         model_mean_type=ModelMeanType.EPSILON,
                         model_var_type=ModelVarType.FIXED_LARGE,
                         loss_type=LossType.MSE)
model = UNet3D(3 + 3, 3, mults=MULTS).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"  UNet3D {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, "
      f"crops {CROP}^3, mults {MULTS}, batch {BATCH}, {STEPS} steps")

t0 = time.time()
for step in range(1, STEPS + 1):
    c = idx[torch.randint(len(idx), (BATCH,), device=dev)]
    lo = (c - CROP // 2).clamp(min=torch.zeros(3, dtype=torch.long, device=dev),
                               max=torch.tensor([X - CROP, Y - CROP, Z - CROP], device=dev))
    xb = torch.stack([x[:, a:a+CROP, b:b+CROP, d:d+CROP] for a, b, d in lo.tolist()])
    cb = torch.stack([cond[:, a:a+CROP, b:b+CROP, d:d+CROP] for a, b, d in lo.tolist()])
    t = torch.randint(0, diff.num_timesteps, (BATCH,), device=dev)
    loss = diff.training_losses(lambda z, tt: model(z, tt, cond=cb), xb, t)["loss"].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 250 == 0 or step == 1:
        print(f"  step {step:6d}  loss {float(loss):.4f}  {time.time()-t0:.0f}s", flush=True)
    if step % 2000 == 0 or step == STEPS:
        torch.save({"model": model.state_dict(), "mean": D["mean"], "std": D["std"],
                    "crop": CROP, "step": step}, f"{W}/latdiff_rgb_mt2_{OBJ}.pt")
print(f"latdiff_rgb_mt2_{OBJ}.pt written, {time.time()-t0:.0f}s")
