"""Generate a colour volume from the 3-D SinDiffusion, and render its cross-sections.

The 3-D SinDiffusion was trained on one 3-D example -- the Stage 1 fitted latent volume -- to
denoise random 32^3 crops of it, exactly as SinDiffusion trains on crops of a single image. This
samples a NEW volume from noise under that model, so its 3-D patch statistics match the fit's; if
the model learned the distribution, the cross-sections of the sampled volume look like orange
cross-sections without any of them being copied.

Two things the earlier band-inpainting got wrong and this fixes: the model is run at the training
crop size in overlapping tiles (GroupNorm reads the same statistics it trained on, not those of a
57%-empty full box), Hann-weighted where tiles overlap; and the denoised estimate is clamped to the
latent's own range, because the cosine schedule multiplies a noise-prediction error by ~300 near
t=1000 and the sample explodes otherwise. Occupancy and the two axis channels are the fixed
conditioning; only the 8-D colour latent is sampled.
"""
import os, sys, time
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/sindiff")
from unet3d import UNet3D
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType,
                                                 ModelVarType, LossType, get_named_beta_schedule)
import ovnative as ON, anchor
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"
OBJ = "orange_sp"
CROP = int(os.environ.get("CROP", "32"))
STRIDE = CROP // 2
CLAMP = float(os.environ.get("CLAMP", "12"))
dev = "cuda"

D = np.load(f"{W}/lat_{OBJ}.npz")
vol = torch.from_numpy(D["vol"]).to(dev)
occ = torch.from_numpy(D["occ"]).to(dev)
mean = torch.from_numpy(D["mean"]).to(dev); std = torch.from_numpy(D["std"]).to(dev)
hc = float(D["hc"]); org = torch.from_numpy(D["org"]).to(dev)
X, Y, Z, Cc = vol.shape

# conditioning: occupancy + radius + axial about the object's own axis (same as training)
cam = np.load(f"{W}/cams_{OBJ}_v2.npz")
axis = torch.from_numpy(cam["h_planes"][0, :3].astype(np.float32)).to(dev); axis = axis / axis.norm()
g = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in (X, Y, Z)], indexing="ij"), -1).float()
pos = (g + 0.5) * hc + org
mid = pos[occ].mean(0); rel = pos - mid
ax = rel @ axis; rad = (rel - ax[..., None] * axis).norm(dim=-1)
R = float(rad[occ].max())
cond = torch.stack([occ.float(), rad / R, ax / R], 0)[None]

ck = torch.load(f"{W}/latdiff_{OBJ}.pt", map_location=dev, weights_only=False)
model = UNet3D(Cc + 3, Cc).to(dev)
model.load_state_dict(ck["model"]); model.eval()
diff = GaussianDiffusion(betas=get_named_beta_schedule("cosine", 1000),
                         model_mean_type=ModelMeanType.EPSILON,
                         model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
print(f"{OBJ}: 3-D SinDiffusion trained to step {ck['step']}, sampling {X}x{Y}x{Z}", flush=True)

# tile origins that contain object
occ_np = occ.cpu().numpy()
origins = []
for a in range(0, max(X - CROP, 0) + 1, STRIDE):
    for b in range(0, max(Y - CROP, 0) + 1, STRIDE):
        for c in range(0, max(Z - CROP, 0) + 1, STRIDE):
            if occ_np[a:a+CROP, b:b+CROP, c:c+CROP].any():
                origins.append((a, b, c))
_w1 = torch.hann_window(CROP, periodic=False, device=dev) + 1e-3
win = (_w1[:, None, None] * _w1[None, :, None] * _w1[None, None, :])[None, None]
ctiles = torch.stack([cond[0, :, a:a+CROP, b:b+CROP, c:c+CROP] for a, b, c in origins])
print(f"  {len(origins)} tiles of {CROP}^3", flush=True)


def tiled_eps(z, tt):
    acc = torch.zeros_like(z); wsum = torch.zeros(1, 1, X, Y, Z, device=dev)
    for i in range(0, len(origins), 32):
        o = origins[i:i+32]
        zt = torch.stack([z[0, :, a:a+CROP, b:b+CROP, c:c+CROP] for a, b, c in o])
        e = model(zt, tt.expand(len(o)), cond=ctiles[i:i+32])
        for k, (a, b, c) in enumerate(o):
            acc[0, :, a:a+CROP, b:b+CROP, c:c+CROP] += e[k] * win[0]
            wsum[0, :, a:a+CROP, b:b+CROP, c:c+CROP] += win[0]
    return acc / wsum.clamp_min(1e-6)


t0 = time.time()
with torch.no_grad():
    img = torch.randn(1, Cc, X, Y, Z, device=dev)
    for i in reversed(range(diff.num_timesteps)):
        t = torch.full((1,), i, device=dev, dtype=torch.long)
        img = diff.p_sample(tiled_eps, img, t, clip_denoised=False, model_kwargs={},
                            denoised_fn=lambda z: z.clamp(-CLAMP, CLAMP))["sample"]
        if i % 200 == 0:
            print(f"    t={i:4d}  |x| {float(img.abs().max()):.2f}  {time.time()-t0:.0f}s", flush=True)

# denormalise, write into the interior decoder's feat, save as a run
gen = (img[0].permute(1, 2, 3, 0) * std + mean)
solid = torch.from_numpy(D["solid"]).long().to(dev)
feat = gen[solid[:, 0], solid[:, 1], solid[:, 2]]

p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
out = f"{W}/s_v2_gen3d_{OBJ}"; os.makedirs(out, exist_ok=True)
q = dict(p); q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = feat.detach().clone()
torch.save(q, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   {time.time()-t0:.0f}s", flush=True)
