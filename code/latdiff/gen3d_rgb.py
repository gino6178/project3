"""Generate an RGB colour volume from the RGB 3-D SinDiffusion and render its cross-sections."""
import os, sys, time
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from unet3d import UNet3D
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType,
                                                 ModelVarType, LossType, get_named_beta_schedule)
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W = "/workspace/ovoxel_native"; OBJ = "orange_sp"; dev = "cuda"
CROP = int(os.environ.get("CROP", "32")); STRIDE = CROP // 2
MULTS = tuple(int(x) for x in os.environ.get("MULTS", "1,2,4").split(","))
D = np.load(f"{W}/lat_rgb_{OBJ}.npz")
vol = torch.from_numpy(D["vol"]).to(dev); occ = torch.from_numpy(D["occ"]).to(dev)
mean = torch.from_numpy(D["mean"]).to(dev); std = torch.from_numpy(D["std"]).to(dev)
hc = float(D["hc"]); org = torch.from_numpy(D["org"]).to(dev)
X, Y, Z, _ = vol.shape
cam = np.load(f"{W}/cams_{OBJ}_v2.npz")
axis = torch.from_numpy(cam["h_planes"][0, :3].astype(np.float32)).to(dev); axis = axis / axis.norm()
g = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in (X, Y, Z)], indexing="ij"), -1).float()
pos = (g + 0.5) * hc + org; mid = pos[occ].mean(0); rel = pos - mid
ax = rel @ axis; rad = (rel - ax[..., None] * axis).norm(dim=-1); R = float(rad[occ].max())
cond = torch.stack([occ.float(), rad / R, ax / R], 0)[None]
ck = torch.load(f"{W}/latdiff_rgb_{OBJ}.pt", map_location=dev, weights_only=False)
model = UNet3D(3 + 3, 3, mults=MULTS).to(dev); model.load_state_dict(ck["model"]); model.eval()
diff = GaussianDiffusion(betas=get_named_beta_schedule("cosine", 1000),
                         model_mean_type=ModelMeanType.EPSILON,
                         model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
print(f"{OBJ}: RGB 3-D SinDiffusion step {ck['step']}, sampling {X}x{Y}x{Z}", flush=True)
occ_np = occ.cpu().numpy(); origins = []
for a in range(0, max(X - CROP, 0) + 1, STRIDE):
    for b in range(0, max(Y - CROP, 0) + 1, STRIDE):
        for c in range(0, max(Z - CROP, 0) + 1, STRIDE):
            if occ_np[a:a+CROP, b:b+CROP, c:c+CROP].any():
                origins.append((a, b, c))
_w = torch.hann_window(CROP, periodic=False, device=dev) + 1e-3
win = (_w[:, None, None] * _w[None, :, None] * _w[None, None, :])[None, None]
ctiles = torch.stack([cond[0, :, a:a+CROP, b:b+CROP, c:c+CROP] for a, b, c in origins])
print(f"  {len(origins)} tiles", flush=True)
def eps(z, tt):
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
    img = torch.randn(1, 3, X, Y, Z, device=dev)
    for i in reversed(range(diff.num_timesteps)):
        t = torch.full((1,), i, device=dev, dtype=torch.long)
        # RGB is in [0,1] normalised; the denoised estimate is clamped to that range in unit space
        img = diff.p_sample(eps, img, t, clip_denoised=False, model_kwargs={},
                            denoised_fn=lambda z: z.clamp(-4, 4))["sample"]
        if i % 200 == 0:
            print(f"    t={i:4d}  |x| {float(img.abs().max()):.2f}  {time.time()-t0:.0f}s", flush=True)
gen = (img[0].permute(1, 2, 3, 0) * std + mean).clamp(0, 1)
solid = torch.from_numpy(D["solid"]).long().to(dev)
rgb = gen[solid[:, 0], solid[:, 1], solid[:, 2]]
# save without a decoder: ovcut takes the direct interior path when there is no dec_i
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
dsr = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
dsr.load_state_dict(p["dec_s"])
with torch.no_grad():
    surf = dsr().clamp(0, 1)
out = f"{W}/s_v2_gen3d_{OBJ}"; os.makedirs(out, exist_ok=True)
torch.save({"dual_v": p["dual_v"], "split_w": p["split_w"],
            "interior": rgb.detach().cpu(), "surf_rgb": surf.detach().cpu()}, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   {time.time()-t0:.0f}s", flush=True)
