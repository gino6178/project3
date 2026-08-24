"""Stage 2, part two: re-sample the cells a held-out plane exposes, and score the face.

The held-out longitudinal planes are the ones with no photograph, and the ones Table 2 shows
carrying a gap. This takes the latent stage 1 fitted, masks a band of cells about one such plane,
re-samples that band from the diffusion model conditioned on every cell outside it, decodes, and
renders the same plane again. The photograph is used once, at the end, to say whether the face got
closer to it -- never as an input to the sampling, or the comparison would be circular.

The whole volume goes through the network at every reverse step rather than tiles: a generative
fill has to agree across the band, and a tile seam is exactly the kind of structure this is
supposed to be producing rather than inventing.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/sindiff")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
from unet3d import UNet3D
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType,
                                                 ModelVarType, LossType,
                                                 get_named_beta_schedule)
import ovnative as ON, anchor, realism
from PIL import Image

W = "/workspace/ovoxel_native"
OBJ = os.environ.get("OBJ", "orange_sp")
BAND = float(os.environ.get("BAND", "4"))        # coarse cells either side of the plane
PLANE = int(os.environ.get("PLANE", "0"))        # which held-out longitudinal plane
RES = int(os.environ.get("RES", "512"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext(device=dev)

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
di.load_state_dict(p["dec_i"])
ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
ds.load_state_dict(p["dec_s"])
with torch.no_grad():
    st["surf_rgb"] = ds()
    rgb0 = di()

D = np.load(f"{W}/lat_{OBJ}.npz")
vol = torch.from_numpy(D["vol"]).to(dev)
occ = torch.from_numpy(D["occ"]).to(dev)
mean = torch.from_numpy(D["mean"]).to(dev)
std = torch.from_numpy(D["std"]).to(dev)
hc, org = float(D["hc"]), torch.from_numpy(D["org"]).to(dev)
solid = torch.from_numpy(D["solid"]).long().to(dev)
X, Y, Z, Cc = vol.shape

# the checkpoint carries the latent mean and std as numpy arrays, which torch 2.6 refuses to
# unpickle under its new weights_only default
ck = torch.load(f"{W}/latdiff_{OBJ}.pt", map_location=dev, weights_only=False)
model = UNet3D(Cc + 3, Cc).to(dev)
model.load_state_dict(ck["model"]); model.eval()
print(f"{OBJ}: diffusion trained to step {ck['step']}")

axis = torch.from_numpy(C["h_planes"][0, :3].astype(np.float32)).to(dev)
axis = axis / axis.norm()
g = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in (X, Y, Z)],
                               indexing="ij"), -1).float()
pos = (g + 0.5) * hc + org
mid = pos[occ].mean(0)
rel = pos - mid
ax = rel @ axis
rad = (rel - ax[..., None] * axis).norm(dim=-1)
R = float(rad[occ].max())
cond = torch.stack([occ.float(), rad / R, ax / R], 0)[None]

# the held-out plane, and the band of cells it runs through
evp = C["ev_planes"]
n = torch.from_numpy(evp[PLANE, :3].astype(np.float32)).to(dev)
n = n / n.norm(); d = float(evp[PLANE, 3])
mvp = torch.as_tensor(C["ev_mvp"][PLANE], dtype=torch.float32, device=dev)
dist = (pos @ n + d).abs()
band = (dist <= BAND * hc) & occ
print(f"  plane {PLANE} of {len(evp)}: {int(band.sum()):,} cells within {BAND} of it "
      f"({float(band.sum())/float(occ.sum())*100:.1f}% of the object)")

x0 = ((vol - mean) / std) * occ[..., None].float()
x0 = x0.permute(3, 0, 1, 2)[None].contiguous()
m = band.float()[None, None]                     # 1 = re-sampled, 0 = kept

diff = GaussianDiffusion(betas=get_named_beta_schedule("cosine", 1000),
                         model_mean_type=ModelMeanType.EPSILON,
                         model_var_type=ModelVarType.FIXED_LARGE,
                         loss_type=LossType.MSE)

# The model must be run at the size it was trained at. GroupNorm normalises over every spatial
# position, and a 32^3 crop of the object and the whole 117x128x120 box -- 57% of which is exactly
# zero -- present it with different statistics: on a crop the predicted noise falls to 0.43 of unit
# by t=100, on the full volume it stays at 0.96, so the reverse process never stops injecting noise
# and the latent ends 158 times larger than the one it started from. Tiles at the training size,
# with their predictions averaged under a Hann window wherever they overlap, keep the statistics
# right and leave no seam for the average to have to hide.
CROP = int(ck["crop"])
STRIDE = CROP // 2
origins = []
for a in range(0, max(X - CROP, 0) + 1, STRIDE):
    for b in range(0, max(Y - CROP, 0) + 1, STRIDE):
        for c in range(0, max(Z - CROP, 0) + 1, STRIDE):
            if band[a:a+CROP, b:b+CROP, c:c+CROP].any():
                origins.append((a, b, c))
print(f"  {len(origins)} tiles of {CROP}^3 cover the band, stride {STRIDE}")

_w1 = torch.hann_window(CROP, periodic=False, device=dev) + 1e-3
win = (_w1[:, None, None] * _w1[None, :, None] * _w1[None, None, :])[None, None]
ctiles = torch.stack([cond[0, :, a:a+CROP, b:b+CROP, c:c+CROP] for a, b, c in origins])


def tiled_eps(z, tt):
    acc = torch.zeros_like(z)
    wsum = torch.zeros(1, 1, X, Y, Z, device=dev)
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
    img = torch.randn_like(x0)
    for i in reversed(range(diff.num_timesteps)):
        t = torch.full((1,), i, device=dev, dtype=torch.long)
        img = m * img + (1 - m) * diff.q_sample(x0, t)     # outside the band: what stage 1 fitted
        # x0 has to be bounded or the reverse process is explosive: under the cosine schedule
        # sqrt(1/alpha_bar) is over three hundred at t=999, so an error in the predicted noise is
        # multiplied by that. clip_denoised assumes images in [-1, 1]; this latent has unit
        # standard deviation and an observed extreme of 10.4, so 12 is its equivalent.
        img = diff.p_sample(tiled_eps, img, t, clip_denoised=False, model_kwargs={},
                            denoised_fn=lambda z: z.clamp(-12, 12))["sample"]
        if i % 200 == 0:
            print(f"    t={i:4d}  |x| {float(img.abs().max()):.2f}  {time.time()-t0:.0f}s",
                  flush=True)
    img = m * img + (1 - m) * x0

new = img[0].permute(1, 2, 3, 0) * std + mean
feat = new[solid[:, 0], solid[:, 1], solid[:, 2]]
changed = band[solid[:, 0], solid[:, 1], solid[:, 2]]
print(f"  {int(changed.sum()):,} cells re-sampled; latent moved "
      f"{float((feat[changed] - di.feat.detach()[changed]).norm(dim=1).mean()):.4f} on average "
      f"against a per-cell norm of {float(di.feat.detach()[changed].norm(dim=1).mean()):.4f}")

with torch.no_grad():
    st["interior"] = rgb0
    a0, _, _, _ = ON.render_section(st, glctx, mvp, n, d, RES)
    di.feat.data = feat
    st["interior"] = di()
    a1, _, _, _ = ON.render_section(st, glctx, mvp, n, d, RES)

out = f"{W}/fill_{OBJ}_{PLANE}"
os.makedirs(out, exist_ok=True)
for tag, a in (("before", a0), ("after", a1)):
    Image.fromarray((a.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
        f"{out}/{tag}.png")

# The photograph enters here and nowhere else. It was not available to the sampling, so this is a
# measurement of the fill rather than a restatement of its input.
# the photographs sit under the worktree, not under the code tree
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
conf = open(f"/workspace/rebuild/project3/code/objects/{OBJ}.conf").read()


def spec(k, default=""):
    for line in conf.splitlines():
        if line.strip().startswith(k):
            return line.split("=", 1)[1].strip().strip('"')
    return default


ref = realism._paths(os.path.join(FN, spec("EVAL_REF_V=", spec("EVAL_REF="))))
if ref:
    b = realism._dreamsim(ref, [f"{out}/before.png"], dev)
    a = realism._dreamsim(ref, [f"{out}/after.png"], dev)
    print(f"  held-out longitudinal plane {PLANE}, DreamSim against {len(ref)} photograph(s): "
          f"{b:.4f} before, {a:.4f} after  ({a-b:+.4f})")
else:
    print("  no held-out longitudinal photograph for this object; images only")
print(f"  wrote {out}/before.png and after.png  ({time.time()-t0:.0f}s)")
