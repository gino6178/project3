"""3-D SinDiffusion, supervised by 2-D cross-section photographs through planar slicing.

SinDiffusion learns the internal patch distribution of a single example. There is no 3-D example
here -- only cross-section photographs -- so the 3-D volume is defined implicitly: it is the volume
whose slices, along either family, look like they were drawn from that family's photographs. The
2-D SinDiffusion trained on those photographs is the patch prior; planar slicing is the renderer
that turns the 3-D volume into the 2-D images the prior scores; and the volume is optimised from a
neutral fill, not from the fitted lattice, so what fills it comes from the photographs.

Distillation is Variational Score Distillation (ProlificDreamer, Wang et al. 2023, arxiv
2305.16213), not plain SDS. SDS pushes each slice toward the single highest-likelihood image of the
prior, which for a one-photograph SinDiffusion is that one photograph -- every slice collapses to
it and the volume loses its interior. VSD keeps a second score network psi that is continuously
fine-tuned on the volume's OWN current slices; the update is the difference between where the
photographs pull (phi) and where the volume already is (psi), so a slice is only moved by what the
photographs add over what the volume already shows, and the diversity across depths survives.
"""
import os, sys, time, argparse, copy
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/sindiff")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
ap = argparse.ArgumentParser()
ap.add_argument("--long_ckpt", required=True)
ap.add_argument("--trans_ckpt", required=True)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--planes_per_step", type=int, default=2)   # random slices per gradient step
ap.add_argument("--lr", type=float, default=1e-2)           # on the colour volume (feat)
ap.add_argument("--lr_psi", type=float, default=1e-5)       # on the variational score net
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--tv", type=float, default=2.0)          # 3-D smoothness on the colour volume
ap.add_argument("--tag", default="s_v2_vsd")
a = ap.parse_args()
S = 256; dev = "cuda"; OBJ = "orange_sp"
ON.FDG = ON._load_ovoxel()
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
    rgb0 = di().detach()

# Neutral init: the interior latent is reset so the decoded colour is the mean interior colour
# everywhere, i.e. the volume starts with no interior structure and VSD has to build it. The shape
# (occupancy) and the shell are untouched -- those came from photographs and geometry, not from a
# generative prior.
with torch.no_grad():
    mean_rgb = rgb0.mean(0, keepdim=True)
    di.feat.mul_(0.0)                    # a flat latent -> the decoder's bias colour, ~mean

hc = float(st["hc"])
org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
solid = st["solid"].long()
cen = (solid.float() + 0.5) * hc + org
mid = cen.mean(0)

# dense-grid indexing for the TV regulariser: each solid cell's neighbours in +x/+y/+z
dims = [int(solid[:, i].max()) + 2 for i in range(3)]
lin = (solid[:, 0] * dims[1] + solid[:, 1]) * dims[2] + solid[:, 2]
cell_at = torch.full((dims[0] * dims[1] * dims[2],), -1, dtype=torch.long, device=dev)
cell_at[lin] = torch.arange(len(solid), device=dev)
strides = torch.tensor([dims[1] * dims[2], dims[2], 1], device=dev)


def tv_loss(rgb):
    """Sum of squared colour differences to the +x, +y, +z neighbour where both cells are solid.

    A per-cell colour field has no coupling between neighbours, so a noisy slice gradient dents one
    cell and leaves the next untouched -- the holes and speckle in the sweep. This couples them:
    the field is pushed to vary smoothly through the volume, which is what fills the neutral holes
    the slice gradients never reached and removes the per-cell speckle.
    """
    loss = 0.0
    for ax in range(3):
        nb = cell_at[(lin + strides[ax]).clamp(max=len(cell_at) - 1)]
        ok = nb >= 0
        loss = loss + ((rgb[ok] - rgb[nb[ok]]) ** 2).mean()
    return loss


def load_diff(ckpt):
    d = model_and_diffusion_defaults()
    d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
             attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=False,
             use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
             noise_schedule="linear", learn_sigma=False, class_cond=False)
    m, D = create_model_and_diffusion(**d)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.cuda(), D


# phi: frozen prior per family. psi: a trainable copy per family (the variational score).
phiL, diff = load_diff(a.long_ckpt); phiL.eval()
phiT, _ = load_diff(a.trans_ckpt); phiT.eval()
for m in (phiL, phiT):
    for pr in m.parameters():
        pr.requires_grad_(False)
psiL = copy.deepcopy(phiL); psiT = copy.deepcopy(phiT)
for m in (psiL, psiT):
    m.train()
    for pr in m.parameters():
        pr.requires_grad_(True)
opt_psi = torch.optim.AdamW(list(psiL.parameters()) + list(psiT.parameters()), lr=a.lr_psi)

for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=a.lr)

HL, HH = int(C["h_lo"][0]), int(C["h_hi"][0])
fams = {
    "long":  (np.concatenate([C["v_planes"], C["ev_planes"]], 0),
              np.concatenate([C["v_mvp"], C["ev_mvp"]], 0), phiL, psiL),
    "trans": (np.concatenate([C["h_planes"][HL:HH], C["eh_planes"]], 0),
              np.concatenate([np.broadcast_to(C["h_mvp"][None], (HH - HL, 4, 4)),
                              C["eh_mvp"]], 0), phiT, psiT),
}


touched = torch.zeros(len(solid), dtype=torch.bool, device=dev)


def sample_plane(name):
    planes, mvps, phi, psi = fams[name]
    d_lo, d_hi = float(planes[:, 3].min()), float(planes[:, 3].max())
    dd = float(np.random.uniform(d_lo, d_hi))
    idx = int(np.argmin(np.abs(planes[:, 3] - dd)))
    n = torch.as_tensor((planes[idx, :3] / np.linalg.norm(planes[idx, :3])).astype(np.float32),
                        device=dev)
    mvp = torch.as_tensor(mvps[idx].copy(), dtype=torch.float32, device=dev).contiguous()
    # every cell within a coarse cell of this plane counts as touched, so coverage is measured
    touched[(((cen @ n) + dd).abs() <= 1.5 * hc)] = True
    return n, dd, mvp, phi, psi


def render(mvp, n, dd, exterior=True):
    st["interior"] = di()
    return ON.render_section(st, glctx, mvp, n, float(dd), S, exterior=exterior)


alphas = torch.as_tensor(diff.alphas_cumprod, device=dev).float()
t0 = time.time()
print(f"  VSD-3D from neutral init, {a.steps} steps, {a.planes_per_step} planes/step", flush=True)
for step in range(1, a.steps + 1):
    vsd_grad_planes = []
    # ---- (1) VSD gradient on the colour volume, from random slices of both families
    for _pi in range(a.planes_per_step):
        name = "long" if _pi % 2 == 0 else "trans"
        n, dd, mvp, phi, psi = sample_plane(name)
        try:
            img, af, _, _ = render(mvp, n, dd)
            _, af2, _, _ = render(mvp, n, dd, exterior=False)
        except RuntimeError:
            continue
        mask = (af2[:1] > 0).float()[None]
        if float(mask.sum()) < 100:
            continue
        x = (img[None] * 2 - 1)
        t = torch.randint(20, diff.num_timesteps - 20, (1,), device=dev)
        noise = torch.randn_like(x)
        x_t = diff.q_sample(x, t, noise)
        with torch.no_grad():
            eps_phi = phi(x_t, t)
            eps_psi = psi(x_t, t)
            wt = (1 - alphas[t])[:, None, None, None]
            grad = (wt * (eps_phi - eps_psi)).nan_to_num() * mask
        vsd_grad_planes.append((x, grad, x_t.detach(), t.detach(), noise.detach(), mask,
                                phi, psi))
    if not vsd_grad_planes:
        continue
    # surrogate whose backward on feat is the VSD gradient
    surrogate = sum((x * g).sum() / m.sum().clamp_min(1)
                    for (x, g, *_rest, m, _phi, _psi) in vsd_grad_planes)
    opt.zero_grad(set_to_none=True)
    (surrogate + a.tv * tv_loss(di())).backward()
    opt.step()

    # ---- (2) fine-tune psi on the current slices (predict the noise that was added)
    loss_psi = 0.0
    for (x, g, x_t, t, noise, mask, phi, psi) in vsd_grad_planes:
        pred = psi(x_t, t)
        loss_psi = loss_psi + (((pred - noise) ** 2) * mask).sum() / mask.sum().clamp_min(1)
    opt_psi.zero_grad(set_to_none=True)
    (loss_psi / len(vsd_grad_planes)).backward()
    opt_psi.step()

    if step % 200 == 0 or step == 1:
        with torch.no_grad():
            drift = float((di() - rgb0).abs().mean())
            tvv = float(tv_loss(di()))
        cov = float(touched.float().mean()) * 100
        print(f"  step {step:5d}  psi {float(loss_psi)/len(vsd_grad_planes):.4f}  "
              f"drift {drift:.4f}  tv {tvv:.4f}  coverage {cov:.1f}%  {time.time()-t0:.0f}s",
              flush=True)
    if step % 2000 == 0:
        _o = f"{W}/{a.tag}_{OBJ}"; os.makedirs(_o, exist_ok=True)
        _q = dict(p); _q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
        _q["dec_i"]["feat"] = di.feat.detach().clone()
        torch.save(_q, f"{_o}/params.pt")
        open(f"{_o}/run.env", "w").write("CAMS_SUFFIX=_v2\n")

out = f"{W}/{a.tag}_{OBJ}"
os.makedirs(out, exist_ok=True)
q = dict(p)
q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = di.feat.detach().clone()
torch.save(q, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   {time.time()-t0:.0f}s")
