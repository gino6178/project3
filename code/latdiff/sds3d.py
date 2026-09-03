"""Sweep the 2-D SinDiffusion over every cell with SDS, from the fitted field.

The 2-D SinDiffusion already produces detailed cross-section faces. Applying its score to a random
plane pushes that slice toward the family's patch distribution; applying it to a stream of random
planes that between them touch every cell pushes the whole field there, and because one field has
to satisfy every plane at once, the result is 3-D consistent. That is SDS (DreamFusion), with a
SinDiffusion as the 2-D model and planar slicing as the renderer.

Two choices that matter here. It starts from the fitted field, not noise -- the fit already carries
the interior detail the photographs supervised, and SDS then only has to make every plane agree
with the patch prior, not invent structure from scratch (which is what plateaued from neutral).
And the prior is a SinDiffusion, whose mode is a textured patch rather than a semantic average, so
the mode-seeking that over-smooths ordinary SDS should here pull toward texture, not away from it.
No anchor, no smoothness term: the point is to see what the 2-D model alone does to the field.
"""
import os, sys, time, argparse
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
ap = argparse.ArgumentParser()
ap.add_argument("--long_ckpt", required=True)
ap.add_argument("--trans_ckpt", required=True)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--planes", type=int, default=4)           # random planes per gradient step
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--w_sds", type=float, default=1.0)
ap.add_argument("--t_min", type=float, default=0.02)
ap.add_argument("--t_max", type=float, default=0.75)       # cap t: high-t SDS is the smoothing part
ap.add_argument("--tag", default="s_v2_sds")
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
dsr = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
dsr.load_state_dict(p["dec_s"])
with torch.no_grad():
    st["surf_rgb"] = dsr()

hc = float(st["hc"]); org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
solid = st["solid"].long(); cen = (solid.float() + 0.5) * hc + org


def load_diff(ckpt):
    d = model_and_diffusion_defaults()
    d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
             attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=False,
             use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
             noise_schedule="linear", learn_sigma=False, class_cond=False)
    m, D = create_model_and_diffusion(**d)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.cuda().eval()
    for pr in m.parameters():
        pr.requires_grad_(False)
    return m, D


phiL, diff = load_diff(a.long_ckpt); phiT, _ = load_diff(a.trans_ckpt)
alphas = torch.as_tensor(diff.alphas_cumprod, device=dev).float()

HL, HH = int(C["h_lo"][0]), int(C["h_hi"][0])
fams = {"long":  (np.concatenate([C["v_planes"], C["ev_planes"]], 0),
                  np.concatenate([C["v_mvp"], C["ev_mvp"]], 0), phiL),
        "trans": (C["h_planes"][HL:HH], np.broadcast_to(C["h_mvp"][None], (HH - HL, 4, 4)), phiT)}
touched = torch.zeros(len(solid), dtype=torch.bool, device=dev)


def sample(name):
    planes, mvps, phi = fams[name]
    lo, hi = float(planes[:, 3].min()), float(planes[:, 3].max())
    dd = float(np.random.uniform(lo, hi))
    idx = int(np.argmin(np.abs(planes[:, 3] - dd)))
    n = torch.as_tensor((planes[idx, :3] / np.linalg.norm(planes[idx, :3])).astype(np.float32), device=dev)
    mvp = torch.as_tensor(mvps[idx].copy(), dtype=torch.float32, device=dev).contiguous()
    touched[(((cen @ n) + dd).abs() <= 1.5 * hc)] = True
    return n, dd, mvp, phi


def render(mvp, n, dd, exterior=True):
    st["interior"] = di()
    return ON.render_section(st, glctx, mvp, n, float(dd), S, exterior=exterior)


for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=a.lr)
tlo, thi = int(a.t_min * diff.num_timesteps), int(a.t_max * diff.num_timesteps)

t0 = time.time()
print(f"  SDS sweep from the fit, {a.steps} steps, {a.planes} planes/step, "
      f"t in [{tlo},{thi}]", flush=True)
for step in range(1, a.steps + 1):
    surustub = 0.0
    got = 0
    for pi in range(a.planes):
        name = "long" if pi % 2 == 0 else "trans"
        n, dd, mvp, phi = sample(name)
        try:
            with torch.no_grad():
                _, af, _, _ = render(mvp, n, dd, exterior=False)
            mask = (af[:1] > 0).float()[None]
            if float(mask.sum()) < 100:
                continue
            img, _, _, _ = render(mvp, n, dd)
        except RuntimeError:
            continue
        x = (img[None] * 2 - 1)
        t = torch.randint(tlo, thi, (1,), device=dev)
        noise = torch.randn_like(x)
        with torch.no_grad():
            x_t = diff.q_sample(x, t, noise)
            eps = phi(x_t, t)
            wt = (1 - alphas[t])[:, None, None, None]
            grad = (wt * (eps - noise)).nan_to_num() * mask
        surustub = surustub + (x * grad).sum() / mask.sum().clamp_min(1)
        got += 1
    if got == 0:
        continue
    opt.zero_grad(set_to_none=True)
    (a.w_sds * surustub / got).backward()
    opt.step()
    if step % 200 == 0 or step == 1:
        print(f"  step {step:5d}  coverage {float(touched.float().mean())*100:.1f}%  "
              f"{time.time()-t0:.0f}s", flush=True)

out = f"{W}/{a.tag}_{OBJ}"; os.makedirs(out, exist_ok=True)
q = dict(p); q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = di.feat.detach().clone()
torch.save(q, f"{out}/params.pt"); open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   {time.time()-t0:.0f}s", flush=True)
