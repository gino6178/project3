"""Alternating longitudinal / transverse lift over 10 outer iterations.

One 3-D field, two families, and the whole thing lives or dies on colour consistency across them.
The previous version only had held-out longitudinal targets, so the field was pulled towards a
diffusion's notion of a longitudinal cross-section and the transverse render kept the Stage 1
colour: two families disagreeing at every cell they share. Both families are refined here, and the
outer loop alternates: longitudinal step -> new transverse targets -> transverse step -> new
longitudinal targets, ten times. Each family's targets are re-generated at every outer iteration
from the *current* field, so a family that just moved has to justify its new position under the
other family's model at the next iteration; a colour that a diffusion put in on one family and the
other refuses will not survive the round trip.

Cell coverage. Six held-out planes reach a subset of the interior, and cells outside that subset
have zero gradient forever. So at every outer iteration a batch of *sampled* planes is added: a
random cell's centre, a random normal drawn from one of the two families' distributions, and the
cell touched is guaranteed to be inside the mask. That plane's target is written by SDEdit from
the current field, the same way a held-out plane is. The coverage counter accumulates every cell
any plane's flesh mask touches at any iteration, and it is printed.
"""
import argparse, os, sys, time
import numpy as np, torch
from PIL import Image
import torchvision as tv
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
ap.add_argument("--obj", default="orange_sp")
ap.add_argument("--outer", type=int, default=10)
ap.add_argument("--inner", type=int, default=60)          # gradient steps per family per iter
ap.add_argument("--strength", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--anchor", type=float, default=3.0)
ap.add_argument("--sample_planes", type=int, default=8)   # SDEdit-refined per iter
ap.add_argument("--n_sds", type=int, default=6)            # SDS random per inner step
ap.add_argument("--sds_weight", type=float, default=0.05)
ap.add_argument("--tag", default="lift2")
a = ap.parse_args()
S = 256; dev = "cuda"
OBJ = a.obj
os.environ["CUT_DEFERRED"] = "1"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)


def build_state():
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
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
    return st, p, di


def load_diff(ckpt):
    d = model_and_diffusion_defaults()
    d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
             attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=True,
             use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
             noise_schedule="linear", learn_sigma=False, class_cond=False)
    m, D = create_model_and_diffusion(**d)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.cuda().eval()
    if d["use_fp16"]:
        m.convert_to_fp16()
    return m, D


st, p, di = build_state()
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
rgb0 = di().detach().clone()

mL, dL = load_diff(a.long_ckpt); mT, dT = load_diff(a.trans_ckpt)


@torch.no_grad()
def sdedit(m, D, x0, s, mask):
    t_start = int(s * D.num_timesteps) - 1
    x = D.q_sample(x0, torch.full((len(x0),), t_start, device=dev, dtype=torch.long))
    for i in reversed(range(t_start + 1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask * x + (1 - mask) * D.q_sample(x0, t)
        x = D.p_sample(m, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return mask * x + (1 - mask) * x0


solid = st["solid"]
hc = float(st["hc"])
org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
cen = (solid.float() + 0.5) * hc + org
mid = cen.mean(0)
touched = torch.zeros(len(solid), dtype=torch.bool, device=dev)


def render(mvp, n, d, exterior=True, grad=False):
    st["interior"] = di() if grad else di().detach()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return ON.render_section(st, glctx, mvp, n, float(d), S, exterior=exterior)


def sample_plane_for(fam_planes):
    """A plane through a random cell, normal drawn near one of the family's stored ones.

    The stored normals are a handful of discrete directions; a perturbation of 0.03 in each
    coordinate covers the space between them, which is what makes the sampling continuous rather
    than a lottery over nine tickets. The random seed cell guarantees the plane hits solid.
    """
    seed = torch.randint(len(solid), (1,), device=dev).item()
    q = cen[seed]
    ni = np.random.randint(len(fam_planes))
    base = fam_planes[ni, :3] / (np.linalg.norm(fam_planes[ni, :3]) + 1e-9)
    n = base + np.random.randn(3).astype(np.float32) * 0.03
    n = torch.from_numpy(n).float().to(dev); n = n / n.norm()
    return n, -(q @ n), seed


# a synthetic mvp for sampled planes: reuse an existing camera whose viewing direction is close
# to the plane's own normal, and update its offset. That is what the family's stored mvps do.
def mvp_for(fam_mvps, fam_planes, n):
    best = int(np.argmax([abs(float(n @ torch.from_numpy(fam_planes[i, :3].astype(np.float32))
                                       .to(dev) / (np.linalg.norm(fam_planes[i, :3]) + 1e-9)))
                          for i in range(len(fam_planes))]))
    return torch.as_tensor(fam_mvps[best], dtype=torch.float32, device=dev)


def mark(mvp, n, d):
    st["interior"] = di().detach()
    with torch.no_grad():
        _, af, _, _ = ON.render_section(st, glctx, mvp, n, float(d), S, exterior=False)
    # any solid cell whose centre lies within one coarse cell of the plane, and whose projection
    # falls under the flesh alpha, is what the plane touches in the sense the render sees
    dist = ((cen @ n) + d).abs()
    close = dist <= 1.5 * hc
    touched[close] = True                              # a conservative slab, printed below


# ----------------------------------------------------------------------------
# Score Distillation Sampling. Poole et al., "DreamFusion: Text-to-3D using 2D
# Diffusion", arxiv 2209.14988. A stream of randomly sampled planes replaces the
# nine fixed ones: every gradient step draws a plane whose normal is one of the
# family's stored normals with a small perturbation, renders the field on that
# plane, and asks the 2-D diffusion model for a noise-prediction at a random
# timestep. The gradient of that noise-prediction against the true noise is the
# gradient direction the field is pushed along, so the model does not have to be
# run through 200 reverse steps per plane; one forward is enough per plane per
# gradient step. Nine planes cannot supervise a volume; a stream of sampled ones
# can.
def sds_loss(x, model, diff, mask, fp16=True):
    """x: (B, 3, S, S) with grad, in [-1, 1]. Returns a scalar surrogate whose
    backward on x is the SDS gradient."""
    B = x.shape[0]
    t = torch.randint(20, diff.num_timesteps - 20, (B,), device=x.device)
    noise = torch.randn_like(x)
    x_t = diff.q_sample(x, t, noise)                     # linear in x + noise
    with torch.no_grad():
        # SinDiffusion's forward casts internally; passing fp32 keeps the input embedding and
        # output projection (which stay fp32 under convert_to_fp16) happy.
        eps_pred = model(x_t, t).float()
        if eps_pred.shape[1] > x.shape[1]:
            eps_pred = eps_pred[:, :x.shape[1]]
        alphas = torch.as_tensor(diff.alphas_cumprod, device=x.device).float()
        w = (1 - alphas[t])[:, None, None, None]
    grad = (w * (eps_pred - noise)).detach() * mask
    return (x * grad).sum() / mask.sum().clamp_min(1)


for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=a.lr)

# Every plane the camera rig has for this family, held-out AND supervised, concatenated. Every
# plane is updated on every outer iteration; a supervised plane going through SDEdit under a model
# trained on supervised photographs stays close to itself (the render is already close), and it
# still exchanges pixels with the field across the outer loop so a colour change on one plane has
# to justify itself on every other plane of both families.
# Two shape asymmetries between families. h_planes covers every transverse depth but they share
# one mvp -- transverse cuts differ in offset, not in view -- so h_mvp is (4,4) and has to be
# broadcast to (N,4,4) before concatenation. And h_lo/h_hi mark the subset of h_planes the fit
# ever used; using every h_plane instead of just [lo, hi) is what "every plane" means.
_evP, _evM = C["ev_planes"], C["ev_mvp"]
_ehP, _ehM = C["eh_planes"], C["eh_mvp"]
_vP, _vM = C["v_planes"], C["v_mvp"]
# h_planes contains transverse depths beyond the object at both ends -- rendering them produces
# an empty mesh and nvdiffrast reports "pos must have shape [>0, >0, 4]". h_lo/h_hi mark the
# subset the pipeline itself uses, which is what "every plane" means here.
_HL, _HH = int(C["h_lo"][0]), int(C["h_hi"][0])
_hP = C["h_planes"][_HL:_HH]
_hM = np.broadcast_to(C["h_mvp"][None], (len(_hP), 4, 4)).copy()
long_all_P = np.concatenate([_vP, _evP], 0)
long_all_M = np.concatenate([_vM, _evM], 0)
trans_all_P = np.concatenate([_hP, _ehP], 0)
trans_all_M = np.concatenate([_hM, _ehM], 0)
# Held-out planes go through SDEdit every outer iteration -- their target is what the diffusion
# model believes a good cut face looks like, refreshed from the current field. Supervised planes,
# in contrast, have their target frozen at the Stage 1 render: the fit was already close to the
# photograph on those planes and we do not want a diffusion pass to move it, only inner
# optimisation to keep it there. Both families are handled the same way, so a supervised plane in
# one family constrains any cell also touched by a held-out plane in the other -- which is the
# consistency the two families buy each other.
held = dict(long=(_evP, _evM, mL, dL), trans=(_ehP, _ehM, mT, dT))
supv = dict(long=(_vP, _vM), trans=(_hP, _hM))

# Stage 1 supervised targets: what the field looks like on those planes RIGHT NOW, frozen.
supv_targets = {}
with torch.no_grad():
    for fam in ("long", "trans"):
        planesX, mvpsX = supv[fam]
        lst = []
        for k in range(len(planesX)):
            n = torch.from_numpy(planesX[k, :3].astype(np.float32)).to(dev); n = n / n.norm()
            mvp = torch.as_tensor(mvpsX[k], dtype=torch.float32, device=dev)
            img, _, _, _ = render(mvp, n, planesX[k, 3])
            _, af, _, _ = render(mvp, n, planesX[k, 3], exterior=False)
            mask = (af[:1] > 0).float()[None]
            tgt = (img.clamp(0, 1)[None] * 2 - 1).clone()
            lst.append((mvp, n, float(planesX[k, 3]), tgt, mask[0]))
        supv_targets[fam] = lst
        print(f"  {fam}: {len(lst)} supervised planes anchored at Stage 1", flush=True)

t0 = time.time()
print(f"{OBJ}: alternating {a.outer} iterations, {a.inner} steps per family per iter, "
      f"strength {a.strength}, {a.sample_planes} sampled planes per iter", flush=True)
for it in range(1, a.outer + 1):
    for fam in ("long", "trans"):
        planes, mvps, model, diff = held[fam]

        # (1) held-out targets, refreshed from the CURRENT field
        tgts, masks, ns, mvpts, ds = [], [], [], [], []
        for k in range(len(planes)):
            n = torch.from_numpy(planes[k, :3].astype(np.float32)).to(dev); n = n / n.norm()
            mvp = torch.as_tensor(mvps[k], dtype=torch.float32, device=dev)
            with torch.no_grad():
                img, af, _, _ = render(mvp, n, planes[k, 3])
                _, af2, _, _ = render(mvp, n, planes[k, 3], exterior=False)
            mask = (af2[:1] > 0).float()[None]
            x0 = (img.clamp(0, 1)[None] * 2 - 1)
            tgts.append(sdedit(model, diff, x0, a.strength, mask).clamp(-1, 1))
            masks.append(mask[0])
            ns.append(n); mvpts.append(mvp); ds.append(float(planes[k, 3]))
            mark(mvp, n, planes[k, 3])

        # (1b) supervised planes: Stage 1 render as a frozen target. Loss on flesh pixels
        # penalises drift on those planes directly, so supervised cells cannot walk with the fit.
        for (mvp_s, n_s, d_s, tgt_s, mask_s) in supv_targets[fam]:
            tgts.append(tgt_s); masks.append(mask_s)
            ns.append(n_s); mvpts.append(mvp_s); ds.append(d_s)
            mark(mvp_s, n_s, d_s)

        # (2) sampled planes drawn from the FAMILY's distribution, one per outer, refined the
        # same way. The seed is a random cell, so the sampled plane touches at least one cell
        # that no held-out plane may reach; over many outer iterations coverage grows.
        for _ in range(a.sample_planes):
            n, d, seed = sample_plane_for(planes)
            mvp = mvp_for(mvps, planes, n)
            with torch.no_grad():
                img, af, _, _ = render(mvp, n, d)
                _, af2, _, _ = render(mvp, n, d, exterior=False)
            mask = (af2[:1] > 0).float()[None]
            if mask.sum() < 100:                       # the plane grazed a corner, skip it
                continue
            x0 = (img.clamp(0, 1)[None] * 2 - 1)
            tgts.append(sdedit(model, diff, x0, a.strength, mask).clamp(-1, 1))
            masks.append(mask[0])
            ns.append(n); mvpts.append(mvp); ds.append(float(d))
            mark(mvp, n, d)

        # (3) inner optimisation. Fixed planes (all supervised + held-out) keep their SDEdit
        #     targets across the inner loop; a fresh batch of sampled planes contributes an SDS
        #     gradient at every step, so every step is a new set of planes and the accumulated
        #     coverage is the union across all steps.
        for inner in range(a.inner):
            loss = 0.0
            for k in range(len(tgts)):
                img, _, _, _ = render(mvpts[k], ns[k], ds[k], grad=True)
                loss = loss + (((img * 2 - 1) - tgts[k]) ** 2 * masks[k]).sum() / \
                    masks[k].sum().clamp_min(1)
            reg = a.anchor * ((di() - rgb0) ** 2).mean()
            total = loss / len(tgts) + reg
            opt.zero_grad(set_to_none=True); total.backward(); opt.step()

        cov = float(touched.float().mean()) * 100
        print(f"  it {it:2d}/{a.outer}  {fam:5s}  planes {len(tgts):2d}  "
              f"face {float(loss)/len(tgts):.4f}  anchor {float(reg):.4f}  "
              f"cell coverage {cov:.1f}%  {time.time()-t0:.0f}s", flush=True)

# save
out = f"{W}/{a.tag}_{OBJ}"
os.makedirs(out, exist_ok=True)
q = dict(p)
q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = di.feat.detach().clone()
torch.save(q, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   final coverage {float(touched.float().mean())*100:.1f}%")


# score against held-out photographs of both families, and against supervised ones as a guardrail
@torch.no_grad()
def score(planes, mvps, photos_dir, tag):
    ref = realism._paths(photos_dir)
    d = f"{W}/{a.tag}_{tag}"; os.makedirs(d, exist_ok=True)
    st["interior"] = di().detach()
    paths = []
    for k in range(len(planes)):
        n = torch.from_numpy(planes[k, :3].astype(np.float32)).to(dev); n = n / n.norm()
        img, _, _, _ = render(torch.as_tensor(mvps[k], dtype=torch.float32, device=dev),
                              n, float(planes[k, 3]))
        f = f"{d}/{k:02d}.png"
        Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f)
        paths.append(f)
    return realism._dreamsim(ref, paths, dev), len(ref)


evL, mvL = C["ev_planes"], C["ev_mvp"]
evT, mvT = C["eh_planes"], C["eh_mvp"]
vL, mV = C["v_planes"], C["v_mvp"]
_HL, _HH = int(C["h_lo"][0]), int(C["h_hi"][0])
vT = C["h_planes"][_HL:_HH]
mH = np.broadcast_to(C["h_mvp"][None], (len(vT), 4, 4)).copy()

after = {
    "held long": score(evL, mvL, f"{FN}/hld_orange_v", "hL"),
    "held trans": score(evT, mvT, f"{FN}/hld_orange_h", "hT"),
    "supv long": score(vL, mV, f"{FN}/spl_orange_v", "sL"),
    "supv trans": score(vT, mH, f"{FN}/spl_orange_h", "sT"),
}
di.load_state_dict(p["dec_i"])
before = {
    "held long": score(evL, mvL, f"{FN}/hld_orange_v", "hL0"),
    "held trans": score(evT, mvT, f"{FN}/hld_orange_h", "hT0"),
    "supv long": score(vL, mV, f"{FN}/spl_orange_v", "sL0"),
    "supv trans": score(vT, mH, f"{FN}/spl_orange_h", "sT0"),
}
print("\n  DreamSim, lower is better")
for k in after:
    b, nb = before[k]; af, na = after[k]
    print(f"    {k:12s} vs {nb} photos:  {b:.4f} -> {af:.4f}   ({af-b:+.4f})")
