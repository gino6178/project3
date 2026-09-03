"""Iterative Dataset Update, after Instruct-NeRF2NeRF (Haque et al. 2023, arxiv 2303.12789).

The dataset is a fixed set of planes, each with a target image. Two kinds:
  * anchor planes -- every supervised plane of both families -- whose target is the Stage 1 render
    and never changes. These are IN2N's unedited ground-truth views: they hold the fit in place on
    the cells photographs already constrained, and that is what keeps the edit from drifting to the
    single photograph the 2-D model was trained on. My earlier dense lift omitted them and drifted.
  * edit planes -- a dense set of longitudinal depths with no photograph -- whose target is an
    SDEdit of the CURRENT render, refreshed a few at a time. IN2N's insight is that early edits
    disagree, but because each is re-generated from the current field, they converge to one
    3-D-consistent appearance over iterations.

Every outer iteration refreshes a random handful of edit targets, then takes K gradient steps on
the whole dataset. Only the per-cell latent moves.
"""
import os, sys, time, argparse
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torchvision as tv
from PIL import Image
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/sindiff")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
ap = argparse.ArgumentParser()
ap.add_argument("--long_ckpt", required=True)
ap.add_argument("--n_edit", type=int, default=40)       # dense longitudinal depths, refreshed
ap.add_argument("--outer", type=int, default=10)
ap.add_argument("--inner", type=int, default=50)
ap.add_argument("--refresh", type=int, default=6)       # edit targets refreshed per outer iter
ap.add_argument("--strength", type=float, default=0.4)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--anchor", type=float, default=1.0)
ap.add_argument("--tag", default="s_v2_lift")
ap.add_argument("--no_anchor_planes", action="store_true")
ap.add_argument("--held_only", action="store_true")
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
    rgb0 = di().detach().clone()

d = model_and_diffusion_defaults()
d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
         attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=True,
         use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
         noise_schedule="linear", learn_sigma=False, class_cond=False)
model, diff = create_model_and_diffusion(**d)
model.load_state_dict(torch.load(a.long_ckpt, map_location="cpu"))
model.cuda().eval()
if d["use_fp16"]:
    model.convert_to_fp16()


def render(mvp, n, dd, exterior=True, grad=False):
    st["interior"] = di() if grad else di().detach()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return ON.render_section(st, glctx, mvp, n, float(dd), S, exterior=exterior)


@torch.no_grad()
def sdedit(x0, s, mask):
    t_start = int(s * diff.num_timesteps) - 1
    x = diff.q_sample(x0, torch.full((len(x0),), t_start, device=dev, dtype=torch.long))
    for i in reversed(range(t_start + 1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask * x + (1 - mask) * diff.q_sample(x0, t)
        x = diff.p_sample(model, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return (mask * x + (1 - mask) * x0).clamp(-1, 1)


# ---- build the dataset -------------------------------------------------------
def flesh_mask(mvp, n, dd):
    _, af, _, _ = render(mvp, n, dd, exterior=False)
    return (af[:1] > 0).float()[None]


anchors, edits = [], []            # each entry: dict(mvp, n, d, target, mask, edit:bool)

# anchor planes: every supervised plane, both families, target = Stage 1 render (frozen)
HL, HH = int(C["h_lo"][0]), int(C["h_hi"][0])
with torch.no_grad():
    for planes, mvps in ((C["v_planes"], C["v_mvp"]),
                         (C["h_planes"][HL:HH], np.broadcast_to(C["h_mvp"][None], (HH - HL, 4, 4)))):
        for k in range(len(planes)):
            n = torch.as_tensor((planes[k, :3] / np.linalg.norm(planes[k, :3])).astype(np.float32),
                                device=dev)
            mvp = torch.as_tensor(mvps[k].copy(), dtype=torch.float32, device=dev).contiguous()
            try:
                img, _, _, _ = render(mvp, n, planes[k, 3])
                m = flesh_mask(mvp, n, planes[k, 3])
            except RuntimeError:
                continue
            if float(m.sum()) < 100:
                continue
            anchors.append(dict(mvp=mvp, n=n, d=float(planes[k, 3]),
                                target=(img.clamp(0, 1)[None] * 2 - 1).clone(), mask=m[0]))
print(f"  {len(anchors)} anchor planes (Stage 1 render, frozen)")

# edit planes: dense longitudinal depths, target filled at first refresh
vd = C["v_planes"][:, 3]
d_min, d_max = float(min(vd.min(), C["ev_planes"][:, 3].min())), \
    float(max(vd.max(), C["ev_planes"][:, 3].max()))
n_ref = torch.as_tensor((C["v_planes"][0, :3] / np.linalg.norm(C["v_planes"][0, :3])).astype(np.float32),
                        device=dev)
allv = np.concatenate([C["v_planes"], C["ev_planes"]], 0)
allm = np.concatenate([C["v_mvp"], C["ev_mvp"]], 0)
if a.held_only:
    _depths = [(C["ev_planes"][k], C["ev_mvp"][k]) for k in range(len(C["ev_planes"]))]
else:
    _depths = []
    for dd in np.linspace(d_min, d_max, a.n_edit):
        idx = int(np.argmin(np.abs(allv[:, 3] - dd)))
        _depths.append((allv[idx], allm[idx]))
for pl, mv in _depths:
    n = torch.as_tensor((pl[:3] / np.linalg.norm(pl[:3])).astype(np.float32), device=dev)
    mvp = torch.as_tensor(mv.copy(), dtype=torch.float32, device=dev).contiguous()
    try:
        m = flesh_mask(mvp, n, pl[3])
    except RuntimeError:
        continue
    if float(m.sum()) < 100:
        continue
    edits.append(dict(mvp=mvp, n=n, d=float(pl[3]), target=None, mask=m[0]))
print(f"  {len(edits)} edit planes (dense longitudinal, {(d_max-d_min)/max(len(edits)-1,1)/float(st['hc']):.1f}-cell spacing)")

# initial SDEdit for every edit plane once, so the dataset is complete from step 1
with torch.no_grad():
    for e in edits:
        img, _, _, _ = render(e["mvp"], e["n"], e["d"])
        e["target"] = sdedit((img.clamp(0, 1)[None] * 2 - 1), a.strength, e["mask"][None])

# ---- iterative dataset update ------------------------------------------------
for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=a.lr)

t0 = time.time()
print(f"  IN2N: {a.outer} outer x {a.inner} inner, refresh {a.refresh}/iter, s={a.strength}",
      flush=True)
for it in range(1, a.outer + 1):
    # refresh a random handful of edit targets from the CURRENT field
    with torch.no_grad():
        for e in [edits[i] for i in np.random.permutation(len(edits))[:a.refresh]]:
            img, _, _, _ = render(e["mvp"], e["n"], e["d"])
            e["target"] = sdedit((img.clamp(0, 1)[None] * 2 - 1), a.strength, e["mask"][None])

    data = (edits if a.no_anchor_planes else anchors + edits)
    for inner in range(a.inner):
        loss = 0.0
        for pl in data:
            img, _, _, _ = render(pl["mvp"], pl["n"], pl["d"], grad=True)
            loss = loss + (((img * 2 - 1) - pl["target"][0]) ** 2 * pl["mask"]).sum() / \
                pl["mask"].sum().clamp_min(1)
        reg = a.anchor * ((di() - rgb0) ** 2).mean()
        (loss / len(data) + reg).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    print(f"  it {it:2d}/{a.outer}  loss {float(loss)/len(data):.4f}  anchor {float(reg):.4f}  "
          f"{time.time()-t0:.0f}s", flush=True)

out = f"{W}/{a.tag}_{OBJ}"
os.makedirs(out, exist_ok=True)
q = dict(p)
q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = di.feat.detach().clone()
torch.save(q, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   {time.time()-t0:.0f}s")
