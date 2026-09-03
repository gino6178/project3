"""Dense sweep lift: SDEdit targets uniformly across the whole depth range of each family.

The observation this fixes: lifting only six held-out planes leaves narrow stripes of refined
cells across a sweep that spans a hundred cells, so the video shows a clear discontinuity between
'lifted' and 'unlifted' depths. Uniformly sampling N depths across each family's full range gives
every cell in the sweep a plane touching it, and the whole video comes down together.

Recipe otherwise identical to the working single-lift: 400 gradient steps, feat-only optimisation,
one SDEdit target per plane (s=0.2, flesh mask, no refresh), a mild global anchor to Stage 1.
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
ap.add_argument("--trans_ckpt", required=True)
ap.add_argument("--n_dense", type=int, default=30)
ap.add_argument("--strength", type=float, default=0.2)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--anchor", type=float, default=3.0)
ap.add_argument("--tag", default="s_v2_lift")
ap.add_argument("--only", default="both", choices=["both","long","trans"])
ap.add_argument("--out", default="/workspace/refineD")
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


mL, dL = load_diff(a.long_ckpt); mT, dT = load_diff(a.trans_ckpt)


def render(mvp, n, d, exterior=True, grad=False):
    st["interior"] = di() if grad else di().detach()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return ON.render_section(st, glctx, mvp, n, float(d), S, exterior=exterior)


@torch.no_grad()
def sdedit(model, diff, x0, s, mask):
    t_start = int(s * diff.num_timesteps) - 1
    x = diff.q_sample(x0, torch.full((len(x0),), t_start, device=dev, dtype=torch.long))
    for i in reversed(range(t_start + 1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask * x + (1 - mask) * diff.q_sample(x0, t)
        x = diff.p_sample(model, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return mask * x + (1 - mask) * x0


os.makedirs(a.out, exist_ok=True)


def build_family(name, ref_planes, ref_mvps, model, diff):
    """Uniform depths across the family's full range, SDEdit each once."""
    d_all = np.array(sorted(ref_planes[:, 3].tolist()))
    d_min, d_max = float(d_all.min()), float(d_all.max())
    depths = np.linspace(d_min, d_max, a.n_dense)
    n_ref = ref_planes[0, :3] / np.linalg.norm(ref_planes[0, :3])
    n_t = torch.as_tensor(n_ref.astype(np.float32), device=dev)
    # each dense depth uses the closest reference mvp (family shares a normal but not a per-plane
    # view; picking nearest keeps the render on-object)
    tgts, masks, ns, mvpts, ds = [], [], [], [], []
    for i, dd in enumerate(depths):
        j = int(np.argmin(np.abs(d_all - dd)))
        # look up original mvp at that index (need to search ref_planes for the value)
        idx = int(np.argmin(np.abs(ref_planes[:, 3] - d_all[j])))
        _m = (ref_mvps[idx] if ref_mvps.ndim == 3 else ref_mvps).copy()
        mvp = torch.as_tensor(_m, dtype=torch.float32, device=dev).contiguous()
        try:
            with torch.no_grad():
                img, _, _, _ = render(mvp, n_t, dd)
                _, af, _, _ = render(mvp, n_t, dd, exterior=False)
        except RuntimeError as e:
            print(f"    depth {dd:+.4f} skipped: {str(e)[:60]}", flush=True)
            continue
        mask = (af[:1] > 0).float()[None]
        if float(mask.sum()) < 100:
            continue
        x0 = (img.clamp(0, 1)[None] * 2 - 1)
        tv.utils.save_image(x0 * 0.5 + 0.5, f"{a.out}/{name}_raw_{i:02d}.png")
        tgt = sdedit(model, diff, x0, a.strength, mask).clamp(-1, 1)
        tv.utils.save_image(tgt * 0.5 + 0.5, f"{a.out}/{name}_ref_{i:02d}.png")
        tgts.append(tgt); masks.append(mask[0])
        ns.append(n_t); mvpts.append(mvp); ds.append(float(dd))
    print(f"  {name}: {len(tgts)} dense targets, depths uniform in [{d_min:.3f}, {d_max:.3f}], "
          f"spacing {(d_max-d_min)/max(len(tgts)-1,1)/float(st['hc']):.1f} cells", flush=True)
    return tgts, masks, ns, mvpts, ds


HL, HH = int(C["h_lo"][0]), int(C["h_hi"][0])
lt_all = np.concatenate([C["v_planes"], C["ev_planes"]], 0)
lm_all = np.concatenate([C["v_mvp"], C["ev_mvp"]], 0)
tt_all = np.concatenate([C["h_planes"][HL:HH], C["eh_planes"]], 0)
tm_h = np.broadcast_to(C["h_mvp"][None], (HH - HL, 4, 4)).copy()
tm_all = np.concatenate([tm_h, C["eh_mvp"]], 0)

t0 = time.time()
tL = build_family("long", lt_all, lm_all, mL, dL) if a.only in ("both", "long") else ([],[],[],[],[])
tT = build_family("trans", tt_all, tm_all, mT, dT) if a.only in ("both", "trans") else ([],[],[],[],[])
tgts = tL[0] + tT[0]; masks = tL[1] + tT[1]
ns = tL[2] + tT[2]; mvpts = tL[3] + tT[3]; ds = tL[4] + tT[4]
print(f"  total {len(tgts)} targets, target generation {time.time()-t0:.0f}s", flush=True)

for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=a.lr)

for step in range(1, a.steps + 1):
    loss = 0.0
    for k in range(len(tgts)):
        img, _, _, _ = render(mvpts[k], ns[k], ds[k], grad=True)
        loss = loss + (((img * 2 - 1) - tgts[k]) ** 2 * masks[k]).sum() / \
            masks[k].sum().clamp_min(1)
    reg = a.anchor * ((di() - rgb0) ** 2).mean()
    total = loss / len(tgts) + reg
    opt.zero_grad(set_to_none=True); total.backward(); opt.step()
    if step % 50 == 0 or step == 1:
        print(f"  step {step:4d}  face {float(loss)/len(tgts):.4f}  "
              f"anchor {float(reg):.4f}  {time.time()-t0:.0f}s", flush=True)

out = f"{W}/{a.tag}_{OBJ}"
os.makedirs(out, exist_ok=True)
q = dict(p)
q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = di.feat.detach().clone()
torch.save(q, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"\nwrote {out}/params.pt   {time.time()-t0:.0f}s total")
