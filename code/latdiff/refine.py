"""Turn a plane with no photograph into a target, using a model trained only on planes that have one.

The 3-D field is the goal; this is the only way to supervise it. Two or three photographs per
family cannot supervise a volume, so the 2-D model trained on those photographs is used to produce
a target on planes they never reached: the plane is rendered from the fitted lattice, noised part of
the way back, and denoised under a model whose notion of a good cut face came from the supervised
photographs. The starting point is the render rather than noise, so the target keeps this object's
own layout and gains the statistics the render is missing.

How far back to noise is the whole question, so it is swept rather than chosen. The held-out
photographs appear once, to score the sweep; they are not available to the model or to the fit.
"""
import argparse, os, sys, time
import numpy as np, torch
import torchvision as tv
from PIL import Image
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/sindiff")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"/workspace/sindiff/OUTPUT/sd-long00/model008000.pt")
ap.add_argument("--obj", default="orange_sp")
ap.add_argument("--family", default="v", choices=["h", "v"])
ap.add_argument("--strengths", default="0.2,0.35,0.5,0.7")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--out", default="/workspace/refine")
a = ap.parse_args()
dev = "cuda"
OBJ = a.obj
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
    st["interior"], st["surf_rgb"] = di(), ds()

pl = C["ev_planes"] if a.family == "v" else C["eh_planes"]
mv = C["ev_mvp"] if a.family == "v" else C["eh_mvp"]
print(f"{OBJ}: {len(pl)} held-out {a.family} planes")

d = model_and_diffusion_defaults()
d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
         attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=True,
         use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
         noise_schedule="linear", learn_sigma=False, class_cond=False)
model, diff = create_model_and_diffusion(**d)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.cuda().eval()
if d["use_fp16"]:
    model.convert_to_fp16()
print(f"  2-D model {os.path.basename(a.ckpt)}")

os.makedirs(a.out, exist_ok=True)
S = a.size
strengths = [float(x) for x in a.strengths.split(",")]


@torch.no_grad()
def sdedit(x0, s, mask):
    """Noise the render to s of the way back, then denoise -- but only inside the mask.

    The shell is not refined. It is the one part of the object taken from photographs directly and
    already correct, so letting the model repaint it only lets it invent a worse outline. mask is 1
    on the cut face and 0 on the peel; at every reverse step the outside is pinned to the render
    diffused to that step, so the shell the model sees as context is the true shell and the shell it
    outputs is the true shell, unchanged. This is RePaint's constraint, applied to keep a region
    rather than to fill one.
    """
    t_start = int(s * diff.num_timesteps) - 1
    x = diff.q_sample(x0, torch.full((len(x0),), t_start, device=dev, dtype=torch.long))
    for i in reversed(range(t_start + 1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask * x + (1 - mask) * diff.q_sample(x0, t)
        x = diff.p_sample(model, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return mask * x + (1 - mask) * x0


t0 = time.time()
raw, made = [], {s: [] for s in strengths}
with torch.no_grad():
    for k in range(len(pl)):
        n = torch.as_tensor(pl[k, :3], dtype=torch.float32, device=dev)
        n = n / n.norm()
        mvpk = torch.as_tensor(mv[k], dtype=torch.float32, device=dev)
        img, _, _, _ = ON.render_section(st, glctx, mvpk, n, float(pl[k, 3]), S)
        # the cut face alone: its alpha is the flesh, everything outside it is peel or background
        _, af, _, _ = ON.render_section(st, glctx, mvpk, n, float(pl[k, 3]), S, exterior=False)
        mask = (af[:1] > 0).float()[None]
        x0 = (img.clamp(0, 1)[None] * 2 - 1)
        f = f"{a.out}/raw_{k}.png"
        tv.utils.save_image(x0 * 0.5 + 0.5, f); raw.append(f)
        if k == 0:
            tv.utils.save_image(mask, f"{a.out}/mask_0.png")
        for s in strengths:
            y = sdedit(x0, s, mask)
            g = f"{a.out}/ref_{k}_{s}.png"
            tv.utils.save_image(y.clamp(-1, 1) * 0.5 + 0.5, g); made[s].append(g)
        print(f"    plane {k}: {time.time()-t0:.0f}s", flush=True)

ref = realism._paths(f"{FN}/hld_orange_{a.family}") if OBJ.startswith("orange") else []
print(f"\n  scored against {len(ref)} held-out photograph(s), lower is better")
print(f"    {'the fitted render':<22}{realism._dreamsim(ref, raw, dev):.4f}")
for s in strengths:
    print(f"    {'refined, s=' + str(s):<22}{realism._dreamsim(ref, made[s], dev):.4f}")
