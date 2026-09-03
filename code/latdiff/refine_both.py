"""SDEdit refinement for both families: longitudinal and transverse held-out planes.

The 2-D model each family is refined under is the one trained on that family's own supervised
photographs, so a longitudinal plane is denoised under sd-long3 and a transverse plane under
sd-trans3. Same mask discipline: the flesh alone, the peel pinned at every reverse step to the
render diffused to that step, so the shell we already have is the shell the target keeps.
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

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
ap = argparse.ArgumentParser()
ap.add_argument("--long_ckpt", required=True)
ap.add_argument("--trans_ckpt", required=True)
ap.add_argument("--obj", default="orange_sp")
ap.add_argument("--strength", type=float, default=0.2)
ap.add_argument("--out", default="/workspace/refine2")
a = ap.parse_args()
S = 256; dev = "cuda"
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
print(f"long {os.path.basename(a.long_ckpt)}  trans {os.path.basename(a.trans_ckpt)}")

os.makedirs(a.out, exist_ok=True)


@torch.no_grad()
def sdedit(m, D, x0, s, mask):
    t_start = int(s * D.num_timesteps) - 1
    x = D.q_sample(x0, torch.full((len(x0),), t_start, device=dev, dtype=torch.long))
    for i in reversed(range(t_start + 1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask * x + (1 - mask) * D.q_sample(x0, t)
        x = D.p_sample(m, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return mask * x + (1 - mask) * x0


for fam, planes, mvps, model, diff, pref in (
        ("long", C["ev_planes"], C["ev_mvp"], mL, dL, "v"),
        ("trans", C["eh_planes"], C["eh_mvp"], mT, dT, "h")):
    with torch.no_grad():
        for k in range(len(planes)):
            n = planes[k, :3] / np.linalg.norm(planes[k, :3])
            mvp = torch.as_tensor(mvps[k], dtype=torch.float32, device=dev)
            nt = torch.as_tensor(n, dtype=torch.float32, device=dev)
            img, _, _, _ = ON.render_section(st, glctx, mvp, nt, float(planes[k, 3]), S)
            _, af, _, _ = ON.render_section(st, glctx, mvp, nt, float(planes[k, 3]), S,
                                            exterior=False)
            mask = (af[:1] > 0).float()[None]
            x0 = (img.clamp(0, 1)[None] * 2 - 1)
            tv.utils.save_image(x0 * 0.5 + 0.5, f"{a.out}/raw_{fam}_{k}.png")
            y = sdedit(model, diff, x0, a.strength, mask)
            tv.utils.save_image(y.clamp(-1, 1) * 0.5 + 0.5,
                                f"{a.out}/ref_{fam}_{k}.png")
    print(f"  {fam}: {len(planes)} planes refined at s={a.strength}", flush=True)
