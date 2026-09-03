"""What the 2-D SinDiffusion does to each cross-section, on its own -- no 3-D.

Render a plane from the fitted field, SDEdit it under that family's 2-D model, and show render vs
2-D output side by side, for several depths of each family. This is the pure 2-D capability the 3-D
methods are trying to lift; if the detail is not here, no 3-D method can produce it.
"""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torchvision as tv
from PIL import Image
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W = "/workspace/ovoxel_native"; OBJ = "orange_sp"; dev = "cuda"; S = 256
STR = float(os.environ.get("STR", "0.5"))
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad():
    st["interior"], st["surf_rgb"] = di(), dsr()


def load(ckpt):
    d = model_and_diffusion_defaults()
    d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
             attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=True,
             use_scale_shift_norm=True, use_checkpoint=True, diffusion_steps=1000,
             noise_schedule="linear", learn_sigma=False, class_cond=False)
    m, D = create_model_and_diffusion(**d); m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.cuda().eval(); m.convert_to_fp16()
    return m, D


mL, diff = load("/workspace/sindiff/OUTPUT/sd-long64/model002000.pt")
mT, _ = load("/workspace/sindiff/OUTPUT/sd-trans64/model002000.pt")


@torch.no_grad()
def sdedit(m, x0, mask):
    ts = int(STR * diff.num_timesteps) - 1
    x = diff.q_sample(x0, torch.full((len(x0),), ts, device=dev, dtype=torch.long))
    for i in reversed(range(ts + 1)):
        t = torch.full((len(x0),), i, device=dev, dtype=torch.long)
        x = mask * x + (1 - mask) * diff.q_sample(x0, t)
        x = m.__self__.p_sample(m, x, t, clip_denoised=True, model_kwargs={})["sample"] if False else diff.p_sample(m, x, t, clip_denoised=True, model_kwargs={})["sample"]
    return (mask * x + (1 - mask) * x0).clamp(-1, 1)


def arr(t):
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


HL, HH = int(C["h_lo"][0]), int(C["h_hi"][0])
rows = []
for fam, planes, mvps, model in (
        ("long", C["v_planes"], C["v_mvp"], mL),
        ("trans", C["h_planes"][HL:HH], np.broadcast_to(C["h_mvp"][None], (HH - HL, 4, 4)), mT)):
    ks = [2, len(planes)//3, 2*len(planes)//3, len(planes)-3]
    for k in ks:
        n = torch.as_tensor((planes[k, :3] / np.linalg.norm(planes[k, :3])).astype(np.float32), device=dev)
        mvp = torch.as_tensor(mvps[k].copy(), dtype=torch.float32, device=dev).contiguous()
        with torch.no_grad():
            img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(planes[k, 3]), S)
            _, af, _, _ = ON.render_section(st, glctx, mvp, n, float(planes[k, 3]), S, exterior=False)
        mask = (af[:1] > 0).float()[None]
        img64 = torch.nn.functional.interpolate(img[None], 64, mode="bilinear", align_corners=False)
        m64 = (torch.nn.functional.interpolate(mask, 64, mode="nearest") > 0.5).float()
        x0 = (img64 * 2 - 1)
        out = sdedit(model, x0, m64)
        out_up = torch.nn.functional.interpolate(out * 0.5 + 0.5, S, mode="bilinear", align_corners=False)
        rows.append(np.concatenate([arr(img), np.full((S, 6, 3), 255, np.uint8), arr(out_up[0])], 1))
        print(f"  {fam} k={k} done", flush=True)
# stack: 8 rows (4 long, 4 trans), each = render | sdedit
n = len(rows)
top = np.concatenate(rows[:4], 0)
bot = np.concatenate(rows[4:], 0)
grid = np.concatenate([top, np.full((top.shape[0], 12, 3), 255, np.uint8), bot], 1)
Image.fromarray(grid).save(f"/workspace/see2d.png")
print(f"see2d.png  left block=longitudinal, right block=transverse; each: render | 2D-SDEdit  (s={STR})")
