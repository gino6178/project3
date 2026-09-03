"""Save the Stage 4 lifted latent as a new run, so ovcut.py can render from it.

The lift updates `dec_i.feat` and leaves everything else at its Stage 1 value. This writes a full
params.pt that is the Stage 1 checkpoint with only feat replaced, so ovcut and every downstream
script can consume it as an ordinary trained run.
"""
import os, sys
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
from PIL import Image
import torchvision as tv
sys.path.insert(0, "/workspace/ovoxel_native")
import ovnative as ON, anchor
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"
OBJ = "orange_sp"
STR = os.environ.get("STR", "0.2")
STEPS = int(os.environ.get("STEPS", "400"))
LR = float(os.environ.get("LR", "5e-3"))
ANCHOR = float(os.environ.get("ANCHOR", "3.0"))
CKPT_2D = os.environ.get("CKPT_2D",
                         "/workspace/sindiff/OUTPUT/sd-long3/model012000.pt")
S = 256
dev = "cuda"
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

# refined targets exist from stage 3 for the released strength
tgt, msk, ns = [], [], []
evp, evm = C["ev_planes"], C["ev_mvp"]
for k in range(len(evp)):
    n = evp[k, :3] / np.linalg.norm(evp[k, :3]); ns.append(n)
    t = tv.transforms.ToTensor()(Image.open(f"{W}/../refine/ref_{k}_{STR}.png").convert("RGB"))
    tgt.append((t.to(dev) * 2 - 1))
    with torch.no_grad():
        _, af, _, _ = ON.render_section(
            st, glctx, torch.as_tensor(evm[k], dtype=torch.float32, device=dev),
            torch.as_tensor(n, dtype=torch.float32, device=dev), float(evp[k, 3]), S,
            exterior=False)
    msk.append((af[:1] > 0).float())

for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=LR)
import time
t0 = time.time()
for step in range(1, STEPS + 1):
    st["interior"] = di()
    loss = 0.0
    for k in range(len(evp)):
        img, _, _, _ = ON.render_section(
            st, glctx, torch.as_tensor(evm[k], dtype=torch.float32, device=dev),
            torch.as_tensor(ns[k], dtype=torch.float32, device=dev), float(evp[k, 3]), S)
        loss = loss + (((img * 2 - 1) - tgt[k]) ** 2 * msk[k]).sum() / msk[k].sum().clamp_min(1)
    reg = ANCHOR * ((di() - rgb0) ** 2).mean()
    total = loss / len(evp) + reg
    opt.zero_grad(set_to_none=True); total.backward(); opt.step()
    if step % 50 == 0 or step == 1:
        print(f"  step {step:4d}  face {float(loss)/len(evp):.4f}  anchor {float(reg):.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)

# a new params.pt that is Stage 1 with feat replaced
out = f"{W}/s_v2_lift_{OBJ}"
os.makedirs(out, exist_ok=True)
q = dict(p)
q["dec_i"] = {k: v.clone().detach() for k, v in p["dec_i"].items()}
q["dec_i"]["feat"] = di.feat.detach().clone()
torch.save(q, f"{out}/params.pt")
open(f"{out}/run.env", "w").write("CAMS_SUFFIX=_v2\n")
print(f"wrote {out}/params.pt")
