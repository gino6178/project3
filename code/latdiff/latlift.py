"""Stage 4: lift the 2-D refined targets back into the one 3-D field.

Two or three photographs per family cannot supervise a volume, so stage 2 made a target on each
plane that had none. This is where those targets become 3-D: the interior latent is optimised so
that the same field, rendered on every held-out plane at once, matches every target at once. A
texture that is plausible on one plane but disagrees with the others cannot survive, because one
field has to render all of them -- that disagreement is exactly the 3-D consistency two views buy
and a single 2-D generation does not have.

The shell is never touched: the targets were refined only inside the flesh, so their gradient is
zero on the peel, and only the interior decoder is optimised. An anchor to the fitted latent keeps
the planes that do have photographs from drifting while the held-out planes are pulled.
"""
import os, sys, time
os.environ["CUT_DEFERRED"] = "1"
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import torchvision as tv
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import ovnative as ON, anchor, realism
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OBJ = "orange_sp"
STR = os.environ.get("STR", "0.2")
STEPS = int(os.environ.get("STEPS", "400"))
LR = float(os.environ.get("LR", "5e-3"))
ANCHOR = float(os.environ.get("ANCHOR", "3.0"))
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
    rgb0 = di().detach()                          # the fitted interior, the anchor

evp, evm = C["ev_planes"], C["ev_mvp"]            # held-out longitudinal, no photograph
vp, vm = C["v_planes"], C["v_mvp"]                # supervised longitudinal, has photographs


def render(mvp, n, d, exterior=True, grad=False):
    st["interior"] = di() if grad else di().detach()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        img, al, _, _ = ON.render_section(
            st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device=dev),
            torch.as_tensor(n, dtype=torch.float32, device=dev), float(d), S,
            exterior=exterior)
    return img, al


# targets and flesh masks for the held-out planes
tgt, msk, ns = [], [], []
for k in range(len(evp)):
    n = evp[k, :3] / np.linalg.norm(evp[k, :3]); ns.append(n)
    t = tv.transforms.ToTensor()(Image.open(f"{W}/../refine/ref_{k}_{STR}.png").convert("RGB"))
    tgt.append((t.to(dev) * 2 - 1))
    _, af = render(evm[k], n, evp[k, 3], exterior=False)
    msk.append((af[:1] > 0).float())
print(f"{OBJ}: lifting {len(evp)} held-out targets (s={STR}) into the interior field, "
      f"{sum(int(m.sum()) for m in msk)/len(msk):.0f} flesh px/plane")

# Only the per-cell latent moves, not the shared decoder. Optimising the MLP changes how every
# cell decodes, so the supervised planes degrade as fast as the held-out ones improve; moving feat
# touches only the cells the held-out planes actually pass through.
for pr in di.parameters():
    pr.requires_grad_(False)
di.feat.requires_grad_(True)
opt = torch.optim.AdamW([di.feat], lr=LR)
t0 = time.time()
for step in range(1, STEPS + 1):
    loss = 0.0
    for k in range(len(evp)):
        img, _ = render(evm[k], ns[k], evp[k, 3], grad=True)
        loss = loss + (((img * 2 - 1) - tgt[k]) ** 2 * msk[k]).sum() / msk[k].sum().clamp_min(1)
    reg = ANCHOR * ((di() - rgb0) ** 2).mean()
    total = loss / len(evp) + reg
    opt.zero_grad(set_to_none=True); total.backward(); opt.step()
    if step % 50 == 0 or step == 1:
        print(f"  step {step:4d}  face {float(loss)/len(evp):.4f}  anchor {float(reg):.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def score(planes, mvps, photos_dir, tag):
    ref = realism._paths(photos_dir)
    out = f"{W}/lift_{tag}"; os.makedirs(out, exist_ok=True)
    st["interior"] = di().detach()
    paths = []
    for k in range(len(planes)):
        n = planes[k, :3] / np.linalg.norm(planes[k, :3])
        img, _ = render(mvps[k], n, planes[k, 3])
        f = f"{out}/{k:02d}.png"
        Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f)
        paths.append(f)
    return realism._dreamsim(ref, paths, dev), len(ref)


# before: reload the fitted decoder; after: the lifted one
after_hld, nh = score(evp, evm, f"{FN}/hld_orange_v", "hld_after")
after_sup, ns_ = score(vp, vm, f"{FN}/spl_orange_v", "sup_after")
di.load_state_dict(p["dec_i"])
before_hld, _ = score(evp, evm, f"{FN}/hld_orange_v", "hld_before")
before_sup, _ = score(vp, vm, f"{FN}/spl_orange_v", "sup_before")

print(f"\n  DreamSim, lower is better")
print(f"    held-out longitudinal (vs {nh} photos):  {before_hld:.4f} -> {after_hld:.4f}  "
      f"({after_hld-before_hld:+.4f})")
print(f"    supervised longitudinal (vs {ns_} photos): {before_sup:.4f} -> {after_sup:.4f}  "
      f"({after_sup-before_sup:+.4f})   [must not rise]")
