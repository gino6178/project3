"""Run the TPDM sampler on one object and look at what came out.

Nothing is written back into the pipeline here. The question this answers is the one the priors
were built for: starting from noise in the volume, do the two families' models agree well enough to
leave a solid whose slices look like sections of this fruit -- in BOTH directions at once, which is
the thing every 2D-only attempt in this line has failed at.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import refsel
import tpdm
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
TAG = os.environ.get("TP_TAG", "_s0.5")
STATE = os.environ.get("STATE", f"{W}/state_r1.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_mv.npz")
STEPS = int(os.environ.get("TP_STEPS", "0")) or None
CHUNK = int(os.environ.get("TP_CHUNK", "1000"))
RESUME = os.environ.get("TP_RESUME", "1") == "1"
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
C = np.load(CAMS)
print(f"{OBJ}: {len(st['interior']):,} interior cells, grid {tuple(st['idx3'].shape)}")

if tpdm.MODE == "prior":
    nets = {f: tpdm.load_prior(f"{W}/pd_{OBJ}_{f}{TAG}.pt", dev) for f in ("h", "v")}
    for f, (_, T, sh) in nets.items():
        print(f"  prior {f}: T={T}, schedule shift {sh}")
else:
    import patchdiff
    T = int(os.environ.get("TP_T", "1000"))
    patchdiff.T = T
    sh = float(os.environ.get("TP_SHIFT", "0.5"))
    nets = {f: (None, T, sh) for f in ("h", "v")}
    print(f"  exemplar targets: T={T}, schedule shift {sh}, no trained prior in the loop")

# how much of a photograph this fruit covers -- the sampler's window is set to match
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
conf = open(f"{OBJDIR}/{OBJ}.conf").read()
photos, fr = {}, []
for fam, key in (("h", "REF_H="), ("v", "REF_V=")):
    d = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)][0]
    ims, mks = [], []
    for q in sorted(refsel.photos_in(f"{FN}/{d}")):
        a = np.asarray(Image.open(q).convert("RGB"), np.float32) / 255.
        t = torch.as_tensor(a).permute(2, 0, 1).to(dev)
        t = torch.nn.functional.interpolate(t[None], (tpdm.P, tpdm.P), mode="area")[0]
        ims.append(t); mks.append(t.min(0).values < 0.98)
        if fam == "h":
            fr.append(float((a.min(2) < 0.98).mean()))
    photos[fam] = (ims, mks)
    print(f"  {fam}: {len(ims)} photographs to draw from")
frac = float(np.mean(fr))

_v0, _m0 = tpdm.dense(st)
_sl = tpdm.Slicer(st, dev)
span, got = _sl.window(frac, st, C, _m0, dev)
print(f"  photographs cover {frac:.3f} of their frame; window {span:.3f} of the object's extent "
      f"gives {got:.3f}")
tpdm.WINDOW = span

sf = f"{W}/tpdm_state_{OBJ}.pt"
state = torch.load(sf, map_location=dev, weights_only=False) \
    if (RESUME and os.path.exists(sf)) else None
if state is not None:
    print(f"  resuming at t={state['t']}, {len(state['hist'])} residuals recorded")
before = state["hist"][-1] if state and state["hist"] else None

vol, mask, state = tpdm.sample(st, C, nets, device=dev, steps=STEPS or T, photos=photos,
                               state=state, chunk=CHUNK)
after = state["hist"][-1]
if before is None:
    print(f"  first chunk: residual h {after[1]:.4f}  v {after[2]:.4f}")
else:
    dh, dv = after[1] - before[1], after[2] - before[2]
    verdict = "both still falling" if (dh < -1e-4 and dv < -1e-4) else \
        ("FLAT -- more steps buy nothing" if (dh > -1e-4 and dv > -1e-4) else
         "one family improving at the other's expense")
    print(f"  residual h {before[1]:.4f} -> {after[1]:.4f}   v {before[2]:.4f} -> {after[2]:.4f}"
          f"   {verdict}")
torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state.items()}, sf)
frames = state["frames"]

if frames:
    gif = []
    for t, a in sorted(frames, key=lambda z: -z[0]):
        im = Image.fromarray(a).resize((a.shape[1] * 2, a.shape[0] * 2), Image.NEAREST)
        gif.append(im)
    gif[0].save(f"{W}/tpdm_{OBJ}.gif", save_all=True, append_images=gif[1:],
                duration=80, loop=0)
    print(f"GIF tpdm_{OBJ}.gif  {len(gif)} frames, t from {frames[0][0]} down to {frames[-1][0]}"
          f"  (left transverse, right longitudinal)")

# slices of the result, in both directions, at the lattice's own resolution
sl = _sl
rows = []
for fam in ("h", "v"):
    pl = tpdm.planes_of(st, C, fam, dev)
    pick = [pl[int(x * (len(pl) - 1))] for x in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    imgs = []
    for n, d, pr in pick:
        flat, ok, _, _ = sl.index(n, d, pr)
        a = sl.gather(vol, flat, ok) + (1 - ok.float()) * 0.6
        imgs.append((a.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    rows.append(np.concatenate(imgs, 1))
sheet = np.concatenate(rows, 0)
Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.NEAREST) \
    .save(f"{W}/tpdm_{OBJ}.jpg", quality=92)
print(f"SHEET tpdm_{OBJ}.jpg  (top row transverse slices, bottom row longitudinal)")

d = vol[:, mask]
print(f"volume: mean RGB {d.mean(1).cpu().numpy().round(3)}, spread {d.std(1).cpu().numpy().round(3)}")
torch.save({"vol": vol.cpu(), "mask": mask.cpu()}, f"{W}/tpdm_{OBJ}.pt")
