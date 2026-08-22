"""Compare denoisers trained under different schedules, on one loss they can all be judged by.

The training loss is not comparable between two schedules: they draw their noise levels from
different distributions, so the same number means different things. What is comparable is the error
at a FIXED noise level. This script fixes the sections, the noise and a list of levels a_bar, and
asks every checkpoint the same question at each of them -- each one given the timestep that its own
schedule assigns to that level, which is the label it was trained to associate with that much noise.

The high-noise columns are the interesting ones. That is where the layout has to be invented rather
than cleaned up, and the whole point of shifting the schedule was to spend more of training there.
"""
import os, sys, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refsel, patchdiff
from PIL import Image

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
FAM = os.environ.get("PD_FAM", "h")
LEVELS = [float(x) for x in os.environ.get("PD_LEVELS", "0.5,0.1,0.02,0.005,0.001").split(",")]
N = int(os.environ.get("PD_CMP_N", "64"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

key = {"h": "REF_H=", "v": "REF_V="}[FAM]
conf = open(f"{OBJDIR}/{OBJ}.conf").read()
d = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)][0]
imgs, masks = [], []
for p in sorted(refsel.photos_in(f"{FN}/{d}")):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    t = torch.as_tensor(a).permute(2, 0, 1).to(dev)
    imgs.append(t); masks.append((t.min(0).values < 0.98))

# one fixed batch and one fixed noise, shared by every arm
np.random.seed(7); torch.manual_seed(7)
x0, m = patchdiff.crops_from(imgs, masks, N)
x0, m = x0.to(dev) * 2 - 1, m.to(dev)
if patchdiff.MASK:
    x0 = m[:, :1] * x0 + (1 - m[:, :1]) * patchdiff.BG
eps = torch.randn(x0.shape, device=dev, generator=torch.Generator(dev).manual_seed(7))
w = m[:, :1].expand_as(eps)

cks = sorted(glob.glob(f"{W}/pd_{OBJ}_{FAM}*.pt"))
print(f"{OBJ} {FAM}: same {N} sections, same noise, {len(cks)} checkpoints\n")
print(f"  {'checkpoint':<26}{'steps':>6}" + "".join(f"{f'a={a:g}':>10}" for a in LEVELS))
for ck in cks:
    st = torch.load(ck, map_location=dev, weights_only=False)
    net = patchdiff.Denoiser(dim=st.get("dim", patchdiff.DIM)).to(dev)
    net.load_state_dict(st["sd"]); net.eval()
    # the schedule this arm was trained under, recovered from its tag
    tag = os.path.basename(ck).rsplit("_", 1)[-1][:-3]
    sh = float(tag[1:]) if tag.startswith("s") and tag[1:].replace(".", "").isdigit() else 1.0
    ab = patchdiff.schedule(st.get("T", patchdiff.T), dev, shift=sh)
    row = []
    for lv in LEVELS:
        t = int((ab - lv).abs().argmin())          # the timestep this arm calls "this much noise"
        a = ab[t]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        if patchdiff.MASK:
            xt = m[:, :1] * xt + (1 - m[:, :1]) * x0
        with torch.no_grad():
            pred = net(xt, torch.full((N,), t / st.get("T", patchdiff.T), device=dev), m)
        row.append(float(((pred - eps) ** 2 * w).sum() / w.sum()))
    print(f"  {os.path.basename(ck):<26}{st.get('done', 0):>6}" + "".join(f"{r:>10.4f}" for r in row))
print("\nLower is better everywhere, but the right-hand columns are the ones that decide whether a"
      "\nmodel can start from noise and produce a section rather than a texture.")
