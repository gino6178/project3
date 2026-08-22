"""Does the forward process actually destroy the section's layout?

A diffusion model only learns to synthesise what the noise takes away. Noise is added per pixel and
independently, but an image is smooth, so the same schedule destroys far less at 128 pixels than at
32 -- and we moved from 32 to 128 without touching the schedule. If the layout is still readable at
the last timestep, the model is never asked to invent one during training, and at sampling time,
starting from pure noise, it has none to read: exactly the samples we are getting.

Readability is measured per scale. Blurring at scale s attenuates white noise far more than it
attenuates the image, so the correlation between blur(x_t) and blur(x_0) at that scale is what a
denoiser could recover there. The membranes live around 8 to 16 pixels of a 128-pixel section.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refsel, patchdiff
from PIL import Image

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
OBJ = os.environ.get("OBJ", "orange_sp")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def blur(x, s):
    if s <= 1:
        return x
    k = min(int(s) * 4 + 1, (x.shape[-1] // 2) * 2 - 1)
    g = torch.exp(-((torch.arange(k, device=x.device) - k // 2) ** 2) / (2. * s * s))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"), g.expand(c, 1, 1, k), groups=c)
    return F.conv2d(F.pad(x, (0, 0) + (k // 2,) * 2, mode="reflect"),
                    g.transpose(-1, -2).expand(c, 1, k, 1), groups=c)


def corr(a, b, m):
    a, b = a[m.expand_as(a)], b[m.expand_as(b)]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


conf = open(f"{OBJDIR}/{OBJ}.conf").read()
d = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith("REF_H=")][0]
files = sorted(refsel.photos_in(f"{FN}/{d}"))
imgs, masks = [], []
for p in files:
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    t = torch.as_tensor(a).permute(2, 0, 1).to(dev)
    imgs.append(t); masks.append((t.min(0).values < 0.98))

for P in (32, 128):
    z = torch.stack([patchdiff.fit_shell(i, m, P) for i, m in zip(imgs, masks)])
    x0, mk = z[:, :3] * 2 - 1, (z[:, 3:] > 0.5)
    sh = float(os.environ.get("PD_SNR_SHIFT", "0")) or (32.0 / P)
    ab = patchdiff.schedule(patchdiff.T, dev, shift=sh)
    print(f"\n{OBJ} at {P}px, schedule shifted by {sh:.3f} -- correlation with the clean section")
    print(f"  {'t/T':>5}  {'a_bar':>7} " + "".join(f"{f's={s}px':>9}" for s in (1, 2, 4, 8, 16)))
    for frac in (0.5, 0.8, 0.95, 1.0):
        t = min(int(frac * patchdiff.T), patchdiff.T - 1)
        a = ab[t]
        g = torch.Generator(dev).manual_seed(0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, device=dev, generator=g)
        row = [corr(blur(xt, s), blur(x0, s), mk) for s in (1, 2, 4, 8, 16)]
        print(f"  {frac:>5.2f}  {float(a):>7.4f} " + "".join(f"{c:>9.3f}" for c in row))
print("\nA correlation still well above zero at the last timestep means the layout survives the "
      "forward process,\nso the model is never trained to create one -- and cannot at sampling time.")
