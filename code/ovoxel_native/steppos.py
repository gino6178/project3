"""Where the steps sit, against where the transverse family changes photograph.

The transverse family's target is chosen by `k = (idx * len(files) // n) % len(files)`, so with
`len(files)` photographs the photograph changes at t = 1/len, 2/len, ... down the stack. If the
plateaus in a longitudinal cut are that switch, their edges sit at those fractions and nowhere
else. The largest downward and upward jumps in the mean axial profile are the edges.
"""
import glob
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

NPHOTO = 6            # orange, data_finetune_images/orange/horizontal


def profile(path, n=512):
    L = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255.
    ys, xs = np.where(L < 0.97)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    w = x1 - x0
    c = L[y0 + (y1 - y0) // 12:y1 - (y1 - y0) // 12, x0 + w // 3:x1 - w // 3]
    p = c.mean(1)
    p = p - ndimage.gaussian_filter1d(p, len(p) / 5.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)


for arm in sys.argv[1:]:
    fs = sorted(glob.glob(f"{arm}/eval_final/rv*.png")) or sorted(glob.glob(f"{arm}/rv*.png"))
    ps = [profile(f) for f in fs[:6]]
    if not ps:
        print(f"  {arm}: nothing"); continue
    m = np.mean(ps, 0)
    d = np.abs(np.diff(ndimage.gaussian_filter1d(m, 2.0)))
    # the five largest jumps, kept apart so one edge is not reported five times
    order, picked = np.argsort(d)[::-1], []
    for i in order:
        if all(abs(i - j) > len(m) / 30 for j in picked):
            picked.append(i)
        if len(picked) == 5:
            break
    pos = sorted(i / len(m) for i in picked)
    switches = [j / NPHOTO for j in range(1, NPHOTO)]
    near = [min(abs(p - s) for s in switches) for p in pos]
    print(f"  {arm:32s} steps at {[f'{p:.3f}' for p in pos]}")
    print(f"  {'':32s} nearest switch off by {[f'{n:.3f}' for n in near]}"
          f"   (a switch every {1 / NPHOTO:.3f}; chance distance {0.25 / NPHOTO:.3f})")
