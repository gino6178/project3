"""Every object, with the depth blend on: is the banding gone, and what did it cost?

    python ovallmeasure.py OBJ [OBJ ...]

The switch depths are not the same for every object -- they are at j/N for N references, and N runs
from 1 (the doughnut's longitudinal family, where there is nothing to blend and the rule falls back
by itself) to 10 (the watermelon's transverse). So each object's are read from its own conf rather
than assumed, and an object with one reference is reported as having no switches instead of being
given the orange's.

The jump is in units of the profile's own median jump, so objects of different contrast are
comparable: 1.0 means the depth where the photograph changes looks like any other depth, which is
what the blend is for.
"""
import glob
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = "/workspace/ovoxel_native"
SKIP = ("_depth", "_mask", "_alpha", "_normal")


def n_refs(obj):
    """How many photographs the transverse family actually has, from the object's own conf."""
    conf = os.path.join(OBJDIR, f"{obj}.conf")
    m = re.search(r"^REF_H=(\S+)", open(conf).read(), re.M)
    d = os.path.join(FN, m.group(1))
    return sum(1 for f in sorted(glob.glob(os.path.join(d, "*")))
               if f.lower().endswith((".png", ".jpg", ".jpeg"))
               and not any(t in os.path.basename(f) for t in SKIP))


def axial(path, n=512):
    L = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255.
    ys, xs = np.where(L < 0.97)
    if len(ys) < 500:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    w = x1 - x0
    c = L[y0 + (y1 - y0) // 12:y1 - (y1 - y0) // 12, x0 + w // 3:x1 - w // 3]
    if min(c.shape) < 16:
        return None
    p = c.mean(1)
    p = p - ndimage.gaussian_filter1d(p, len(p) / 5.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)


def detail(path):
    L = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255.
    fg = ndimage.binary_erosion(L < 0.97, np.ones((9, 9)))
    if fg.sum() < 500:
        return np.nan
    gy, gx = np.gradient(L)
    return float(np.hypot(gy, gx)[fg].mean() / max(L[fg].std(), 1e-6))


print(f"  {'object':18s} {'refs':>4s}  {'jump at switches':>16s}  {'probe 1':>8s} "
      f"{'probe last':>10s}  {'detail rh':>9s} {'rv':>6s}")
for obj in sys.argv[1:]:
    d = os.path.join(W, f"ov_{obj}")
    rv = sorted(glob.glob(f"{d}/eval_final/rv*.png"))
    rh = sorted(glob.glob(f"{d}/eval_final/rh*.png"))
    if not rv:
        print(f"  {obj:18s} no renders"); continue
    N = n_refs(obj)
    ps = [p for p in (axial(f) for f in rv[:6]) if p is not None]
    if not ps:
        print(f"  {obj:18s} no usable sections"); continue
    m = np.mean(ps, 0)
    jd = np.abs(np.diff(ndimage.gaussian_filter1d(m, 2.0)))
    typ = float(np.median(jd))
    if N < 2:
        jump = "no switches"
    else:
        r = max(2, int(len(m) / (2 * N)))
        at = [float(jd[max(0, int(s * len(m)) - r):min(len(m) - 1, int(s * len(m)) + r)].max())
              / typ for s in (j / N for j in range(1, N))]
        jump = f"{np.mean(at):.2f}"
    log = os.path.join(W, f"ovall_{obj}.log")
    pr = re.findall(r"probe curve:(.*)", open(log).read())[-1] if os.path.isfile(log) else ""
    vals = re.findall(r"\d+:([\d.]+)", pr)
    p1, pl = (vals[1], vals[-1]) if len(vals) > 1 else ("-", "-")
    print(f"  {obj:18s} {N:>4d}  {jump:>16s}  {p1:>8s} {pl:>10s}  "
          f"{np.nanmean([detail(f) for f in rh[:6]]):>9.4f} "
          f"{np.nanmean([detail(f) for f in rv[:6]]):>6.4f}")
