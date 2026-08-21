"""Every object, the block rule against the blend, paired.

    python ovpair.py OBJ [OBJ ...]

`ovallmeasure` reported the blend arms alone, which says what each model scores and not what the
change was worth. This reads `ov0_<obj>` and `ov_<obj>` -- identical but for REF_TRANS_BLEND -- and
prints them side by side.

The jump is measured at the depths where the BLOCK rule changed photograph, which is the only place
the two arms can differ for this reason, and in units of each arm's own median jump so contrast
does not enter. An object with two references has one such depth and it is the middle of the
object, where an apple or a cake changes structure anyway; the pairing is what separates that from
an artefact, since both arms see the same object and only one sees the discontinuity.
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
    m = re.search(r"^REF_H=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
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


def jump(tag, obj, N):
    fs = sorted(glob.glob(f"{W}/{tag}_{obj}/eval_final/rv*.png"))[:6]
    ps = [p for p in (axial(f) for f in fs) if p is not None]
    if not ps or N < 2:
        return None
    m = np.mean(ps, 0)
    d = np.abs(np.diff(ndimage.gaussian_filter1d(m, 2.0)))
    r = max(2, int(len(m) / (2 * N)))
    return float(np.mean([d[max(0, int(s * len(m)) - r):min(len(m) - 1, int(s * len(m)) + r)].max()
                          for s in (j / N for j in range(1, N))]) / float(np.median(d)))


def probe(tag, obj):
    lg = f"{W}/ovall0_{obj}.log" if tag == "ov0" else f"{W}/ovall_{obj}.log"
    if not os.path.isfile(lg):
        return None
    v = re.findall(r"\d+:([\d.]+)", (re.findall(r"probe curve:(.*)", open(lg).read()) or [""])[-1])
    return float(v[-1]) if v else None


def fmt(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


print(f"  {'object':18s} {'refs':>4s}  {'jump block':>10s} {'jump blend':>10s}  "
      f"{'probe block':>11s} {'probe blend':>11s}")
for obj in sys.argv[1:]:
    N = n_refs(obj)
    j0, j1 = jump("ov0", obj, N), jump("ov", obj, N)
    p0, p1 = probe("ov0", obj), probe("ov", obj)
    mark = ""
    if j0 and j1:
        mark += "  banding down" if j1 < j0 else "  banding up"
    if p0 and p1:
        mark += ", probe down" if p1 < p0 else ", probe up"
    print(f"  {obj:18s} {N:>4d}  {fmt(j0, 2):>10s} {fmt(j1, 2):>10s}  "
          f"{fmt(p0, 5):>11s} {fmt(p1, 5):>11s}{mark}")
