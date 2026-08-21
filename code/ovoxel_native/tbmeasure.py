"""The two things the sheet shows, as numbers.

`steppos` reported the five largest jumps, which exist in any profile, so it can say a step moved
but not that it shrank. This asks the sharper question: at the five depths where the transverse
family changes photograph, and nowhere else, how big is the jump in each arm?

The second half is there because the sheet showed the transverse row changing more than the
longitudinal one, which is the opposite of what was predicted -- a transverse cut is perpendicular
to the stacking axis and crosses no switch. What it does cross is the framing: `_canonical` puts
every photograph's section at the same size in the same place, so the radial walls of neighbouring
planes agree instead of averaging out. Gradient energy inside the face measures whether they
survived.
"""
import glob
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

NPHOTO = 6
SWITCH = [j / NPHOTO for j in range(1, NPHOTO)]


def axial(path, n=512):
    L = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255.
    ys, xs = np.where(L < 0.97)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    w = x1 - x0
    c = L[y0 + (y1 - y0) // 12:y1 - (y1 - y0) // 12, x0 + w // 3:x1 - w // 3]
    p = c.mean(1)
    p = p - ndimage.gaussian_filter1d(p, len(p) / 5.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)


def detail(path):
    """Mean gradient magnitude inside the face, normalised by its own contrast.

    Normalised because an arm that is uniformly darker has larger gradients everywhere without
    resolving anything; what is wanted is how much structure there is per unit of range.
    """
    L = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255.
    fg = ndimage.binary_erosion(L < 0.97, np.ones((9, 9)))
    if fg.sum() < 500:
        return np.nan
    gy, gx = np.gradient(L)
    g = np.hypot(gy, gx)[fg]
    return float(g.mean() / max(L[fg].std(), 1e-6))


for arm in sys.argv[1:]:
    rv = sorted(glob.glob(f"{arm}/eval_final/rv*.png")) or sorted(glob.glob(f"{arm}/rv*.png"))
    rh = sorted(glob.glob(f"{arm}/eval_final/rh*.png")) or sorted(glob.glob(f"{arm}/rh*.png"))
    if not rv:
        print(f"  {arm}: nothing"); continue
    m = np.mean([axial(f) for f in rv[:6]], 0)
    d = np.abs(np.diff(ndimage.gaussian_filter1d(m, 2.0)))
    n = len(m)
    # the largest jump within half a switch spacing of each switch, against the profile's own
    # typical jump, so the number is "how many times an ordinary step" and not a raw luminance
    typical = float(np.median(d))
    at = []
    for s in SWITCH:
        i = int(s * n)
        r = max(2, int(n / (2 * NPHOTO)))
        at.append(float(d[max(0, i - r):min(n - 1, i + r)].max()) / typical)
    print(f"  {arm:16s} jump at the five switches, in units of the profile's median jump:")
    print(f"  {'':16s}   {'  '.join(f'{v:5.1f}' for v in at)}      mean {np.mean(at):5.2f}")
    print(f"  {'':16s} transverse detail {np.mean([detail(f) for f in rh[:6]]):.4f}   "
          f"longitudinal detail {np.mean([detail(f) for f in rv[:6]]):.4f}")
