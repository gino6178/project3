"""Do the seeds sit at depths, or are they smeared along the whole sweep?

    python seeds.py DIR_A=name DIR_B=name ...

Each directory holds one dense sweep of transverse sections. Seeds are the dark pixels inside
the flesh, so for every section this measures how much of it is seed, and for every pair of
sections how much their seed masks overlap. A volume built from one photograph can only repeat
that photograph's seeds down the whole axis, and then the masks of two distant sections agree; a
volume that has seeds at particular depths does not.

The overlap is intersection over union between the two masks, which is the quantity a claim
about "the same seeds everywhere" is actually about.
"""
import glob
import os
import sys

import cv2
import numpy as np


def masks(d, thr=0.42, lo=0.10, hi=0.86):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "d*.png"))):
        a = cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255.
        fg = a.min(2) < 0.94
        if fg.sum() < 500:
            continue
        ys, xs = np.where(fg)
        cy, cx = ys.mean(), xs.mean()
        r = np.hypot(ys - cy, xs - cx)
        R = np.percentile(r, 98)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        rr = np.hypot(yy - cy, xx - cx)
        band = fg & (rr > lo * R) & (rr < hi * R)     # the flesh, not the rind
        out.append(((a.mean(2) < thr) & band, band))
    return out


def main(specs):
    for spec in specs:
        d, _, name = spec.partition("=")
        m = masks(d)
        if len(m) < 8:
            print(f"  {name}: only {len(m)} sections"); continue
        frac = np.array([s.sum() / max(b.sum(), 1) for s, b in m])
        n = len(m)
        far = []
        for i in range(0, n - 20, 4):
            a, b = m[i][0], m[i + 20][0]
            u = (a | b).sum()
            far.append((a & b).sum() / u if u else 1.0)
        near = []
        for i in range(0, n - 1, 4):
            a, b = m[i][0], m[i + 1][0]
            u = (a | b).sum()
            near.append((a & b).sum() / u if u else 1.0)
        print(f"  {name:<10} seed fraction mean {frac.mean()*100:5.2f}%  sd {frac.std()*100:4.2f}  "
              f"| IoU neighbours {np.mean(near):.3f}  20 apart {np.mean(far):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
