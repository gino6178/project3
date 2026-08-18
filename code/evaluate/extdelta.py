"""Distance between two six-view sheets, away from the silhouette, and stray bright specks."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

A = np.asarray(Image.open(sys.argv[1]).convert("RGB"), np.float32) / 255
B = np.asarray(Image.open(sys.argv[2]).convert("RGB"), np.float32) / 255
body = (np.abs(A - 1).max(2) > 0.06) & (np.abs(B - 1).max(2) > 0.06)
inner = ndimage.binary_erosion(body, np.ones((9, 9), bool))
d = np.linalg.norm(A - B, axis=2)[inner]
S = A.shape[0] // 2
specks = 0
for k in range(6):
    r, c = divmod(k, 3)
    sl = (slice(r * S, (r + 1) * S), slice(c * S, (c + 1) * S))
    b, inn = B[sl], ndimage.binary_erosion(body[sl], np.ones((5, 5), bool))
    if inn.sum() < 100:
        continue
    lum = b.mean(2)
    lab, _ = ndimage.label(inn & (lum - ndimage.uniform_filter(lum, 21) > 0.10))
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    for j in np.nonzero(sz >= 4)[0]:
        m = lab == j
        if (b[m].mean(0) - A[sl][m].mean(0)).mean() > 0.08:
            specks += 1
print(f"  {int(inner.sum()):,} px   mean {d.mean():.4f}   p95 {np.quantile(d,0.95):.4f}   "
      f"over 0.15 {100*(d>0.15).mean():.3f}%   specks {specks}")
