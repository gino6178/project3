"""Rank a reference set by how much of each photograph is seed.

    python rankseeds.py REF_DIR

Seeds are the dark pixels inside the fruit's own silhouette, so the fraction is of the fruit
and not of the frame. A subset chosen this way gives the depth-assignment figure a feature that
moves, rather than an even field of red where any two photographs look alike.
"""
import glob
import os
import sys

import cv2
import numpy as np


def main(ref):
    fs = [f for f in sorted(glob.glob(os.path.join(ref, "*.png")))
          if not f.endswith("_depth.png")]
    rows = []
    for f in fs:
        a = cv2.imread(f)[:, :, ::-1].astype(np.float32) / 255.
        m = a.min(2) < 0.90
        if m.sum() < 500:
            continue
        rows.append((float(((a.mean(2) < 0.34) & m).sum() / m.sum()), int(m.sum()),
                     os.path.basename(f)))
    rows.sort(reverse=True)
    for d, n, f in rows:
        print(f"  {f:24s} seeds {d*100:5.2f}%   area {n}")
    print("  top three:", " ".join(r[2] for r in rows[:3]))


if __name__ == "__main__":
    main(sys.argv[1])
