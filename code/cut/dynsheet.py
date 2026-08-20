"""A contact sheet from the dynamic-cut frames, at the moments the topology changes.

    python dynsheet.py FRAME_DIR OUT.png
"""
import glob
import os
import sys

import cv2
import numpy as np


def main(d, out, picks=(0, 25, 60, 90, 110, 150, 205), w=300):
    fs = sorted(glob.glob(os.path.join(d, "d_*.png")))
    print(f"  {len(fs)} frames, {cv2.imread(fs[0]).shape}")
    picks = [p for p in picks if p < len(fs)]
    row = np.hstack([cv2.resize(cv2.imread(fs[i]), (w, w)) for i in picks])
    cv2.imwrite(out, row)
    print(f"  -> {out}  frames {picks}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
