"""Continuous slicing consistency: does the interior hold still as the plane moves?

Every appearance number in this paper scores one cut against a photograph, which says whether a
section looks right and says nothing about whether the next section along is the same object. A
representation can score well on both and still flicker between them, and flicker is what a viewer
sees when they drag a cutting plane rather than place one.

So this steps a plane through the object in small increments and measures how much consecutive
sections differ. Two numbers, because they fail differently:

    mean 1-SSIM      how much a section changes per step. A large value can be honest -- the
                     object really does change -- so it is not by itself a fault.
    jerk             the standard deviation of the step-to-step change. A representation that
                     varies smoothly has a small one; a representation that jumps as the plane
                     crosses a cell boundary or a primitive's support does not. This is the
                     quantity flicker actually is.

Both are computed on the same plane sequence for every arm, and the step is a fraction of the cell
size so that a lattice is asked about its own discretisation rather than about something coarser.

    python method/common/eval/slicing_consistency.py MODEL.ply CFG DEMO OUT_DIR [n] [size]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

N_STEPS = int(_os.environ.get("CSC_STEPS", "24"))
LO, HI = (float(v) for v in _os.environ.get("CSC_BAND", "0.44,0.56").split(","))


def _ssim(a, b):
    from skimage.metrics import structural_similarity as ss
    return float(ss(a, b, channel_axis=2, data_range=1.0))


def curve(paths):
    import cv2
    ims = [cv2.imread(p).astype(np.float32) / 255.0 for p in paths]
    d = np.array([1.0 - _ssim(ims[i], ims[i + 1]) for i in range(len(ims) - 1)])
    return d


def main(model, cfg, demo, out_dir, n=N_STEPS, size=512):
    import random_cuts as rc
    n, size = int(n), int(size)
    _os.makedirs(out_dir, exist_ok=True)
    # a fixed azimuth and a depth walked in equal steps, so consecutive frames differ only by the
    # plane's position. random_cuts chooses depths at random, so the sequence is built here.
    import types
    depths = np.linspace(LO, HI, n)
    saved = rc.main
    frames = []

    def one(depth, i):
        # a degenerate band falls back to random depths, so open it by a hair
        _os.environ["HELDOUT_BAND"] = f"{depth:.6f},{depth + 1e-5:.6f}"
        d = _os.path.join(out_dir, f"s{i:03d}")
        rc.main(model, cfg, demo, d, n=2, size=size)
        got = sorted(glob.glob(_os.path.join(d, "rh*_init_0.png")))
        return got[0] if got else None

    for i, dp in enumerate(depths):
        p = one(float(dp), i)
        if p:
            frames.append(p)
    if len(frames) < 3:
        print("  too few frames"); return
    d = curve(frames)
    print(f"  {len(frames)} sections over depths {LO:g} to {HI:g}")
    print(f"  mean 1-SSIM per step  {d.mean():.4f}")
    print(f"  jerk (sd of the step) {d.std():.4f}")
    print(f"  worst single step     {d.max():.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:5], *(sys.argv[5:7]))
