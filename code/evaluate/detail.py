"""How much high-frequency content a set of renders carries, by the trainer's own definition.

`train_voxel.log_ref` reports a detail figure for every target it writes: the mean absolute
Laplacian of luminance over the foreground, times a thousand. That is the statistic the band term
of equation (9) exists to protect, and the ablation that removes the term is asking whether the
renders lose it. So the same statistic is computed here, on renders rather than on targets, and on
the reference photographs as the level to read the arms against -- a number that is only
comparable to itself needs something to be compared with.

Foreground is the same test the trainer uses, |colour - white| > 0.06, so a white background
cannot inflate or deflate the mean.

    python code/evaluate/detail.py refs=REF_DIR name=RENDER_DIR ...

Renders are matched as rh*_init_0.png and photographs as any image, which is what the rest of the
harness does.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np

sys.path += [_FN_ROOT]


def _paths(d):
    out = [p for p in sorted(glob.glob(_os.path.join(d, "rh*_init_0.png")))]
    if out:
        return out
    return [p for p in sorted(glob.glob(_os.path.join(d, "*")))
            if _os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg")
            and not _os.path.splitext(_os.path.basename(p))[0].endswith(
                ("_depth", "_mask", "_alpha", "_ref"))]


def detail(path):
    import cv2
    a = cv2.imread(path)
    if a is None:
        return None
    a = a[:, :, ::-1].astype(np.float32) / 255.0
    fg = np.abs(a - 1).max(2) > 0.06
    if fg.sum() < 500:
        return None
    lum = a.mean(2)
    return float(np.abs(cv2.Laplacian(lum, cv2.CV_32F))[fg].mean()) * 1000


def main(*specs):
    print(f"  {'set':<26} {'images':>7} {'detail, e-3':>13} {'sd':>8}")
    for spec in specs:
        name, _, d = spec.partition("=")
        v = [x for x in (detail(p) for p in _paths(d)) if x is not None]
        if not v:
            print(f"  {name:<26} {'nothing read':>21}")
            continue
        v = np.array(v)
        print(f"  {name:<26} {len(v):>7} {v.mean():>13.2f} {v.std():>8.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
