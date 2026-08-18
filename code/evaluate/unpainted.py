"""How much of a cut face a representation leaves unpainted, through each method's own renderer.

`slab_compare.py` answers a different question and was reported as if it answered this one. It
paints every primitive at its own footprint, which is the honest way to ask what a representation
*covers*, but at sigma = 1e-7 the radius rounds to zero and the primitive becomes one pixel, so the
number it returns is a statement about sampling density at one image size. A real rasteriser never
draws a Gaussian that small: it clamps the screen-space footprint and applies an antialiasing
filter, which is precisely why sub-pixel Gaussians reach the screen as a solid surface. Reporting
the footprint number against methods that ship a rasteriser compares them to something they do not
do.

So this measures the same quantity on the images each method's own renderer produced. A cut face is
whatever is inside the silhouette; unpainted is a pixel inside it that came back background. The
silhouette is the painted region's own closing and convex fill, so the measure is holes *in the
section*, not the background around it.

Resolution is the parameter that matters and it is swept rather than chosen: a coverage failure
that is real gets worse as pixels get smaller, and one that is an artefact of sampling does not.

    python method/common/eval/unpainted.py "name=dir" ...
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np

sys.path += [_FN_ROOT]

BG = float(_os.environ.get("BG_THRESH", "0.96"))


def unpainted(path):
    import cv2
    a = cv2.imread(path).astype(np.float32) / 255.0
    if a is None:
        return None
    painted = a.min(2) < BG
    if not painted.any():
        return None
    # The section is what the object covers, holes included, so it is the painted region closed
    # and filled. A convex hull would swallow a genuine concavity; a closing at a fixed fraction
    # of the image bridges the gaps a coverage failure leaves without inventing area.
    k = max(3, int(round(a.shape[0] * 0.02)) | 1)
    fill = cv2.morphologyEx(painted.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    sil = np.zeros_like(fill)
    cv2.drawContours(sil, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    inside = sil > 0
    return 100.0 * float((inside & ~painted).sum()) / max(int(inside.sum()), 1)


def main(*specs):
    print(f"  background above {BG:.2f} luminance; the section is the closed painted region\n")
    print(f"  {'render set':<44} {'n':>4}  {'unpainted %':>12}  {'px':>6}")
    for spec in specs:
        name, d = spec.split("=", 1)
        ps = sorted(glob.glob(_os.path.join(d, "rh*_init_0.png")))
        vals = [v for v in (unpainted(p) for p in ps) if v is not None]
        if not vals:
            print(f"  {name:<44} {'-':>4}  {'nothing read':>12}")
            continue
        import cv2
        px = cv2.imread(ps[0]).shape[0]
        print(f"  {name:<44} {len(vals):>4}  {np.mean(vals):>11.2f}%  {px:>6}"
              f"   (min {min(vals):.2f}, max {max(vals):.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
