"""Which photographs each plane's target is actually built from.

Not by recomputing the formula -- that tests a copy of it -- but by handing `_depth_pick` a probe
in place of the real photograph loader and recording which indices it asks for and at what weight.
The transverse family is a stack from one side of the object to the other, so the pair must move
forward and the last plane must land on the last photograph.
"""
import os
import sys

import numpy as np
from PIL import Image

FN = "/workspace/rebuild/worktree"
import refsel

spec, n = sys.argv[1], int(sys.argv[2])
d = os.path.join(FN, spec)
L = len(sorted(refsel.photos_in(d)))
print(f"  {spec}: {L} photographs, {n} planes, REF_TRANS_BLEND="
      f"{os.environ.get('REF_TRANS_BLEND', '1')}")

asked = []
_orig = refsel._blend_images
refsel._blend_images = lambda a, b, w: (asked.append((a, b, w))
                                        or Image.new("RGB", (8, 8), (255, 255, 255)))
back = 0
for idx in range(n):
    asked.clear()
    refsel._PHOTOS.clear()
    refsel._depth_pick(d, idx, n, lambda j: j)
    if not asked:
        print(f"    plane {idx:3d}  single photograph (no blend)"); continue
    j0, j1, w = asked[0]
    bad = j1 < j0
    back += bad
    print(f"    plane {idx:3d}  {j0} -> {j1} at w {w:.3f}"
          + ("   <-- runs backwards" if bad else ""))
refsel._blend_images = _orig
print(f"  {back} of {n} planes run backwards")
