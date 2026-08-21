"""How much does a cut face change when the plane is turned?

    python cutvary.py CUTDIR

Gino's observation about the loaf is that every direction gives the same face. If that is so, the
interior has no directional structure for the two families to disagree about, and calling one cut
transverse and another longitudinal describes nothing -- which is the same fact as its two
reference families being the same files, seen from the other side.

The cut grid is already rendered: 16 azimuths by 5 elevations of the plane's normal, each drawn
face-on. Every pair of directions is compared inside the object, and the answer is reported against
the object's own contrast so a pale interior is not mistaken for a uniform one.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

d = sys.argv[1]
meta = json.load(open(os.path.join(d, "cut_meta.json")))
print(f"  {'object':16s} {'spread across directions':>24} {'own contrast':>13} {'ratio':>7}")
for obj, m in meta.items():
    a = np.asarray(Image.open(os.path.join(d, f"cut_{obj}.jpg")).convert("RGB"), np.float32) / 255.
    naz, els, res = m["naz"], m["els"], m["res"]
    tiles = []
    for ei in range(len(els)):
        for ai in range(naz):
            t = a[ei * res:(ei + 1) * res, ai * res:(ai + 1) * res]
            fg = t.min(2) < 0.97
            if fg.sum() > 0.02 * res * res:
                tiles.append((t, fg))
    if len(tiles) < 4:
        print(f"  {obj:16s} too few usable directions"); continue
    # pairwise difference where both faces have object, so the silhouette's own change does not
    # masquerade as a change of content
    ds = []
    for i in range(0, len(tiles), 3):
        for j in range(i + 3, len(tiles), 7):
            m2 = tiles[i][1] & tiles[j][1]
            if m2.sum() > 0.02 * res * res:
                ds.append(float(np.abs(tiles[i][0] - tiles[j][0])[m2].mean()))
    spread = float(np.mean(ds))
    contrast = float(np.mean([t[f].std() for t, f in tiles]))
    print(f"  {obj:16s} {spread:24.4f} {contrast:13.4f} {spread / max(contrast, 1e-6):7.2f}")
print("\n  ratio near 0 means every direction cuts the same face; the higher it is, the more the"
      "\n  interior actually depends on which way it is cut")
