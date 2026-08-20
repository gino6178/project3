"""The dynamic-cut frames as one animation, cropped to what the object actually occupies.

    python dyngif.py FRAME_DIR OUT.gif [STRIDE]

The renderer frames a domain rather than the object, so the pieces occupy a fraction of it and
move outward as they separate. One box over every frame keeps the scale fixed -- a piece must
not appear to grow because the crop followed it -- while removing the margin no frame uses.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image


def main(d, out, stride=2, pad=12, w=420):
    fs = sorted(glob.glob(os.path.join(d, "d_*.png")))[::int(stride)]
    lo = np.array([10 ** 9, 10 ** 9]); hi = np.array([-1, -1])
    for p in fs:
        a = np.asarray(Image.open(p).convert("RGB"))
        m = a.min(2) < 238
        ys, xs = np.where(m)
        if not len(ys):
            continue
        lo = np.minimum(lo, [ys.min(), xs.min()]); hi = np.maximum(hi, [ys.max(), xs.max()])
    y0, x0 = max(int(lo[0]) - pad, 0), max(int(lo[1]) - pad, 0)
    y1, x1 = int(hi[0]) + pad, int(hi[1]) + pad
    print(f"  {len(fs)} frames, one box x {x0}..{x1}, y {y0}..{y1}")
    ims = []
    for p in fs:
        im = Image.open(p).convert("RGB").crop((x0, y0, x1, y1))
        im = im.resize((w, max(1, int(w * im.height / im.width))), Image.LANCZOS)
        ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=70, loop=0,
                optimize=True, disposal=2)
    print(f"  -> {out}  {ims[0].size}  {len(ims)} frames  {os.path.getsize(out)/2**20:.2f} MiB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 2)
