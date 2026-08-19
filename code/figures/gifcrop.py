"""Trim the render overlay and the empty margin from the cut animations.

    python gifcrop.py OUT_DIR IN.gif [IN.gif ...]

`multicut_gif` draws a block of counters in the top-left for the person watching it render, and
frames the object loosely so that the pieces have room at full separation. Neither belongs in a
figure. This finds the content of every frame of every input below the overlay band, takes one
box over all of them, and writes each file back cropped to that same box -- the same box, so
that three animations shown side by side stay comparable and a piece does not appear to change
size between them.
"""
import os
import sys

import numpy as np
from PIL import Image

OVERLAY = 118          # the counters occupy the top of the frame and nothing else does
PAD = 10


def content_box(paths):
    lo = np.array([10 ** 9, 10 ** 9])
    hi = np.array([-1, -1])
    for p in paths:
        im = Image.open(p)
        for f in range(im.n_frames):
            im.seek(f)
            a = np.asarray(im.convert("RGB"))[OVERLAY:]
            m = a.min(2) < 235
            ys, xs = np.where(m)
            if not len(ys):
                continue
            lo = np.minimum(lo, [ys.min(), xs.min()])
            hi = np.maximum(hi, [ys.max(), xs.max()])
    return lo, hi


def main(out_dir, paths):
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = content_box(paths)
    y0, x0 = int(lo[0]) + OVERLAY - PAD, int(lo[1]) - PAD
    y1, x1 = int(hi[0]) + OVERLAY + PAD, int(hi[1]) + PAD
    print(f"  one box for all: x {x0}..{x1}, y {y0}..{y1}")
    for p in paths:
        im = Image.open(p)
        fs, dur = [], []
        for f in range(im.n_frames):
            im.seek(f)
            fs.append(im.convert("RGB").crop((x0, y0, x1, y1)).convert(
                "P", palette=Image.ADAPTIVE, colors=128))
            dur.append(im.info.get("duration", 60))
        q = os.path.join(out_dir, os.path.basename(p))
        fs[0].save(q, save_all=True, append_images=fs[1:], duration=dur, loop=0,
                   optimize=True, disposal=2)
        print(f"  -> {q}  {fs[0].size}  {im.n_frames} frames  "
              f"{os.path.getsize(q) / 2**20:.2f} MiB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
