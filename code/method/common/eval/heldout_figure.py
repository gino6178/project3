"""Held-out cuts, three rows: ours, the released model, the photographs.

The rows have to be built from the same cuts, so the two models are rendered by `random_cuts.py`
with the same seed and the same `HELDOUT_BAND` and this only lays them out. The band matters: with
none set, `random_cuts` takes the outer eighths, which on a round fruit are shallow caps of rind
and pith and are not the same kind of picture as the photographs, all of which are of the middle.

    python report/heldout_figure.py OURS_DIR THEIRS_DIR PHOTO_DIR OUT.png [label]
"""
import glob
import os
import sys

import cv2
import numpy as np

INK, MUT = (26, 26, 26), (110, 110, 110)
S = 190


def strip(paths, n=6, crop=True):
    out = []
    for p in paths[:n]:
        a = cv2.imread(p)
        if a is None:
            a = np.full((S, S, 3), 255, np.uint8)
        else:
            if crop:
                # Fit each image to what is actually in it, so a render that frames the object
                # small is not shown smaller than a photograph that fills the frame. The
                # comparison is of interiors, not of framings.
                ys, xs = np.where(a.astype(int).sum(2) < 720)
                if len(xs):
                    pad = 6
                    a = a[max(0, ys.min() - pad):ys.max() + pad,
                          max(0, xs.min() - pad):xs.max() + pad]
            a = cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA)
        out.append(a)
    while len(out) < n:
        out.append(np.full((S, S, 3), 255, np.uint8))
    return np.hstack(out)


def _photos(d):
    """The photographs in a reference directory, and nothing derived from them.

    The trainer and the DreamBooth script both cache depth maps beside the images they were
    computed from, so `*.png` in a reference directory is half photographs and half greyscale
    depth. `fid_eval` has always filtered them by name; this figure did not, and published a
    "Photographs" row with two depth maps in it.
    """
    return [p for p in sorted(glob.glob(os.path.join(d, "*.png")))
            if not os.path.splitext(os.path.basename(p))[0].endswith(
                ("_depth", "_mask", "_alpha", "_normal"))]


def label(row, text):
    bar = np.full((26, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (2, 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return np.vstack([bar, row])


def main(ours, theirs, photos, out, tag="Ours"):
    rows = [
        label(strip(sorted(glob.glob(os.path.join(ours, "rh*_init_0.png")))), tag),
        label(strip(sorted(glob.glob(os.path.join(theirs, "rh*_init_0.png")))),
              "FruitNinja (released model)"),
        label(strip(_photos(photos)), "Photographs"),
    ]
    gap = np.full((16, rows[0].shape[1], 3), 255, np.uint8)
    img = np.vstack([rows[0], gap, rows[1], gap, rows[2]])
    cv2.imwrite(out, img)
    print(f"  -> {out}  {img.shape[1]}x{img.shape[0]}")


def rows(out, *specs):
    """`label=dir` pairs, in the order they should stack.

    The three-row form above answers one question and the comparison against prior work needs
    more rows than three. Every row is the same cut index of the same plane sequence, which holds
    because `random_cuts.py` seeds from a constant and every model here was rendered through the
    same camera and the same configuration file; a row whose directory holds photographs is laid
    out as photographs instead.
    """
    out_rows = []
    for spec in specs:
        tag, d = spec.split("=", 1)
        paths = sorted(glob.glob(os.path.join(d, "rh*_init_0.png")))
        out_rows.append(label(strip(paths) if paths else strip(_photos(d)), tag))
    gap = np.full((16, out_rows[0].shape[1], 3), 255, np.uint8)
    img = np.vstack(sum([[r, gap] for r in out_rows[:-1]], []) + [out_rows[-1]])
    cv2.imwrite(out, img)
    print(f"  -> {out}  {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    if sys.argv[1].endswith(".png"):
        rows(sys.argv[1], *sys.argv[2:])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
             tag=sys.argv[5] if len(sys.argv) > 5 else "Ours")
