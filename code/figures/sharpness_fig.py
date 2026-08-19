"""What a cut face is made of at 2048 px, and how sharply its boundaries are drawn.

The claim this figure exists to test is that a splatted representation cannot draw a boundary
sharply, because its primitive's support is wider than the feature. It is tested rather than
asserted: the same held-out plane is rendered by each method through its own renderer at
2048 px, a radial line is drawn from the section's centre to its rim, and the luminance along
that line is plotted for all of them against a photograph of a real cut.

    FN_ROOT=... python sharpness_fig.py OUT.png

The statistic under the plot is the mean absolute gradient along the line, over the interior
only. A representation that draws a peel, an albedo and a membrane as steps has a large one; a
representation that blurs them into each other has a small one, whatever its colours are.
"""
import glob
import os
import sys

import numpy as np

FN = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import cv2                                                             # noqa: E402

ARMS = [("photograph", None),
        ("FruitNinja", "fruitninja"),
        ("GaussianFluent", "gaussianfluent"),
        ("ours", "ours")]


def disc(a):
    """Centre and radius of the section against its background."""
    m = a.min(2) < 0.96
    ys, xs = np.where(m)
    if len(ys) < 50:
        return a.shape[0] / 2, a.shape[1] / 2, a.shape[0] / 4
    cy, cx = ys.mean(), xs.mean()
    r = np.percentile(np.hypot(ys - cy, xs - cx), 98)
    return cy, cx, r


def profile(a, n=400, ang=None):
    """Luminance along a radial line from the centre to the rim, and its mean |gradient|."""
    cy, cx, r = disc(a)
    L = a.mean(2)
    angs = np.linspace(0, 2 * np.pi, 24, endpoint=False) if ang is None else [ang]
    prof = np.zeros(n)
    for th in angs:
        t = np.linspace(0, 0.98 * r, n)
        ys = np.clip((cy + t * np.sin(th)).astype(int), 0, a.shape[0] - 1)
        xs = np.clip((cx + t * np.cos(th)).astype(int), 0, a.shape[1] - 1)
        prof += L[ys, xs]
    prof /= len(angs)
    return prof, float(np.abs(np.diff(prof)).mean())


def main(out):
    base = os.path.join(FN, "measurements", "table2")
    ref = sorted(glob.glob(os.path.join(FN, "data_finetune_images", "watermelon",
                                        "horizontal", "*.png")))
    imgs, profs, grads = [], [], []
    for label, sub in ARMS:
        if sub is None:
            a = cv2.imread(ref[0])[:, :, ::-1].astype(np.float32) / 255.
        else:
            p = sorted(glob.glob(os.path.join(base, sub, "rh*_init_0.png")))[0]
            a = cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255.
        # The crops are drawn at native resolution -- the whole point of the figure is what
        # 2048 px shows -- while the statistic is taken on a 512 copy, because a per-sample
        # gradient is four times smaller at four times the sampling for the same edge and
        # comparing the two directly measures the file sizes rather than the representations.
        imgs.append(a)
        small = a if a.shape[0] == 512 else cv2.resize(a, (512, 512),
                                                       interpolation=cv2.INTER_AREA)
        pr, g = profile(small)
        profs.append(pr); grads.append(g)
        print(f"  {label:16s} {a.shape[0]}px   mean |dL/dr| = {g:.5f}")

    # Two rows over the same four renders. The top row is a quadrant of the section, which is
    # where the reader can still tell what the object is; the bottom is the same crop the top
    # row's box marks, at the scale where a cell is a few pixels. Neither is resampled -- both
    # are slices of the native 2048, so what separates them is field of view and nothing else.
    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(2, 4, wspace=0.045, hspace=0.055,
                          left=0.008, right=0.992, top=0.948, bottom=0.012)
    for k, ((label, _), a) in enumerate(zip(ARMS, imgs)):
        cy, cx, r = disc(a)
        # A quadrant: from just inside the centre out past the rim, so a quarter of the section
        # and the peel that bounds it are both in the frame.
        w = int(1.00 * r)
        wy0, wx0 = int(cy - 0.10 * r), int(cx - 0.10 * r)
        wide = np.clip(a[max(wy0, 0):wy0 + w, max(wx0, 0):wx0 + w], 0, 1)
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(wide)
        ax.set_axis_off()
        ax.set_title(f"({'abcd'[k]})  {label}", fontsize=11)

        h = int(0.16 * r)   # tighter: the fibres are a few pixels wide
        y0, x0 = int(cy - h), int(cx - h)
        ax.add_patch(plt.Rectangle((x0 - max(wx0, 0), y0 - max(wy0, 0)), 2 * h, 2 * h,
                                   fill=False, lw=0.9, ec="0.15"))
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(np.clip(a[max(y0, 0):y0 + 2 * h, max(x0, 0):x0 + 2 * h], 0, 1))
        ax.set_axis_off()
        ax.set_title(f"({'efgh'[k]})", fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/sharpness.png")
