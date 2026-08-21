"""Are the reference photographs' patch distributions consistent with each other, or not?

The objection is exact and worth settling before any of this is trained on: the references are
photographs of different oranges, so their cross-sections do not agree, and a loss that compares
distributions inherits that disagreement rather than escaping it.

What settles it is the ratio of two distances in the same units:

    within    two disjoint halves of ONE photograph's patches -- the floor, the difference a
              distribution has from itself at this sample size, including everything that varies
              across one real section
    between   two different photographs' patches

If between/within is near 1 the photographs are the same distribution and the objection does not
bite; if it is large they are genuinely different and it does. The number to compare it against is
already measured in pixel space: the axial profile's jump where the supervising photograph changes
is 5.14 times its jump between planes that share one, so pixel-wise between/within is about 5.

Everything is done on the common disc `_canonical` puts the targets on, because that is what the
loss would actually see.
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchdist as pd                                               # noqa: E402
import refsel                                                        # noqa: E402

N = 1500


def sets(path, seed):
    """Two disjoint halves of one photograph's patches, on the common disc."""
    a = refsel._canonical(Image.open(path).convert("RGB"))
    t = torch.from_numpy(a).permute(2, 0, 1).float()
    m = (t.min(0).values < 0.98).float()
    g = torch.Generator().manual_seed(seed)
    u = pd.patches(t, m, n=2 * N, generator=g)
    k = u.shape[0] // 2
    perm = torch.randperm(u.shape[0], generator=g)
    return u[perm[:k]], u[perm[k:2 * k]]


def main(spec):
    files = sorted(refsel.photos_in(spec))
    print(f"  {len(files)} photographs in {spec}")
    halves = [sets(f, 100 + i) for i, f in enumerate(files)]
    for kind in ("sw", "chamfer", "js"):
        fn = pd._FN[kind]
        within = [float(fn(a, b)) for a, b in halves]
        between = [float(fn(halves[i][0], halves[j][0]))
                   for i in range(len(files)) for j in range(i + 1, len(files))]
        w, b = float(np.mean(within)), float(np.mean(between))
        print(f"  {kind:8s} within one photograph {w:.6f}   between photographs {b:.6f}   "
              f"ratio {b / max(w, 1e-12):5.2f}")
    # the same photographs pixel-wise, for the row the ratio has to be read against
    ims = [refsel._canonical(Image.open(f).convert("RGB")) for f in files]
    mse_b = np.mean([float(((ims[i] - ims[j]) ** 2).mean())
                     for i in range(len(ims)) for j in range(i + 1, len(ims))])
    # a photograph against itself turned by one degree: the smallest change that is still a
    # different picture of the same section, which is the nearest thing pixels have to a floor
    mse_w = np.mean([float(((a - refsel._canonical(
        Image.fromarray((a * 255).astype("uint8")).rotate(1, fillcolor=(255, 255, 255)))) ** 2
    ).mean()) for a in ims])
    print(f"  {'MSE':8s} within one photograph {mse_w:.6f}   between photographs {mse_b:.6f}   "
          f"ratio {mse_b / max(mse_w, 1e-12):5.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
