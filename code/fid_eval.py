"""FID and KID against the cross-section photographs, as the paper reports them.

CLIP says whether a slice reads as an orange's cross-section and it says so at the ceiling for
everything tried here, 31.1 to 31.5 against the photographs' own 31.4. It is insensitive to the
differences that are actually in dispute -- a continuous pith ring against a broken one, detail
overshooting the photographs by 44%, saturation running high -- so it cannot choose between
candidates.

FID and KID compare *distributions* of Inception features. That is the right shape for this
problem: our orange is not the photographed one and its segments are somewhere else, so no
per-pixel score is meaningful, but "does this look like it was drawn from the same population
of cross-section images" is exactly the question.

The caveat is the sample size. There are six transverse photographs; FID's covariance estimate
is badly biased at that count and the number will be large and noisy. KID is unbiased and is
the one to read -- which is presumably why the paper reports both.

    python fid_eval.py secref_orraw_hsep run/snap/iter_0029/h*_init_0.png
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image


def load(paths, size=299):
    out = []
    for p in paths:
        a = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        out.append(torch.from_numpy(np.asarray(a).copy()).permute(2, 0, 1))
    return torch.stack(out).to(torch.uint8)


def main(real_dir, fake_paths, subset=None):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    real = load([p for p in sorted(glob.glob(os.path.join(real_dir, "*")))
                 if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg")
                 and not os.path.splitext(os.path.basename(p))[0].endswith(
                     ("_depth", "_mask", "_alpha"))])
    fake = load(fake_paths)
    n = subset or max(2, min(len(real), len(fake)) // 2)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    fid.update(real.to(dev), real=True)
    fid.update(fake.to(dev), real=False)
    kid = KernelInceptionDistance(feature=2048, subset_size=n, subsets=50,
                                  normalize=False).to(dev)
    kid.update(real.to(dev), real=True)
    kid.update(fake.to(dev), real=False)
    km, ks = kid.compute()
    print(f"  {len(real)} photographs vs {len(fake)} renders   "
          f"FID {float(fid.compute()):7.1f}   KID {float(km) * 1e3:6.1f}e-3 "
          f"(+-{float(ks) * 1e3:.1f}, subset {n})")
    return float(fid.compute()), float(km)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("real_dir")
    ap.add_argument("fake", nargs="+")
    ap.add_argument("--subset", type=int, default=None)
    a = ap.parse_args()
    main(a.real_dir, a.fake, a.subset)
