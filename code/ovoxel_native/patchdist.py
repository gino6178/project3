"""The cut face scored as a distribution of patches rather than as an array of pixels.

The section loss compares a render with a photograph pixel by pixel -- SSIM and MSE on crops.
That asks the interior to reproduce *this* photograph at *this* position, and the reference is a
photograph of an orange, not of the orange being reconstructed, so the request is partly
impossible and the optimiser satisfies it the only way it can: per-plane decoration that fits one
view and means nothing between views. The band term already in `secloss` was the first answer to
that -- compare finer octaves in quantity rather than in place -- and it is the largest single
longitudinal improvement measured here (0.2535 -> 0.2230).

This is that idea carried to the whole local structure. Take the patches of the render and the
patches of the photograph as two point sets in R^(p*p*3) and ask how far apart the two *sets* are,
with no correspondence between them. A face that has the same kinds of local structure in the same
proportions scores well wherever it puts them.

Three distances, because they disagree about what "far apart" means and the disagreement is worth
measuring rather than assuming:

    sw        sliced Wasserstein -- the earth mover's distance, projected onto random directions
              where it is a sort. This is the one to reach for first: it is the actual optimal
              transport cost in 1-D, averaged over projections, and it costs O(n log n).
    chamfer   each patch to its nearest patch in the other set, both ways. Cheaper to state and
              more forgiving: it does not care whether the *proportions* match, only that nothing
              is far from everything, so a render can drop a rare structure and not be charged.
    js        Jensen-Shannon between soft histograms of the same projections. Bounded, symmetric,
              and the one that most directly says "these are the same distribution" -- but it needs
              a bin width, which is a parameter the other two do not have.

All three were measured and none of them helps. Off by default for that reason, and kept rather
than deleted because the reason is worth more than the code.

The references settle sw and js before any training. Their patch distributions were compared
against the floor -- two disjoint halves of one photograph's own patches -- over the orange's six
transverse references on the common disc:

    sw        within 0.000250   between 0.008865   ratio 35.4
    js        within 0.001433   between 0.086100   ratio 60.1
    MSE       within 0.001364   between 0.014618   ratio 10.7
    chamfer   within 0.159661   between 0.389924   ratio  2.44

sw and js hold the six photographs FURTHER apart than the pixel loss they would replace, so a
target built from them is less consistent across the family, not more. Which is what the arm did:
`r1_sw` tracked its control to five decimals until the term switched on and then ended slightly
worse. The reason is not that they are bad distances -- it is that they are sensitive to the
proportions of each kind of patch, and the proportions are a property of whichever orange was
photographed rather than of oranges.

Chamfer is the one whose premise survives that test, at 2.44: it asks only whether the vocabulary
of patches matches and not in what amounts, and the six oranges do share a vocabulary. It was
trained on that basis, at two weights, against `r1_tb1`:

    held-out probe    0.02958  ->  0.03014 (w 0.2)  ->  0.03090 (w 1.0)
    banding           3.78     ->  4.12            ->  4.38
    detail            0.1772   ->  0.1617          ->  0.1574

Identical to the control until the term switches on at half way, then worse in every column and
monotonically in the weight. Blind to proportions turns out to mean blind to *amount of
structure*: a face that covers the vocabulary thinly satisfies it, so it lets the interior smooth
out, and the detail column is that happening.

What none of them can do is place anything. All three are invariant to where a patch sits, so from
a flat initialisation they have no gradient that says the pith belongs in the middle; they would
be satisfied by pith-coloured texture at the rim. That is why the pixel term runs first and why
`SEC_DIST_MIX` keeps some of it afterwards rather than dropping it.

    SEC_DIST        sw | chamfer | js | 0     which distance, 0 = off (default)
    SEC_DIST_W      weight of the term                                        (default 1.0)
    SEC_DIST_START  fraction of training before it switches on                (default 0.5)
    SEC_DIST_MIX    fraction of the pixel term kept once it is on             (default 0.3)
    SEC_DIST_P      patch side                                                (default 5)
    SEC_DIST_N      patches sampled per image per step                        (default 1024)
    SEC_DIST_DIR    projections for sw and js                                 (default 64)
"""
import os

import torch
import torch.nn.functional as F

P = int(os.environ.get("SEC_DIST_P", "5"))
NSAMP = int(os.environ.get("SEC_DIST_N", "1024"))
NDIR = int(os.environ.get("SEC_DIST_DIR", "64"))


def patches(img, mask=None, n=None, p=None, generator=None):
    """`n` patches of side `p` from a (3,H,W) image, as rows of length p*p*3.

    Drawn where `mask` is set rather than uniformly: a patch of background is the same constant
    block in both sets and matching it says nothing, while it does move the proportions the
    distributional distances are measuring.
    """
    p = P if p is None else p
    n = NSAMP if n is None else n
    C, H, Wd = img.shape
    u = F.unfold(img.unsqueeze(0), p).squeeze(0).T                    # (L, p*p*C)
    L = u.shape[0]
    if mask is not None:
        # a patch is inside the object when its centre is, which is the mask cropped by the
        # margin unfold removes
        m = mask[p // 2:H - p // 2, p // 2:Wd - p // 2].reshape(-1) > 0.5
        idx = m.nonzero(as_tuple=True)[0]
        if idx.numel() >= 16:
            u = u[idx]
            L = u.shape[0]
    if L > n:
        sel = torch.randint(0, L, (n,), device=img.device, generator=generator)
        u = u[sel]
    return u


def _quantiles(v, q):
    """`q` evenly spaced quantiles of each column of `v`, by linear interpolation of the sort."""
    s, _ = v.sort(0)
    n = s.shape[0]
    t = torch.linspace(0, n - 1, q, device=v.device, dtype=v.dtype)
    lo = t.floor().long().clamp(0, n - 1)
    hi = t.ceil().long().clamp(0, n - 1)
    w = (t - lo.to(v.dtype)).unsqueeze(1)
    return s[lo] * (1 - w) + s[hi] * w


def _project(X, Y, n_dir=None):
    n_dir = NDIR if n_dir is None else n_dir
    V = torch.randn(X.shape[1], n_dir, device=X.device, dtype=X.dtype)
    V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return X @ V, Y @ V


def sliced_wasserstein(X, Y, n_dir=None, q=None):
    """The 2-Wasserstein distance on random 1-D projections, where it is a sort.

    The two sets need not be the same size: both are read at the same `q` quantiles, which is the
    same estimator and lets the reference contribute all its patches while the render contributes
    however many the mask allowed.
    """
    px, py = _project(X, Y, n_dir)
    q = min(px.shape[0], py.shape[0]) if q is None else q
    return ((_quantiles(px, q) - _quantiles(py, q)) ** 2).mean()


def chamfer(X, Y):
    """Each patch to its nearest in the other set, both ways, squared."""
    d = torch.cdist(X, Y) ** 2
    return d.min(1).values.mean() + d.min(0).values.mean()


def sliced_js(X, Y, n_dir=None, bins=64, sharp=2.0):
    """Jensen-Shannon between soft histograms of the same random projections.

    The histogram is soft -- a patch contributes a Gaussian of width `sharp` bins rather than
    landing in one -- because a hard bin has no gradient with respect to where the patch sits
    inside it, which is the whole quantity being optimised.
    """
    px, py = _project(X, Y, n_dir)
    lo = torch.minimum(px.min(0).values, py.min(0).values)
    hi = torch.maximum(px.max(0).values, py.max(0).values)
    w = ((hi - lo) / bins).clamp_min(1e-6)
    centres = lo + (torch.arange(bins, device=X.device, dtype=X.dtype)[:, None] + 0.5) * w

    def hist(pr):
        z = (pr.unsqueeze(1) - centres.unsqueeze(0)) / (sharp * w)    # (n, bins, n_dir)
        h = torch.exp(-0.5 * z ** 2).sum(0)
        return h / h.sum(0, keepdim=True).clamp_min(1e-12)

    p, q = hist(px), hist(py)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(0)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


_FN = {"sw": sliced_wasserstein, "chamfer": chamfer, "js": sliced_js}


def distance(rendering, ground_truth, kind=None, mask=None, n=None, p=None):
    """One distributional distance between the two images' patch sets."""
    kind = os.environ.get("SEC_DIST", "0") if kind is None else kind
    if kind not in _FN:
        return rendering.new_zeros(())
    if mask is None:
        mask = ((ground_truth.min(0).values < 0.98) |
                (rendering.min(0).values < 0.98)).float()
    X = patches(rendering, mask, n, p)
    Y = patches(ground_truth, mask, n, p)
    if X.shape[0] < 16 or Y.shape[0] < 16:
        return rendering.new_zeros(())
    return _FN[kind](X, Y)


def schedule(step, iters):
    """(weight on the distributional term, weight on the pixel term) at this step.

    Two stages, as proposed: the pixel term alone until `SEC_DIST_START` of the run has passed,
    then the distributional term with `SEC_DIST_MIX` of the pixel term retained. The pixel term
    does not go to zero because nothing else here can say where anything belongs.
    """
    if os.environ.get("SEC_DIST", "0") not in _FN:
        return 0.0, 1.0
    start = float(os.environ.get("SEC_DIST_START", "0.5"))
    if iters <= 0 or step < start * iters:
        return 0.0, 1.0
    return float(os.environ.get("SEC_DIST_W", "1.0")), \
        float(os.environ.get("SEC_DIST_MIX", "0.3"))


def _selftest():
    """The property that matters: blind to position, not blind to content.

    A shuffled copy of an image has exactly the same patches in a different arrangement, so a
    distributional distance must score it near zero while MSE scores it badly. A different texture
    must score badly on both. If the first does not hold the distance is measuring position after
    all; if the second does not hold it is measuring nothing.
    """
    torch.manual_seed(0)
    H = 128
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    a = torch.stack([((xx // 8 + yy // 8) % 2).float()] * 3) * 0.6 + 0.2   # 8 px checks
    b = torch.stack([((xx // 3 + yy // 3) % 2).float()] * 3) * 0.6 + 0.2   # 3 px checks
    # the same patches, rearranged: roll by a whole check, so no patch is new
    a_shift = torch.roll(a, shifts=(16, 24), dims=(1, 2))
    m = torch.ones(H, H)

    bad = 0
    print(f"  {'':10s} {'MSE':>10s} {'sw':>12s} {'chamfer':>12s} {'js':>12s}")
    row = {}
    for nm, other in (("shifted", a_shift), ("different", b)):
        mse = float(F.mse_loss(a, other))
        vals = {k: float(distance(a, other, k, m, n=2048)) for k in _FN}
        row[nm] = (mse, vals)
        print(f"  {nm:10s} {mse:10.5f} " + " ".join(f"{vals[k]:12.6f}" for k in _FN))
    for k in _FN:
        near, far = row["shifted"][1][k], row["different"][1][k]
        ok = far > 5 * max(near, 1e-9)
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {k}: a rearrangement of the same patches scores "
              f"{near:.6f}, a different texture {far:.6f} ({far / max(near, 1e-9):.0f}x)")
    ok = row["shifted"][0] > 0.5 * row["different"][0]
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} MSE cannot tell them apart: {row['shifted'][0]:.5f} "
          f"against {row['different'][0]:.5f}, which is what the distances are for")

    z = {k: float(distance(a, a.clone(), k, m, n=2048)) for k in _FN}
    for k, v in z.items():
        ok = v < 1e-3
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {k} of an image with itself is {v:.2e}")
    return bad


if __name__ == "__main__":
    raise SystemExit(_selftest())
