"""The section loss stage_train actually trains on, ported from train_voxel.py.

This build was using plain L1 to the mapped photograph. That is not what the pipeline does, and
the difference is not only SEC_PATCH:

    whole frame     0.7 * (1 - SSIM) + 0.3 * MSE          -- get_patch_loss's own fallback
    SEC_PATCH>0     the same pair scored on SEC_PATCH_N crops of side SEC_PATCH drawn from the
                    foreground, plus SEC_PATCH_STAT * band term per crop

stages.sh sets SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3, so every number this repository
reports was measured under the crop form.

The band term is the colour-consistency mechanism of the three: low frequencies compared where
they sit, finer octaves compared only in quantity, on a mask eroded inside both silhouettes. Its
reason is exactly the reason the free-per-cell field speckled -- "the reference is a photograph of
*an* orange, not of this one ... the optimiser still has a million free colours, so it does the
only thing that lowers the error" -- and statistics have no phase, so no single view can impose
its own pattern on a cell.

EXT_BAND_SW is 0 by default, so the octave term is the mean-magnitude form; EXT_BAND_REACH is 0.5
and get_patch_loss passes sig=(0.5, 1.0, 2.0, 4.0), so every band here is two-sided.
"""
import os

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

SEC_PATCH = int(os.environ.get("SEC_PATCH", "0"))
SEC_PATCH_N = int(os.environ.get("SEC_PATCH_N", "4"))
SEC_PATCH_STAT = float(os.environ.get("SEC_PATCH_STAT", "0"))
BAND_REACH = float(os.environ.get("EXT_BAND_REACH", "0.5"))
_BLUR_K = {}


def _blur(t, sigma):
    """Gaussian blur of a (3,H,W) tensor, separable, differentiable."""
    if sigma not in _BLUR_K:
        r = max(int(3 * sigma), 1)
        x = torch.arange(-r, r + 1, device=t.device, dtype=torch.float32)
        k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
        _BLUR_K[sigma] = (k / k.sum(), r)
    k, r = _BLUR_K[sigma]
    k = k.to(t.device)
    u = t.unsqueeze(0)
    u = F.conv2d(u, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), padding=(0, r), groups=3)
    u = F.conv2d(u, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), padding=(r, 0), groups=3)
    return u.squeeze(0)


def band_loss(rendering, ground_truth, mask, w_stat=1.0, sig=(0.5, 1.0, 2.0, 4.0)):
    lo_r, lo_g = _blur(rendering, sig[-1]), _blur(ground_truth, sig[-1])
    m = mask
    den = m.sum().clamp_min(1.0)
    loss = ((lo_r - lo_g) ** 2 * m).sum() / den / 3.0
    stat = rendering.new_zeros(())
    pr, pg = rendering, ground_truth
    for s in sig:                                   # ascending, so each band is an octave
        br, bg = _blur(rendering, s), _blur(ground_truth, s)
        tgt = ((pg - bg).abs() * m).sum() / den / 3.0
        e = (pr - br).abs()
        d = e - tgt if s >= BAND_REACH else F.relu(e - tgt)
        stat = stat + (d ** 2 * m).sum() / den / 3.0
        pr, pg = br, bg
    return loss + w_stat * stat


def ssim_loss(rendering, ground_truth):
    return 1 - ssim(rendering.unsqueeze(0), ground_truth.unsqueeze(0),
                    data_range=1, size_average=True)


def whole_loss(rendering, ground_truth):
    return 0.7 * ssim_loss(rendering, ground_truth) \
        + 0.3 * F.mse_loss(rendering, ground_truth)


# The pixel term is a squared error, so where several planes demand different things of one cell
# the solution it converges to is their weighted MEAN. That is the right estimator when the
# disagreement is noise around one answer and the wrong one when it is a mixture: two photographs of
# a different orange laid on the same plane differ by 0.0600 transverse and 0.0964 longitudinal,
# measured, so the disagreement here is a mixture and the mean of it is a blend of both.
#
# ROBUST replaces the squared error with a redescending one -- Geman-McClure, whose influence
# function falls back towards zero as the residual grows, so a demand far from the current value is
# down-weighted rather than averaged in. The estimator then settles at a mode of the demands
# instead of their mean. SEC_ROBUST is the scale at which a residual stops counting, in the units
# of the image; at 0 the loss is the ordinary squared error.
ROBUST = float(os.environ.get("SEC_ROBUST", "0"))


def _sq(r, g):
    """Squared error, or Geman-McClure's redescending version of it."""
    d2 = (r - g) ** 2
    if ROBUST <= 0:
        return d2.mean()
    c2 = ROBUST ** 2
    return (d2 / (d2 + c2)).mean() * c2      # scaled so the small-residual limit matches MSE


def patch_loss(rendering, ground_truth, n=None, size=None, stat_w=None, wfun=None):
    """Score the section in pieces instead of all at once.

    Crops come from the foreground only: a crop of background is two constant images and scores
    perfectly, so including them dilutes the gradient in proportion to how much of the frame the
    object does not fill.
    """
    n = SEC_PATCH_N if n is None else n
    size = SEC_PATCH if size is None else size
    stat_w = SEC_PATCH_STAT if stat_w is None else stat_w
    if size <= 0:
        return whole_loss(rendering, ground_truth)
    H, W = rendering.shape[-2:]
    size = min(size, H, W)
    fg = (ground_truth.min(0).values < 0.98) | (rendering.min(0).values < 0.98)
    ys, xs = fg.nonzero(as_tuple=True)
    if ys.numel() < 16:
        return whole_loss(rendering, ground_truth)
    pick = torch.randint(0, ys.numel(), (n,), device=ys.device)
    total = 0.0
    _w = []
    for k in range(n):
        y0 = int(ys[pick[k]]) - size // 2
        x0 = int(xs[pick[k]]) - size // 2
        y0 = max(0, min(y0, H - size))
        x0 = max(0, min(x0, W - size))
        _w.append(1.0 if wfun is None else float(wfun(y0 + size / 2, x0 + size / 2)))
        r = rendering[:, y0:y0 + size, x0:x0 + size]
        g = ground_truth[:, y0:y0 + size, x0:x0 + size]
        _t = 0.7 * ssim_loss(r, g) + 0.3 * _sq(r, g)
        if stat_w > 0:
            m = ((g.min(0).values < 0.98) | (r.min(0).values < 0.98)).float()[None]
            _t = _t + stat_w * band_loss(r, g, m, w_stat=1.0, sig=(0.5, 1.0, 2.0, 4.0))
        total = total + _w[k] * _t
    # normalised by the weights actually drawn, so a weighted family keeps the same total say as
    # an unweighted one and only its distribution over the section changes
    return total / max(sum(_w), 1e-6)
