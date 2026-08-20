"""A differentiable Φ, so the question "why not optimise the alignment too" has an answer.

The method fits each reference to the silhouette its own plane renders, by moments, and the fit
is not differentiable: it is a LANCZOS resample and an integer paste. Equation (8)'s stop-gradient
records that rather than imposing it. The obvious objection is that the fit *should* be
differentiable and learned jointly, and the obvious reply -- that the optimiser would then move
the target instead of the volume -- is a prediction until someone runs it.

This is that run. Each supervised plane gets three parameters, a scale and a translation, applied
to its canonical reference by `grid_sample` so the target is a differentiable function of them.
They are initialised at the moment fit, so the arm starts exactly where the method starts, and
they are optimised by the same loss that optimises the volume. Whether they stay put or drift is
the measurement.

    DIFF_ALIGN=1        enable
    DIFF_ALIGN_LR=0.01  step size for the three parameters (Adam)

Not to be combined with PHASE_ALIGN, which rotates the target after it is built and would detach
the warp from its parameters. PHASE_ALIGN is off by default and no run in the paper sets it.

Nothing here is used when the flag is off; `target()` is the only entry point the trainer calls.
"""
import os

import torch
import torch.nn.functional as F

ON = os.environ.get("DIFF_ALIGN", "0") == "1"
LR = float(os.environ.get("DIFF_ALIGN_LR", "0.01"))

_P = {}          # plane key -> dict(par=Parameter(3), opt=Adam, init=tensor(3))


def _disc(t, thr=0.98):
    """Centre and radius of the section in an image, in pixels, as `sds_demo._disc` does it."""
    m = t.min(0).values < thr
    idx = m.nonzero(as_tuple=False)
    if idx.numel() < 8:
        h, w = t.shape[-2:]
        return t.new_tensor([h / 2, w / 2, min(h, w) / 4])
    ys, xs = idx[:, 0].float(), idx[:, 1].float()
    cy, cx = ys.mean(), xs.mean()
    r = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).quantile(0.98)
    return torch.stack([cy, cx, r])


def _moment_init(canon, render):
    """The fit the method already performs, expressed as (log scale, tx, ty) in grid units.

    `grid_sample` samples the source at grid coordinates, so a target that is *larger* on screen
    is a source sampled over a *smaller* window: the scale that appears here is the reciprocal of
    the one `_fit_disc` applies, and the translations carry its sign.
    """
    H, W = render.shape[-2:]
    ccy, ccx, cr = _disc(canon)
    tcy, tcx, tr = _disc(render)
    s = torch.clamp(cr / torch.clamp(tr, min=1e-3), 0.05, 20.0)
    # Centre offset, from pixels to the [-1, 1] grid the sampler uses. `affine_grid` maps an
    # output coordinate p to the input coordinate s*p + t, so a reference feature at c_c appears
    # at (c_c - t)/s; asking for it to appear at the render's centre c_r gives t = c_c - s*c_r,
    # not the other way round. Written the other way round the radius still matches and only the
    # centre is wrong, which is why this is checked against `_fit_disc` rather than eyeballed.
    tx = (2.0 * ccx / (W - 1) - 1.0) - s * (2.0 * tcx / (W - 1) - 1.0)
    ty = (2.0 * ccy / (H - 1) - 1.0) - s * (2.0 * tcy / (H - 1) - 1.0)
    return torch.stack([torch.log(s), tx, ty]).detach()


def _warp(canon, par):
    """Sample the canonical reference under (log scale, tx, ty). White outside, as a section is."""
    s = torch.exp(par[0])
    theta = canon.new_zeros(1, 2, 3)
    theta[0, 0, 0] = s
    theta[0, 1, 1] = s
    theta[0, 0, 2] = par[1]
    theta[0, 1, 2] = par[2]
    grid = F.affine_grid(theta, (1, 3) + tuple(canon.shape[-2:]), align_corners=True)
    # sample (canon - 1) so that everything outside the source pads to zero and comes back white
    out = F.grid_sample((canon - 1.0).unsqueeze(0), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return (out.squeeze(0) + 1.0).clamp(0.0, 1.0)


def target(key, canon, render):
    """The differentiable target for one plane. `canon` is its phase-aligned reference."""
    if key not in _P:
        p = torch.nn.Parameter(_moment_init(canon, render).clone())
        _P[key] = dict(par=p, opt=torch.optim.Adam([p], lr=LR),
                       init=p.detach().clone())
    d = _P[key]
    d["opt"].zero_grad(set_to_none=True)
    return _warp(canon, d["par"])


_N = [0, 0, 0]          # calls, parameters that had a gradient, parameters that did not


def step():
    """Take the alignment step. Called after the loss has been backpropagated.

    The count is not decoration. If the target is differentiable in name only -- because some
    step between it and the loss detaches, or because the crops miss it -- every parameter comes
    back with no gradient and the arm silently becomes the method it was meant to contrast with.
    """
    if not ON:
        # `_bw` calls this after every backward, so the ordinary run reaches it too. Without
        # this the counter still advances and the report still fires, putting a line about an
        # arm that is not running into every training log.
        return
    _N[0] += 1
    for d in _P.values():
        if d["par"].grad is not None and float(d["par"].grad.abs().sum()) > 0:
            _N[1] += 1
            d["opt"].step()
        else:
            _N[2] += 1
    if _N[0] % 200 == 0:
        print(f"  align: {_N[1]:,} parameter steps taken, {_N[2]:,} skipped for want of a "
              f"gradient", flush=True)
        report(_N[0])


def report(j):
    """How far each plane's alignment has moved from the moment fit that initialised it."""
    if not _P:
        return
    ds, dt = [], []
    for d in _P.values():
        delta = (d["par"].detach() - d["init"])
        ds.append(float(delta[0].abs()))
        dt.append(float(delta[1:].norm()))
    n = len(ds)
    print(f"  align j={j}: {n} planes, |dlog s| mean {sum(ds)/n:.4f} max {max(ds):.4f}, "
          f"|dt| mean {sum(dt)/n:.4f} max {max(dt):.4f}", flush=True)
