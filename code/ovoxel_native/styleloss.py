"""A statistic the family agrees on, asked of the cut face: second-order band structure.

The photographs are of different fruit, so a cell fitted through one and the cell beside it fitted
through another are under no obligation to agree -- and on a plane nobody photographed, that
disagreement is the blocks and stripes. The pixel loss cannot fix it because the thing it asks for
is impossible: it wants this orange to be that orange. What the family does agree on is texture, so
this asks for the texture and leaves the arrangement free.

Which statistic, decided by measurement before anything was trained. For each candidate, the
distance between two halves of one photograph against the distance between halves of different
photographs of the same family -- a ratio near 1 means the statistic describes the class rather than
the individual:

                              orange h   orange v   melon h   melon v
    pixel loss (for scale)        10.7          -         -         -
    sliced Wasserstein            35.4          -         -         -
    Jensen-Shannon                60.1          -         -         -
    Chamfer on patch sets          2.44         -         -         -
    band energy, raw               7.68      2.99      2.86      2.48
    Gram, raw                      9.14      2.64      2.59      2.35
    Gram, AdaIN normalised         5.69      2.39      2.71      1.26

AdaIN first, then Gram. Removing each image's own channel mean and deviation over the section takes
out the fruit's colour, which is most of what separates two photographs of the same family, and the
watermelon's twelve longitudinal photographs then agree almost as well between themselves as within
one of them. The orange's three transverse photographs do not come down -- 5.69 after normalisation
-- because they differ in structure and not only in tone, and three is not many. That is a stated
failure case, not a solved one.

    SEC_STYLE       weight; 0 (default) leaves the objective as it was
    SEC_STYLE_SIG   the octave scales, ascending
"""
import os

import torch
import torch.nn.functional as F

WEIGHT = float(os.environ.get("SEC_STYLE", "0"))
SIG = tuple(float(x) for x in os.environ.get("SEC_STYLE_SIG", "0.5,1,2,4").split(","))
_FAM = {}


def _blur(x, s):
    k = int(2 * round(3 * s) + 1)
    g = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(g ** 2) / (2 * s * s))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), padding=(0, k // 2), groups=c)
    return F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), padding=(k // 2, 0), groups=c)


def _adain(img, m):
    """Remove the section's own channel mean and deviation, so what is left is structure."""
    mm = m[None, None]
    den = mm.sum().clamp_min(1.0)
    mu = (img * mm).sum((0, 2, 3), keepdim=True) / den
    x = (img - mu) * mm
    sd = (((x ** 2) * mm).sum((0, 2, 3), keepdim=True) / den).sqrt().clamp_min(1e-4)
    return x / sd


def gram(img, m):
    """(S*C, S*C) second-order statistic of the octave band responses over the section."""
    x = _adain(img[None] if img.dim() == 3 else img, m)
    resp, prev = [], x
    for s in SIG:
        b = _blur(x, s)
        resp.append(prev - b)
        prev = b
    B = torch.stack(resp)                                   # (S,1,C,H,W)
    S_, _, C, H, W = B.shape
    f = B.reshape(S_ * C, -1) * m.reshape(1, -1)
    den = m.sum().clamp_min(1.0)
    return (f @ f.T) / den


def family(refs, key, device, dtype):
    """The family's pooled Gram: every photograph normalised, then averaged.

    Pooled and not per-photograph on purpose. A plane's own neighbour is one fruit; the family is
    what they have in common, and it is the common part this term is for.
    """
    if key in _FAM:
        return _FAM[key]
    acc = None
    for r in refs:
        t = r if torch.is_tensor(r) else torch.as_tensor(r)
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        t = t.to(device=device, dtype=dtype)
        if float(t.max()) > 1.5:
            t = t / 255.0
        m = (t.min(0).values < 0.98).to(dtype)
        if float(m.sum()) < 400:
            continue
        g = gram(t, m)
        acc = g if acc is None else acc + g
    _FAM[key] = acc / max(len(refs), 1)
    print(f"  style: pooled Gram for {key} from {len(refs)} photographs, "
          f"{_FAM[key].shape[0]}x{_FAM[key].shape[0]}, weight {WEIGHT}", flush=True)
    return _FAM[key]


def penalty(img, refs, key):
    """Squared distance to the family's Gram, divided by its own size so the weight is scale free."""
    if WEIGHT <= 0:
        return img.new_zeros(())
    m = (img.min(0).values < 0.98).to(img.dtype)
    if float(m.sum()) < 400:
        return img.new_zeros(())
    G = family(refs, key, img.device, img.dtype)
    g = gram(img, m)
    return ((g - G) ** 2).sum() / (G ** 2).sum().clamp_min(1e-9)
