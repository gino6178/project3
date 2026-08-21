"""Where on one cut face the other family's planes cross it.

The conflict between the two families is real and Chamfer does relax it -- measured on the orange's
interior colours, the cosine between the two families' gradients on cells both touch is -0.4285
under the pixel loss and -0.1160 under Chamfer. What that measurement also showed is why applying
Chamfer everywhere made things worse: **only 1.0% of the cells either family touches are touched by
both**, and that is geometry rather than an accident. Two planes meet in a line, so on a face of a
few thousand cells the shared set is the few dozen along one line.

So the term belongs where the disagreement is. This computes the band around those lines in the
image the loss is being taken in: the intersection of the plane being drawn with each plane of the
other family, projected through the same camera and thickened to a few pixels.

Everywhere else the pixel loss is left alone, which is the part Chamfer was damaging -- a cell
supervised by one family only needs to be told what structure to have, and a distance blind to the
amount of structure cannot tell it.
"""
import numpy as np


def line_of(n1, d1, n2, d2):
    """A point and a direction on the intersection of two planes, or None if they are parallel."""
    u = np.cross(n1, n2)
    s = float(u @ u)
    if s < 1e-12:
        return None
    p0 = np.cross(d2 * np.asarray(n1, float) - d1 * np.asarray(n2, float), u) / s
    return p0, u / np.sqrt(s)


def band(mvp, n1, d1, others, res, half_px=6, span=None, nsamp=768):
    """A mask of the pixels within `half_px` of where any plane in `others` crosses this one.

    `span` is how far along each line to walk, in world units; the object's radius is the sensible
    value and is what the caller has.
    """
    m = np.zeros((res, res), bool)
    if span is None:
        span = 1.0
    yy, xx = np.mgrid[0:res, 0:res]
    for n2, d2 in others:
        got = line_of(np.asarray(n1, float), float(d1), np.asarray(n2, float), float(d2))
        if got is None:
            continue
        p0, u = got
        t = np.linspace(-span, span, nsamp)[:, None]
        pts = p0[None, :] + t * u[None, :]
        h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ np.asarray(mvp, float)
        w = h[:, 3:4]
        ok = (w[:, 0] > 1e-9)
        if not ok.any():
            continue
        nd = h[ok, :3] / w[ok]
        px = (nd[:, 0] * 0.5 + 0.5) * res
        py = (0.5 - nd[:, 1] * 0.5) * res
        keep = (px > -res) & (px < 2 * res) & (py > -res) & (py < 2 * res)
        if not keep.any():
            continue
        px, py = px[keep], py[keep]
        # a thick polyline, drawn by marking every pixel within half_px of a sample; the samples
        # are dense enough along the line that gaps between them cannot open
        ix = np.clip(np.round(px).astype(int), 0, res - 1)
        iy = np.clip(np.round(py).astype(int), 0, res - 1)
        seed = np.zeros((res, res), bool)
        seed[iy, ix] = True
        if half_px > 0:
            from scipy import ndimage
            k = 2 * half_px + 1
            seed = ndimage.binary_dilation(seed, np.ones((k, k), bool))
        m |= seed
    return m
