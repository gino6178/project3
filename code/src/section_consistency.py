"""Make the two families of section targets agree where they describe the same material.

A transverse plane and a longitudinal plane meet in a line, and the cells on that line are in
both sections. Each section's target was produced on its own -- from its own photograph, or from
its own diffusion draw -- and neither was ever asked to agree with the other, so along that line
they generally ask those cells for two different colours. The model cannot satisfy both: a cell
has one colour. What it does instead is settle on the average, and the average of two structures
at different phases is no structure, which is why sections come out flatter than either target.

The published answer is to regenerate the targets from the model's own render every iteration,
so both families descend from one 3-D state. It works and it costs the structure: the loop has
no fixed point, the change has a persistent direction, and measured here the transverse FID went
from 126 to 172 over two hundred iterations.

This does the same job without a generator in the loop. The intersection is geometry, so the
contradiction can be found exactly and removed from the targets before they are used: sample the
line, project it into both images, and replace both with their mean along it. Nothing is
invented, one of the two is not preferred, and the operation is idempotent -- once they agree,
it does nothing, which is the fixed point the regeneration loop lacks.
"""
import torch


def _plane_normal_d(plane):
    a, b, c, d = plane
    n = torch.tensor([float(a), float(b), float(c)])
    return n / n.norm().clamp_min(1e-12), float(d) / float(n.norm().clamp_min(1e-12))


def intersection_samples(plane_a, plane_b, points, n=256, band=None):
    """Points along the line where two planes meet, kept to where the object actually is.

    A line is infinite and the object is not, so the samples are taken between the extremes of
    the model's own primitives projected onto the line's direction. `band` restricts to points
    within that distance of both planes, which is what makes them representative of the cells
    the two sections actually share rather than of the mathematical line.
    """
    dev = points.device
    na, da = _plane_normal_d(plane_a)
    nb, db = _plane_normal_d(plane_b)
    na, nb = na.to(dev), nb.to(dev)
    dirv = torch.cross(na, nb, dim=0)
    if float(dirv.norm()) < 1e-6:
        return None                                  # parallel: no line
    dirv = dirv / dirv.norm()
    # A point on both planes: solve the 2x2 system in the plane spanned by the normals.
    A = torch.stack([na, nb])
    rhs = -torch.tensor([da, db], device=dev)
    p0 = A.T @ torch.linalg.solve(A @ A.T, rhs)

    t = (points - p0) @ dirv
    if band is not None:
        near = ((points @ na + da).abs() < band) & ((points @ nb + db).abs() < band)
        if int(near.sum()) < 16:
            return None
        t = t[near]
    lo, hi = float(t.quantile(0.02)), float(t.quantile(0.98))
    if hi - lo < 1e-6:
        return None
    return p0[None] + torch.linspace(lo, hi, n, device=dev)[:, None] * dirv[None]


def project(cam, pts):
    """Pixel coordinates of world points in a rendered section, and who is in frame."""
    P = cam.full_proj_transform.to(pts.device)
    h = torch.cat([pts, torch.ones_like(pts[:, :1])], 1) @ P
    w = h[:, 3:4].clamp_min(1e-8)
    ndc = h[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * cam.image_width
    v = (ndc[:, 1] * 0.5 + 0.5) * cam.image_height
    ok = (h[:, 3] > 0) & (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height)
    return u, v, ok


def reconcile(tgt_a, cam_a, plane_a, tgt_b, cam_b, plane_b, points,
              to_world=None, radius=3, n=256, band=None, weight=1.0, mode="mean"):
    """Make the two targets agree along the line the two planes share.

    `mode` decides how, and the choice matters more than it looks. Averaging is the neutral
    thing to do and it destroys exactly what the disagreement is made of: a seed is a small
    dark blob whose position is arbitrary between two photographs of two different melons, and
    the average of a seed and no seed is a smudge. Repeated over a run that is the same
    mechanism that removes seeds from the model in the first place, applied one level up.

    `mode="copy"` instead lets one side win: `tgt_b` adopts what `tgt_a` says along the line.
    The families end up agreeing and the structure survives with a definite position, which is
    the property averaging cannot give. Which side wins is the caller's to decide and should
    be fixed for the whole run, or the two will simply take turns overwriting each other.

    Writes in place and returns how many samples were used and how far apart the two were.
    `radius` spreads each sample over a small disc, because a single pixel of a 512-wide image
    is a tenth of a cell and the two sections do not agree to that precision anyway; the point
    is to remove a disagreement about structure, not to align a sampling grid.
    """
    pts = intersection_samples(plane_a, plane_b, points, n=n, band=band)
    if pts is None:
        return 0, 0.0
    # The planes are stated in the frame the cut is computed in; the cameras project the frame
    # the rasteriser is handed. They are not the same frame, and passing points from one to the
    # other silently produced no intersections at all rather than an error.
    wpts = pts if to_world is None else to_world(pts)
    ua, va, oka = project(cam_a, wpts)
    ub, vb, okb = project(cam_b, wpts)
    ok = oka & okb
    if int(ok.sum()) < 8:
        return 0, 0.0
    ua, va, ub, vb = ua[ok].long(), va[ok].long(), ub[ok].long(), vb[ok].long()

    H, W = tgt_a.shape[-2:]
    off = torch.arange(-radius, radius + 1, device=tgt_a.device)
    dy, dx = torch.meshgrid(off, off, indexing="ij")
    keep = (dy ** 2 + dx ** 2) <= radius ** 2
    dy, dx = dy[keep].reshape(-1), dx[keep].reshape(-1)

    ya = (va[:, None] + dy[None]).clamp(0, H - 1).reshape(-1)
    xa = (ua[:, None] + dx[None]).clamp(0, W - 1).reshape(-1)
    yb = (vb[:, None] + dy[None]).clamp(0, H - 1).reshape(-1)
    xb = (ub[:, None] + dx[None]).clamp(0, W - 1).reshape(-1)

    with torch.no_grad():
        ca = tgt_a[:, ya, xa]
        cb = tgt_b[:, yb, xb]
        # Only where both are foreground: the background of one section is not a statement
        # about the material, and averaging it in would paint the other section white.
        fg = (ca.min(0).values < 0.98) & (cb.min(0).values < 0.98)
        if int(fg.sum()) < 8:
            return 0, 0.0
        disagree = float((ca[:, fg] - cb[:, fg]).abs().mean())
        if mode == "copy":
            tgt_b[:, yb[fg], xb[fg]] = (1 - weight) * cb[:, fg] + weight * ca[:, fg]
        else:
            m = 0.5 * (ca + cb)
            tgt_a[:, ya[fg], xa[fg]] = (1 - weight) * ca[:, fg] + weight * m[:, fg]
            tgt_b[:, yb[fg], xb[fg]] = (1 - weight) * cb[:, fg] + weight * m[:, fg]
    return int(fg.sum()), disagree
