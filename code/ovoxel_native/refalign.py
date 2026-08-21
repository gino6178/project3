"""Put each reference the right way up against the object's shell, before it supervises anything.

`_same_topology_map` carries a reference across at the angle it sits at, so it preserves
orientation exactly: the photograph's up becomes the render's up. Nothing in the pipeline
establishes that the two ups are the same one. The transverse family is phase-aligned, but that
aligns the photographs to EACH OTHER; the longitudinal family is not aligned at all. So a
photograph shot upside down, or mirrored, is mapped in upside down or mirrored and the loss cannot
tell -- it is computed after the map and therefore grades the map's own output.

What makes this fixable rather than circular is that the shell is not what is being learned.
`SHELL_PIN` holds the exterior at the released model's, so the silhouette of any cut and the rind
inside it are the object's own shape from the first iteration. A reference can be registered
against that without using anything the interior is supposed to supply.

The registration is over the four flips of the image plane and nothing else. Rotation is left
alone: the transverse family already has its own angular alignment, and for the longitudinal family
a rotation would tilt the polar axis, which is a different and larger claim than "this photograph
was stored mirrored".

    REF_ORIENT=1   register each reference to the shell before use (default 0)

Two scores, multiplied, because either alone is fooled by a round object: the intersection over
union of the two silhouettes once both are centred and scaled, and the correlation of their radial
profiles, which is what tells a rind from a core when the outline says nothing.
"""
import os

import numpy as np
from scipy import ndimage

ENABLED = os.environ.get("REF_ORIENT", "0") == "1"

# The manual setting, and it takes precedence over anything this file can work out. Which way up a
# photograph was stored is not a claim the paper makes or needs to defend, and the automatic search
# below can only decide it where the reference and the shell already agree in shape -- which is not
# where it is needed. `REF_H_FLIP` and `REF_V_FLIP` name the flip for a whole family:
#
#     none      as stored
#     ud        top to bottom
#     lr        left to right
#     rot180    both, which is a half turn
#
# Set them in objects/<obj>.conf beside the reference directories they apply to.
BY_NAME = {"none": lambda a: a, "ud": lambda a: a[::-1],
           "lr": lambda a: a[:, ::-1], "rot180": lambda a: a[::-1, ::-1]}
FLIPS = (("as stored", lambda a: a),
         ("flipped top to bottom", lambda a: a[::-1]),
         ("flipped left to right", lambda a: a[:, ::-1]),
         ("turned half round", lambda a: a[::-1, ::-1]))


def _norm(a, n=192):
    """One image's mask and radial profile, on a common frame so two can be compared."""
    L = np.asarray(a, np.float32).mean(2)
    m = L < 0.97
    if m.sum() < 400:
        return None, None
    ys, xs = np.where(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    mm = m[sl].astype(np.float32)
    zoom = (n / mm.shape[0], n / mm.shape[1])
    mm = ndimage.zoom(mm, zoom, order=1) > 0.5
    ll = ndimage.zoom(np.where(m, L, np.nan)[sl], zoom, order=1)

    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
    prof = np.full(24, np.nan)
    for k in range(24):
        sel = mm & (r >= k / 24) & (r < (k + 1) / 24)
        if sel.sum() > 20:
            v = ll[sel]
            v = v[~np.isnan(v)]
            if len(v) > 10:
                prof[k] = v.mean()
    return mm, prof


def _score(ref, shell):
    mr, pr = _norm(ref)
    ms, ps = _norm(shell)
    if mr is None or ms is None:
        return -np.inf
    iou = float((mr & ms).sum()) / max(float((mr | ms).sum()), 1.0)
    ok = ~np.isnan(pr) & ~np.isnan(ps)
    if ok.sum() < 6:
        return iou
    a, b = pr[ok], ps[ok]
    c = float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 \
        else 0.0
    return iou * (0.5 + 0.5 * max(c, -1.0))


MARGIN = float(os.environ.get("REF_ORIENT_MARGIN", "0.05"))


def orient_family(refs, shells, tag="", manual=None):
    """One flip for a whole family, or none, decided over every plane in it at once.

    Per plane was the wrong unit and said so out loud: on the orange it chose left-to-right for the
    first six transverse planes and top-to-bottom for the next three, on margins of one per cent.
    Which way a photograph was stored is a property of the photograph, not of the plane it happens
    to supervise, so the scores are summed over the family and one answer comes out.

    And it has to be allowed to come out empty. A round object's silhouette cannot tell a flip from
    its mirror, so a win of a few parts in a thousand is noise; `REF_ORIENT_MARGIN` is the relative
    margin over leaving it alone that a flip has to clear, and below it the family is left as it is
    and the line says the shell could not answer.
    """
    refs = [np.asarray(r) for r in refs]
    if not refs:
        return refs, None
    if manual:
        m = manual.strip().lower()
        if m not in BY_NAME:
            return refs, f"  {tag}: '{manual}' is not one of {sorted(BY_NAME)}; left as stored"
        if m == "none":
            return refs, f"  {tag}: set to none, left as stored"
        f = BY_NAME[m]
        return [f(r) for r in refs], f"  {tag}: {m}, set by hand"
    if not ENABLED:
        return refs, None
    tot = {}
    for nm, f in FLIPS:
        tot[nm] = float(np.mean([_score(f(r), sh) for r, sh in zip(refs, shells)]))
    base = tot["as stored"]
    nm, best = max(tot.items(), key=lambda kv: kv[1])
    rel = (best - base) / max(abs(base), 1e-6)
    if nm == "as stored" or rel < MARGIN:
        second = max((k for k in tot if k != "as stored"), key=lambda k: tot[k])
        return refs, (f"  {tag}: left as stored ({base:.4f}); best alternative {second} at "
                      f"{tot[second]:.4f}, {100 * (tot[second] - base) / max(abs(base), 1e-6):+.1f}%"
                      f", under the {100 * MARGIN:.0f}% the shell has to win by")
    f = dict(FLIPS)[nm]
    return [f(r) for r in refs], (f"  {tag}: {nm} for the whole family "
                                  f"({best:.4f} against {base:.4f}, {100 * rel:+.1f}%)")
