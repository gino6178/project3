"""Build the cross-section target by matching the reference to the section actually rendered.

The released pairing rescales the reference so its disc matches the render's disc, taking the
centroid and the 98th-percentile radius over every foreground pixel. That is a correspondence
only when both sides are a single round blob. A ring cut through its axis renders two separate
tube faces: the centroid of "every foreground pixel" lands in the hole between them and the
radius spans both, so the reference is pasted over the gap. The target then carries colour
where the render has background, a difference no arrangement of the model can close, and the
only way the loss can shrink is to pull the whole object toward the background -- which is
what happened, the doughnut washing out to cream while its profile error went 0.016 -> 0.122.

The correspondence has to be per connected component, and it has to exist. A pointwise map
between the reference and a component is a bijection between their domains, so it exists iff
the two are homeomorphic. Topology is therefore the branch condition, and it is computable --
the number of holes in a 2-D mask is the number of components of its complement inside the
bounding box, less the outside.

  same topology   the angle about the centroid, and the fraction of the way across the
                  region along that ray. Both are coordinates any planar region has, and
                  together they are a bijection between two regions that agree: a disc onto
                  a disc, an annulus onto an annulus. On a solid of revolution it reduces to
                  the polar mapping.
  otherwise       normalised distance from the boundary, and the reference's colour at the
                  same depth -- the one coordinate that survives when no bijection exists.

That is one rule, and it used to be two. The simply-connected case was handled by normalised
bounding-box coordinates and *everything else* fell to the depth mapping, which meant a ring's
transverse section -- an annulus supervised against an annulus, homeomorphic and perfectly
mappable -- had its two-dimensional pattern thrown away and came back as a smooth band of
concentric colour. Under one rule it comes back as crumb.

Backgrounds line up too: the target carries colour exactly where the render has primitives, so
the unsatisfiable difference disappears with it.
"""
import hashlib
import os

import cv2
import numpy as np
import torch
from scipy import ndimage

_REF = {}


def _holes(mask, min_frac=0.05):
    """Number of holes in a 2-D binary mask, counting only ones big enough to be real.

    Thresholding a photograph against its background leaves speckle: the references have 11
    to 245 enclosed regions each, and every one is small -- the largest is 2.3% of the mask.
    A genuine hole is not: a ring's is about a fifth of its outer disc. Counting every
    enclosed pixel made all four references look multiply connected, which sent every
    section down the depth branch and threw away the two-dimensional pattern entirely.
    """
    lab, n = ndimage.label(~mask)
    if n == 0:
        return 0
    border = set(lab[0].tolist()) | set(lab[-1].tolist()) | \
        set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    border.discard(0)
    area = max(int(mask.sum()), 1)
    cnt = 0
    for i in range(1, n + 1):
        if i in border:
            continue
        if int((lab == i).sum()) >= min_frac * area:
            cnt += 1
    return cnt


def _reference_mask(a, tol=12, fill=True):
    """Where the reference's section is, found by what the background is connected to.

    This used to threshold against the median of the top and bottom eight rows: background is
    whatever sits within 0.08 of that, section is everything else. Two things break it, and
    both are in every run.

    The soft halo a generator leaves around the disc is outside the tolerance, so it joins the
    section -- at iteration 0 the mask came out 399x383 for an object that is about 300x320,
    and 44% of the frame for one covering 30%. And the generator does not hold the background
    still: over fifty iterations it drifted from (0.996, 1.000, 1.000) to (0.922, 0.957, 0.957)
    and stopped being uniform, at which point one median cannot describe it and a whole side of
    the background fell outside the tolerance and attached itself to the disc. Taking the
    largest connected component does not help -- the background is attached, so it is part of
    the largest component.

    That mask is not cosmetic. `_ray_coords` measures the region's outer radius along every ray
    from its centroid, so an inflated lopsided mask stretches the mapping, and the target then
    carries the reference's *background* over part of the render -- a grey crescent with a
    straight edge, telling the model to delete the half of itself that lies under it.

    Background is the region touching the frame edge, so find it that way instead: flood from
    the four corners with a tolerance on the step between neighbouring pixels, which follows a
    gradient without crossing the object's boundary. Holes are then filled, so a pale core
    stays part of the section. Measured across the same run this returns 295-302 by 313-328 at
    every iteration, against the threshold's 399x383, 309x328 and 389x336.

    `fill` is what the filling costs on an object that has a real hole. An orange's core is
    pale and enclosed and must be kept; a doughnut's hole is pale and enclosed and must not,
    and nothing in this image tells the two apart. Filling both makes every reference simply
    connected, so the transverse doughnut photograph reported no hole while the render's
    annulus reported one, they never matched, and `section_target` took its depth fallback for
    the whole run -- which is a lookup on distance to the boundary and can only draw bands
    parallel to the silhouette. That is the concentric ring pattern the trained doughnut had
    instead of crumb. The caller settles it by asking for both and keeping whichever agrees
    with the render, which does know the difference.
    """
    img = (np.clip(a, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    h, w = img.shape[:2]
    ff = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        # FIXED_RANGE: compare each candidate with the seed, not with the neighbour it came
        # from. Without it the tolerance is a step size and the flood walks a gradient: a
        # photograph is a hard-edged cutout and stops it, but a *generated* section has a soft
        # edge, and twelve levels per step is enough to creep through the rind and eat the
        # fruit. Measured on the reference regenerated at iteration 177: the mask came out at
        # 842 pixels for a disc 223 pixels across. `_ray_coords` then measured its radii on
        # those 842 pixels and `_same_topology_map` stretched them over the whole section --
        # a target with 6.977 of radial streaking against the photograph's 0.838, which the
        # model then dutifully learned. Every regeneration failure tonight is this line.
        cv2.floodFill(img, ff, seed, 0, (tol,) * 3, (tol,) * 3,
                      4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
    m = ndimage.binary_fill_holes(~(ff[1:-1, 1:-1] > 0))
    # And a floor under it, whatever the cause. A section that covers a fifth of what plainly
    # differs from the background is not a section, and carrying on with it produces a target
    # that looks like nothing and is silently trained against.
    _bg = np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), 0)
    _plain = ndimage.binary_fill_holes(
        np.abs(img.astype(np.float32) - _bg).max(2) > max(tol, 16))
    if m.sum() < 0.2 * max(int(_plain.sum()), 1):
        print(f"  reference mask collapsed ({int(m.sum()):,} px against {int(_plain.sum()):,} "
              f"plainly non-background); falling back to the plain test")
        m = _plain
    if not fill:
        # The flood is four-connected from the corners, so a hole enclosed by the section is
        # never reached however exactly its colour matches the background -- the mask arrives
        # here already filled, and dropping `binary_fill_holes` alone changes nothing. Carve
        # instead: inside the mask, what sits at the background's own colour is a hole. Then
        # put back anything too small to be one, so speckle in the crumb does not perforate
        # the mask that `_same_topology_map` measures its radii on.
        bgc = np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), 0)
        m = m & ~(np.abs(img.astype(np.float32) - bgc).max(2) <= tol)
        lab, n = ndimage.label(~m)
        if n:
            border = set(lab[0].tolist()) | set(lab[-1].tolist()) | \
                set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
            keep = max(int(m.sum()), 1) * 0.05
            for i in range(1, n + 1):
                sel = lab == i
                if i not in border and sel.sum() < keep:
                    m[sel] = True
    lab, k = ndimage.label(m)
    if k > 1:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


def _ref_info(ref_rgb, nb=48, fill=True):
    """Mask, bounding box, hole count and depth profile of the reference, cached.

    `fill` selects which reading of an enclosed pale region the caller wants -- part of the
    section, or a hole through it. See `_reference_mask`.
    """
    # Key on the whole array, not its first 64 bytes. Every one of these photographs opens
    # with white background, so their first 64 bytes are identical and the cache returned
    # whichever reference was seen first for all of them. Training runs the ten longitudinal
    # planes before the sixteen transverse ones, so the longitudinal photograph was cached
    # first and every transverse plane was then supervised against it -- which is exactly
    # what the trained orange showed, a transverse section carrying a lengthwise core.
    key = (hashlib.blake2b(np.ascontiguousarray(ref_rgb).tobytes(),
                           digest_size=16).digest(), ref_rgb.shape, fill)
    if key in _REF:
        return _REF[key]
    a = ref_rgb
    m = _reference_mask(a, fill=fill)
    ys, xs = np.where(m)
    box = (ys.min(), ys.max(), xs.min(), xs.max())
    dt = ndimage.distance_transform_edt(m)
    dn = dt / max(dt.max(), 1e-6)
    lut = np.zeros((nb, 3), np.float32)
    for i in range(nb):
        sel = m & (dn >= i / nb) & (dn < (i + 1) / nb)
        lut[i] = a[sel].mean(0) if sel.sum() > 10 else np.nan
    for c in range(3):
        v = lut[:, c]; idx = np.arange(nb); good = ~np.isnan(v)
        lut[:, c] = np.interp(idx, idx[good], v[good]) if good.any() else 1.0
    out = (a, m, box, _holes(m), lut, nb)
    _REF[key] = out
    return out


def _ray_coords(mask, nb_a=1440, smooth_deg=None):
    """Centroid, and the inner and outer radius of the region along each ray from it.

    The two coordinates any planar region has. For a simply connected region the inner radius
    is zero and this is the polar mapping; for an annulus it is the inner boundary, and the
    fraction between the two runs 0 to 1 across the material either way.
    """
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    th = np.arctan2(ys - cy, xs - cx)
    r = np.hypot(ys - cy, xs - cx)
    ai = np.clip(((th + np.pi) / (2 * np.pi) * nb_a).astype(np.int64), 0, nb_a - 1)
    r_in = np.full(nb_a, np.inf)
    r_out = np.zeros(nb_a)
    np.minimum.at(r_in, ai, r)
    np.maximum.at(r_out, ai, r)
    seen = np.isfinite(r_in)
    if not seen.any():
        return (cy, cx), np.zeros(nb_a), np.ones(nb_a)
    if not seen.all():
        idx = ndimage.distance_transform_edt(~seen, return_distances=False,
                                             return_indices=True)[0]
        r_in, r_out = r_in[idx], r_out[idx]
    # How wide the circular smoothing is, in degrees of arc rather than in bins. At nb_a=1440
    # and a disc of radius 200px one bin subtends 0.87px, so the per-bin radius is pixel noise
    # and the nine-tap kernel this used covered under eight pixels of it. What the map then
    # does with that noise is visible: adjacent rays scale the reference by slightly different
    # factors, and the reference's content is torn along the rays -- seeds stretched into
    # streaks, fibres into spokes radiating from the centre. A real section's radius varies
    # smoothly with angle, so smoothing over degrees costs nothing real and removes the noise.
    # Estimate on coarse bins and interpolate back, rather than estimating on fine bins and
    # smoothing the result. At nb_a=1440 a bin subtends under a pixel of arc, so its extreme
    # radius is the extreme of one or two anti-aliased pixels -- a noisy statistic, and the
    # noise is what makes neighbouring rays sample different parts of the reference and tear
    # its content into spokes. Smoothing that afterwards removes the spokes and displaces the
    # content: measured, 9 degrees of smoothing costs 23% of the edge the map is copying.
    # Binning coarsely instead gives each estimate enough pixels to be stable, and the radius
    # of a section really does vary slowly with angle, so nothing real is lost.
    est = max(int(os.environ.get("SEC_RAY_BINS", "90")), 8)
    deg = float(os.environ.get("SEC_RAY_SMOOTH_DEG", "0")) if smooth_deg is None \
        else float(smooth_deg)
    idx = np.linspace(0, nb_a, est, endpoint=False).astype(np.int64)
    def coarse(v):
        w = np.array([v[a:b].min() if a < b else v[a]
                      for a, b in zip(idx, list(idx[1:]) + [nb_a])])
        return w
    def coarse_max(v):
        return np.array([v[a:b].max() if a < b else v[a]
                         for a, b in zip(idx, list(idx[1:]) + [nb_a])])
    ci, co = coarse(r_in), coarse_max(r_out)
    k = max(int(round(deg / 360.0 * est)) | 1, 1)
    if k > 1:
        ker = np.ones(k) / k; h = k // 2
        wrap = lambda v: np.convolve(np.concatenate([v[-h:], v, v[:h]]), ker, "same")[h:-h]
        ci, co = wrap(ci), wrap(co)
    xs = np.arange(nb_a) / nb_a * est
    xp = np.arange(est)
    lift = lambda v: np.interp(xs, xp, v, period=est)
    return (cy, cx), lift(ci), lift(co)


# Off. Two ways of aligning the layers radially were measured and neither is general; the
# note in _same_topology_map has the numbers. Kept so the next person can re-measure rather
# than re-derive the idea, which is a convincing one.
PATH_MATCH = os.environ.get("SEC_PATH_MATCH", "0") == "1"

def _band_profile(a, mask, ri, ro, cy, cx, nb_s=64):
    """Mean colour of the region as a function of how far across it a pixel sits.

    The one coordinate the ray map already uses, collapsed over angle. A fruit's layers are
    bands in this coordinate -- peel, pith, flesh -- so a layer boundary is a step in this
    profile no matter what shape the outline is.
    """
    ys, xs = np.where(mask)
    th = np.arctan2(ys - cy, xs - cx)
    nb = ri.shape[0]
    ai = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(np.int64), 0, nb - 1)
    r = np.hypot(ys - cy, xs - cx)
    s = np.clip((r - ri[ai]) / np.maximum(ro[ai] - ri[ai], 1e-6), 0.0, 1.0)
    b = np.clip((s * nb_s).astype(np.int64), 0, nb_s - 1)
    # bincount, not np.add.at: the same sum over a quarter of a million pixels, and the
    # unbuffered version costs more than the map it is supporting -- 126 ms of the 216 this
    # added per target, against 90 for the map itself.
    v = a[ys, xs]
    cnt = np.bincount(b, minlength=nb_s).astype(np.float64)
    tot = np.stack([np.bincount(b, weights=v[:, c], minlength=nb_s)
                    for c in range(v.shape[1])], axis=1)
    ok = cnt > 0
    if not ok.any():
        return None, None
    prof = np.zeros_like(tot)
    prof[ok] = tot[ok] / cnt[ok, None]
    if not ok.all():                      # bins no pixel landed in: carry the nearest
        idx = ndimage.distance_transform_edt(~ok, return_distances=False,
                                             return_indices=True)[0]
        prof = prof[idx]
    return prof, cnt


def _colour_path(a, mask, ri, ro, cy, cx):
    """How far the colour has travelled by the time it reaches each radius, 0 to 1.

    A section's layers are steps in the radial profile, so the profile's own arc length rises
    quickly across a boundary and slowly inside a layer. Two photographs of the same kind of
    fruit pass the same boundaries in the same order, whatever fraction of the radius each one
    gives them, so this coordinate lines their layers up without anyone deciding where a layer
    is. That matters more than the alignment does: every threshold that finds a boundary is a
    number tuned on the object it was found with, and there is none here.
    """
    nb_s = int(os.environ.get("SEC_PATH_BINS", "64"))
    prof, _ = _band_profile(a, mask, ri, ro, cy, cx, nb_s)
    if prof is None:
        return None
    g = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    if g[-1] < 1e-9:
        return None                     # one flat colour: no layers, nothing to line up
    g = g / g[-1]
    # np.interp needs its sample points strictly increasing, and a layer's interior is flat
    return g + np.linspace(0.0, 1e-6, g.shape[0])




def _sample(a_ref, fy, fx):
    """Read the reference at the mapped positions, however SEC_MAP_INTERP says to.

    Lifted out of the per-ray map so the similarity map can use the same reader: the two
    differ in where they look, and nothing else should differ with them.
    """
    mode = os.environ.get("SEC_MAP_INTERP", "bilinear")
    if mode == "bilinear":
        y0 = fy.astype(np.int64); x0 = fx.astype(np.int64)
        wy = (fy - y0)[:, None]; wx = (fx - x0)[:, None]
        y1 = np.minimum(y0 + 1, a_ref.shape[0] - 1); x1 = np.minimum(x0 + 1, a_ref.shape[1] - 1)
        return ((a_ref[y0, x0] * (1 - wx) + a_ref[y0, x1] * wx) * (1 - wy) +
                (a_ref[y1, x0] * (1 - wx) + a_ref[y1, x1] * wx) * wy)
    if mode == "nearest":
        return a_ref[np.rint(fy).astype(np.int64).clip(0, a_ref.shape[0] - 1),
                     np.rint(fx).astype(np.int64).clip(0, a_ref.shape[1] - 1)]
    flag = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
            "area": cv2.INTER_AREA}.get(mode, cv2.INTER_CUBIC)
    out = cv2.remap(np.ascontiguousarray(a_ref, dtype=np.float32),
                    fx.astype(np.float32).reshape(-1, 1),
                    fy.astype(np.float32).reshape(-1, 1),
                    flag, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out.reshape(-1, a_ref.shape[2]), 0.0, 1.0)


def _same_topology_map(comp, a_ref, m_ref, a_dst=None):
    """Colour a component from a reference of the same topology, by their own coordinates.

    One rule for every case the branch used to split. A disc onto a disc and an annulus onto
    an annulus are both bijections and are both this map; the bounding-box version it replaces
    was a second rule that only ever covered the first of those, which is why a doughnut's
    transverse section -- an annulus supervised against an annulus, homeomorphic and perfectly
    mappable -- was being sent to the depth fallback and having its two-dimensional pattern
    thrown away.
    """
    # Smooth the two sides differently, because they are different things. The destination is
    # our own render: a lattice-quantised silhouette whose per-ray radius is jagged, and a
    # jagged radius makes neighbouring rays scale the reference differently, which is what
    # tears its content into spokes. The source is a photograph of a round fruit, already
    # smooth, and smoothing it again only moves its content off the radius it actually sits at
    # -- which is how a one-pixel pith edge becomes a soft band.
    #
    # Smoothing the estimate is the wrong repair either way, and splitting it between the two
    # sides does not help: measured, 9 degrees on both matches the streaking and loses 23% of
    # the edge, 0.5 on the source keeps the edge and puts the streaking back. Estimating on 90
    # coarse bins and interpolating fixes both, and it was checked on all three objects rather
    # than on the one it was found with -- targets against their own references, per-ray edge
    # and streaking: watermelon 0.1685/0.870 against 0.2092/0.899, orange 0.2046/1.229 against
    # 0.2164/1.218, doughnut 0.1321/0.545 against 0.1256/0.590. Each tracks its own reference,
    # which is the property a shared default has to have.
    (dcy, dcx), dri, dro = _ray_coords(comp)
    (scy, scx), sri, sro = _ray_coords(
        m_ref, smooth_deg=float(os.environ.get("SEC_RAY_SMOOTH_SRC_DEG", "0")))
    ys, xs = np.where(comp)
    th = np.arctan2(ys - dcy, xs - dcx)
    r = np.hypot(ys - dcy, xs - dcx)

    # SEC_MAP=similarity: one scale and one translation for the whole component, so the
    # photograph keeps its own shape and the component's outline only crops it.
    #
    # The per-ray map below aligns the two outlines, which is what makes a photograph of one
    # fruit sit on the silhouette of another -- but it scales every angle by its own factor, so
    # wherever the rendered silhouette is not round the photograph is stretched by a different
    # amount at each angle. On this watermelon the transverse silhouette is visibly not round,
    # and the target it produces is not round either: a circular photograph arrives bent.
    #
    # Under similarity the target keeps the photograph's geometry and loses the outline
    # alignment, so the rind lands where the photograph put it rather than where this cut's
    # own edge is. Which of those matters more is a measurement, not a preference, and this
    # exists so it can be made.
    _map = os.environ.get("SEC_MAP", "ray")
    if _map == "affine":
        # Match the two masks' second-moment ellipses: one linear map for the whole component, so
        # the outline still lines up in extent and orientation while the content keeps its
        # straight lines. The per-ray map has ninety independent scales and bends a round
        # photograph; the similarity map has one and loses the rind, because a photograph of an
        # elongated section scaled by a single factor puts its edge inside the mask along one
        # direction and outside it along the other. An affine has three, which is exactly the
        # elongation and the turn that separate two ellipses.
        def _mom(m):
            yy, xx = np.where(m)
            c = np.array([yy.mean(), xx.mean()])
            d = np.stack([yy - c[0], xx - c[1]])
            return c, (d @ d.T) / max(len(yy), 1)

        def _sqrtm(A):
            w, V = np.linalg.eigh(A)
            return V @ np.diag(np.sqrt(np.maximum(w, 1e-9))) @ V.T

        c_d, C_d = _mom(comp)
        c_s, C_s = _mom(m_ref)
        M = _sqrtm(C_s) @ np.linalg.inv(_sqrtm(C_d))
        q = np.stack([ys - c_d[0], xs - c_d[1]])
        src = M @ q
        fy = np.clip(c_s[0] + src[0], 0, a_ref.shape[0] - 1.001)
        fx = np.clip(c_s[1] + src[1], 0, a_ref.shape[1] - 1.001)
        return _sample(a_ref, fy, fx)

    if _map == "similarity":
        d_rad = float(np.mean(dro[dro > 0])) if np.any(dro > 0) else 1.0
        s_rad = float(np.mean(sro[sro > 0])) if np.any(sro > 0) else 1.0
        k_sim = s_rad / max(d_rad, 1e-6)
        fy = np.clip(scy + (ys - dcy) * k_sim, 0, a_ref.shape[0] - 1.001)
        fx = np.clip(scx + (xs - dcx) * k_sim, 0, a_ref.shape[1] - 1.001)
        return _sample(a_ref, fy, fx)

    nb = dri.shape[0]
    ai = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(np.int64), 0, nb - 1)
    s = np.clip((r - dri[ai]) / np.maximum(dro[ai] - dri[ai], 1e-6), 0.0, 1.0)
    # Match the layers, not just the outline. The map above is the identity in `s`, so it
    # aligns the two boundaries and then leaves each reference's layers at whatever fraction of
    # the radius they happened to fall at -- across the watermelon's twenty transverse
    # photographs the peel runs from 3.9% to 19.5%, five to one.
    #
    # Reparametrising by the colour path sends each reference's layers onto the render's own.
    # It is monotone, fixes both ends, needs no threshold and no idea of what a layer is, so it
    # is the same code for a watermelon's pith and skin, an orange's peel and a loaf that has
    # neither -- and on an object with one flat colour it returns nothing and the map is
    # untouched.
    #
    # It does not work, and it is off. Measured as the spread between references mapped onto one
    # shape, over every object whose reference set is large enough to ask: the watermelon's
    # nineteen give 0.0734 to 0.0732, and the orange's five give 0.0540 to 0.0632 across and
    # 0.0377 to 0.0453 along -- seventeen and twenty percent worse. A version before this one
    # put a single knot on a boundary found by a threshold, and was worse still on the
    # watermelon: 0.0723 to 0.0908 just inside the boundary, because its outside is pith and
    # skin in proportions that vary as much as their total, so the threshold landed on the pith
    # in one photograph and the skin in another.
    #
    # The premise was wrong, not the implementation. These references disagree about colour,
    # everywhere at once and by 0.07, because they are photographs of different fruit, and no
    # reparametrisation of the radius reaches that -- while any reparametrisation resamples,
    # and resampling adds disagreement of its own. That is the whole of both results.
    #
    # The phase alignment above works because the cut angle really is the only thing wrong with
    # a rotated photograph, and undoing it costs nothing. There is no such quantity here.
    if PATH_MATCH and a_dst is not None:
        g_dst = _colour_path(a_dst, comp, dri, dro, dcy, dcx)
        g_src = _colour_path(a_ref, m_ref, sri, sro, scy, scx)
        if g_dst is not None and g_src is not None:
            nb_s = g_dst.shape[0]
            grid = np.arange(nb_s) / nb_s
            s = np.interp(np.interp(s, grid, g_dst), g_src, grid)
    rs = sri[ai] + s * (sro[ai] - sri[ai])
    # Sample the reference bilinearly, not at the nearest pixel. The map sends neighbouring
    # destination pixels to source positions that differ by a fraction of a pixel; rounding
    # that fraction quantises the map into bands, which is the second half of the streaking.
    fy = np.clip(scy + rs * np.sin(th), 0, a_ref.shape[0] - 1.001)
    fx = np.clip(scx + rs * np.cos(th), 0, a_ref.shape[1] - 1.001)
    # How the reference is sampled decides how much of its layering survives the map, and the
    # layering is the thing being copied: a watermelon's pith is a band a few pixels wide and
    # the whole difference between a section that reads right and one that does not is whether
    # its edge stays an edge. Bilinear interpolation is a box filter, so under the magnification
    # this map applies it spreads that edge over two or three pixels -- measured on the run this
    # was written for, the reference falls at 0.1478 per 0.02R and the target it produced fell
    # at 0.1059, twenty-eight percent of the layering gone before the model saw anything. The
    # model then reproduced its target to within seven percent, so nearly all of the softness
    # was made here and none of it was the model's.
    #
    # Nearest-neighbour keeps the edge and brings back the aliasing that made the map streak in
    # the first place. Cubic keeps most of the edge and stays smooth along the rays, which is
    # the combination this needs; `SEC_MAP_INTERP` is here so the choice can be re-measured
    # rather than believed.
    return _sample(a_ref, fy, fx)


def section_target(render, ref_rgb, alpha=None, min_frac=0.002, bg_tol=0.03, bg=None):
    """Target image for one rendered section, assembled component by component.

    render   (3, H, W) float tensor, on whatever SECTION_BG the run renders against
    ref_rgb  (h, w, 3) float array, the generated section reference
    alpha    (1, H, W) accumulated opacity from the rasteriser, if available

    A section is rendered on white, so "differs from white" is only a guess at where the
    primitives are, and it is wrong exactly where the section is pale. The orange's core is
    white: read as background it splits the disc into two half-moons, each of which then got
    the whole reference squeezed into it sideways, and the target came out flat orange with
    no radial structure at all. The rasteriser already returns the accumulated opacity next
    to the colour, which is the coverage itself and costs nothing to keep, so use it when the
    caller can supply it and fall back to the colour test when it cannot.
    """
    dev = render.device
    r = render.permute(1, 2, 0).detach().cpu().numpy()
    H, W, _ = r.shape
    if alpha is not None:
        fg = alpha.reshape(alpha.shape[-2], alpha.shape[-1]).detach().cpu().numpy() > 0.5
    else:
        # `SECTION_BG` lets the sections render against something other than white, so the
        # colour test has to be told what background it is looking for rather than assuming
        # the value it used to be
        _bg = float(os.environ.get("SECTION_BG", "1.0")) if bg is None else bg
        fg = np.abs(r - _bg).max(2) > bg_tol
    _rgb = np.asarray(ref_rgb, dtype=np.float32) / (255.0 if np.max(ref_rgb) > 1.5 else 1.0)
    _filled = _ref_info(_rgb, fill=True)
    _open = _ref_info(_rgb, fill=False)

    # Start from the render, not from white. A component too small to establish a
    # correspondence gets skipped, and against a white canvas "skipped" reads as "should be
    # background" -- the loss then works to delete it. On the loaf that was 5,689 pixels, a
    # fifth of the section's own area, told to disappear. Copying the render leaves those
    # pixels with no gradient instead of a wrong one, and costs nothing elsewhere: the
    # background already matches in both, whatever it is.
    tgt = r.copy()
    lab, k = ndimage.label(fg)
    if k == 0:
        return torch.from_numpy(tgt).permute(2, 0, 1).to(dev)
    min_px = max(int(H * W * min_frac), 64)

    for j in range(1, k + 1):
        comp = lab == j
        if comp.sum() < min_px:
            continue
        ys, xs = np.where(comp)
        # Read the reference the way this component does. Filling its enclosed regions is
        # right for a pale core and wrong for a hole, and only the render knows which it is
        # looking at, so try both and keep the one whose topology matches. Preferring the
        # filled reading leaves every object that has no hole exactly as it was.
        h_comp = _holes(comp)
        ref_info = next((ri for ri in (_filled, _open) if ri[3] == h_comp), None)
        if ref_info is not None:
            a_ref, m_ref = ref_info[0], ref_info[1]
            # same topology, so a bijection exists: map them by their own coordinates
            tgt[ys, xs] = _same_topology_map(comp, a_ref, m_ref, r)
        else:
            lut, nb = _filled[4], _filled[5]
            # no bijection exists; fall back to the one coordinate both shapes have
            dt = ndimage.distance_transform_edt(comp)
            dn = dt[ys, xs] / max(dt.max(), 1e-6)
            tgt[ys, xs] = lut[np.clip((dn * nb).astype(np.int64), 0, nb - 1)]
    return torch.from_numpy(tgt).permute(2, 0, 1).to(dev)
