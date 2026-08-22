"""Which photograph supervises which plane, and at what angle.

Lifted from sds_demo.py rather than reinvented, because the assignment is the thing equation (27)
solves and a second implementation of it is a second answer.  Three pieces:

  _solved_photo   REF_PHASE_MODE=solve, the transverse family: phase_opt.npz gives a permutation
                  (which photograph belongs at which depth) and a phase per photograph (how far to
                  turn it), both minimised against the longitudinal family's chords before any
                  gradient is taken.  The orange's is already solved and is used as it stands.
  _photo          the longitudinal family, which the phases do not cover -- sds_demo only applies
                  them when view_cut == "horizontal".  REF_DEPTH_BLEND=1 by default, so a plane at
                  t = idx * M / N takes photograph floor(t) and the next, both brought onto a
                  common disc, mixed at the fractional part.
  _photos_in      colour photographs only; the _depth maps beside them are not references.

Nothing here samples a diffusion model: REF_WARMUP is 10^7 in stage_train, so the photograph is
the target for the whole run and the released regeneration path is never reached.
"""
import glob
import os

import numpy as np
from PIL import Image

_PHOTOS = {}
# REF_SAMPLE draws a photograph instead of averaging two of them; see `_depth_pick`. The common disc
# is kept either way -- re-centring and re-scaling each photograph is a separate change from mixing
# them, and it was the half of the blend that moved the held-out probe.
import random as _random
SAMPLE = os.environ.get("REF_SAMPLE", "0") == "1"
_RNG = _random.Random(int(os.environ.get("REF_SAMPLE_SEED", "0")))


def photos_in(spec):
    return [p for p in glob.glob(os.path.join(spec, "*"))
            if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg", ".webp")
            and not os.path.splitext(os.path.basename(p))[0].endswith(
                ("_depth", "_mask", "_alpha", "_normal"))]


def _disc(img):
    """Centre and radius of the object against its own border colour."""
    from scipy import ndimage
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.
    h, w, _ = a.shape
    k = max(2, h // 20)
    bg = np.median(np.concatenate([a[:k].reshape(-1, 3), a[-k:].reshape(-1, 3)]), 0)
    m = ndimage.binary_fill_holes(np.abs(a - bg).max(2) >= 0.10)
    if m.sum() < 100:
        return (h / 2, w / 2, h / 4)
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return (cy, cx, float(np.percentile(rr, 98)))


def _canonical(im, size=512, frac=0.38):
    """One photograph on a common disc: its section centred and scaled to a fixed radius.

    Two photographs of two different oranges are framed differently, so mixing them as they sit
    mixes a section with its neighbour's background. On the disc they are the same size in the
    same place and the mix is between the sections.
    """
    cy, cx, r = _disc(im)
    scale = (frac * size) / max(r, 1e-6)
    w, h = im.size
    im2 = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    out = Image.new("RGB", (size, size), (255, 255, 255))
    out.paste(im2, (int(round(size / 2 - cx * scale)), int(round(size / 2 - cy * scale))))
    return np.asarray(out, np.float32) / 255.0


def _blend_canonical(path, size=512, frac=0.38):
    return _canonical(Image.open(path).convert("RGB"), size, frac)


def _blend_images(a, b, w):
    """Two photographs already in hand, mixed on the common disc."""
    m = np.clip((1.0 - w) * _canonical(a) + w * _canonical(b), 0, 1)
    return Image.fromarray((m * 255).astype("uint8"))


def _blend_on_disc(pa, pb, w):
    a, b = _blend_canonical(pa), _blend_canonical(pb)
    return Image.fromarray((np.clip((1.0 - w) * a + w * b, 0, 1) * 255).astype("uint8"))


def _solved(spec):
    f = os.path.join(spec, "phase_opt.npz")
    if not os.path.isfile(f):
        return None
    z = np.load(f, allow_pickle=False)
    now = [os.path.basename(p) for p in sorted(photos_in(spec))]
    was = [str(x) for x in z["files"]]
    if now != was:
        print(f"  phase_opt.npz is stale for {spec}: {len(was)} solved, {len(now)} present")
        return None
    return z["phases"], z["perm"]


def _angular_profile(img, nb=360, r_lo=0.25, r_hi=0.80):
    """Mean brightness as a function of angle, over the annulus the walls live in."""
    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.
    m = a < 0.98
    if m.sum() < 400:
        return None
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    R = np.sqrt(m.sum() / np.pi)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    rr = np.hypot(yy - cy, xx - cx) / R
    th = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)
    sel = m & (rr > r_lo) & (rr < r_hi)
    if sel.sum() < 400:
        return None
    b = (th[sel] * nb).astype(int) % nb
    prof = np.bincount(b, a[sel], minlength=nb) / np.maximum(np.bincount(b, minlength=nb), 1)
    prof = prof - prof.mean()
    nn = np.linalg.norm(prof)
    return prof / nn if nn > 1e-9 else None


def _depth_pick(spec, idx, n, one):
    """This depth's target, from whichever per-photograph rule `one` is.

    `REF_TRANS_BLEND` extends the continuous depth assignment to the transverse family. It had
    only ever been given to the longitudinal one -- `photo` blends and this family did not -- and
    the rule here is `idx * len(files) // n`, integer division, so the photograph changes abruptly
    at len(files) - 1 depths and every plane between two changes is supervised by whichever
    photograph its side of the switch fell on.

    Measured on the O-Voxel arms, that is what the horizontal banding in a longitudinal cut is:
    the mean axial profile of `r1_free` steps at 0.500, 0.658 and 0.816 against switches at
    0.500, 0.667 and 0.833, while the pipeline's own profile over the same six held-out planes
    has no step there. The transverse planes stack along the polar axis, so each switch draws a
    line across it and a longitudinal cut crosses every one.

        1  the two photographs either side of the depth, mixed at the fractional part (default)
        2  the nearer photograph alone, but on the same common disc as 1
        0  the block rule, and what every arm before r1_tb1 was trained under

    2 exists because 1 changes two things at once: it interpolates, and it re-centres and
    re-scales each photograph onto a common disc, which is a change to the target even where the
    mixing weight is 0 or 1. Without 2 an improvement cannot be attributed to either -- and it
    could not have been. Measured against r1_pin_full on the orange, the jump at the five switch
    depths, in units of the profile's own median jump, is 5.14 for the block rule, 5.20 for the
    disc alone and 3.78 for the blend, against the pipeline's 3.38: the re-framing does nothing
    for the banding. What the re-framing does do is the held-out probe, 0.03081 -> 0.03023, and
    the blend carries that further to 0.02958, lowest at every checkpoint after the twentieth.

    1 by default from that measurement. It does not touch the other defect in the same picture:
    the vertical streaks are a cut plane meeting a per-cell field on an axis-aligned lattice,
    they are identical in all three arms, and no assignment rule reaches them.

    It wraps the rule rather than living inside one, because which rule is in force is not the
    caller's choice to make: `REF_PHASE_MODE=solve` falls through to the greedy alignment whenever
    a reference set has no `phase_opt.npz`, and none of them on this box has one, so every arm so
    far has run `phase_aligned` under a label that says `solved_photo`.
    """
    files = sorted(photos_in(spec))
    L = len(files)
    mode = os.environ.get("REF_TRANS_BLEND", "1")
    if mode not in ("1", "2") or L < 2:
        return one((idx * L // max(n, 1)) % L)
    # The stack is a stack and not a cycle. `idx * L / n` with the pair taken modulo L sends the
    # last segment back to the first photograph, so the deepest planes are supervised by the
    # shallowest section: on the apple, with two references and sixteen planes, planes 8 to 15
    # blend 1 -> 0 while planes 0 to 7 blend 0 -> 1, and the assignment runs backwards over half
    # the object. Spanning [0, L-1] instead puts the first plane on the first photograph and the
    # last on the last, monotonically, and agrees with the block rule at both ends.
    t = idx * (L - 1) / max(n - 1, 1)
    j0 = min(int(t), L - 2)
    j1 = j0 + 1
    w = float(t - j0)
    if mode == "2":
        w = float(round(w))
    if SAMPLE:
        # Draw one of the two rather than mixing them, with probability equal to the mixing weight.
        #
        # The expectation is the blend, so the depth assignment is unchanged in the mean and the
        # banding the blend was introduced to remove stays removed. What changes is that the target
        # on any given step is a photograph rather than the average of two, and an average of two
        # sections of two different oranges is a picture neither of them is. Measured on the
        # families: the blended target carries 86% of the transverse photographs' gradient on the
        # orange, 90% on the watermelon, 97% on the apple. That is structure the loss can never ask
        # for because it is not in the target.
        #
        # A plane sees both photographs over a run, at the right proportions, so nothing is lost --
        # it is the difference between fitting a mean and fitting a sample from a distribution
        # whose mean is the same. The variance it adds is what the field prior is there to absorb.
        j = j1 if _RNG.random() < w else j0
        key = (spec, "tbs", j)
        if key not in _PHOTOS:
            _PHOTOS[key] = Image.fromarray(
                (_canonical(one(j)) * 255).astype("uint8"))
        return _PHOTOS[key]
    key = (spec, "tb", mode, j0, j1, round(w, 3))
    if key not in _PHOTOS:
        _PHOTOS[key] = _blend_images(one(j0), one(j1), w)
    return _PHOTOS[key]


def phase_aligned(spec, idx, n):
    """Equation (11): this plane's own photograph, turned to the family's angular phase.

    The greedy per-family alignment sds_demo falls through to when nothing has been solved. It is
    here so that an object with no phase_opt.npz runs the same experiment this one did -- falling
    through to `photo`, with no alignment at all, would quietly be a different method.
    """
    return _depth_pick(spec, idx, n, lambda k: _phase_one(spec, k))


def _phase_one(spec, k):
    """Photograph k of the family, turned to the phase of the first."""
    files = sorted(photos_in(spec))
    key = (spec, k, "phase")
    if key in _PHOTOS:
        return _PHOTOS[key]
    ref_key = (spec, 0, "phase")
    ref = _PHOTOS.get(ref_key)
    if ref is None:
        ref = Image.open(files[0]).convert("RGB")
        _PHOTOS[ref_key] = ref
    if k == 0:
        return ref
    img = Image.open(files[k]).convert("RGB")
    pr, pi = _angular_profile(ref), _angular_profile(img)
    if pr is None or pi is None:
        _PHOTOS[key] = img
        return img
    cc = np.fft.irfft(np.fft.rfft(pr) * np.conj(np.fft.rfft(pi)), n=len(pr))
    deg = 360.0 * int(np.argmax(cc)) / len(pr)
    out = img.rotate(-deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    _PHOTOS[key] = out
    return out


def _solved_one(spec, files, phases, perm, j):
    """Photograph j of the transverse family, turned to its own solved phase."""
    k = int(perm[j]) if j < len(perm) else j
    key = (spec, k, "solved")
    if key not in _PHOTOS:
        deg = float(np.degrees(phases[k])) if k < len(phases) else 0.0
        img = Image.open(files[k]).convert("RGB")
        if abs(deg) > 1e-6:
            img = img.rotate(-deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        _PHOTOS[key] = img
    return _PHOTOS[key]


def solved_photo(spec, idx, n):
    """The transverse family: equation (27)'s assignment, at its solved phase.

    `REF_TRANS_BLEND` extends the continuous depth assignment to this family. It had only ever
    been given to the longitudinal one -- `photo` blends and this did not -- and the assignment
    here is `idx * len(files) // n`, integer division, so the photograph changes abruptly at
    len(files) - 1 depths and every plane between two changes is supervised by whichever
    photograph its side of the switch fell on.

    Measured on the O-Voxel arms, that is what the horizontal banding in a longitudinal cut is:
    the mean axial profile of `r1_free` steps at 0.500, 0.658 and 0.816 against switches at
    0.500, 0.667 and 0.833, while the pipeline's own profile over the same six held-out planes
    has no step there. The transverse planes stack along the polar axis, so each switch draws a
    line across it and a longitudinal cut crosses every one.

        1  the two photographs either side of the depth, mixed at the fractional part (default)
        2  the nearer photograph alone, but on the same common disc as 1
        0  the block rule, and what every arm before r1_tb1 was trained under

    2 exists because 1 changes two things at once: it interpolates, and it re-centres and
    re-scales each photograph onto a common disc, which is a change to the target even where the
    mixing weight is 0 or 1. Without 2 an improvement cannot be attributed to either -- and it
    could not have been. Measured against r1_pin_full on the orange, the jump at the five switch
    depths, in units of the profile's own median jump, is 5.14 for the block rule, 5.20 for the
    disc alone and 3.78 for the blend, against the pipeline's 3.38: the re-framing does nothing
    for the banding. What the re-framing does do is the held-out probe, 0.03081 -> 0.03023, and
    the blend carries that further to 0.02958, lowest at every checkpoint after the twentieth.

    1 by default from that measurement. It does not touch the other defect in the same picture:
    the vertical streaks are a cut plane meeting a per-cell field on an axis-aligned lattice,
    they are identical in all three arms, and no assignment rule reaches them.
    """
    got = _solved(spec)
    if got is None:
        return phase_aligned(spec, idx, n)
    phases, perm = got
    files = sorted(photos_in(spec))
    return _depth_pick(spec, idx, n,
                       lambda j: _solved_one(spec, files, phases, perm, j))


def photo(spec, idx, n):
    """The longitudinal family: the continuous depth assignment, equation (14)."""
    files = sorted(photos_in(spec))
    if os.environ.get("REF_DEPTH_BLEND", "1") == "1" and len(files) > 1:
        t = idx * len(files) / max(n, 1)
        k0 = int(t) % len(files)
        k1 = (k0 + 1) % len(files)
        w = float(t - int(t))
        if SAMPLE:
            k = k1 if _RNG.random() < w else k0
            key = (spec, "blends", k)
            if key not in _PHOTOS:
                _PHOTOS[key] = Image.fromarray(
                    (_blend_canonical(files[k]) * 255).astype("uint8"))
            return _PHOTOS[key]
        key = (spec, "blend", k0, k1, round(w, 3))
        if key not in _PHOTOS:
            _PHOTOS[key] = _blend_on_disc(files[k0], files[k1], w)
        return _PHOTOS[key]
    k = (idx * len(files) // max(n, 1)) % len(files)
    key = (spec, k)
    if key not in _PHOTOS:
        _PHOTOS[key] = Image.open(files[k]).convert("RGB")
    return _PHOTOS[key]


def as_array(img, res):
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    return np.asarray(img.convert("RGB"), np.float32) / 255.0
