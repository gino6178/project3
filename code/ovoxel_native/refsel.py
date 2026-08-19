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


def _blend_canonical(path, size=512, frac=0.38):
    im = Image.open(path).convert("RGB")
    cy, cx, r = _disc(im)
    scale = (frac * size) / max(r, 1e-6)
    w, h = im.size
    im2 = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    out = Image.new("RGB", (size, size), (255, 255, 255))
    out.paste(im2, (int(round(size / 2 - cx * scale)), int(round(size / 2 - cy * scale))))
    return np.asarray(out, np.float32) / 255.0


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


def phase_aligned(spec, idx, n):
    """Equation (11): this plane's own photograph, turned to the family's angular phase.

    The greedy per-family alignment sds_demo falls through to when nothing has been solved. It is
    here so that an object with no phase_opt.npz runs the same experiment this one did -- falling
    through to `photo`, with no alignment at all, would quietly be a different method.
    """
    files = sorted(photos_in(spec))
    k = (idx * len(files) // max(n, 1)) % len(files)
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


def solved_photo(spec, idx, n):
    """The transverse family: equation (27)'s assignment, at its solved phase."""
    got = _solved(spec)
    files = sorted(photos_in(spec))
    if got is None:
        return phase_aligned(spec, idx, n)
    phases, perm = got
    k = (idx * len(files) // max(n, 1)) % len(files)
    k = int(perm[k]) if k < len(perm) else k
    key = (spec, k, "solved")
    if key in _PHOTOS:
        return _PHOTOS[key]
    deg = float(np.degrees(phases[k])) if k < len(phases) else 0.0
    img = Image.open(files[k]).convert("RGB")
    if abs(deg) > 1e-6:
        img = img.rotate(-deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    _PHOTOS[key] = img
    return img


def photo(spec, idx, n):
    """The longitudinal family: the continuous depth assignment, equation (14)."""
    files = sorted(photos_in(spec))
    if os.environ.get("REF_DEPTH_BLEND", "1") == "1" and len(files) > 1:
        t = idx * len(files) / max(n, 1)
        k0 = int(t) % len(files)
        k1 = (k0 + 1) % len(files)
        w = float(t - int(t))
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
