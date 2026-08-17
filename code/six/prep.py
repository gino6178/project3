"""Step 2: put every reference on the white background the pipeline assumes, and repoint the confs.

`_fit_disc` in sds_demo.py locates a reference's slice by thresholding against white, so a
photograph shot on a dark or textured backdrop has no background as far as the fit is concerned:
the "disc" is the whole frame, its radius is the frame's, and the render is aligned to the wrong
circle. Eight of the released references are like this -- cake 3/3, pomegranate vertical 4/5,
apple horizontal 1/9 -- and the measured radius inflation on the cake is 1.25x, 1.53x and 1.82x.

The backdrop is removed by a flood from the frame border in chromaticity rather than RGB: a wood
backdrop varies by a factor of two in brightness while holding its hue, so an RGB ball around the
border's median colour covers only its lighter half. Luminance is kept as a second test so a dark
backdrop is not confused with a dark object of the same hue. The flooded region is then dilated by
two pixels, because the pixels where a black backdrop meets the slice are a blend of the two, lie
outside any tolerance around either, and survive a flood alone as a dark contour ringing the object
that the section loss then tries to reproduce.

Files already on white are copied unchanged, so each output directory is a drop-in replacement for
the one it came from and no object loses a reference. Idempotent: rerunning rebuilds refs_white/
and leaves the confs as they are.

    /usr/bin/python3 six/prep.py [--check]

`--check` audits and prints without writing anything.
"""
import glob, os, re, shutil, sys
from collections import deque

import numpy as np
from PIL import Image

ROOT = os.environ.get("FN_ROOT", "/workspace/fn_voxel")
CHROMA_TOL = 0.055      # chromaticity units
LUM_TOL = 0.30          # luminance units on [0, 1]
GROW = 2                # pixels of background dilation, to absorb the antialiased seam
WHITE = 0.93            # a border median above this is already white

# Which conf keys point at which directory, for the objects that have a non-white reference.
FIX = {
    "cake":        (["REF_H", "REF_V", "EVAL_REF", "EVAL_REF_V"], "cake"),
    "pomegranate": (["REF_V", "EVAL_REF_V"], "pomegranate/vertical"),
    "apple":       (["REF_H", "EVAL_REF"], "apple/horizontal"),
}
NOTE = ("#\n"
        "# References repointed to refs_white/. Some of this object's photographs were shot on a\n"
        "# dark backdrop; the section loss finds the slice by thresholding against white, so the\n"
        "# whole frame read as one disc and the render was aligned to the frame's radius rather\n"
        "# than the slice's. refs_white/ is the same set with the backdrop flooded to white.\n")


def photos(d):
    return [p for p in sorted(glob.glob(os.path.join(d, "*.png"))) if "depth" not in p]


def chroma(a):
    return (a / (a.sum(2, keepdims=True) + 1e-6))[..., :2]


def disc_fraction(a):
    """Fraction of the frame that reads as slice. Above ~0.9 means the backdrop is being counted."""
    return float((a.mean(2) < 0.95).mean())


def flood_bg(a):
    """Return (whitened image or None if already white, border colour, fraction removed)."""
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(edge, 0)
    if bg.mean() > WHITE:
        return None, bg, 0.0
    cand = ((np.linalg.norm(chroma(a) - chroma(bg[None, None])[0, 0][None, None], axis=2) <= CHROMA_TOL)
            & (np.abs(a.mean(2) - bg.mean()) <= LUM_TOL))
    H, W = cand.shape
    seen = np.zeros((H, W), bool)
    q = deque()
    for x in range(W):
        for y in (0, H - 1):
            if cand[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    for y in range(H):
        for x in (0, W - 1):
            if cand[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and cand[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))
    for _ in range(GROW):
        g = seen.copy()
        g[1:] |= seen[:-1]; g[:-1] |= seen[1:]
        g[:, 1:] |= seen[:, :-1]; g[:, :-1] |= seen[:, 1:]
        seen = g
    out = a.copy()
    out[seen] = 1.0
    return out, bg, float(seen.mean())


def audit():
    """Every reference directory, and which files are not on white."""
    bad = []
    pats = [os.path.join(ROOT, "data_finetune_images", "*", "*.png"),
            os.path.join(ROOT, "data_finetune_images", "*", "*", "*.png")]
    for p in sorted(sum((glob.glob(x) for x in pats), [])):
        if "depth" in p:
            continue
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        c = np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3),
                            a[-8:, :8].reshape(-1, 3), a[-8:, -8:].reshape(-1, 3)]).mean(0)
        f = disc_fraction(a)
        if c.mean() < 0.92 or f > 0.90:
            bad.append((os.path.relpath(p, ROOT), c, f))
    return bad


def whiten_dir(rel):
    src = os.path.join(ROOT, "data_finetune_images", rel)
    dst = os.path.join(ROOT, "refs_white", rel)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for p in photos(src):
        b = os.path.basename(p)
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        before = disc_fraction(a)
        out, bg, frac = flood_bg(a)
        if out is None:
            shutil.copy(p, os.path.join(dst, b))
            print(f"   {b:22s} already white                        disc {100*before:5.1f}%")
            continue
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(os.path.join(dst, b))
        after = disc_fraction(np.asarray(Image.open(os.path.join(dst, b)).convert("RGB"), np.float32) / 255)
        print(f"   {b:22s} backdrop {bg.round(2)} removed {100*frac:5.1f}%   "
              f"disc {100*before:5.1f}% -> {100*after:5.1f}%")
    n = len(photos(dst))
    assert n == len(photos(src)), f"{rel}: {n} out of {len(photos(src))} references survived"
    return n


def repoint(obj, keys, rel):
    p = os.path.join(ROOT, "method/objects", f"{obj}.conf")
    s = open(p).read()
    for k in keys:
        s = re.sub(rf"^{k}=.*$", f"{k}=refs_white/{rel}", s, flags=re.M)
    if "refs_white" not in s.split("SRC=")[0]:
        s = s.replace("SRC=", NOTE + "SRC=", 1)
    open(p, "w").write(s)
    print(f"   {obj}: {', '.join(keys)} -> refs_white/{rel}")


def main(check_only=False):
    bad = audit()
    print(f"{len(bad)} references are not on white:")
    for rel, c, f in bad:
        print(f"   {rel:46s} corner {c.round(2)}  disc {100*f:5.1f}%")
    if check_only:
        return 0 if not bad else 1
    for obj, (keys, rel) in FIX.items():
        print(f"\n{obj}")
        whiten_dir(rel)
        repoint(obj, keys, rel)
    print("\nre-auditing refs_white/")
    for _, (_, rel) in FIX.items():
        for p in photos(os.path.join(ROOT, "refs_white", rel)):
            a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
            f = disc_fraction(a)
            assert f < 0.90, f"{p} still fills {100*f:.1f}% of the frame"
    print("   every whitened reference is below 90% frame coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
