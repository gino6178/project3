"""Isolate the fruit in each cube reference and drop the shadow around it.

Measuring the radial profile showed what the darkening actually is. On the orange's front
face the fruit only reaches 0.65 of the detected radius; past it the mask is empty, and then
there is another ring of pixels out at 0.88 to 1.00. So the generation drew the fruit small
and laid a shadow ring around it, detached from the fruit by clear background. Nothing dark
is on the fruit at all -- the shadow is a separate object in the picture.

That means no tone correction is called for; the earlier attempt at one moved the edge ratio
by five percent and introduced masking artefacts, because it was correcting a thing that was
not there. Taking the largest connected component keeps the fruit and discards the ring
outright, which is exact rather than approximate.

Then rescale so the fruit fills the frame. The projection samples each face over the cone it
owns -- 0.816 of the radius, from the cube-corner angle -- and that fraction has to be
measured against the fruit, not against a frame the fruit occupies two thirds of.
"""
import sys, os, argparse
import numpy as np
import cv2
from scipy import ndimage

# Whatever the generator wrote, not a list fixed here. A hard-coded fourteen silently dropped
# the twenty-six scattered directions of a thirty-two direction set, and the run that used the
# output looked like a fair test of scattering while being six faces.
FACES = None


def fruit_mask(a, bg_tol=0.10, erode=3):
    """The fruit alone: the largest connected component, minus its boundary pixels."""
    bg = np.median(np.concatenate([a[:10].reshape(-1, 3), a[-10:].reshape(-1, 3)]), 0)
    m = np.abs(a - bg).max(2) > bg_tol
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_erosion(m, iterations=erode)
    return m


def prep(path, out_path, size=512):
    a = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    m = fruit_mask(a)
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    R = float(np.percentile(rr, 99))

    # crop a square around the fruit and resize so it fills the frame
    half = int(np.ceil(R)) + 1
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    pad = max(0, -y0, -x0, y1 - a.shape[0], x1 - a.shape[1])
    if pad:
        a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), constant_values=1.0)
        m = np.pad(m, pad)
        y0 += pad; y1 += pad; x0 += pad; x1 += pad
    ca = a[y0:y1, x0:x1].copy()
    cm = m[y0:y1, x0:x1]
    ca[~cm] = 1.0                                  # everything but the fruit is background
    out = cv2.resize(ca, (size, size), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(out_path, cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_RGB2BGR) * 255)

    # how much of the frame the fruit occupied before, and the darkness that got dropped
    a2 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    bg = np.median(np.concatenate([a2[:10].reshape(-1, 3), a2[-10:].reshape(-1, 3)]), 0)
    all_obj = ndimage.binary_fill_holes(np.abs(a2 - bg).max(2) > 0.10)
    dropped = all_obj & ~m[pad:pad + a2.shape[0], pad:pad + a2.shape[1]] if pad else \
        all_obj & ~m
    lum = a2.mean(2)
    return (R / (a2.shape[0] / 2), int(dropped.sum()),
            float(lum[dropped].mean()) if dropped.sum() else float("nan"),
            float(lum[m[pad:pad + a2.shape[0], pad:pad + a2.shape[1]] if pad else m].mean()))


def main(src_dir, out_dir, size=512):
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'面':<8}{'水果原本佔半徑':>16}{'丟掉的像素':>12}{'丟掉的平均亮度':>16}{'水果平均亮度':>14}")
    print("-" * 68)
    import glob as _g, shutil as _sh
    for _m in ("dirs.json", "prompts.json"):
        if os.path.exists(os.path.join(src_dir, _m)):
            _sh.copy(os.path.join(src_dir, _m), os.path.join(out_dir, _m))
    names = ([f[:-len("_ref.png")] for f in sorted(
                os.path.basename(x) for x in _g.glob(os.path.join(src_dir, "*_ref.png")))]
             if FACES is None else FACES)
    for f in names:
        p = os.path.join(src_dir, f"{f}_ref.png")
        if not os.path.exists(p):
            continue
        frac, nd, ld, lf = prep(p, os.path.join(out_dir, f"{f}_ref.png"), size)
        print(f"{f:<8}{frac:>16.2f}{nd:>12,}{ld:>16.3f}{lf:>14.3f}")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    main(a.src, a.out, a.size)
