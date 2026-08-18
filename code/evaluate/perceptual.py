"""PSNR, SSIM and LPIPS on the cut planes -- the metrics the specification actually asked for.

Section 11.2's first line is "Interior appearance: PSNR / SSIM / LPIPS on training and held-out
cut planes". The project reports FID, KID and CLIP instead, computed on six renders against six
photographs -- a sample size at which the reference sets score 46 to 70 against *themselves* split
in half, so a margin of a few points carries nothing.

These three are paired: each render is compared against one image rather than against the
distribution of a set. That removes the sample-size problem entirely, at the cost of needing a
correspondence, which the held-out cuts do not have -- a cut at an arbitrary depth is not a
photograph of any particular slice. So the honest use is the one the specification names first,
*training* planes, where the reference the model was fitted to is known, plus a nearest-reference
reading on held-out planes that is reported as what it is: an upper bound on agreement, not a
measure of held-out quality.

    python method/common/eval/perceptual.py REF_DIR RENDER_DIR [pattern]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]


def _load(path, size):
    import cv2
    a = cv2.imread(path)
    if a is None:
        return None
    return cv2.resize(a[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def ssim(a, b):
    """The standard windowed form, on luminance, with the usual constants."""
    import cv2
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
    saa = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a * mu_a
    sbb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b * mu_b
    sab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(s.mean())


def _lpips_fn():
    try:
        import torch
        import lpips as _l
        net = _l.LPIPS(net="alex").cuda().eval()

        def f(a, b):
            ta = torch.from_numpy(a).permute(2, 0, 1)[None].cuda() * 2 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None].cuda() * 2 - 1
            with torch.no_grad():
                return float(net(ta, tb).item())
        return f
    except Exception as e:                                       # noqa: BLE001
        print(f"  LPIPS unavailable ({type(e).__name__}); reporting PSNR and SSIM only")
        return None


def main(ref_dir, render_dir, pattern="rh*_init_0.png", size=512):
    refs = [p for p in sorted(glob.glob(_os.path.join(ref_dir, "*")))
            if p.lower().endswith((".png", ".jpg", ".jpeg")) and not p.endswith("_depth.png")]
    rend = sorted(glob.glob(_os.path.join(render_dir, pattern)))
    if not refs or not rend:
        raise SystemExit(f"nothing to compare: {len(refs)} references, {len(rend)} renders")
    R = [x for x in (_load(p, size) for p in refs) if x is not None]
    F = [x for x in (_load(p, size) for p in rend) if x is not None]
    lp = _lpips_fn()

    # Each render against its *best* reference. On held-out cuts there is no correspondence to
    # use, so this is an upper bound on agreement rather than a paired score, and it is reported
    # as such -- a number that can only flatter is still comparable between two models measured
    # the same way.
    rows = []
    for i, f in enumerate(F):
        best = max(range(len(R)), key=lambda j: psnr(f, R[j]))
        rows.append((psnr(f, R[best]), ssim(f, R[best]),
                     lp(f, R[best]) if lp else float("nan"), best))
    p = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    l = np.array([r[2] for r in rows])
    print(f"  {len(F)} renders against {len(R)} references, each to its nearest")
    print(f"    PSNR  {p.mean():6.2f} dB  (min {p.min():.2f}, max {p.max():.2f})")
    print(f"    SSIM  {s.mean():6.4f}     (min {s.min():.4f}, max {s.max():.4f})")
    if lp:
        print(f"    LPIPS {np.nanmean(l):6.4f}     (min {np.nanmin(l):.4f}, max {np.nanmax(l):.4f})")
    # the same three between the references themselves, which is the floor these sit against
    if len(R) > 1:
        pp, ss, ll = [], [], []
        for i in range(len(R)):
            j = max((k for k in range(len(R)) if k != i), key=lambda k: psnr(R[i], R[k]))
            pp.append(psnr(R[i], R[j])); ss.append(ssim(R[i], R[j]))
            if lp:
                ll.append(lp(R[i], R[j]))
        print(f"    -- the references against each other: PSNR {np.mean(pp):.2f} dB, "
              f"SSIM {np.mean(ss):.4f}" + (f", LPIPS {np.mean(ll):.4f}" if lp else ""))
    return rows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "rh*_init_0.png")
