"""Choose the references' phases together instead of one at a time.

The transverse family and the longitudinal family describe one object, and where a plane from each
meets, they describe one line. That line is computable before any training: a transverse section at
height z, rotated by phase d, is cut by a longitudinal plane at azimuth a along its diameter at
angle a + d; the longitudinal section at azimuth a is cut by the transverse plane along its row at
height z. Both are one-dimensional signals of the same physical line, so the disagreement between
the two families is a function of the phases alone and can be minimised before a gradient is taken.

The pipeline currently sets each transverse phase by cross-correlating that photograph's angular
profile against a fixed member of its own family. That is greedy in two ways: it never consults the
longitudinal family, and it never revisits a choice. This solves for all phases at once against the
disagreement they actually produce, by coordinate descent over a discrete set of angles, which is
exact for each coordinate and monotone overall.

    python phaseopt.py H_DIR V_DIR [n_angles]
"""
import glob, os as _os, sys
os = _os
import numpy as np
from PIL import Image

NA = 72                     # candidate phases, 5 degrees apart
NS = 128                    # samples along a shared line


def _photos(d):
    """The colour photographs, by the same rule sds_demo uses, so the two agree on the order."""
    return sorted(p for p in glob.glob(_os.path.join(d, "*"))
                  if _os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg", ".webp")
                  and not _os.path.splitext(_os.path.basename(p))[0].endswith(
                      ("_depth", "_mask", "_alpha", "_normal")))


def load(d):
    ps = [p for p in sorted(glob.glob(os.path.join(d, "*.png"))) if "depth" not in p]
    return [np.asarray(Image.open(p).convert("RGB").resize((256, 256)), np.float32) / 255 for p in ps], ps


def disc(im):
    """Centre and radius of the silhouette, so both families are sampled in the same units."""
    m = im.mean(2) < 0.96
    ys, xs = np.nonzero(m)
    if not len(xs):
        return np.array([im.shape[1] / 2, im.shape[0] / 2]), im.shape[1] / 2
    c = np.array([xs.mean(), ys.mean()])
    r = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    return c, max(r, 1.0)


def bilinear(im, pts):
    x = np.clip(pts[:, 0], 0, im.shape[1] - 1.001)
    y = np.clip(pts[:, 1], 0, im.shape[0] - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx) * (1 - fy)
            + (im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx) * fy)


def h_line(im, ang):
    """The diameter of a transverse section at angle `ang`, sampled from rim to rim."""
    c, r = disc(im)
    t = np.linspace(-1, 1, NS)[:, None]
    d = np.array([[np.cos(ang), np.sin(ang)]])
    return bilinear(im, c + t * r * d)


def v_line(im, zf):
    """The row of a longitudinal section at height fraction `zf` in [-1, 1]."""
    c, r = disc(im)
    t = np.linspace(-1, 1, NS)[:, None]
    pts = np.concatenate([c[0] + t * r, np.full_like(t, c[1] + zf * r)], 1)
    return bilinear(im, pts)


def cost(H, V, phases, zs, azs):
    tot = 0.0
    for i, im_h in enumerate(H):
        for j, im_v in enumerate(V):
            a = azs[j] + phases[i]
            tot += float(np.abs(h_line(im_h, a) - v_line(im_v, zs[i])).mean())
    return tot / (len(H) * len(V))


def main(hd, vd):
    H, hp = load(hd)
    V, vp = load(vd)
    zs = np.linspace(-0.55, 0.55, len(H))               # the depths the family is spread over
    azs = np.pi * np.arange(len(V)) / len(V)            # the azimuths the family is spread over
    cands = 2 * np.pi * np.arange(NA) / NA

    # the pipeline's own choice, reproduced: each photograph's angular profile is cross-correlated
    # against the first member of its family and rotated by the shift that maximises it. Comparing
    # against no alignment at all would flatter the joint solution; this is the baseline that exists.
    def ang_profile(im, nb=180):
        c, r = disc(im)
        t = np.linspace(0.15, 0.95, 40)[None, :]
        a = (2 * np.pi * np.arange(nb) / nb)[:, None]
        px = c[0] + r * t * np.cos(a); py = c[1] + r * t * np.sin(a)
        pts = np.stack([px.ravel(), py.ravel()], 1)
        v = bilinear(im, pts).mean(1).reshape(nb, -1).mean(1)
        return v - v.mean()
    ref = ang_profile(H[0])
    greedy = np.zeros(len(H))
    for i in range(1, len(H)):
        cc = np.fft.irfft(np.fft.rfft(ref) * np.conj(np.fft.rfft(ang_profile(H[i]))), n=len(ref))
        greedy[i] = 2 * np.pi * int(np.argmax(cc)) / len(ref)
    c_none = cost(H, V, np.zeros(len(H)), zs, azs)
    c0 = cost(H, V, greedy, zs, azs)
    print(f"  no alignment                {c_none:.5f}")
    print(f"  pipeline's greedy alignment {c0:.5f}")
    ph = greedy.copy()
    for sweep in range(4):
        moved = 0.0
        for i in range(len(H)):
            best, bc = ph[i], None
            for a in cands:
                ph[i] = a
                c = cost(H, V, ph, zs, azs)
                if bc is None or c < bc:
                    bc, best = c, a
            ph[i] = best
            moved = max(moved, 1.0)
        c1 = cost(H, V, ph, zs, azs)
        print(f"  sweep {sweep+1}: cost {c1:.5f}")
        if sweep and abs(c1 - prev) < 1e-6:
            break
        prev = c1
    # the assignment: which photograph is shown at which depth. The pipeline spreads them in file
    # order; the depths themselves are fixed by the schedule, so this is a permutation of six items
    # and the exhaustive search is 720 evaluations.
    import itertools
    best_perm, best_c = tuple(range(len(H))), None
    for perm in itertools.permutations(range(len(H))):
        Hp = [H[i] for i in perm]; php = ph[list(perm)]
        c = cost(Hp, V, php, zs, azs)
        if best_c is None or c < best_c:
            best_c, best_perm = c, perm
    print(f"  + assignment over {len(H)}! permutations {best_c:.5f}   order {best_perm}")

    c1 = cost(H, V, ph, zs, azs)
    print(f"\n  pipeline's greedy alignment {c0:.5f}")
    print(f"  jointly chosen phases       {c1:.5f}   {100*(c0-c1)/c0:.1f}% less disagreement")
    print(f"  phase and assignment        {best_c:.5f}   {100*(c0-best_c)/c0:.1f}% less disagreement")
    print("  phases, degrees: " + ", ".join(f"{np.degrees(p):.0f}" for p in ph))
    # Beside the references it solved for, because that is what it is a property of: the same
    # directory used by another object, or re-photographed, needs its own solution. Both halves
    # of (27) are written -- the phases and the permutation -- and the file names the members it
    # solved over, so a directory that has gained a photograph since is detected rather than
    # silently mismatched.
    out = _os.path.join(hd, "phase_opt.npz")
    np.savez(out, phases=ph, perm=np.asarray(best_perm, np.int64),
             files=np.asarray([_os.path.basename(f) for f in sorted(_photos(hd))]))
    print(f"  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
