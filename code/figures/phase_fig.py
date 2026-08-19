"""How a family of cross-section photographs is brought to one angular phase.

Equations (11) and (27) are two ways of choosing the same thing -- the angle each reference is
turned through before it becomes a target -- and the paper states what they do without showing
one worked example. This is that example, on a family the pipeline actually trains on, drawn by
calling `sds_demo`'s own functions rather than reimplementing them, so the figure and the
training cannot disagree about what was aligned.

    python phase_fig.py secref_orraw_hsep out/phase_align.png

Four rows:

  a  the photographs as they arrive, in the order the directory sorts them.
  b  each one's angular profile -- mean brightness on the annulus the segment walls live in,
     which is the only thing in a transverse section that is a function of angle alone. The
     first photograph's profile is drawn behind each of the others, so the shift between them
     is the thing being measured and not an abstraction.
  c  the circular cross-correlation against that first profile, with its maximum marked. That
     maximum is the shift of equation (11): one number per photograph, in degrees.
  d  the photographs turned by it. What (11) does is a rotation and nothing else, so the seeds,
     the pith and the flesh each photograph happens to show all survive at a different angle.

If the family has a solved `phase_opt.npz` beside it, row d also carries equation (27)'s angle
for comparison, and the caption prints both. They are not the same problem: (11) aligns a family
to its own first member, and (27) chooses the angles and the depth assignment together so that
the two families agree where their planes cross. Where they disagree it is (27) that training
used, because REF_PHASE_MODE defaults to solve.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from PIL import Image                                                 # noqa: E402

from sds_demo import _angular_profile, _photos_in, _solved            # noqa: E402


def shift_of(prof_ref, prof):
    """Equation (11): the circular cross-correlation's argmax, in degrees."""
    cc = np.fft.irfft(np.fft.rfft(prof_ref) * np.conj(np.fft.rfft(prof)), n=len(prof_ref))
    m = int(np.argmax(cc))
    return 360.0 * m / len(prof_ref), cc, m


def main(spec, out):
    files = sorted(_photos_in(spec))
    if len(files) < 2:
        raise SystemExit(f"{spec} holds {len(files)} photographs; alignment needs at least two")
    imgs = [Image.open(f).convert("RGB") for f in files]
    profs = [_angular_profile(im) for im in imgs]
    if any(p is None for p in profs):
        raise SystemExit("a photograph has too little of the annulus to profile")

    greedy, ccs = [], []
    for p in profs:
        d, cc, m = shift_of(profs[0], p)
        greedy.append(d); ccs.append((cc, m))

    got = _solved(spec)
    solved = None
    if got is not None:
        phases, perm = got
        solved = [float(np.degrees(phases[k])) % 360 for k in range(len(files))]

    n = len(files)
    fig, ax = plt.subplots(4, n, figsize=(2.05 * n, 8.6),
                           gridspec_kw=dict(height_ratios=[1.25, 0.85, 0.85, 1.25]))
    if n == 1:
        ax = ax.reshape(4, 1)
    deg = np.arange(360)

    for k in range(n):
        ax[0, k].imshow(imgs[k]); ax[0, k].set_axis_off()
        ax[0, k].set_title(os.path.basename(files[k]), fontsize=7)

        ax[1, k].plot(deg, profs[0], lw=0.8, color="0.72",
                      label="the first, for reference" if k == 0 else None)
        ax[1, k].plot(deg, profs[k], lw=1.0, color="#c0392b")
        ax[1, k].set_xlim(0, 360); ax[1, k].set_xticks([0, 180, 360])
        ax[1, k].tick_params(labelsize=6)
        if k: ax[1, k].set_yticklabels([])

        cc, m = ccs[k]
        ax[2, k].plot(np.arange(len(cc)) * 360.0 / len(cc), cc, lw=0.9, color="#4a7ba7")
        ax[2, k].axvline(greedy[k], color="#c0392b", lw=1.0, ls="--")
        ax[2, k].set_xlim(0, 360); ax[2, k].set_xticks([0, 180, 360])
        ax[2, k].tick_params(labelsize=6)
        if k: ax[2, k].set_yticklabels([])
        ax[2, k].set_title(f"(11): {greedy[k]:.0f}°", fontsize=7)

        turned = imgs[k].rotate(-greedy[k], resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        ax[3, k].imshow(turned); ax[3, k].set_axis_off()
        t = f"turned {greedy[k]:.0f}°"
        if solved is not None:
            t += f"\n(27) chose {solved[k]:.0f}°"
        ax[3, k].set_title(t, fontsize=7)

    for r, lab in enumerate(["the photographs as they arrive",
                             "angular profile, the first drawn behind each",
                             "circular cross-correlation with the first",
                             "turned to the family's phase"]):
        ax[r, 0].set_ylabel(lab, fontsize=7.5)
        if r in (0, 3):
            ax[r, 0].set_axis_on(); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
            for s in ax[r, 0].spines.values():
                s.set_visible(False)

    sub = f"{spec}: {n} photographs"
    if solved is not None:
        d = np.abs((np.array(solved) - np.array(greedy) + 180) % 360 - 180)
        sub += (f"   —   (11) and (27) differ by {d.mean():.0f}° on average, "
                f"up to {d.max():.0f}°; training used (27)")
    else:
        sub += "   —   no phase_opt.npz here, so training would use (11)"
    fig.suptitle(sub, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=145)
    print(f"  -> {out}")
    print("  (11) degrees:", " ".join(f"{d:.0f}" for d in greedy))
    if solved is not None:
        print("  (27) degrees:", " ".join(f"{d:.0f}" for d in solved))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out/phase_align.png")
