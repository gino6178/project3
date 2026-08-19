"""Moving the plane, under the two rules for handing photographs to planes.

Equation (7) mixes the two photographs a plane falls between at the fractional part of its depth.
The rule it replaced deals them out in blocks -- integer division, so two or three adjacent planes
get the same photograph and the target changes all at once where the block does. The claim is that
the block rule puts a step in the interior at each of those boundaries and the continuous rule
does not.

That claim has two halves and this measures both, because only the first is a property of the
method and the second is what a viewer sees.

  targets     what the rules ask for. `sds_demo`'s own canonicalisation is applied under both, so
              the two differ in the assignment and in nothing else, and the difference between
              consecutive targets is computed with no model and no renderer involved. The block
              rule's curve is zero inside a block and jumps at its edge, by construction; what is
              worth measuring is how large the jump is against the object's own rate of change.

  sections    what the trained models do. Two runs of the same program on the same lattice with
              the same references, differing only in REF_DEPTH_BLEND, each swept over the same
              depths, and the difference between consecutive rendered sections. A model that
              inherited the steps shows them here; one that smoothed them away does not, and that
              is a result either way.

    python code/figures/depthsweep.py targets  OUT.png REF_DIR [n_planes]
    python code/figures/depthsweep.py sections OUT.png CFG DEMO name=MODEL.ply ...
    python code/figures/depthsweep.py both OUT.png REF_DIR n_planes CFG DEMO name=MODEL.ply ...

`both` draws the two beside each other and is what the page carries. It reuses whatever `sections`
already rendered, so running it after `sections` costs the targets alone.

`jerk` is the standard deviation of the per-step difference, the same quantity
`evaluate/slicing_consistency.py` reports, so the two are comparable.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path += [_os.path.join(_os.path.dirname(_HERE), "src"), _FN_ROOT,
             _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]

LO, HI = (float(v) for v in _os.environ.get("SWEEP_BAND", "0.30,0.70").split(","))


def _ssim(a, b):
    from skimage.metrics import structural_similarity as ss
    return float(ss(a, b, channel_axis=2, data_range=1.0))


def _stats(name, d):
    # The block rule makes most of its steps exactly zero, so a ratio against the median is a
    # division by nothing and says less than the two counts beside it.
    flat = int((d < 1e-6).sum())
    print(f"  {name:<26} mean 1-SSIM {d.mean():.4f}   jerk {d.std():.4f}   "
          f"worst {d.max():.4f}   steps below 1e-6: {flat}/{len(d)}")
    print(f"    per step  {' '.join(f'{v:.4f}' for v in d)}")
    return dict(mean=float(d.mean()), jerk=float(d.std()), worst=float(d.max()), flat=flat,
                n=len(d))


def targets(out, ref_dir, n_planes=16, ax=None):
    """The two rules' targets, at the plane indices the trainer actually uses."""
    import sds_demo as sd
    n_planes = int(n_planes)
    files = sorted(sd._photos_in(ref_dir))
    print(f"  {len(files)} photographs over {n_planes} planes")

    block, cont = [], []
    for i in range(n_planes):
        t = i * len(files) / n_planes
        k0 = int(t) % len(files)
        k1 = (k0 + 1) % len(files)
        w = float(t - int(t))
        # _blend_canonical returns the photograph already centred and scaled onto a common
        # disc, in [0, 1]; _blend_on_disc is that same function applied to both neighbours and
        # mixed. So the two lists differ in the assignment rule and in nothing else.
        block.append(np.asarray(sd._blend_canonical(files[k0]), dtype=np.float32))
        cont.append(np.asarray(sd._blend_on_disc(files[k0], files[k1], w),
                               dtype=np.float32) / 255.0)

    db = np.array([1.0 - _ssim(block[i], block[i + 1]) for i in range(n_planes - 1)])
    dc = np.array([1.0 - _ssim(cont[i], cont[i + 1]) for i in range(n_planes - 1)])
    print("  target to target, between adjacent planes:")
    sb = _stats("block assignment", db)
    sc = _stats("continuous, equation (7)", dc)
    _plot(out, [("block assignment", np.arange(n_planes - 1) + 0.5, db),
                ("continuous, equation (7)", np.arange(n_planes - 1) + 0.5, dc)],
          "plane index", "1 - SSIM between adjacent targets",
          "what the two rules ask adjacent planes for", ax=ax)
    return sb, sc


def sections(out, cfg, demo, *specs, n=None, ax=None):
    """The trained models, swept over the same depths."""
    import random_cuts as rc
    import cv2
    n = int(n or _os.environ.get("SWEEP_STEPS", "48"))
    depths = np.linspace(LO, HI, n)
    curves, res = [], {}
    for spec in specs:
        name, _, model = spec.partition("=")
        d = _os.path.join(_os.path.dirname(out) or ".", f"sweep_{name}")
        frames = []
        for i, dp in enumerate(depths):
            # a degenerate band falls back to random depths, so open it by a hair
            _os.environ["HELDOUT_BAND"] = f"{dp:.6f},{dp + 1e-5:.6f}"
            _os.environ["FULL_SH"] = "1"
            sub = _os.path.join(d, f"s{i:03d}")
            if not glob.glob(_os.path.join(sub, "rh*_init_0.png")):
                rc.main(model, cfg, demo, sub, n=2, size=512)
            got = sorted(glob.glob(_os.path.join(sub, "rh*_init_0.png")))
            if got:
                frames.append(cv2.imread(got[0]).astype(np.float32) / 255.0)
        if len(frames) < 3:
            print(f"  {name}: too few frames"); continue
        dd = np.array([1.0 - _ssim(frames[i], frames[i + 1]) for i in range(len(frames) - 1)])
        res[name] = _stats(name, dd)
        curves.append((name, (depths[:-1] + depths[1:]) / 2, dd))
    _plot(out, curves, "depth along the axis", "1 - SSIM between adjacent sections",
          f"the same plane walked from {LO:g} to {HI:g}", ax=ax)
    return res


def _plot(out, curves, xlab, ylab, title, ax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, (name, x, y) in enumerate(curves):
        ax.plot(x, y, "o-", ms=3, lw=1.4, label=f"{name}  (jerk {y.std():.4f})",
                color=plt.get_cmap("tab10")(i))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.5)
    if own:
        ax.figure.tight_layout(); ax.figure.savefig(out, dpi=150)
        print("  ->", out)


def both(out, ref_dir, n_planes, cfg, demo, *specs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8))
    targets(out, ref_dir, n_planes, ax=axes[0])
    sections(out, cfg, demo, *specs, ax=axes[1])
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("  ->", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "targets":
        targets(*sys.argv[2:])
    elif mode == "sections":
        sections(sys.argv[2], sys.argv[3], sys.argv[4], *sys.argv[5:])
    elif mode == "both":
        both(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], *sys.argv[7:])
    else:
        raise SystemExit(__doc__)
