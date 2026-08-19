"""What the continuous assignment buys: the interior between two supervised planes.

Equation (14) gives a plane at depth t = j M_f / N_f the photograph floor(t) and the next one,
brought onto a common disc and mixed at the fractional part. The block rule it replaces gives
two or three adjacent planes the same photograph, so the interior has no reason to differ
between them and every reason to change where the block does -- a step, at a depth chosen by
integer division rather than by the fruit.

The claim is about continuity, so it is drawn as one: a dense sweep of transverse sections
through the trained model, across a depth where the photograph the trainer would hand a plane
changes over. If the step were there it would be at that depth, and the read-out below the
strip is what would show it -- the mean absolute difference between neighbouring sections along
the sweep, which is flat if the interior varies smoothly and spikes if it does not.

    FN_ROOT=... python blend_fig.py watermelon out/blend_watermelon.png

Three panels:

  a  the weights of equation (14) over the depth range: which two photographs a plane at each
     depth is supervised by, and at what mixture. The crossing points are where the block rule
     would have stepped.
  b  the targets themselves at a run of consecutive depths -- what supervision asks for.
  c  the model's own sections at those same depths, rendered through random_cuts' renderer, and
     under them the neighbour-to-neighbour difference along the whole sweep. This is the panel
     that carries the claim: the targets can be smooth and the volume still not be.

The sweep is denser than the trained planes on purpose. Most of these depths were never
supervised by anything, which is the case continuity is about.
"""
import os
import sys

import numpy as np

FN = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FN)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import cv2                                                            # noqa: E402

import random_cuts                                                    # noqa: E402
from sds_demo import _blend_canonical, _photos_in                     # noqa: E402


def conf(key, obj):
    p = os.path.join(os.path.dirname(HERE), "objects", f"{obj}.conf")
    for line in open(p):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].split("#")[0].strip()
    return None


def main(obj, out, n_sweep=24, lo=0.34, hi=0.66, size=384):
    ref_h = conf("REF_H", obj)
    cfg, demo = conf("CFG", obj), conf("DEMO", obj)
    iters = 200
    ply = os.path.join(FN, obj, f"orange_demo_epoch_{iters - 1}.ply")
    if not os.path.exists(ply):
        raise SystemExit(f"no trained model at {ply}")
    files = sorted(_photos_in(os.path.join(FN, ref_h)))
    M = len(files)
    N_f = int(os.environ.get("H_HI", "20")) - int(os.environ.get("H_LO", "4"))
    plt.rcParams["figure.constrained_layout.use"] = False

    # --- (a) the weights, over the trainer's own plane index -------------------------------
    j = np.arange(N_f)
    t = j * M / max(N_f, 1)
    k0 = (t.astype(int)) % M
    w = t - t.astype(int)

    # --- the sweep, in the trainer's depth fraction ----------------------------------------
    depths = np.linspace(lo, hi, n_sweep)
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), f"_sweep_{obj}")
    # Four times the trainer's 24, so consecutive sweep depths are distinct planes and
    # panel (d) reads the volume rather than the indexing.
    paths = random_cuts.sweep(ply, cfg, demo, tmp, depths, size=size, n_depth=4 * n_sweep)
    imgs = [cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255. for p in paths]
    d = [float(np.abs(imgs[i + 1] - imgs[i]).mean()) for i in range(len(imgs) - 1)]

    # --- which depths in the sweep sit at a photograph changeover --------------------------
    # depth fraction -> trainer plane index -> t -> the integer part changes here
    tj = depths * (N_f - 1) * M / max(N_f, 1)
    cross = [i for i in range(len(tj) - 1) if int(tj[i]) != int(tj[i + 1])]

    show = min(6, n_sweep)
    pick = np.linspace(0, n_sweep - 1, show).astype(int)

    fig = plt.figure(figsize=(2.15 * show, 8.4))
    gs = fig.add_gridspec(4, show, height_ratios=[1.05, 1.15, 1.15, 0.95], hspace=0.30)

    axw = fig.add_subplot(gs[0, :])
    for m in range(M):
        wm = np.where(k0 == m, 1 - w, 0.0) + np.where((k0 + 1) % M == m, w, 0.0)
        axw.plot(j, wm, lw=1.3, label=os.path.basename(files[m]))
    axw.set_xlabel("transverse plane index $j$", fontsize=8)
    axw.set_ylabel("weight in (14)", fontsize=8)
    axw.tick_params(labelsize=7); axw.set_ylim(-0.03, 1.03)
    if M <= 8:
        axw.legend(fontsize=6, ncol=min(M, 4), loc="upper center", framealpha=0.9)
    else:
        axw.set_title("", fontsize=1)
    axw.set_title("(a) every plane is a mixture of two photographs, and the mixture is "
                  "continuous in depth", fontsize=9)

    for c, i in enumerate(pick):
        ti = depths[i] * (N_f - 1) * M / max(N_f, 1)
        a, b = int(ti) % M, (int(ti) + 1) % M
        tgt = np.clip((1 - (ti - int(ti))) * _blend_canonical(files[a])
                      + (ti - int(ti)) * _blend_canonical(files[b]), 0, 1)
        axt = fig.add_subplot(gs[1, c]); axt.imshow(tgt); axt.set_axis_off()
        axt.set_title(f"{os.path.basename(files[a])[:-4]} × {1-(ti-int(ti)):.2f}\n"
                      f"{os.path.basename(files[b])[:-4]} × {ti-int(ti):.2f}", fontsize=6.5)
        axr = fig.add_subplot(gs[2, c]); axr.imshow(imgs[i]); axr.set_axis_off()
        axr.set_title(f"depth {depths[i]:.3f}", fontsize=7)

    fig.text(0.008, 0.545, "(b) what supervision asks for", fontsize=9, rotation=90, va="center")
    fig.text(0.008, 0.325, "(c) what the volume renders", fontsize=9, rotation=90, va="center")

    axd = fig.add_subplot(gs[3, :])
    axd.plot(depths[:-1], d, lw=1.4, color="#c0392b", marker="o", ms=2.6)
    for i in cross:
        axd.axvline(depths[i], color="#4a7ba7", lw=1.0, ls="--")
    axd.set_xlabel("depth fraction", fontsize=8)
    axd.set_ylabel("mean |difference|\nto the next section", fontsize=8)
    axd.tick_params(labelsize=7)
    mu, sd = float(np.mean(d)), float(np.std(d))
    axd.axhline(mu, color="0.6", lw=0.8)
    axd.set_title(f"(d) neighbour-to-neighbour difference along the sweep: mean {mu:.4f}, "
                  f"spread {sd:.4f}; dashed lines are where the photograph changes over",
                  fontsize=9)

    fig.suptitle(f"{obj}: {n_sweep} sections between depth {lo} and {hi}, "
                 f"{M} transverse photographs over {N_f} supervised planes", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.06)
    fig.savefig(out, dpi=145)
    print(f"  -> {out}")
    at = [d[i] for i in cross if i < len(d)]
    if at:
        print(f"  at the changeovers: {np.mean(at):.4f}   elsewhere: "
              f"{np.mean([x for i, x in enumerate(d) if i not in cross]):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f"out/blend_{sys.argv[1]}.png")
