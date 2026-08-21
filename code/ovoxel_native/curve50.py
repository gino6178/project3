"""The two fifty-iteration runs: what the loss did, and what came out.

    python curve50.py OUT.png

The control is the configuration that was measured on all seven objects. The other adds the two
things this session established: the longitudinal planes turn about the axis, which takes that
family's reach from 15.9% of the cells to 97.7%, and each plane's reference is chosen at the
position it was jittered to rather than at its unjittered index, for both families by the same
rule.

Eight hundred gradient steps is short. What it can show is whether the curves separate and which
way; it cannot show where either ends up.
"""
import json
import os
import re
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

W = "/workspace/ovoxel_native"
ARMS = [("c50", "control", "0.45"), ("n50", "turning + reference follows", "tab:blue")]

fig = plt.figure(figsize=(13.6, 8.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.22, wspace=0.06)
axl = fig.add_subplot(gs[0, :2])
axp = fig.add_subplot(gs[0, 2:])

for tag, label, col in ARMS:
    h = json.load(open(f"{W}/{tag}_orange_sp/hist.json"))
    loss = np.asarray(h["loss"], float)
    k = 16                                        # one outer iteration, so the curve is per pass
    n = len(loss) // k * k
    axl.plot(np.arange(n // k) + 1, loss[:n].reshape(-1, k).mean(1), color=col, lw=1.6,
             label=label)
    pr = np.asarray(h["probe"], float)
    axp.plot(pr[:, 0], pr[:, 1], color=col, lw=1.6, marker="o", ms=3, label=label)

axl.set_xlabel("outer iteration"); axl.set_ylabel("training loss, mean over the pass")
axl.legend(fontsize=9, frameon=False); axl.set_title("what it is fitting", fontsize=10.5)
axp.set_xlabel("outer iteration"); axp.set_ylabel("held-out probe")
axp.legend(fontsize=9, frameon=False)
axp.set_title("what it has not seen", fontsize=10.5)

for c, (tag, label, _) in enumerate(ARMS):
    for r, fam in enumerate(("rh", "rv")):
        import glob
        f = sorted(glob.glob(f"{W}/{tag}_orange_sp/eval_final/{fam}0_*.png"))
        ax = fig.add_subplot(gs[1, 2 * c + r])
        if f:
            a = np.asarray(Image.open(f[0]).convert("RGB"), np.float32) / 255.
            m = a.min(2) < 0.97
            ys, xs = np.where(m)
            s = int(0.05 * max(ys.max() - ys.min(), xs.max() - xs.min()))
            ax.imshow(np.clip(a[max(ys.min()-s,0):ys.max()+s, max(xs.min()-s,0):xs.max()+s], 0, 1))
        ax.set_axis_off()
        ax.set_title(f"{label}\n{'transverse' if fam == 'rh' else 'longitudinal'}", fontsize=9.5)
fig.savefig(sys.argv[1], dpi=150, facecolor="white", bbox_inches="tight")
print("  ->", sys.argv[1])
for tag, label, _ in ARMS:
    h = json.load(open(f"{W}/{tag}_orange_sp/hist.json"))
    pr = np.asarray(h["probe"], float)
    print(f"  {label:30s} probe {pr[1,1]:.5f} -> {pr[-1,1]:.5f}")
