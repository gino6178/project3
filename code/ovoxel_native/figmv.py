"""The held-out cuts, three ways, and what the supervision now reaches.

Left to right is one plane -- the same plane in every row, because the cameras and the depths come
out of `random_cuts.py`'s own sequence under HELDOUT_BAND=0.30,0.70 for all three.  The photograph
row is a different orange and is not of these planes; nothing here is a per-pixel comparison and
DreamSim is not one either.
"""
import glob, json, os, sys
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OUT = W + "/out"
os.makedirs(OUT, exist_ok=True)


def g(d, pat):
    return sorted(glob.glob(os.path.join(d, pat)))


def sheet(pat, refdir, tag, title):
    rows = [("photograph", g(refdir, "*.png")),
            ("O-Voxel-native, initialised", g(W + "/mv_pin/eval_init", pat)),
            ("O-Voxel-native, exterior pinned", g(W + "/mv_pin/eval_final", pat)),
            ("O-Voxel-native, exterior trained", g(W + "/mv_free/eval_final", pat)),
            ("existing pipeline", g(os.path.join(FN, "eval_orange"), pat))]
    n = max(len(p) for _, p in rows[1:])
    fig, ax = plt.subplots(len(rows), n, figsize=(1.55 * n, 1.72 * len(rows)))
    for r, (name, ps) in enumerate(rows):
        for c in range(n):
            a = ax[r, c]
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if c < len(ps):
                a.imshow(cv2.resize(cv2.imread(ps[c])[:, :, ::-1], (320, 320)))
            if c == 0:
                a.set_ylabel(name.replace(", ", ",\n"), fontsize=7.5, rotation=0,
                             ha="right", va="center", labelpad=6)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/{tag}.png", dpi=115, bbox_inches="tight")
    print("wrote", f"{OUT}/{tag}.png")


sheet("rh*_init_0.png", os.path.join(FN, "secref_orraw_hsep"), "heldout_transverse",
      "six held-out transverse cuts, HELDOUT_BAND=0.30,0.70")
sheet("rv*_init_0.png", os.path.join(FN, "secref_orraw_vsep"), "heldout_longitudinal",
      "six held-out longitudinal cuts, azimuths off the trained grid")

# ---- coverage and the training curves ---------------------------------------------------
Z = np.load(W + "/out/coverage.npz")
KEYS = ("dual_v", "split_w", "surf_rgb", "interior")
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
x = np.arange(len(KEYS))
b = [100 * Z[f"before_{k}"].mean() for k in KEYS]
s = [100 * Z[f"sec_{k}"].mean() for k in KEYS]
a = [100 * Z[f"after_{k}"].mean() for k in KEYS]
T = np.load(W + "/mv_free/touch.npz")
j = [100 * T[k].mean() for k in KEYS]
ax[0].bar(x - 0.30, b, 0.19, label="one transverse camera, 17 planes")
ax[0].bar(x - 0.10, s, 0.19, label="both section families (16 + 10)")
ax[0].bar(x + 0.10, a, 0.19, label="+ the six exterior views")
ax[0].bar(x + 0.30, j, 0.19, label="the 200-iteration schedule, jittered")
ax[0].set_xticks(x); ax[0].set_xticklabels(KEYS, fontsize=8)
ax[0].set_ylabel("% of rows that ever receive a gradient")
ax[0].set_title("what the supervision reaches", fontsize=9)
ax[0].legend(fontsize=6.5); ax[0].grid(alpha=.3, axis="y"); ax[0].set_ylim(0, 108)
for i in range(len(KEYS)):
    ax[0].text(i - 0.30, b[i] + 1.5, f"{b[i]:.0f}", ha="center", fontsize=7)
    ax[0].text(i + 0.30, j[i] + 1.5, f"{j[i]:.0f}", ha="center", fontsize=7)

for nm, c in (("mv_pin", "C0"), ("mv_free", "C1")):
    p = f"{W}/{nm}/hist.json"
    if not os.path.exists(p):
        continue
    h = json.load(open(p))
    L = np.array(h["loss"]); k = 200
    ax[1].plot(np.arange(k - 1, len(L)), np.convolve(L, np.ones(k) / k, "valid"), lw=1.4,
               color=c, label=nm)
    P = np.array(h["probe"])
    ax[2].plot(P[:, 0], P[:, 1], "o-", ms=3, color=c, label=nm)
ax[1].set_xlabel("gradient step"); ax[1].set_ylabel("L1 to the mapped photograph")
ax[1].set_title("training loss (200-step mean)", fontsize=9)
ax[2].set_xlabel("outer iteration"); ax[2].set_ylabel("L1, twelve held-out cuts")
ax[2].set_title("fixed probe", fontsize=9)
for q in ax[1:]:
    q.grid(alpha=.3); q.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/coverage_and_training.png", dpi=115)
print("wrote", f"{OUT}/coverage_and_training.png")
print("FIG_OK")
