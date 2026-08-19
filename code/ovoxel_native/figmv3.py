"""Both routes, both families, every arm that exists -- and the crop panel for each.

The longitudinal sheet is the one that was missing, and it is the column where this loses to the
pipeline, so it is the one worth having.
"""
import glob, json, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OUT = W + "/out"


def g(d, pat):
    return sorted(glob.glob(os.path.join(d, pat))) if os.path.isdir(d) else []


ARMS = {
    "1": [("photograph", None),
          ("free per-cell RGB, trained", f"{W}/mv_free/eval_final"),
          ("decoder, initialised", f"{W}/r1_free/eval_init"),
          ("decoder, trained", f"{W}/r1_free/eval_final"),
          ("decoder, exterior pinned", f"{W}/r1_pin/eval_final"),
          ("decoder, flat interior, pinned", f"{W}/r1flat_pin/eval_final"),
          ("decoder, full parity", f"{W}/r1_pin_full/eval_final"),
          ("existing pipeline (eval_orange_b)", f"{FN}/eval_orange_b")],
    "2": [("photograph", None),
          ("decoder, initialised", f"{W}/r2_free/eval_init"),
          ("decoder, trained", f"{W}/r2_free/eval_final"),
          ("decoder, exterior pinned", f"{W}/r2_pin/eval_final"),
          ("decoder, full parity", f"{W}/r2_pin_full/eval_final"),
          ("existing pipeline, route 2", f"{FN}/eval_orange_r2")],
}
FAM = {"rh": ("transverse", "secref_orraw_hsep"), "rv": ("longitudinal", "secref_orraw_vsep")}


def build(route, fam):
    name, refdir = FAM[fam]
    rows = []
    for lab, d in ARMS[route]:
        ps = g(os.path.join(FN, refdir), "*.png") if d is None else g(d, f"{fam}*_init_0.png")
        if ps:
            rows.append((lab, ps))
    return rows, name


def sheet(route, fam, n=6):
    rows, name = build(route, fam)
    fig, ax = plt.subplots(len(rows), n, figsize=(1.55 * n, 1.72 * len(rows)))
    for r, (lab, ps) in enumerate(rows):
        for c in range(n):
            a = ax[r, c]
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if c < len(ps):
                a.imshow(cv2.resize(cv2.imread(ps[c])[:, :, ::-1], (320, 320)))
            if c == 0:
                a.set_ylabel(lab.replace(", ", ",\n"), fontsize=7.5, rotation=0,
                             ha="right", va="center", labelpad=6)
    fig.suptitle(f"route {route} -- six held-out {name} cuts", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = f"{OUT}/route{route}_{name}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def crops(route, fam, which=0, half=96):
    rows, name = build(route, fam)
    fig, ax = plt.subplots(1, len(rows), figsize=(2.4 * len(rows), 3.0))
    for i, (lab, ps) in enumerate(rows):
        im = cv2.imread(ps[which])[:, :, ::-1]
        h, w, _ = im.shape
        cy, cx = h // 2, w // 2
        c = im[cy - half:cy + half, cx - half:cx + half]
        ax[i].imshow(cv2.resize(c, (384, 384), interpolation=cv2.INTER_NEAREST))
        ax[i].set_title(lab.replace(", ", ",\n"), fontsize=7)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.suptitle(f"route {route}, {name}: middle {2*half} px of one held-out cut, 2x nearest",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{OUT}/route{route}_{name}_crop.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


for route in ("1", "2"):
    for fam in ("rh", "rv"):
        sheet(route, fam); crops(route, fam)

# probe curves for every arm that has a history
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for nm in sorted(os.listdir(W)):
    p = f"{W}/{nm}/hist.json"
    if not os.path.exists(p):
        continue
    h = json.load(open(p))
    if "probe" not in h or "loss" not in h:
        continue
    L = np.array(h["loss"]); k = 200
    if len(L) < k:
        continue
    ax[0].plot(np.arange(k - 1, len(L)), np.convolve(L, np.ones(k) / k, "valid"), lw=1.1, label=nm)
    P = np.array(h["probe"])
    ax[1].plot(P[:, 0], P[:, 1], "-", lw=1.1, label=nm)
ax[0].set_xlabel("gradient step"); ax[0].set_ylabel("training loss (200-step mean)")
ax[0].set_title("training loss -- note the arms with SEC_PATCH are on a different loss", fontsize=8)
ax[1].set_xlabel("outer iteration"); ax[1].set_ylabel("L1, twelve held-out cuts")
ax[1].set_title("fixed probe (L1, comparable across every arm)", fontsize=8)
for q in ax:
    q.grid(alpha=.3); q.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/probes_all.png", dpi=115)
print("wrote", f"{OUT}/probes_all.png")
print("FIG_OK")
