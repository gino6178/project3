"""The held-out cuts per route, in the layout Gino read the last one in, plus a crop panel at the
scale the complaint is about.

A whole 512px section shown at 320px hides exactly the thing in question: the speckle is per
lattice cell, a few pixels wide. So each sheet is followed by the same rows cropped to the middle
192px of one cut and shown at 3x, where a per-cell field and a coupled one look different or do
not.
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


def g(d, pat="rh*_init_0.png"):
    return sorted(glob.glob(os.path.join(d, pat))) if os.path.isdir(d) else []


ROUTES = {
    "route1": ("route 1 -- the released ply quantised: six held-out transverse cuts", [
        ("photograph", g(os.path.join(FN, "secref_orraw_hsep"), "*.png")),
        ("free per-cell RGB, trained", g(W + "/mv_free/eval_final")),
        ("anchor decoder, initialised", g(W + "/r1_free/eval_init")),
        ("anchor decoder, trained", g(W + "/r1_free/eval_final")),
        ("anchor decoder, exterior pinned", g(W + "/r1_pin/eval_final")),
        ("existing pipeline", g(os.path.join(FN, "eval_orange"))),
    ]),
    "route2": ("route 2 -- shape from the SDF, exterior from the six views: six held-out "
               "transverse cuts", [
        ("photograph", g(os.path.join(FN, "secref_orraw_hsep"), "*.png")),
        ("anchor decoder, initialised", g(W + "/r2_free/eval_init")),
        ("anchor decoder, trained", g(W + "/r2_free/eval_final")),
        ("anchor decoder, exterior pinned", g(W + "/r2_pin/eval_final")),
        ("existing pipeline, route 2", g(os.path.join(FN, "eval_orange_r2"))),
    ]),
}


def sheet(tag, title, rows, n=6):
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
    fig.savefig(f"{OUT}/{tag}_transverse.png", dpi=115, bbox_inches="tight")
    print("wrote", f"{OUT}/{tag}_transverse.png")


def crops(tag, rows, which=0, half=96):
    keep = [(nm, ps) for nm, ps in rows if ps]
    fig, ax = plt.subplots(1, len(keep), figsize=(2.5 * len(keep), 3.0))
    for i, (nm, ps) in enumerate(keep):
        im = cv2.imread(ps[which])[:, :, ::-1]
        h, w, _ = im.shape
        cy, cx = h // 2, w // 2
        c = im[cy - half:cy + half, cx - half:cx + half]
        ax[i].imshow(cv2.resize(c, (384, 384), interpolation=cv2.INTER_NEAREST))
        ax[i].set_title(nm.replace(", ", ",\n"), fontsize=7.5)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.suptitle(f"{tag}: the middle {2*half} px of one held-out cut, at 2x nearest neighbour",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/{tag}_crop.png", dpi=120, bbox_inches="tight")
    print("wrote", f"{OUT}/{tag}_crop.png")


for tag, (title, rows) in ROUTES.items():
    sheet(tag, title, rows)
    crops(tag, rows)

# ---- probe curves, all arms, plus the free-RGB pair they replace ------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for nm, c, ls in (("mv_free", "0.6", "--"), ("mv_pin", "0.75", "--"),
                  ("r1_free", "C0", "-"), ("r1_pin", "C1", "-"),
                  ("r2_free", "C2", "-"), ("r2_pin", "C3", "-")):
    p = f"{W}/{nm}/hist.json"
    if not os.path.exists(p):
        continue
    hh = json.load(open(p))
    L = np.array(hh["loss"]); k = 200
    lab = nm + (" (free RGB)" if nm.startswith("mv") else "")
    ax[0].plot(np.arange(k - 1, len(L)), np.convolve(L, np.ones(k) / k, "valid"),
               lw=1.3, color=c, ls=ls, label=lab)
    P = np.array(hh["probe"])
    ax[1].plot(P[:, 0], P[:, 1], "o-", ms=2.5, color=c, ls=ls, label=lab)
ax[0].set_xlabel("gradient step"); ax[0].set_ylabel("L1 to the mapped photograph")
ax[0].set_title("training loss (200-step mean)", fontsize=9)
ax[1].set_xlabel("outer iteration"); ax[1].set_ylabel("L1, twelve held-out cuts")
ax[1].set_title("fixed probe", fontsize=9)
for q in ax:
    q.grid(alpha=.3); q.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/probes_all.png", dpi=115)
print("wrote", f"{OUT}/probes_all.png")
print("FIG_OK")
