"""The rendered section beside the photograph, and the loss curve."""
import json, glob, os, sys
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/workspace/ovoxel_native"
RUN = sys.argv[1] if len(sys.argv) > 1 else W + "/run2"
refs = sorted(glob.glob("/workspace/rebuild/worktree/secref_orraw_hsep/*.png"))
rows = [("photograph", refs),
        ("O-Voxel-native, init", sorted(glob.glob(RUN + "/eval_init/rh*_init_0.png"))),
        (f"O-Voxel-native, trained", sorted(glob.glob(RUN + "/eval_final/rh*_init_0.png"))),
        ("existing pipeline", sorted(glob.glob("/workspace/rebuild/worktree/eval_orange/rh*_init_0.png")))]
n = 4
fig, ax = plt.subplots(len(rows), n, figsize=(2.1 * n, 2.25 * len(rows)))
for r, (name, ps) in enumerate(rows):
    for c in range(n):
        a = ax[r, c]; a.axis("off")
        if c < len(ps):
            im = cv2.imread(ps[c])[:, :, ::-1]
            a.imshow(cv2.resize(im, (384, 384)))
        if c == 0:
            a.set_title(name, fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(W + "/out/sections.png", dpi=110)
print("wrote", W + "/out/sections.png")

h = json.load(open(RUN + "/hist.json"))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
L = np.array(h["loss"]); k = 25
ax[0].plot(L, lw=0.4, color="0.75")
ax[0].plot(np.arange(k - 1, len(L)), np.convolve(L, np.ones(k) / k, "valid"), lw=1.6)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("L1 to the mapped photograph")
ax[0].set_title("training loss (25-iteration mean)", fontsize=9)
P = np.array(h["probe"])
ax[1].plot(P[:, 0], P[:, 1], "o-", ms=3)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("L1, six held-out planes")
ax[1].set_title("fixed probe: the six evaluation cuts", fontsize=9)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(W + "/out/loss.png", dpi=110)
print("wrote", W + "/out/loss.png")
