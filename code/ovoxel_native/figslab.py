"""Plane against slab, both directions, on one longitudinal cut -- where the blocking lives."""
import glob, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
COLS = [
    ("photograph", sorted(glob.glob(f"{FN}/secref_orraw_vsep/*.png"))[0]),
    ("O-Voxel r1_pin\nplane (0 cells)", f"{W}/slabeval/r1_pin_plane/rv0_init_0.png"),
    ("O-Voxel r1_pin\nslab (5.6 cells)", f"{W}/slabeval/r1_pin_slab/rv0_init_0.png"),
    ("O-Voxel r1_pin_full\nplane", f"{W}/slabeval/r1_pin_full_plane/rv0_init_0.png"),
    ("O-Voxel r1_pin_full\nslab", f"{W}/slabeval/r1_pin_full_slab/rv0_init_0.png"),
    ("pipeline\nslab (5.6 cells)", f"{W}/thin/s1/rv0_init_0.png"),
    ("pipeline\nthinned (1.4 cells)", f"{W}/thin/s0.25/rv0_init_0.png"),
]
COLS = [(t, p) for t, p in COLS if os.path.exists(p)]
fig, ax = plt.subplots(2, len(COLS), figsize=(2.3 * len(COLS), 5.2))
for i, (t, p) in enumerate(COLS):
    im = cv2.imread(p)[:, :, ::-1]
    ax[0, i].imshow(cv2.resize(im, (330, 330)))
    ax[0, i].set_title(t, fontsize=7.5)
    h, w, _ = im.shape
    c = im[h // 2 - 96:h // 2 + 96, w // 2 - 96:w // 2 + 96]
    ax[1, i].imshow(cv2.resize(c, (330, 330), interpolation=cv2.INTER_NEAREST))
    for r in (0, 1):
        ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
ax[1, 0].set_ylabel("middle 192 px, 2x", fontsize=7.5)
fig.suptitle("one held-out longitudinal cut: the grid blocking, and what the slab does to it",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{W}/out/slab_longitudinal.png", dpi=120, bbox_inches="tight")
print("wrote", f"{W}/out/slab_longitudinal.png")
print("FIG_OK")
