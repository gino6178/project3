"""The exterior, every arm, from the six cameras that supervised it.

Two renderers appear on this sheet and the rows say which. Rows drawn from the O-Voxel dual surface
by nvdiffrast are marked (O-Voxel); the pipeline's own exterior is Gaussians and is drawn by
`project3/code/src/exterior_views.py` at the same six azimuths, marked (Gaussian). They are not the
same instrument and are not presented as if they were -- exterior_views.py builds its cameras with
the same `get_camera_view(az, el, init_radius)` construction `mvcams.py` used for `e_mvp`, so the
viewpoints match, but everything downstream of that differs.
"""
import glob, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
EXT = W + "/out/ext"
NAMES = ["up", "down", "front", "right", "back", "left"]
# exterior_views.DIRS order, as the sheet is laid out: [up, front, right] over [down, back, left]
SHEET = {"up": (0, 0), "front": (0, 1), "right": (0, 2),
         "down": (1, 0), "back": (1, 1), "left": (1, 2)}


def tiles_from_sheet(p, size=512):
    a = cv2.imread(p)
    return {n: a[r * size:(r + 1) * size, c * size:(c + 1) * size] for n, (r, c) in SHEET.items()}


def from_dir(d):
    return {n: cv2.imread(f"{d}/{n}.png") for n in NAMES} if os.path.isdir(d) else None


ROWS = [
    ("the six reference views\n(what the exterior was supervised against)",
     {n: cv2.imread(f"{FN}/cube_or6_prep/{n}_ref.png") for n in NAMES}),
    ("route 1 exterior, as built\n(from the released ply)  (O-Voxel)", from_dir(f"{EXT}/seed_r1")),
    ("r1_pin  (O-Voxel)", from_dir(f"{EXT}/r1_pin")),
    ("r1flat_pin  (O-Voxel)", from_dir(f"{EXT}/r1flat_pin")),
    ("r1_pin_full  (O-Voxel)", from_dir(f"{EXT}/r1_pin_full")),
    ("r1_free, exterior trained  (O-Voxel)", from_dir(f"{EXT}/r1_free")),
    ("route 2 exterior, as built\n(projected from six photographs)  (O-Voxel)",
     from_dir(f"{EXT}/seed_r2")),
    ("r2_pin_full  (O-Voxel)", from_dir(f"{EXT}/r2_pin_full")),
    ("r2_free, exterior trained  (O-Voxel)", from_dir(f"{EXT}/r2_free")),
    ("existing pipeline, route 1\n(orange_b)  (Gaussian)",
     tiles_from_sheet(f"{EXT}/pipe_r1_sheet.png") if os.path.exists(f"{EXT}/pipe_r1_sheet.png") else None),
    ("existing pipeline, route 2\n(orange_r2)  (Gaussian)",
     tiles_from_sheet(f"{EXT}/pipe_r2_sheet.png") if os.path.exists(f"{EXT}/pipe_r2_sheet.png") else None),
]
ROWS = [(lab, d) for lab, d in ROWS if d is not None and all(d[n] is not None for n in NAMES)]

fig, ax = plt.subplots(len(ROWS), len(NAMES), figsize=(1.62 * len(NAMES), 1.72 * len(ROWS)))
for r, (lab, tiles) in enumerate(ROWS):
    for c, n in enumerate(NAMES):
        a = ax[r, c]
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.imshow(cv2.resize(tiles[n][:, :, ::-1], (320, 320)))
        if r == 0:
            a.set_title(n, fontsize=9)
        if c == 0:
            a.set_ylabel(lab, fontsize=6.8, rotation=0, ha="right", va="center", labelpad=6)
fig.suptitle("the exterior, from the six cameras that supervised it -- "
             "(O-Voxel) rows are the dual surface via nvdiffrast, "
             "(Gaussian) rows are exterior_views.py", fontsize=9, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(f"{W}/out/exterior_views.png", dpi=115, bbox_inches="tight")
print("wrote", f"{W}/out/exterior_views.png", f"({len(ROWS)} rows)")
print("FIG_OK")
