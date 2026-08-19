"""Both routes, both DreamSim families, and the speckle measured rather than looked at.

DreamSim comes from `evaluate/realism.py` -- imported, not reimplemented, and called with the same
reference set and the same glob `realism.main` uses.

The speckle number is the one the complaint is about. A cut face of a real orange is smooth at the
scale of a lattice cell; a free per-cell RGB field fitted to a stencil target is not, because
nothing in that loss couples one cell to the next. So: mean |Laplacian| over the foreground of each
render, in the same units for every row. The photographs give the value a real section has and the
existing pipeline gives the value a coupled decoder reaches.
"""
import glob, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
W = "/workspace/ovoxel_native"
dev = "cuda"
REFH = os.path.join(FN, "secref_orraw_hsep")
REFV = os.path.join(FN, "secref_orraw_vsep")
refs_h = realism._paths(REFH)
refs_v = realism._paths(REFV)


def ds(refs, paths):
    return realism._dreamsim(refs, sorted(paths), dev) if paths else float("nan")


def speckle(paths):
    """Mean |Laplacian| inside the foreground, x1000. Scale-free enough to compare rows."""
    out = []
    for p in sorted(paths):
        a = cv2.imread(p).astype(np.float32) / 255.
        fg = (np.abs(a - 1).max(2) > 0.06)
        if fg.sum() < 100:
            continue
        g = a.mean(2)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        e = cv2.erode(fg.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if e.sum() < 100:
            e = fg
        out.append(float(lap[e].mean()))
    return 1000 * float(np.mean(out)) if out else float("nan")


ARMS = [
    # tag,                                         dir,                          route
    ("the photographs, split in half",             None,                          None),
    ("existing pipeline",                          f"{FN}/eval_orange",           "1"),
    ("pipeline's colours, our repr + renderer",    f"{W}/diag_transplant/dc",     "1"),
    ("r1  free RGB, exterior trained (ANCHOR=0)",  f"{W}/mv_free/eval_final",     "1"),
    ("r1  free RGB, exterior pinned  (ANCHOR=0)",  f"{W}/mv_pin/eval_final",      "1"),
    ("r1  decoder, initialised",                   f"{W}/r1_free/eval_init",      "1"),
    ("r1  decoder, exterior trained",              f"{W}/r1_free/eval_final",     "1"),
    ("r1  decoder, exterior pinned",               f"{W}/r1_pin/eval_final",      "1"),
    ("existing pipeline, route 2",                 f"{FN}/eval_orange_r2",        "2"),
    ("r2  decoder, initialised",                   f"{W}/r2_free/eval_init",      "2"),
    ("r2  decoder, exterior trained",              f"{W}/r2_free/eval_final",     "2"),
    ("r2  decoder, exterior pinned",               f"{W}/r2_pin/eval_final",      "2"),
]

print(f"{len(refs_h)} transverse references, {len(refs_v)} longitudinal\n")
print(f"  {'arm':<44} {'DreamSim rh':>11} {'DreamSim rv':>11} {'speckle':>8}")
h = len(refs_h) // 2
print(f"  {'the photographs, split in half':<44} "
      f"{realism._dreamsim(refs_h[:h], refs_h[h:], dev):>11.4f} "
      f"{realism._dreamsim(refs_v[:3], refs_v[3:], dev):>11.4f} "
      f"{speckle(refs_h):>8.2f}")

res = {}
for tag, d, route in ARMS[1:]:
    ph = sorted(glob.glob(os.path.join(d, "rh*_init_0.png"))) if d and os.path.isdir(d) else []
    pv = sorted(glob.glob(os.path.join(d, "rv*_init_0.png"))) if d and os.path.isdir(d) else []
    if not ph:
        print(f"  {tag:<44} {'(missing)':>11}")
        continue
    a, b, s = ds(refs_h, ph), ds(refs_v, pv), speckle(ph)
    res[tag] = (a, b, s)
    print(f"  {tag:<44} {a:>11.4f} {b if b == b else float('nan'):>11.4f} {s:>8.2f}")

print("\nwhat changed")


def delta(x, y, what):
    if x in res and y in res:
        print(f"  {what:<52} {res[y][0] - res[x][0]:+.4f} DreamSim, "
              f"{res[y][2] - res[x][2]:+.2f} speckle")


delta("r1  free RGB, exterior trained (ANCHOR=0)", "r1  decoder, exterior trained",
      "route 1, the decoder, exterior trained")
delta("r1  free RGB, exterior pinned  (ANCHOR=0)", "r1  decoder, exterior pinned",
      "route 1, the decoder, exterior pinned")
delta("r1  decoder, exterior trained", "r1  decoder, exterior pinned",
      "route 1, pinning the exterior")
delta("r2  decoder, exterior trained", "r2  decoder, exterior pinned",
      "route 2, pinning the exterior")
delta("r2  decoder, initialised", "r2  decoder, exterior trained",
      "route 2, what training is worth")
print("EVAL_OK")
