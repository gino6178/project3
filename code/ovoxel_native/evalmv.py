"""One table, scored by the pipeline's own instrument.

`evaluate/realism.py` is imported rather than reimplemented, and `_dreamsim` is called with the
same reference set and the same `rh*_init_0.png` glob `realism.main` uses, so every row here is on
the footing the pipeline's numbers are on.

The decomposition the rows are for:

    photographs, split in half      the floor -- what one real orange scores against another
    existing pipeline               what is being matched
    pipeline's appearance in our    the same trained colours, carried into the O-Voxel container
      representation and renderer   and drawn by the cut-polygon renderer.  Whatever this costs
                                    over the row above is the representation and the renderer.
    ours, trained                   whatever this costs over the row above is our training.
    ours, initialised               where training started.
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
W = "/workspace/ovoxel_native"
REF = os.path.join(FN, "secref_orraw_hsep")
dev = "cuda"
refs = realism._paths(REF)
print(f"{len(refs)} references from {REF}")


def ds(paths):
    if not paths:
        return float("nan")
    return realism._dreamsim(refs, sorted(paths), dev)


def row(name, d, pat="rh*_init_0.png"):
    ps = sorted(glob.glob(os.path.join(d, pat))) if os.path.isdir(d) else []
    v = ds(ps)
    print(f"  {name:<52} {v:>8.4f}   ({len(ps)} renders)")
    return v


h = len(refs) // 2
floor = realism._dreamsim(refs[:h], refs[h:], dev)
print(f"\n  {'the photographs, split in half':<52} {floor:>8.4f}   ({h} vs {len(refs)-h})")

pipe = row("existing pipeline (eval_orange)", os.path.join(FN, "eval_orange"))
trans = row("pipeline's appearance, our representation + renderer", W + "/diag_transplant/dc")
init = row("ours, initialised (f_dc per cell, released skin)", W + "/mv_pin/eval_init")
old = row("ours, one transverse camera, 3000 steps (run2)", W + "/run2/eval_final")
free = row("ours, multi-view, exterior trained", W + "/mv_free/eval_final")
pin = row("ours, multi-view, exterior pinned (SHELL_PIN=1)", W + "/mv_pin/eval_final")

best = min(v for v in (free, pin) if v == v)
print(f"\ndecomposition of the gap to the pipeline ({pipe:.4f})")
print(f"  representation + renderer   {trans - pipe:+.4f}   "
      f"(pipeline's own colours, drawn by us, against the pipeline)")
print(f"  our training                {best - trans:+.4f}   (our best, against those same colours)")
print(f"  total                       {best - pipe:+.4f}")
print(f"\ngap to the photographs' own floor ({floor:.4f})")
print(f"  existing pipeline           {pipe - floor:+.4f}")
print(f"  ours                        {best - floor:+.4f}")
print(f"\npinning the exterior: {free:.4f} free -> {pin:.4f} pinned  ({pin - free:+.4f})")

# and the longitudinal family, which nothing has ever scored: same instrument, its own references
REFV = os.path.join(FN, "secref_orraw_vsep")
rv = realism._paths(REFV)
if len(rv) > 1:
    print(f"\nlongitudinal held-out cuts, against {len(rv)} longitudinal photographs")

    def dsv(d):
        ps = sorted(glob.glob(os.path.join(d, "rv*_init_0.png")))
        return realism._dreamsim(rv, ps, dev) if ps else float("nan")
    for nm, d in [("existing pipeline", os.path.join(FN, "eval_orange")),
                  ("ours, exterior trained", W + "/mv_free/eval_final"),
                  ("ours, exterior pinned", W + "/mv_pin/eval_final"),
                  ("ours, initialised", W + "/mv_pin/eval_init")]:
        print(f"  {nm:<52} {dsv(d):>8.4f}")

print("EVAL_OK")
