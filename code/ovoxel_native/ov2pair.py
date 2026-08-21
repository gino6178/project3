"""Every object, the old loop against the joint step and the field prior.

    python ov2pair.py

`ov_<obj>` is one plane per gradient step with no spatial prior; `ov2_<obj>` is one transverse and
one longitudinal plane per step with SEC_TV=0.1, at the same 5,200-step budget. Same object, same
references, same held-out planes, so the difference is the two changes and nothing else.

The floor is the object's own photographs split in half: two disjoint halves of one reference
family scored against each other, which is how far apart two real sections of one object are.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
W = "/workspace/ovoxel_native"
OBJDIR = "/workspace/rebuild/project3/code/objects"
OBJS = ["orange_sp", "watermelon_sp", "apple1_sp", "bread_sp", "cake2_sp",
        "pomegranate2_sp", "doughnut"]


def refs(obj, which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    return realism._paths(os.path.join(FN, m.group(1)))


def shots(tag, obj, fam):
    d = f"{W}/{tag}_{obj}/eval_final"
    return sorted(glob.glob(f"{d}/{fam}*_init_0.png")) or sorted(glob.glob(f"{d}/{fam}*.png"))


def fmt(a, b):
    """Print the pair with the better one marked, so the direction is not left to the reader."""
    if np.isnan(a) or np.isnan(b):
        return f"{a:8.4f} {b:8.4f}   "
    # a tie is a tie: marking it as worse reads as a regression that is not there
    mark = "=" if abs(b - a) < 5e-5 else ("v" if b < a else "^")
    return f"{a:8.4f} {b:8.4f} {mark:>3}"


print(f"  {'object':16s} {'floor':>8} {'old rh':>8} {'new rh':>8}     "
      f"{'floor':>8} {'old rv':>8} {'new rv':>8}")
tot = {"rh": [], "rv": []}
for obj in OBJS:
    cols = []
    for which, fam in (("REF_H", "rh"), ("REF_V", "rv")):
        R = refs(obj, which)
        h = len(R) // 2
        floor = (realism._dreamsim(R[:h], R[h:], "cuda")
                 if h >= 1 and len(R) - h >= 1 else np.nan)
        v = []
        for tag in ("ov", "ov2"):
            f = shots(tag, obj, fam)
            v.append(realism._dreamsim(R, f, "cuda") if f else np.nan)
        if not any(np.isnan(x) for x in v):
            tot[fam].append((v[0], v[1]))
        cols.append(f"{floor:8.4f} " + fmt(v[0], v[1]))
    print(f"  {obj:16s} {cols[0]}   {cols[1]}")
for fam in ("rh", "rv"):
    if tot[fam]:
        a = np.mean([x[0] for x in tot[fam]]); b = np.mean([x[1] for x in tot[fam]])
        w = sum(1 for x in tot[fam] if x[1] < x[0] - 5e-5)
        e = sum(1 for x in tot[fam] if abs(x[1] - x[0]) < 5e-5)
        print(f"  mean {fam}: {a:.4f} -> {b:.4f}   better on {w} of {len(tot[fam])}"
              + (f", level on {e}" if e else ""))
