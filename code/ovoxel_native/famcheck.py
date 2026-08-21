"""Do the two reference families actually contain different photographs?

    python famcheck.py

Two objects reported the same floor in both columns to four decimal places, which is what happens
when the transverse and longitudinal families are the same files: the floor is a family scored
against itself, so an identical floor means an identical family. If they are the same, nothing in
the data distinguishes a transverse cut from a longitudinal one for that object, and the model has
no way to learn the difference no matter what the loss does.
"""
import glob
import hashlib
import os
import re

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
SKIP = ("_depth", "_mask", "_alpha", "_normal")
OBJS = ["orange_sp", "watermelon_sp", "apple1_sp", "bread_sp", "cake2_sp",
        "pomegranate2_sp", "doughnut"]


def fam(obj, which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    d = os.path.join(FN, m.group(1))
    fs = [f for f in sorted(glob.glob(os.path.join(d, "*")))
          if f.lower().endswith((".png", ".jpg", ".jpeg"))
          and not any(t in os.path.basename(f) for t in SKIP)]
    return m.group(1), {hashlib.md5(open(f, "rb").read()).hexdigest() for f in fs}


print(f"  {'object':16s} {'REF_H':22s} {'REF_V':22s} {'n_h':>3} {'n_v':>3} {'shared':>7}")
for obj in OBJS:
    dh, hh = fam(obj, "REF_H")
    dv, hv = fam(obj, "REF_V")
    share = len(hh & hv)
    flag = ""
    if share and share == len(hh) == len(hv):
        flag = "   <-- the same family twice"
    elif share:
        flag = f"   <-- {share} file(s) in both"
    print(f"  {obj:16s} {dh:22s} {dv:22s} {len(hh):>3} {len(hv):>3} {share:>7}{flag}")
