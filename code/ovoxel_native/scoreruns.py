"""Score any set of runs on the held-out half, with the metric the page reports.

`evalmv3.py` names its arms and its frames by hand, from before the validation/test split existed:
it reads `rh*_init_0.png`, which is the initial render, and its arm list is fixed. Everything
trained since writes `test_rh*.png` and `test_rv*.png` -- the twelve cuts nothing looked at -- and
this scores whichever runs it is given against the object's held-out photographs.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RUNS = [r for r in os.environ.get("SR_RUNS", "").split(",") if r]
dev = "cuda"

conf = open(f"{OBJDIR}/{OBJ}.conf").read()


def spec(key, default=None):
    v = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)]
    return v[0] if v else default


refs_h = realism._paths(os.path.join(FN, spec("EVAL_REF=")))
refs_v = realism._paths(os.path.join(FN, spec("EVAL_REF_V=", spec("EVAL_REF="))))
print(f"{OBJ}: scoring against {len(refs_h)} held-out transverse and {len(refs_v)} longitudinal "
      f"photographs, which no run has opened")
hh = len(refs_h) // 2
print(f"  {'the photographs against themselves':<34} "
      f"{realism._dreamsim(refs_h[:hh], refs_h[hh:], dev):>7.4f} "
      f"{realism._dreamsim(refs_v[:len(refs_v) // 2], refs_v[len(refs_v) // 2:], dev):>7.4f}")
print(f"\n  {'run':<34} {'DS transverse':>13} {'DS longitudinal':>16} {'probe':>9}")
for r in RUNS:
    d = f"{W}/{r}/eval_final"
    ph = sorted(glob.glob(f"{d}/test_rh*.png"))
    pv = sorted(glob.glob(f"{d}/test_rv*.png"))
    if not ph:
        print(f"  {r:<34} {'(no test frames)':>13}")
        continue
    a = realism._dreamsim(refs_h, ph, dev)
    b = realism._dreamsim(refs_v, pv, dev) if pv else float("nan")
    pr = ""
    hp = f"{W}/{r}/hist.json"
    if os.path.exists(hp):
        j = json.load(open(hp))
        if "probe" in j:
            pr = f"{j['probe'][-1][1]:.5f}" if isinstance(j["probe"][-1], list) else f"{j['probe']:.5f}"
    print(f"  {r:<34} {a:>13.4f} {b:>16.4f} {pr:>9}")
