"""What each family's photographs agree with each other about, and how far our renders sit from it.

A render is scored against a photograph of a different fruit of the same kind, so no method can beat
the distance between two such photographs. That distance is not the same for the two families: a
transverse cut of a watermelon looks much like another transverse cut, while two central sections at
different azimuths differ by more. Comparing the families' raw scores therefore compares two
different questions.

This reports, per family and per split, the floor and the score, and their ratio -- which is the
only quantity in which the two families are asking the same thing.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = os.path.dirname(os.path.abspath(__file__))
OBJS = [o for o in os.environ.get("FL_OBJS", "watermelon_sp,orange_sp").split(",") if o]
RUN = os.environ.get("FL_RUN", "s_rs")
dev = "cuda"


def floor_of(paths):
    """Half against half, averaged over a few splits, so one unlucky split cannot decide it."""
    if len(paths) < 2:
        return float("nan")
    vals = []
    idx = np.arange(len(paths))
    for s in range(min(5, len(paths))):
        rng = np.random.default_rng(s)
        p = rng.permutation(idx)
        a = [paths[i] for i in p[: len(paths) // 2]]
        b = [paths[i] for i in p[len(paths) // 2:]]
        vals.append(realism._dreamsim(a, b, dev))
    return float(np.mean(vals))


for OBJ in OBJS:
    conf = open(f"{OBJDIR}/{OBJ}.conf").read()

    def spec(k, d=None):
        v = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)]
        return v[0] if v else d

    sets = {
        ("shown", "h"): spec("REF_H="), ("shown", "v"): spec("REF_V="),
        ("unseen", "h"): spec("EVAL_REF="),
        ("unseen", "v"): spec("EVAL_REF_V=", spec("EVAL_REF=")),
    }
    print(f"\n{OBJ} ({RUN})")
    print(f"  {'split':<8}{'family':<8}{'photos':>7}{'floor':>9}{'score':>9}{'score/floor':>13}")
    for (split, fam), sp in sets.items():
        if sp is None:
            continue
        paths = realism._paths(os.path.join(FN, sp))
        fl = floor_of(paths)
        d = f"/tmp/sf/{RUN}_{OBJ}_{'s' if split == 'shown' else 'h'}{fam}"
        got = sorted(glob.glob(f"{d}/*.png"))
        sc = realism._dreamsim(paths, got, dev) if got else float("nan")
        print(f"  {split:<8}{fam:<8}{len(paths):>7}{fl:>9.4f}{sc:>9.4f}"
              f"{sc / fl if fl == fl else float('nan'):>13.2f}")
