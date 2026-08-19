"""Score the thickness sweep. Rendering happens in the pipeline's env, scoring in this one."""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
rh = realism._paths(f"{FN}/secref_orraw_hsep")
rv = realism._paths(f"{FN}/secref_orraw_vsep")
print(f"  {'x avg/2':>8} {'cells':>7} {'DS rh':>7} {'DS rv':>7}")
for sc in (1.0, 0.5, 0.25, 0.125, 0.0625):
    out = f"{W}/thin/s{sc:g}"
    ph = sorted(glob.glob(out + "/rh*_init_0.png"))
    pv = sorted(glob.glob(out + "/rv*_init_0.png"))
    if not ph:
        print(f"  {sc:>8.4g} missing"); continue
    a = realism._dreamsim(rh, ph, "cuda")
    b = realism._dreamsim(rv, pv, "cuda")
    print(f"  {sc:>8.4g} {0.02174*sc*1.5281/0.0118:>7.2f} {a:>7.4f} {b:>7.4f}", flush=True)
print("SCORE_OK")
