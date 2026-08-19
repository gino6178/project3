"""The pipeline's own model, rendered with the slab thinned, on the same twelve held-out cuts.

`random_cuts.py` is used unmodified -- `project3` is not touched. The one thing changed is the
`surf_dis` it passes to `plane_filter`, by rebinding the name in the module's own namespace, so the
plane sequence, the seed, the band, the camera and the rasteriser are all the repository's.

The question is whether the pipeline's advantage on the longitudinal column survives at zero
thickness. There is a floor: a Gaussian point cloud cannot render a mathematically thin plane. The
lattice spacing is 0.0077 in the frame surf_dis is measured in, so below about half of that the
band stops containing a primitive in every column and the render degenerates into a sparse
stipple rather than becoming a sharper section. So this is a sweep, and the count of primitives
that survive the filter is reported alongside, because that is what says whether a row is a
thinner picture or a broken one.
"""
import glob, os, sys
import numpy as np

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
W = "/workspace/ovoxel_native"
sys.path.insert(0, "/workspace/rebuild/project3/code/src")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
os.environ.setdefault("HELDOUT_BAND", "0.30,0.70")
os.environ.setdefault("FULL_SH", "1")
import random_cuts as rc                                              # noqa: E402

PLY = os.environ.get("PLY", f"{W}/baseline/orange_b.ply")
CFG = os.environ.get("CFG", "config/orange_physics.json")
DEMO = os.environ.get("DEMO", "config/sphere_demo")

_orig = rc.plane_filter
_state = {"scale": 1.0, "kept": []}


def patched(plane, tpos, raw, surf_dis, include_double=True):
    m, ms = _orig(plane, tpos, raw, surf_dis=surf_dis * _state["scale"],
                  include_double=include_double)
    _state["kept"].append(int(ms.sum()))
    return m, ms


rc.plane_filter = patched
print(f"model {PLY}")
print(f"  {'surf_dis':>10} {'x avg/2':>8} {'cells':>7} {'primitives in the band':>24}")
runs = {}
for sc in (1.0, 0.5, 0.25, 0.125, 0.0625):
    _state["scale"] = sc
    _state["kept"] = []
    out = f"{W}/thin/s{sc:g}"
    if not (os.path.isdir(out) and len(glob.glob(out + "/r*_init_0.png")) == 12):
        rc.main(PLY, CFG, DEMO, out, 12, 512)
    k = _state["kept"]
    # surf_dis in the transformed frame is avg/2 = 0.02174 for the transverse camera
    sd = 0.02174 * sc
    print(f"  {sd:>10.5f} {sc:>8.4g} {sd*1.5281/0.0118:>7.2f} "
          f"{(np.mean(k) if k else float('nan')):>24,.0f}")
    runs[sc] = out

import realism                                                        # noqa: E402
rh = realism._paths(f"{FN}/secref_orraw_hsep")
rv = realism._paths(f"{FN}/secref_orraw_vsep")
print(f"\n  {'x avg/2':>8} {'cells':>7} {'DS rh':>7} {'DS rv':>7}")
for sc, out in runs.items():
    a = realism._dreamsim(rh, sorted(glob.glob(out + "/rh*_init_0.png")), "cuda")
    b = realism._dreamsim(rv, sorted(glob.glob(out + "/rv*_init_0.png")), "cuda")
    print(f"  {sc:>8.4g} {0.02174*sc*1.5281/0.0118:>7.2f} {a:>7.4f} {b:>7.4f}", flush=True)
print("THIN_OK")
