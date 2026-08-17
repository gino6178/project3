"""What the operations cost -- the specification's runtime metrics, which were never measured.

Section 11.2 asks for four: initial load, single-cut update, O-Voxel conversion, and render rate.
The page repeatedly claims cheapness -- "connectivity, piece identity and collision are integer
operations", "collision is a floor division and an occupancy test and no point-in-polyhedron query
anywhere" -- and there is not one wall-clock number anywhere in the project to support it. A speed
claim with no timing is an assertion.

Each is timed the way it is actually used, not in a microbenchmark: the cut is a whole cut on a
whole object, the collision query is against the number of particles the solver has, and the
conversion is of the entire boundary.

    python method/common/eval/timing.py LATTICE [MODEL.ply]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys
import time

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]


class T:
    """A timer that reports the median of several runs, because the first is never typical."""

    def __init__(self, reps=3):
        self.reps = reps
        self.out = []

    def __call__(self, name, fn, unit=None, n=None):
        ts = []
        r = None
        for i in range(self.reps):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            r = fn()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        extra = ""
        if n:
            extra = f"   {n / med:,.0f} {unit or 'items'}/s"
        print(f"  {name:<44} {1000 * med:9.1f} ms   (min {1000 * min(ts):.1f}, "
              f"max {1000 * max(ts):.1f}){extra}")
        self.out.append((name, med))
        return r


def main(lattice_dir, model_ply=None):
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid
    from method.common.cube import subdivide as sd
    from method.common.cube import cutmesh as cm
    from method.common.cube.physics import CollisionIndex, particles_to_pieces

    t = T(reps=int(_os.environ.get("REPS", "3")))
    ply = model_ply or _os.path.join(lattice_dir, "gs_fill.ply")

    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])

    def load():
        el = PlyData.read(ply).elements[0]
        return np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)

    xyz = t("initial load: read the model from disk", load, "cells", None)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    print(f"      {len(xyz):,} cells, h_c {hc:.5f}, h_f {hf:.5f}")

    org = xyz[lvl == 0].min(0) - 0.5 * hf
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)

    def fill():
        occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
        return close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1

    solid = t("occupancy: close and fill", fill, "cells", len(coords))
    print(f"      {len(coords):,} quantised -> {len(solid):,} solid")

    n = np.array([0.13, 0.97, -0.21])
    n /= np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n)

    r = t("single cut: refine the band, label the pieces",
          lambda: sd.cut(solid, hc, n, d, hf), "leaves", None)
    print(f"      {len(solid):,} cells -> {len(r['leaf']):,} leaves, {r['K']} pieces")

    t("cut face: the exact polygons",
      lambda: cm.cut_mesh(solid, hc, n, d, hf), "polygons", None)

    ix = t("collision index: build it", lambda: CollisionIndex(r, hc, org=org, plane=(n, d)))
    pid, _ = t("collision: label every particle by piece",
               lambda: particles_to_pieces(xyz, ix), "particles", len(xyz))

    q = xyz + np.array([0.0, 0.3 * hc, 0.0])
    t("collision: one occupancy query over the particle set",
      lambda: ix.occupied(q, piece=0), "queries", len(q))

    if _os.environ.get("SKIP_OVOX", "0") != "1":
        try:
            from method.common.cube import globalovox as go
            t("O-Voxel conversion: the whole boundary",
              lambda: go.main(lattice_dir, None, colour_from=model_ply), "cells", len(xyz))
        except SystemExit as e:
            print(f"  O-Voxel conversion skipped: {e}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  O-Voxel conversion skipped: {type(e).__name__}: {e}")

    # --- render rate, the fourth runtime line the design document asks for ------------------
    if _os.environ.get("SKIP_FPS", "0") != "1" and model_ply:
        try:
            import cv2  # noqa: F401
            from method.common.eval import ovox_cuts as oc
            from method.common.eval import random_cuts as rc
            for size in (512, 1024):
                fps = {}
                for name, hook in (("cube volume, marching the face", True),
                                   ("Gaussian rasteriser", False)):
                    frames = []
                    oc.RENDER_COUNT = 0
                    t0 = time.perf_counter()
                    # one frame is one held-out cut, which is what the evaluator renders and so
                    # the unit a reader can compare against the tables
                    if hook:
                        rc.RENDER_HOOK[0] = oc.make_hook(model_ply, lattice_dir, _os.environ["OVOX_NPZ"]) \
                            if _os.environ.get("OVOX_NPZ") else None
                    else:
                        rc.RENDER_HOOK[0] = None
                    if hook and rc.RENDER_HOOK[0] is None:
                        continue
                    rc.main(model_ply, _os.environ["FPS_CFG"], _os.environ["FPS_DEMO"],
                            f"/tmp/_fps_{size}", n=4, size=size)
                    dt = (time.perf_counter() - t0) / 4
                    fps[name] = dt
                    print(f"  render, {size}px, {name:<34} {1000 * dt:8.1f} ms/frame  "
                          f"{1 / dt:6.2f} fps")
                rc.RENDER_HOOK[0] = None
        except Exception as e:                                   # noqa: BLE001
            print(f"  render rate not measured: {type(e).__name__}: {e}")

    print("\n  summary")
    for name, sec in t.out:
        print(f"    {name:<44} {1000 * sec:9.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
