"""What a cut costs in device memory, and at what rate it can be repeated.

    python cutmem.py LATTICE [Q]

Reports the allocator's own peak for one whole operator -- refinement, adjacency, labelling and
the polygons -- rather than what nvidia-smi shows, which includes the CUDA context and whatever
the caching allocator is holding from earlier work and is therefore a property of the process
and not of the cut.
"""
import os
import statistics
import sys
import time

import numpy as np
import torch

FN = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
sys.path += [FN]

import gpumulti as gmu                                                # noqa: E402
import gpumesh as gm                                                  # noqa: E402


def main(lattice_dir, q=1):
    from plyfile import PlyData
    from occupancy import close_and_fill, to_grid
    q = int(q)
    lat = torch.load(os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lv = torch.load(os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    org = xyz[lv == 0].min(0) - 0.5 * hc
    co = np.unique(np.floor((xyz[lv == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(co).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + co.min(0) - 1
    planes = gmu.make_planes(solid, hc, q)

    for _ in range(2):
        gmu.cut_mesh(solid, hc, planes, hf)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = gmu.cut_mesh(solid, hc, planes, hf)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    t = statistics.median(ts)
    print(f"  {os.path.basename(os.path.dirname(lattice_dir))}  {len(solid):,} cells, {q} plane(s)"
          f"   {t*1000:7.1f} ms  = {1/t:5.2f} cuts/s"
          f"   peak allocated {torch.cuda.max_memory_allocated()/2**20:7.1f} MiB"
          f"   reserved {torch.cuda.max_memory_reserved()/2**20:7.1f} MiB"
          f"   {r['tris']:,} triangles")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 1)
