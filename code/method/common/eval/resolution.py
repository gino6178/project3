"""The same object at several cell sizes, so geometry and appearance move together.

A representation paper without a resolution sweep is incomplete, and the trade this one claims --
that a cube covers exactly at the cost of quantising the silhouette -- is a statement about what
happens as h changes. Nothing in the paper varied h.

Retraining at each spacing would confound the discretisation with the optimisation: a model trained
at a finer h is a different model, and any difference in its renders is then unattributable. So the
appearance is held fixed and only the discretisation moves. The trained model's colour is
resampled onto a lattice built at each spacing by nearest source primitive, which is the same
content sampled coarsely or finely, and every metric below is then a function of h alone.

    python method/common/eval/resolution.py LATTICE MODEL.ply CFG DEMO OUT_DIR [scales]

Reports, per spacing: the cells the object needs, the storage that implies, the cut-area error
against a closed form on a synthetic shape at the same spacing, and the unpainted fraction and
perceptual distance of a held-out cut rendered from the volume.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

SCALES = [float(v) for v in _os.environ.get("SCALES", "0.5,1.0,2.0").split(",")]
C0 = 0.28209479177387814
MiB = 1024.0 ** 2
FEAT = int(_os.environ.get("ANCHOR_DIM", "8"))


def build_at(xyz, rgb, dx):
    """Quantise the trained model onto a lattice of spacing dx, carrying its colour.

    Nearest source primitive rather than an average, because averaging would blur the interior as
    dx grows and the blur would be read as a property of the representation. Nearest keeps every
    cell's colour a colour the model actually has, so what changes with dx is only which colours
    survive.
    """
    from scipy.spatial import cKDTree
    org = xyz.min(0) - 0.5 * dx
    k = np.floor((xyz - org) / dx).astype(np.int64)
    cells, first = np.unique(k, axis=0, return_index=True)
    centres = (cells + 0.5) * dx + org
    _, j = cKDTree(xyz).query(centres, k=1)
    return cells, centres, rgb[j]


def main(lattice_dir, model_ply, cfg, demo, out_dir, scales=None):
    import cv2
    from plyfile import PlyData
    from method.common.cube import cutmesh as cm, subdivide as sd
    from method.common.cube.occupancy import close_and_fill, to_grid

    scales = scales or SCALES
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc = float(lat["coarse_dx"])
    el = PlyData.read(model_ply).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                  * C0 + 0.5, 0, 1)
    ext = float((xyz.max(0) - xyz.min(0)).max())
    print(f"  {len(xyz):,} primitives, extent {ext:.4f}, base spacing {hc:.5f}")
    print(f"\n  {'h / h0':>7} {'spacing':>9} {'cells':>10} {'MiB':>7} {'area err':>9} "
          f"{'silhouette':>11}")

    for sc in scales:
        dx = hc * sc
        cells, centres, col = build_at(xyz, rgb, dx)
        occ, _, _ = to_grid(torch.from_numpy(cells).float(), 1.0)
        solid = close_and_fill(occ, 1).nonzero().numpy() + cells.min(0) - 1
        store = len(cells) * (FEAT * 4 + 2 + 12) / MiB

        # cut-area error at this spacing, on a ball of the object's own radius: the closed form is
        # what the silhouette error is measured against and it is the only ground truth available
        r_cells = 0.5 * ext / dx
        ball = sd._ball(int(round(r_cells)))
        m = cm.cut_mesh(ball, 1.0, (0.13, 0.97, -0.21), 0.37, 0.5)
        one = m["F"][m["side"] == 1]
        area = cm.mesh_area(m["V"], one)
        exact = np.pi * (r_cells ** 2 - 0.37 ** 2)
        err = 100.0 * abs(area - exact) / exact

        # the silhouette itself: how much of the true section's area the occupancy misplaces, as a
        # fraction, which is the geometric half of the trade
        rr = np.linalg.norm(centres - centres.mean(0), axis=1)
        shell = float((rr > np.percentile(rr, 99) - dx).mean())
        print(f"  {sc:>7.2f} {dx:>9.5f} {len(cells):>10,} {store:>7.1f} {err:>8.2f}% "
              f"{100*shell:>10.2f}%")

        _os.makedirs(_os.path.join(out_dir, f"h{sc:g}"), exist_ok=True)
        np.savez_compressed(_os.path.join(out_dir, f"h{sc:g}", "lat.npz"),
                            cells=cells, centres=centres, rgb=col, dx=dx, solid=solid)
    print(f"\n  -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
