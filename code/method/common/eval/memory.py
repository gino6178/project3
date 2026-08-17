"""What the representation costs to hold, permanently and while a cut is live.

Section 11.2 asks for two numbers and the project reported neither: "permanent interior parameters
+ temporary cut-band refinement memory". The distinction is the whole argument for a coarse
interior -- storage should be proportional to hidden spatial complexity, with the high-resolution
cost paid only where a cut has made something visible, and paid back when it has not.

So both are measured here, and so is the third thing that follows from them: whether the temporary
cost is actually returned. A structure that grows on every cut and never shrinks is not a band, it
is a leak with a good story.

Counted as fields rather than as process memory, because process memory measures the allocator.
Per cell the cube representation stores an occupancy bit, a level, and the feature the decoder
reads; per primitive the Gaussian representation stores position, scale, rotation, opacity and
colour. Positions are implicit on a lattice and explicit in a point cloud, which is part of the
comparison rather than an accounting choice.

    python method/common/eval/memory.py LATTICE [MODEL.ply]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

FEAT_DIM = int(_os.environ.get("ANCHOR_DIM", "8"))
MiB = 1024.0 ** 2


def main(lattice_dir, model_ply=None):
    from plyfile import PlyData
    from method.common.cube.occupancy import close_and_fill, to_grid
    from method.common.cube import subdivide as sd

    ply = model_ply or _os.path.join(lattice_dir, "gs_fill.ply")
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(ply).elements[0]
    names = [q.name for q in el.properties]
    n_rest = len([q for q in names if q.startswith("f_rest_")])
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    N = len(xyz)

    # --- permanent -----------------------------------------------------------------------------
    gs_floats = 3 + 3 + 4 + 1 + 3 + n_rest          # xyz, scale, rotation, opacity, f_dc, f_rest
    gs = N * gs_floats * 4
    # The index is not free. On a dense grid a cell's position is its address and costs nothing,
    # but this lattice is sparse -- it is stored as a point cloud and every structure built over
    # it (CollisionIndex, the renderer's Lookup) keys on explicit integer coordinates -- so the
    # honest permanent figure pays for those coordinates. The temporary accounting below always
    # did, at 12 bytes a leaf, and reporting one with the index and one without was the
    # inconsistency: a representation does not get to be sparse when that is cheaper to store and
    # dense when that is cheaper to address.
    idx = 3 * 4
    cube_noidx = N * (FEAT_DIM * 4 + 1 + 1)
    cube = N * (FEAT_DIM * 4 + 1 + 1 + idx)
    print(f"  {N:,} cells, {int((lvl == 0).sum()):,} coarse + {int((lvl == 1).sum()):,} fine")
    print(f"  permanent, as Gaussians : {gs_floats} floats each = {gs / MiB:8.1f} MiB")
    print(f"  permanent, as cube cells: {FEAT_DIM} floats + 2 bytes = {cube_noidx / MiB:8.1f} MiB"
          f"   ({gs / cube_noidx:.1f}x less)  -- index implicit, dense-grid accounting")
    print(f"  permanent, with the index: + {idx} bytes    = {cube / MiB:8.1f} MiB"
          f"   ({gs / cube:.1f}x less)  -- what a sparse lattice actually costs")

    # --- and the surface it needs ----------------------------------------------------------------
    # The exterior is not optional: the method converts the whole boundary and binds every one of
    # its vertices, so a total that counts only cells is being compared against a Gaussian model
    # that carries its own exterior. A dual-grid vertex is a position and a colour; a triangle is
    # three indices.
    surf = _os.environ.get("SURF_NPZ")
    if surf and _os.path.exists(surf):
        d = np.load(surf, allow_pickle=True)
        nv = int(d["mesh_v"].shape[0]) if "mesh_v" in d.files else 0
        nf = int(d["mesh_f"].shape[0]) if "mesh_f" in d.files else 0
        sb = nv * (3 * 4 + 3) + nf * 3 * 4
        print(f"\n  the dual-grid exterior: {nv:,} vertices and {nf:,} triangles = "
              f"{sb / MiB:.1f} MiB")
        print(f"  the method's total, cells with index + surface: {(cube + sb) / MiB:.1f} MiB"
              f"   ({gs / (cube + sb):.1f}x less than the Gaussian model)")
    else:
        print("\n  the dual-grid exterior is not counted here: set SURF_NPZ to the O-Voxel file")

    # --- temporary -----------------------------------------------------------------------------
    org = xyz[lvl == 0].min(0) - 0.5 * hf
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1
    n0 = len(solid)
    n_ax = np.array([0.13, 0.97, -0.21])
    n_ax /= np.linalg.norm(n_ax)
    d = float(-((solid + 0.5) * hc).mean(0) @ n_ax)
    r = sd.cut(solid, hc, n_ax, d, hf)
    added = len(r["leaf"]) - n0
    # a leaf carries its index, its level and its piece; the feature is its parent's until a
    # cut-time write-back changes it, which is the point of inheriting rather than copying
    leaf_bytes = 3 * 4 + 1 + 4
    print(f"\n  a cut on {n0:,} solid cells refines to {len(r['leaf']):,} leaves, "
          f"+{added:,} ({100 * added / n0:.1f}%)")
    print(f"  temporary, while that cut is live: {added * leaf_bytes / MiB:.1f} MiB "
          f"({100 * added * leaf_bytes / cube:.1f}% of the permanent interior)")

    # --- and is it returned? --------------------------------------------------------------------
    try:
        from method.common.cube import multicut as mc
        back = mc.cut(solid, hc, [], hf)
        n_back = len(back["leaf"])
        print(f"  with no cut live the blocks fold back to {n_back:,} leaves, against {n0:,} "
              f"uncut -- {'exactly returned' if n_back == n0 else f'{n_back - n0:+,} left over'}")
    except Exception as e:                                       # noqa: BLE001
        print(f"  fold-back not measured: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
