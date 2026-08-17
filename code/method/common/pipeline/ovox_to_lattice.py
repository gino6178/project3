"""Close the loop: the O-Voxel surface becomes the lattice the next cut is made on.

Two paths built the three lattices and only one of them was ever meant to. The orange and the
watermelon were quantised from released Gaussian models; the doughnut came from the older route --
a generated shell, a ray-cast interior fill, a voxeliser -- whose occupancy thins toward the
surface, and it is the object whose cross-sections still show 9% holes after the occupancy is
closed and filled. Morphological closing is a repair, and a repair is the wrong tool when the
information was never missing.

A closed surface answers inside-or-outside outright. `mesh_to_voxel.occupancy` rasterises the
triangles densely enough that no cell is skipped, floods from outside the grid, and calls
everything the flood cannot reach interior: solid by construction, for any shape, with no radius,
no centre, no axis and no closing. The only thing it needs is a closed surface, and now there is
one -- the whole exterior is an O-Voxel dual grid, and `flexible_dual_grid_to_mesh` hands back its
triangles.

So the pipeline becomes a loop rather than a chain. The surface the object is rendered with is the
surface its next occupancy is built from, and every object goes the same way regardless of whether
it started as a photograph or a mesh.

    python method/common/cube/globalovox.py LATTICE out.npz     # exterior -> O-Voxel + mesh
    python method/common/pipeline/ovox_to_lattice.py out.npz LATTICE NEW_DIR
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]


def write_ply(path, V, F, C=None):
    """The mesh, with vertex colours if it has them, in the form trimesh will read back."""
    V = np.asarray(V, np.float32)
    F = np.asarray(F, np.int32)
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(V)}",
           "property float x", "property float y", "property float z"]
    if C is not None:
        hdr += ["property uchar red", "property uchar green", "property uchar blue"]
    hdr += [f"element face {len(F)}", "property list uchar int vertex_indices", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        if C is None:
            f.write(V.tobytes())
        else:
            c = np.clip(np.asarray(C) * 255, 0, 255).astype(np.uint8)
            rec = np.empty(len(V), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = V[:, 0], V[:, 1], V[:, 2]
            rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
            f.write(rec.tobytes())
        rec = np.empty(len(F), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        rec["n"] = 3
        rec["a"], rec["b"], rec["c"] = F[:, 0], F[:, 1], F[:, 2]
        f.write(rec.tobytes())
    return path


def main(npz_path, ref_lattice, out_dir, cells=None, refine=2):
    import torch
    z = np.load(npz_path)
    if "mesh_v" not in z.files:
        raise SystemExit(f"{npz_path} carries no mesh; run globalovox on a CUDA device")
    V, F = z["mesh_v"].astype(np.float64), z["mesh_f"].astype(np.int64)
    rgb = z["rgb"] if "rgb" in z.files else None
    # a node's colour belongs to its dual vertex, and the mesh's vertices are those nodes
    C = rgb[:len(V)] if rgb is not None and len(rgb) >= len(V) else None
    print(f"  {len(V):,} vertices, {len(F):,} triangles"
          + ("" if C is None else ", with colour"))

    lat = torch.load(_os.path.join(ref_lattice, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    ext = V.max(0) - V.min(0)
    # the same coarse spacing the reference lattice used, expressed the way mesh_to_voxel asks
    n_cells = cells or int(round(np.prod(ext) / hc ** 3))
    print(f"  matching h_c {hc:.5f} and h_f {hf:.5f}: about {n_cells:,} coarse cells over "
          f"an extent of {ext.round(4).tolist()}")

    ply = _os.path.join(_os.path.dirname(out_dir) or ".", _os.path.basename(out_dir) + "_surface.ply")
    write_ply(ply, V, F, C)
    print(f"  -> {ply}")

    if _os.environ.get("EXPORT_ONLY", "0") == "1":
        # Two interpreters, one loop. o_voxel is built for the python that has no simple_knn and
        # mesh_to_voxel imports the Gaussian model, which needs it, so the mesh is written by one
        # and voxelised by the other. Splitting on a file is the same shape globalovox already
        # uses and is why the mesh is saved rather than passed.
        print(f"  export only; voxelise it with:\n"
              f"    python method/common/pipeline/mesh_to_voxel.py {ply} {out_dir} "
              f"--cells {n_cells} --refine {refine} --diameter {float(ext.max()):.6f}")
        return ply
    from method.common.pipeline import mesh_to_voxel as m2v
    m2v.main(ply, out_dir, cells=n_cells, refine=refine, diameter=float(ext.max()))
    return out_dir


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         cells=int(sys.argv[4]) if len(sys.argv) > 4 else None)
