"""Beyond v1: the whole visible boundary as O-Voxel, not only the cut patch.

The first version deliberately left the exterior as the Gaussian model it already was (spec 1.2,
and again as pitfall 13.3), so that debugging the surface conversion would not tangle with
reconstructing the original exterior. That has held up -- M3 to M7 are done and the cut path is
measured -- so this is the step it was deferred in favour of: convert the exterior too, and the
object becomes one representation with two roles.

    cube volume   everything hidden: occupancy, connectivity, piece identity, collision
    O-Voxel       everything visible: the original outer surface and every new cut face

Two of the limitations the spec asks v1 to admit (12.2) are about exactly this split, and both
close here. "原始 exterior 與 cut surface 使用不同 renderer" stops being true when there is one
surface representation. And M6's honest failure -- the skin Gaussians cover only 7% to 20% of the
frame at alpha above a half, so the cut face shows through from the side it should be hidden on
-- was a property of a semi-transparent exterior, not of the depth test. A dual-grid surface is
opaque because it is a surface.

The boundary comes from the occupancy rather than from a mesh library: a cell face whose
neighbour is empty is on the boundary, and there is no ambiguity to resolve and nothing to
install. It is blocky at the cell size, which is the honest shape of the cube representation, and
the dual grid's QEF then places each dual vertex by fitting the faces near it -- which is what
turns a staircase back into a surface. How well is measured here rather than asserted: the
distance from each dual vertex to the nearest primitive of the original model.

    python method/common/cube/globalovox.py LATTICE [OUT.npz] [TRAINED.ply]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

import ovoxel as ov                         # noqa: E402

# the six face directions, and for each the four corner offsets of that face, wound outwards
FACE = [
    (np.array([1, 0, 0]), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    (np.array([-1, 0, 0]), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    (np.array([0, 1, 0]), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    (np.array([0, -1, 0]), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    (np.array([0, 0, 1]), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    (np.array([0, 0, -1]), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]


def boundary_mesh(coords, h, origin=None):
    """The occupancy's own boundary: every face of an occupied cell whose neighbour is empty.

    Exact, and no marching-cubes ambiguity to resolve -- the surface of a set of cubes is a set
    of quads. Returns merged vertices and triangles, wound outwards.
    """
    origin = np.zeros(3) if origin is None else np.asarray(origin, np.float64)
    c = np.asarray(coords, np.int64)
    mn = c.min(0) - 2
    span = (c.max(0) - mn + 3).astype(np.int64)
    key = ((c[:, 0] - mn[0]) * span[1] + (c[:, 1] - mn[1])) * span[2] + (c[:, 2] - mn[2])
    ks = np.sort(key)

    tris = []
    for d, corners in FACE:
        nb = c + d
        k = ((nb[:, 0] - mn[0]) * span[1] + (nb[:, 1] - mn[1])) * span[2] + (nb[:, 2] - mn[2])
        pos = np.clip(np.searchsorted(ks, k), 0, len(ks) - 1)
        exposed = ks[pos] != k
        if not exposed.any():
            continue
        base = c[exposed].astype(np.float64)
        q = [(base + np.asarray(off, np.float64)) * h + origin for off in corners]
        tris.append(np.stack([q[0], q[1], q[2]], 1))
        tris.append(np.stack([q[0], q[2], q[3]], 1))
    tri = np.concatenate(tris)

    flat = tri.reshape(-1, 3)
    k = np.round(flat / (h * 1e-6)).astype(np.int64)
    _, first, inv = np.unique(k, axis=0, return_index=True, return_inverse=True)
    return flat[first], inv.reshape(-1, 3)


def main(lattice_dir, out_npz=None, device="cpu", colour_from=None):
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from occupancy import close_and_fill, to_grid
    import subdivide as sd

    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    C0 = 0.28209479177387814
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                  * C0 + 0.5, 0, 1)
    # The surface's colour, from the model that was trained rather than from the lattice it
    # started on. A boundary cell is not always a skin cell -- the boundary is the occupancy's,
    # at any level -- so on a generated lattice the cells the six views never painted are still
    # the flat grey make_shape wrote, and they show as grey bands wherever the surface is drawn
    # from a direction that reaches them. The trained model is row-aligned with its lattice, so
    # the substitution is by row and the row count is the check.
    if colour_from and _os.path.isfile(colour_from):
        e2 = PlyData.read(colour_from).elements[0]
        if len(e2["x"]) == len(xyz):
            rgb = np.clip(np.stack([e2["f_dc_0"], e2["f_dc_1"], e2["f_dc_2"]], 1)
                          .astype(np.float32) * C0 + 0.5, 0, 1)
            print(f"  colour from {colour_from}")
        else:
            print(f"  {colour_from} has {len(e2['x']):,} rows against the lattice's {len(xyz):,};"
                  f" keeping the lattice's own colour")
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)[:len(xyz)].numpy()

    # The boundary is built at the fine spacing, not the coarse one. The exterior is where the
    # object's detail is, and the skin already exists at h_f; extracting the surface at h_c
    # would throw away the resolution the shell was refined to have in the first place.
    # from a corner, not from a centre: floor of a centre sitting exactly on a cell
    # boundary lets floating point choose the side, and on a lattice whose cells are at
    # (i + 1/2)h that discards 49% of them. Offset by half the finest spacing used here.
    org = xyz.min(0) - 0.5 * hf
    fine = np.floor((xyz - org) / hf).astype(np.int64)
    coarse_solid = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coarse_solid).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coarse_solid.min(0) - 1
    # the filled interior, expressed at the fine spacing, plus the skin's own fine cells
    off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
    interior_fine = (solid[:, None, :] * 2 + off[None]).reshape(-1, 3)
    allf = np.unique(np.concatenate([interior_fine, fine]), axis=0)
    print(f"  {len(solid):,} solid coarse cells -> {len(allf):,} fine cells at h_f {hf:.5f}")

    V, F = boundary_mesh(allf, hf, org)
    print(f"  boundary: {len(F):,} triangles, {len(V):,} merged vertices")

    # The nearest *skin* cell, not the nearest cell. The boundary is the occupancy's, at any
    # level, and 13.1% of the surface's vertices are nearer to an interior cell than to a skin
    # one -- so a lookup over every cell gives those vertices whatever the interior happens to
    # hold. On a generated lattice that is the flat grey make_shape wrote, drawn as blue-grey
    # seams across the peel; with the trained model substituted it is pale pulp instead, drawn
    # as blotches. Neither is the exterior's colour. The cells that carry exterior appearance
    # are the ones the six views painted, and every surface vertex should take its colour from
    # one of those.
    skin = lvl == 1
    if skin.sum() > 0.2 * len(xyz):
        tree = cKDTree(xyz[skin])
        src_rgb = rgb[skin]
        print(f"  colour from the {int(skin.sum()):,} skin cells, not from all {len(xyz):,}")
    else:
        tree, src_rgb = cKDTree(xyz), rgb
        print(f"  only {int(skin.sum()):,} of {len(xyz):,} cells are skin; "
              f"colouring from all of them")
    patch = ov.patch_to_o_voxel(V, F, hf, colour=lambda p: src_rgb[tree.query(p, k=1)[1]],
                                device=device)
    n_ov = len(patch["voxel"])
    print(f"  O-Voxel exterior: {n_ov:,} active voxels, mean RGB "
          f"{patch['rgb'].mean(0).round(3)}")

    # Measured 2026-08-19, and it does not mean what it looks like it means. Against the same
    # 480,287 skin centres: the staircase corners this converter is *given* sit at 0.00609
    # (1.03 h_f), the centroids of those same triangles -- one line, no solver -- sit at 0.00432
    # (0.73), and what the QEF returns sits at 0.00517 (0.88). The dual vertex is closer than the
    # staircase and further than a trivial average of it, so this number is not evidence that the
    # grid recovered sub-cell geometry. It is a smoothing of the occupancy, and `pos = f(occupancy)`
    # exactly: the call below is given (V, F, h_f) and nothing else -- no image, no appearance.
    #
    # How close the dual grid puts the surface to the model it came from. The blockiness is the
    # input's; the QEF is what is being measured.
    dist, _ = tree.query(patch["pos"], k=1)
    print(f"  dual vertices to the nearest primitive of the original model: "
          f"mean {dist.mean():.5f}, 95th {np.percentile(dist, 95):.5f}, "
          f"in units of h_f {dist.mean() / hf:.2f} and {np.percentile(dist, 95) / hf:.2f}")

    ext_o = patch["pos"].max(0) - patch["pos"].min(0)
    ext_m = xyz.max(0) - xyz.min(0)
    print(f"  extent {ext_o.round(4)} against the model's {ext_m.round(4)} "
          f"({100 * np.abs(ext_o - ext_m).max() / ext_m.max():.2f}% worst)")

    # What it costs, against what it replaces. The exterior Gaussians carry position, scale,
    # rotation, opacity and spherical harmonics; a dual-grid voxel carries an index, a dual
    # vertex, three intersection flags and a colour.
    n_skin = int((lvl != 0).sum())
    per_gs = 3 + 3 + 4 + 1 + 3 * (1 + len([q.name for q in el.properties
                                           if q.name.startswith("f_rest_")]) // 3)
    per_ov = 3 + 3 + 3 / 8 + 3
    print(f"  exterior storage: {n_skin:,} skin Gaussians at ~{per_gs} floats = "
          f"{n_skin * per_gs / 2 ** 20:.1f} MiB  ->  {n_ov:,} voxels at ~{per_ov:.1f} = "
          f"{n_ov * per_ov * 4 / 2 ** 20:.1f} MiB")
    # the surface as a surface, not as a point set
    try:
        MV, MF = ov.dual_to_mesh(patch, device="cuda")
        print(f"  as a mesh: {len(MF):,} triangles, {len(MV):,} vertices")
    except Exception as e:
        MV = MF = None
        print(f"  mesh extraction unavailable here: {type(e).__name__}: {e}")

    if out_npz:
        # Saved rather than rendered here, because the O-Voxel extension is built for one
        # interpreter and the Gaussian camera code for another; see method/README.md.
        np.savez_compressed(out_npz, pos=patch["pos"].astype(np.float32),
                            rgb=patch["rgb"].astype(np.float32),
                            voxel=patch["voxel"].astype(np.int32),
                            inter=patch["inter"], voxel_size=patch["voxel_size"],
                            frac=patch["frac"].astype(np.float32),
                            origin=patch["origin"].astype(np.float64),
                            **({} if MV is None else dict(mesh_v=MV.astype(np.float32),
                                                          mesh_f=MF.astype(np.int32))))
        print(f"  -> {out_npz}")
    return patch


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
         colour_from=sys.argv[3] if len(sys.argv) > 3 else None)
