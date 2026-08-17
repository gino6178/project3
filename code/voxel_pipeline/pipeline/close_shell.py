"""Seal the holes a scan leaves in the shell, on the lattice, before anything is painted.

A reconstruction's outer surface is not guaranteed to be closed. The released watermelon's is
not: quantised to the lattice, its skin cells cover 84.5% of the object's own silhouette against
the orange's 100%, and what shows through the remaining 15.5% is whatever the interior happens to
be. That is invisible while the interior is a uniform colour and obvious the moment it is trained
to be red.

Chasing it upstream does not work well. The gaps come from the scan, so a deeper shell extraction
only carries more of the same surface into the fill -- and the fill's cost grows with it, which is
where it ran out of memory. The gaps are also small: a few cells across, which is exactly the
regime a morphological closing handles.

So close the occupancy grid: dilate by `radius`, erode by the same, and keep whatever the round
trip added. A cell added this way is on the boundary by construction, so it is labelled skin, and
because this runs before the exterior projection it is painted from the reference renders like any
other skin cell rather than being given an invented colour.

Closing is the right operator rather than hole-filling by connectivity: a hole in a shell connects
the inside to the outside, so a flood fill from outside reaches straight through it and finds
nothing enclosed to fill.

    python voxel_pipeline/pipeline/close_shell.py vox_in vox_out [radius]
"""
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("/home/gino/project/FruitNinja_clean")
sys.path.append("/home/gino/project/FruitNinja_clean/gaussian-splatting")
os.chdir("/home/gino/project/FruitNinja_clean")

from scene.gaussian_model import GaussianModel   # noqa: E402

DEV = "cuda:0"
C0 = 0.28209479177387814


def main(src, dst, radius=2):
    os.makedirs(dst, exist_ok=True)
    g = GaussianModel(0)
    g.load_ply_zero_sh(os.path.join(src, "gs_fill.ply"))
    lat = torch.load(os.path.join(src, "lattice.pt"))
    lvl = torch.load(os.path.join(src, "cell_level.pt")).reshape(-1)

    xyz = g.get_xyz.detach().to(DEV)
    dx = float(lat["fine_dx"])
    # Close the shell, not the lattice. On a two-level lattice the interior sits on the coarse
    # grid, so on the fine grid its cells are a spacing apart and a closing at this radius fills
    # every gap between them -- 5.4M cells added on the watermelon, 561% of the lattice, which is
    # the interior being made solid at the skin's resolution rather than a shell being sealed.
    # The operator belongs on the skin's own occupancy.
    skin = (lvl.to(DEV).reshape(-1)[:xyz.shape[0]] == 1)
    sx = xyz[skin]
    mn = xyz.min(0).values - dx * (radius + 2)
    idx = ((sx - mn) / dx).round().long()
    all_idx = ((xyz - mn) / dx).round().long()
    dims = (all_idx.max(0).values + radius + 3).tolist()
    occ = torch.zeros(dims, dtype=torch.bool, device=DEV)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    # anything the lattice already holds, at any level, is not a hole
    held = torch.zeros(dims, dtype=torch.bool, device=DEV)
    held[all_idx[:, 0], all_idx[:, 1], all_idx[:, 2]] = True

    k = 2 * radius + 1
    o = occ.float()[None, None]
    dil = (F.max_pool3d(o, k, 1, radius) > 0.5)
    ero = (-F.max_pool3d(-dil.float(), k, 1, radius) > 0.5)[0, 0]
    added = ero & ~held
    n_add = int(added.sum())
    if n_add == 0:
        print(f"  nothing to close at radius {radius}; copying through")
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        return

    ai = added.nonzero()
    new_xyz = mn[None] + ai.float() * dx

    # A new cell takes the mean of the occupied cells around it, which is a placeholder: it sits
    # on the boundary, so the exterior projection overwrites it. Seeding it from its neighbours
    # rather than from nothing only matters if the projection misses it.
    with torch.no_grad():
        old_rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
        # In chunks: at radius 3 the closing adds enough cells that one cdist against the
        # whole lattice asks for 15 GB, and the operation is a nearest-neighbour lookup that
        # has no reason to be done all at once.
        sel = torch.randperm(xyz.shape[0], device=DEV)[:150000]
        ref, refc = xyz[sel], old_rgb[sel]
        new_rgb = torch.empty(n_add, 3, device=DEV)
        for a in range(0, n_add, 4000):
            b = min(a + 4000, n_add)
            new_rgb[a:b] = refc[torch.cdist(new_xyz[a:b], ref).argmin(1)]

        cat = lambda a, b: torch.cat([a, b])
        g._xyz = nn.Parameter(cat(xyz, new_xyz).contiguous())
        g._features_dc = nn.Parameter(
            cat(g._features_dc.detach().to(DEV), ((new_rgb - 0.5) / C0).unsqueeze(1)).contiguous())
        fr = g._features_rest.detach().to(DEV)
        g._features_rest = nn.Parameter(
            cat(fr, torch.zeros(n_add, fr.shape[1], 3, device=DEV)).contiguous()
            if fr.shape[0] else fr)
        g._opacity = nn.Parameter(cat(g._opacity.detach().to(DEV),
                                      torch.full((n_add, 1), 3.0, device=DEV)).contiguous())
        sc = g._scaling.detach().to(DEV)
        g._scaling = nn.Parameter(cat(sc, torch.full((n_add, 3), float(sc.median()),
                                                     device=DEV)).contiguous())
        ro = g._rotation.detach().to(DEV)
        g._rotation = nn.Parameter(cat(ro, ro[:1].repeat(n_add, 1)).contiguous())
        g.max_radii2D = torch.zeros(g._xyz.shape[0], device=DEV)

    g.save_ply(os.path.join(dst, "gs_fill.ply"))
    torch.save(torch.cat([lvl, torch.ones(n_add, dtype=lvl.dtype)]),
               os.path.join(dst, "cell_level.pt"))
    torch.save(torch.ones(g._xyz.shape[0], dtype=torch.bool),
               os.path.join(dst, "is_interior.pt"))
    torch.save(lat, os.path.join(dst, "lattice.pt"))
    print(f"  closed at radius {radius}: {n_add:,} cells added as skin "
          f"({100 * n_add / xyz.shape[0]:.1f}% of the lattice)  -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         radius=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
