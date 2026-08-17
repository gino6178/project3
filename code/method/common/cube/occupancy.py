"""M0's completion condition: make the occupancy a solid, and show that it is.

The lattice is not solid and never was. `internal_filling` places one particle per *empty* cell
and skips cells that already hold a primitive, so an occupied cell has on average 18.5 of its 26
neighbours occupied -- the volume is a sponge. Gaussians hid this: their support overlaps, so a
missing cell was covered by its neighbours' skirts. A cube's support is exactly its own cell, so
the same occupancy renders with 5.5% to 14.9% of each cross-section as holes.

That is not a rendering bug to work around. It is the spec's point arriving early: eq. (2) says
a Gaussian must choose between coverage and blending, and the reason the cube representation can
refuse that choice is that its coverage comes from the occupancy being right rather than from
primitives being fat. So the occupancy has to be made right.

Two operations, in this order, on the coarse grid:

  close   a binary closing (dilate then erode) at radius r seals the pinholes that skipping
          already-occupied cells left behind. It cannot open a passage to the outside, so a
          genuine hole through the object -- the doughnut's -- survives it.
  fill    anything enclosed by the closed shell is interior by definition. A flood fill from
          the grid's border marks the outside; whatever is neither outside nor occupied is a
          void inside the object, and becomes occupied.

`close_and_fill` is idempotent, which is the completion condition worth asserting: running it
twice changes nothing, so the result is a fixed point rather than a step that happens to help.

    python method/common/cube/occupancy.py LATTICE OUT_DIR [radius]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]
_os.chdir(_FN_ROOT)

DEV = "cuda:0"


def to_grid(xyz, h):
    """Occupancy as a dense boolean volume, with the origin and spacing to get back."""
    mn = xyz.min(0).values - h
    idx = ((xyz - mn) / h).round().long()
    dims = (idx.max(0).values + 2).tolist()
    occ = torch.zeros(dims, dtype=torch.bool, device=xyz.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ, mn, idx


def close_and_fill(occ, radius=1, keep_cavities=0):
    """Seal the pinholes, then fill what is enclosed. Idempotent by construction.

    `keep_cavities` is the size, in cells, above which an enclosed void is kept rather than
    filled. It defaults to zero, which fills everything and is what a photogrammetric
    reconstruction needs: the voids behind a scanned surface are the reconstruction's failure to
    see through the object and every one of them is spurious.

    It is not what a real interior needs. Bread has pores, a pepper has a seed cavity, an organ has
    a lumen, and this pass destroys all of them by construction -- the occupancy it produces is the
    complement of the exterior, so genus survives only for holes that pass all the way through.
    Setting a threshold separates the two populations by the only property that distinguishes them
    here: a reconstruction's voids are small and numerous and a real cavity is large and few. The
    number is a property of the object and there is no way to infer it from the occupancy alone,
    which is why it is an argument and not a heuristic.
    """
    k = 2 * radius + 1
    o = occ.float()[None, None]
    dil = F.max_pool3d(o, k, 1, radius) > 0.5
    ero = -F.max_pool3d(-dil.float(), k, 1, radius) > 0.5
    closed = ero[0, 0] | occ

    # Flood the outside from the border. Iterated dilation of the border against the free space
    # reaches everything connected to it; what it never reaches is enclosed.
    free = ~closed
    outside = torch.zeros_like(free)
    outside[0, :, :] = free[0, :, :]; outside[-1, :, :] = free[-1, :, :]
    outside[:, 0, :] |= free[:, 0, :]; outside[:, -1, :] |= free[:, -1, :]
    outside[:, :, 0] |= free[:, :, 0]; outside[:, :, -1] |= free[:, :, -1]
    while True:
        grown = (F.max_pool3d(outside.float()[None, None], 3, 1, 1)[0, 0] > 0.5) & free
        if bool((grown == outside).all()):
            break
        outside = grown
    enclosed = free & ~outside
    if keep_cavities > 0 and bool(enclosed.any()):
        # keep the enclosed components at or above the threshold, fill the rest. Labelling is on
        # the host because an enclosed set is small by definition and this runs once per object.
        import numpy as np
        from scipy import ndimage
        lab, n = ndimage.label(enclosed.cpu().numpy())
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            big = np.isin(lab, np.nonzero(sizes >= keep_cavities)[0])
            kept = int(big.sum())
            print(f"  cavities: {n} enclosed components, {int((sizes >= keep_cavities).sum())} "
                  f"of them >= {keep_cavities} cells kept ({kept:,} cells), the rest filled")
            enclosed = enclosed & ~torch.from_numpy(big).to(enclosed.device)
    return closed | enclosed


def main(lattice_dir, out_dir, radius=1):
    _os.makedirs(out_dir, exist_ok=True)
    from scene.gaussian_model import GaussianModel
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)
    g = GaussianModel(0)
    g.load_ply_zero_sh(_os.path.join(lattice_dir, "gs_fill.ply"))
    xyz = g.get_xyz.detach().to(DEV)
    lvl = lvl[:xyz.shape[0]].to(DEV)

    added = {}
    for level, dxk in ((0, "coarse_dx"), (1, "fine_dx")):
        sel = lvl == level
        if not bool(sel.any()):
            continue
        h = float(lat[dxk])
        occ, mn, _ = to_grid(xyz[sel], h)
        n0 = int(occ.sum())
        solid = close_and_fill(occ, radius)
        n1 = int(solid.sum())
        # the completion condition: a second pass must change nothing
        again = close_and_fill(solid, radius)
        fixed = bool((again == solid).all())
        new = solid & ~occ
        ni = new.nonzero()
        added[level] = (mn.cpu(), h, ni.cpu())
        print(f"  level {level}: {n0:,} occupied -> {n1:,} solid  (+{n1 - n0:,}, "
              f"{100 * (n1 - n0) / max(n0, 1):.1f}%)   fixed point: {fixed}")

    torch.save({"radius": radius,
                "added": {k: (v[0], v[1], v[2]) for k, v in added.items()}},
               _os.path.join(out_dir, "occupancy_fill.pt"))
    print(f"  -> {out_dir}/occupancy_fill.pt")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
