"""Which way a shell cell faces, taken from the occupancy.

The one function route 2 still needs from the cone skinner that preceded skin_project.py.
It is here rather than imported because the other nine hundred lines of that file are the
mapping this pipeline replaced, and nothing in these two routes calls them.
"""
import os

import torch
from scipy import ndimage


def surface_normals(world, xs, dx, centre,
                    sigma=float(os.environ.get("SKIN_NORMAL_SIGMA", "5.0"))):
    """Which way a shell cell faces, from the lattice's occupancy rather than from the centroid.

    `xs - centre` is the surface normal exactly when the centroid lies inside the material.
    A sphere, an apple and a melon satisfy that; a ring does not -- its centroid sits in the
    hole, 0.3373 away from the nearest cell it has, and every direction from there to the
    surface comes out nearly horizontal. The consequence was not a small error: with a 60
    degree cone, `up` and `down` caught *zero* cells of the doughnut, so two of its six
    references were never read at all and its top was coloured by the upper halves of the
    four side views, which is what the white blotches across the glaze were.

    Occupancy has no such assumption. Bin the cells onto the lattice's own grid, smooth, and
    take the gradient: it points from filled toward empty everywhere on the boundary, which
    is the outward normal for any topology.

    But it is computed on a grid, so on a smooth surface it carries the grid's own steps,
    and replacing the analytic rule with it outright put concentric ridges across the orange
    -- the cone counts did not move at all, so that was the noise and nothing else. Neither
    rule is right everywhere and each says where it fails: where the two agree, the centroid
    ray leaves the material at that cell and the analytic normal is both correct and smooth;
    where they disagree, the centroid is not inside the material along that ray and only the
    gradient means anything. So cross over between them by their own agreement, over the same
    cone this file already uses for the faces. A sphere agrees everywhere and gets exactly
    the old behaviour; a ring's inner wall and its top and bottom disagree and get the
    gradient.
    """
    lo = world.min(0).values
    n = ((world.max(0).values - lo) / dx).long() + 3
    idx = ((world - lo) / dx).long() + 1
    occ = torch.zeros(tuple(int(v) for v in n), device=world.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    occ = torch.from_numpy(
        ndimage.gaussian_filter(occ.cpu().numpy(), sigma=sigma)).to(world.device)
    gx, gy, gz = torch.gradient(occ)
    si = ((xs - lo) / dx).long() + 1
    g = -torch.stack([gx[si[:, 0], si[:, 1], si[:, 2]],
                      gy[si[:, 0], si[:, 1], si[:, 2]],
                      gz[si[:, 0], si[:, 1], si[:, 2]]], 1)     # filled -> empty
    ln = g.norm(dim=1, keepdim=True)
    n_c = xs - centre
    n_c = n_c / n_c.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # a cell the gradient cannot resolve keeps the analytic rule rather than a random direction
    n_g = torch.where(ln > 1e-8, g / ln.clamp_min(1e-12), n_c)

    a = (n_c * n_g).sum(1, keepdim=True)
    w = ((a - COS_CONE) / (1.0 - COS_CONE)).clamp(0.0, 1.0)     # 1 -> analytic, 0 -> gradient
    n = w * n_c + (1.0 - w) * n_g
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    print(f"  法向：{int((w > 0.999).sum()):,} 格用質心解析、"
          f"{int((w < 0.001).sum()):,} 格用佔據梯度、"
          f"{int(((w >= 0.001) & (w <= 0.999)).sum()):,} 格混合"
          f"   (梯度解析成功 {int((ln > 1e-8).sum()):,}/{xs.shape[0]:,})")
    return n
