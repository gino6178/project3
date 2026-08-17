"""Emit a physics config for any object, so the pipeline is not tied to one fruit.

Every field the pipeline reads is recoverable from the point cloud: the vertical axis is
the direction of least variance, the fill boundary is the bounding box after the same
normalisation the renderer applies, and the camera distance follows the object's size.
The camera intrinsics are shared -- they describe the lens, not the fruit.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os, json
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)
import torch
from scene.gaussian_model import GaussianModel
from utils.transformation_utils import transform2origin, shift2center111

DEV = "cuda:0"


def main(ply, out_json, lattice_pt=None):
    g = GaussianModel(0); g.load_ply_zero_sh(ply)
    x = g.get_xyz.detach().to(DEV)
    if lattice_pt and os.path.exists(lattice_pt):
        up = torch.load(lattice_pt)["up"].to(DEV)
    else:
        d = x - x.mean(0)
        ev, evec = torch.linalg.eigh((d.T @ d) / d.shape[0])
        up = evec[:, 0]
        # On a round object the least-variance direction is noise. The orange's principal
        # extents are 0.360 / 0.381 / 0.386 -- the shortest is 93% of the longest, which
        # is not a shape with an axis, and PCA still returned a confident-looking
        # [0.96, -0.16, -0.23]. Where the extents agree this closely the object does not
        # name an axis, so use the one the renderer already calls up; that is also the
        # value the hand-tuned orange config carries, arrived at independently.
        # The watermelon is a genuine 0.85, so its own axis is kept.
        if float((ev[0] / ev[2]).sqrt()) > 0.90:
            up = torch.tensor([0., -1., 0.], device=x.device)
            print(f"  near-spherical (extent ratio {float((ev[0]/ev[2]).sqrt()):.3f}), "
                  f"using world up [0,-1,0]")
    up = up / up.norm()

    # Where azimuth zero points, in the plane perpendicular to the vertical. The vertical
    # fixes the slicing axis and says nothing about the rotation about it, and the renderer's
    # default -- the projection of the constant [1, 1, 1] -- is 45 degrees from the loaf's own
    # in-plane axes, which rendered its square cross-section as a diamond and turned its
    # interior against its own shell.
    #
    # One rule, the same shape as the one above it: use the object's in-plane principal axis,
    # and where the object does not have one, say so rather than inventing it. A solid of
    # revolution has isotropic in-plane variance and any direction is as good as any other, so
    # it falls back to exactly what the renderer already used -- which is why writing this
    # field for the sphere and the torus leaves their frames, and every reference generated in
    # them, bit-for-bit unchanged.
    d = x - x.mean(0)
    perp = d - (d @ up)[:, None] * up
    pev, pevec = torch.linalg.eigh((perp.T @ perp) / perp.shape[0])
    aniso = float((pev[1] / pev[2]).sqrt())          # 1.0 = round in plane, lower = has axes
    if aniso < 0.95:
        inplane = pevec[:, 2]
    else:
        inplane = torch.tensor([1.0, 1.0, 1.0], device=x.device)
        print(f"  in-plane isotropic (ratio {aniso:.3f}), keeping the renderer's default")
    inplane = inplane - (inplane @ up) * up
    inplane = inplane / inplane.norm()

    # Frame the object by its size in the coordinates the renderer actually uses.
    # transform2origin normalises only the internal mpm space; the render path inverts
    # that and works in the original scale, where the watermelon is 1.47x the orange.
    # A fixed camera distance therefore filled 87-95% of the frame for the watermelon
    # against 24-30% for the orange, and the cross-section supervision was comparing a
    # blown-up slab against a whole-fruit photograph.
    world_ext = float((x.max(0).values - x.min(0).values).max())
    _tp, so_scale, _om = transform2origin(x.clone())
    m = shift2center111(_tp)
    lo, hi = m.min(0).values, m.max(0).values
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad
    cfg = {
        "opacity_threshold": 0.1,
        "rotation_degree": [0.0], "rotation_axis": [0],
        "grid_lim": 1.0,
        # Frame every object the way the orange is framed. The one quantity that decides
        # how much of the picture the object covers is diameter over camera distance, and
        # the orange -- whose camera the paper's authors set by hand -- sits at 0.5526.
        # init_radius is not that distance: it is scaled on the way through
        # transform2origin by `so`, which is 0.708 for the orange and 0.482 for the
        # watermelon, so carrying the number across unchanged, or scaling it by the
        # diameter ratio, both miss.
        "init_radius": round(world_ext / 0.5526 / float(so_scale), 4),
        "mpm_space_vertical_upward_axis": [round(float(v), 6) for v in up],
        "mpm_space_inplane_axis": [round(float(v), 6) for v in inplane],
        "mpm_space_viewpoint_center": [round(float(v), 4) for v in m.mean(0)],
        "particle_filling": {
            "n_grid": 100, "density_threshold": 1.0, "search_threshold": 1,
            "search_exclude_direction": 2, "ray_cast_direction": 3,
            "max_particles_num": 19000000, "max_partciels_per_cell": 1,
            "smooth": True, "visualize": True,
            "boundary": [round(float(lo[0]), 4), round(float(hi[0]), 4),
                         round(float(lo[1]), 4), round(float(hi[1]), 4),
                         round(float(lo[2]), 4), round(float(hi[2]), 4)],
        },
    }
    json.dump(cfg, open(out_json, "w"), indent=1)
    print(f"{out_json}: up {cfg['mpm_space_vertical_upward_axis']}  "
          f"in-plane {cfg['mpm_space_inplane_axis']}  "
          f"centre {cfg['mpm_space_viewpoint_center']}")
    print(f"  boundary {cfg['particle_filling']['boundary']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
