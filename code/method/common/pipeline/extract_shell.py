"""Recover the exterior scan from a released model that has already been filled.

The released watermelon is 7,959,789 primitives at a flat 2M per unit volume from the centre
right out to the rind: `internal_filling` has already run on it, so there is no density step
to separate the scan from the fill. What still separates them is that the scan only ever sat
on the surface. Binning the primitives by direction from the centroid and keeping those within
a fixed depth of the furthest one in their own bin recovers that surface, and follows the
object's real shape rather than a sphere fitted to it.

The depth is in world units and wants to be a little over one fill layer -- deep enough that
no direction comes back empty where the surface is oblique, shallow enough that the interior
does not come with it.

    python extract_shell.py in.ply out.ply [depth] [n_theta]
"""
import os as _os
# The repository root, so the same file runs here and on the remote box. It was written three
# times in every script -- two sys.path entries and a chdir -- and each one silently pinned the
# script to one machine: on the remote the chdir landed somewhere else and a relative source
# path then could not be found, which surfaces as "no such file" for a file that is plainly
# there.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import os
import sys

import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]
os.chdir(_FN_ROOT)

from scene.gaussian_model import GaussianModel   # noqa: E402
from torch import nn                             # noqa: E402


def main(src, dst, depth=0.045, nth=360):
    g = GaussianModel(0)
    g.load_ply_zero_sh(src)
    x = g.get_xyz.detach()
    c = x.mean(0)
    d = x - c
    r = d.norm(dim=1)
    u = d / r.clamp_min(1e-9)[:, None]
    # equal-area direction bins: cos(elevation) is uniform on the sphere, azimuth already is
    iz = ((u[:, 1] * 0.5 + 0.5) * (nth // 2)).long().clamp(0, nth // 2 - 1)
    ia = ((torch.atan2(u[:, 2], u[:, 0]) / (2 * 3.14159265) + 0.5) * nth).long().clamp(0, nth - 1)
    key = iz * nth + ia
    nb = (nth // 2) * nth
    far = torch.zeros(nb, device=x.device).index_reduce_(
        0, key, r, "amax", include_self=False)
    keep = r >= (far[key] - depth)
    print(f"{src}: {x.shape[0]:,} -> {int(keep.sum()):,} shell primitives "
          f"({100 * float(keep.float().mean()):.1f}%), depth {depth}, "
          f"{int((far > 0).sum()):,}/{nb:,} directions occupied")
    with torch.no_grad():
        g._xyz = nn.Parameter(x[keep].contiguous())
        g._features_dc = nn.Parameter(g._features_dc.detach()[keep].contiguous())
        fr = g._features_rest.detach()
        g._features_rest = nn.Parameter(fr[keep].contiguous() if fr.shape[0] else fr)
        g._opacity = nn.Parameter(g._opacity.detach()[keep].contiguous())
        g._scaling = nn.Parameter(g._scaling.detach()[keep].contiguous())
        g._rotation = nn.Parameter(g._rotation.detach()[keep].contiguous())
    g.save_ply(dst)
    print(f"  -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         depth=float(sys.argv[3]) if len(sys.argv) > 3 else 0.045,
         nth=int(sys.argv[4]) if len(sys.argv) > 4 else 360)
