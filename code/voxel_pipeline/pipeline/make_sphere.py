"""Generate the shell the pipeline starts from, instead of borrowing a reconstruction.

The paper's pipeline begins with a scanned shell and generates the interior behind it. That
made the scan a hidden dependency: the repo ships exactly one raw model, so a second object
could only be tested by stripping a model that had already been trained -- which imports the
answer into the input and, at the trained model's own density, does not even fit in memory.

Both fruits are spheres, and a sphere needs no scan. This emits a hollow voxel shell on a
regular lattice: cells whose distance from the centre falls in the outermost band, one
primitive per cell, at the cell centre. Nothing about the fruit is in it -- the colour is a
flat grey and the geometry is a sphere -- so the same file is the input for both objects and
everything that distinguishes an orange from a watermelon has to come from the cross-section
photographs, which is the claim worth testing.

The default diameter matches the orange raw scan's, so the framing calibration measured
against it (diameter / camera distance = 0.5526) carries over unchanged.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)
import torch
from torch import nn
from scene.gaussian_model import GaussianModel

DEV = "cuda:0"
C0 = 0.28209479177387814


def main(out_ply, radius=0.7065, dx=0.0074, thickness_cells=2.0, grey=0.5):
    n = int(radius / dx) + 2
    a = torch.arange(-n, n + 1, device=DEV, dtype=torch.float32) * dx
    gx, gy, gz = torch.meshgrid(a, a, a, indexing="ij")
    p = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    r = p.norm(dim=1)
    # A scan sees the surface only, so the shell is as thin as the lattice allows while
    # still closing: below about two cells the band develops holes along the diagonals,
    # where the distance between successive shells exceeds dx.
    shell = (r <= radius) & (r > radius - thickness_cells * dx)
    p = p[shell]

    g = GaussianModel(0)
    N = p.shape[0]
    with torch.no_grad():
        g._xyz = nn.Parameter(p.contiguous())
        # flat grey: the shell carries no appearance of its own, so anything the trained
        # model shows has come from the reference photographs
        rgb = torch.full((N, 3), float(grey), device=DEV)
        g._features_dc = nn.Parameter(((rgb - 0.5) / C0).unsqueeze(1).contiguous())
        g._features_rest = nn.Parameter(torch.zeros(N, 0, 3, device=DEV))
        g._opacity = nn.Parameter(torch.full((N, 1), 3.0, device=DEV))     # sigmoid -> 0.95
        g._scaling = nn.Parameter(torch.full((N, 3), float(torch.tensor(dx * 0.5).log()),
                                             device=DEV))
        g._rotation = nn.Parameter(
            torch.tensor([1., 0., 0., 0.], device=DEV).expand(N, 4).contiguous())
        g.max_radii2D = torch.zeros(N, device=DEV)
    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
    g.save_ply(out_ply)
    print(f"sphere shell: radius {radius:.4f} (diameter {2*radius:.4f})  dx {dx:.5f}  "
          f"{thickness_cells:g} cells thick")
    print(f"  {N:,} cells -> {out_ply}")


if __name__ == "__main__":
    main(sys.argv[1],
         radius=float(sys.argv[2]) if len(sys.argv) > 2 else 0.7065,
         dx=float(sys.argv[3]) if len(sys.argv) > 3 else 0.0074,
         thickness_cells=float(sys.argv[4]) if len(sys.argv) > 4 else 2.0)
