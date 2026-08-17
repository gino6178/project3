"""The topology, generated rather than recovered.

Three lattices reached this pipeline by two different routes and neither was the one the method
wants. The orange and the watermelon were quantised from released Gaussian models, so their
occupancy is whatever the reconstruction happened to put there -- a sponge, 5.5% to 14.9% of every
cross-section reading as holes until a morphological closing repairs it. The doughnut came from an
older shell-and-ray-cast route whose occupancy thins toward the surface, and it is the one that
does not repair: 8.7% to 10.2% left after closing, and 5.3% after a radius that costs 38% more
cells.

None of that is necessary. These objects are a sphere and a torus. A solid is what you get by
asking whether a cell's centre is inside the shape, which is a comparison, and there is no scan to
recover, no radius to infer, no closing to repair and no hole to leave. `mesh_to_voxel` already
makes this argument for a downloaded mesh; for a shape with an equation it is one line.

What comes out is the format everything downstream already reads -- gs_fill.ply, cell_level.pt,
lattice.pt, is_interior.pt -- with two levels, because the interior and the surface want different
spacings: coarse cells fill the volume, and cells within `skin` of the boundary are subdivided so
the skin the six views are projected onto is at h_f.

The colour is flat grey on purpose. Nothing about the fruit is in the geometry, so everything that
distinguishes an orange from a watermelon has to arrive from the six views and the cross-section
photographs, which is the claim worth being able to test.

    python method/common/pipeline/make_shape.py sphere OUT --radius 0.7065 --dx 0.0118
    python method/common/pipeline/make_shape.py torus  OUT --radius 0.55 --tube 0.18 --dx 0.00832
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import argparse
import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814


def sdf_sphere(p, radius, **kw):
    """Negative inside. The signed distance is what makes the skin band a distance and not a
    fraction of a radius, so the same number means the same thing on any shape."""
    return p.norm(dim=1) - radius


def sdf_torus(p, radius, tube, axis=1, **kw):
    """A torus about `axis`, radius to the tube's centre, `tube` its own radius."""
    o = [a for a in range(3) if a != axis]
    q = torch.stack([p[:, o].norm(dim=1) - radius, p[:, axis]], 1)
    return q.norm(dim=1) - tube


def sdf_ellipsoid(p, radii, **kw):
    """Radii per axis. The exact-distance form for an ellipsoid has no closed form, and the
    scaled sphere distance is enough here: the sign is exact, which is what decides occupancy,
    and the magnitude is only used to pick a skin band a few cells wide."""
    r = torch.as_tensor(radii, dtype=p.dtype, device=p.device)
    q = p / r
    k = q.norm(dim=1)
    return (k - 1.0) * float(r.min())


SHAPES = {"sphere": sdf_sphere, "torus": sdf_torus, "ellipsoid": sdf_ellipsoid}


def build(shape, dx, refine=2, skin=None, **kw):
    """Cells inside the shape at dx, with those near the boundary subdivided.

    Returns coarse centres, fine centres and the fine spacing. A coarse cell is kept only if it
    is not replaced by its children, so the two levels partition the solid rather than overlap
    -- the same invariant the lattice everything else reads assumes.
    """
    f = SHAPES[shape]
    hf = dx / refine
    skin = skin if skin is not None else 3.0 * hf
    if "radii" in kw:
        ext = float(max(kw["radii"])) + 4 * dx
    else:
        ext = kw.get("radius", 0.7) + kw.get("tube", 0.0) + 4 * dx

    def grid(h):
        n = int(np.ceil(ext / h))
        a = (torch.arange(-n, n + 1, device=DEV, dtype=torch.float64) + 0.5) * h
        gx, gy, gz = torch.meshgrid(a, a, a, indexing="ij")
        return torch.stack([gx, gy, gz], -1).reshape(-1, 3)

    pc = grid(dx)
    dc = f(pc, **kw)
    inside_c = dc <= 0
    near_c = inside_c & (dc > -skin)
    coarse = pc[inside_c & ~near_c]

    pf = grid(hf)
    df = f(pf, **kw)
    inside_f = df <= 0
    # a fine cell is kept where its parent was replaced, which is where the parent was near the
    # boundary -- tested on the parent's own centre so the partition is exact
    parent = torch.floor(pf / dx) * dx + 0.5 * dx
    dp = f(parent, **kw)
    fine = pf[inside_f & (dp <= 0) & (dp > -skin)]
    return coarse.float(), fine.float(), hf


def normals(shape, pts, eps, **kw):
    """The exact surface normal: the gradient of the field that defined the shape.

    A generated shape knows its own normal and nothing has to be inferred from a voxel grid. That
    matters most where the inference is worst: on a torus the direction from the centroid has
    n_z = -0.284 on a face whose normal is -1, and the occupancy gradient recovers only -0.49 at
    the smoothing the skinner defaults to, -0.78 at the smallest. Reading it off the field gives
    -1 with no smoothing scale to choose.
    """
    f = SHAPES[shape]
    g = []
    for a in range(3):
        d = torch.zeros(3, device=pts.device, dtype=pts.dtype)
        d[a] = eps
        g.append((f(pts + d, **kw) - f(pts - d, **kw)) / (2 * eps))
    n = torch.stack(g, 1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)


def write(out_dir, coarse, fine, dx, hf, grey=0.5, nrm=None):
    from scene.gaussian_model import GaussianModel
    from torch import nn
    _os.makedirs(out_dir, exist_ok=True)
    p = torch.cat([coarse, fine]).contiguous()
    lvl = torch.cat([torch.zeros(len(coarse), dtype=torch.int8),
                     torch.ones(len(fine), dtype=torch.int8)])
    scale = torch.cat([torch.full((len(coarse),), float(np.log(dx * 0.5))),
                       torch.full((len(fine),), float(np.log(hf * 0.5)))])

    g = GaussianModel(0)
    N = len(p)
    with torch.no_grad():
        g._xyz = nn.Parameter(p.to(DEV))
        rgb = torch.full((N, 3), float(grey), device=DEV)
        g._features_dc = nn.Parameter(((rgb - 0.5) / C0).unsqueeze(1).contiguous())
        g._features_rest = nn.Parameter(torch.zeros(N, 0, 3, device=DEV))
        g._opacity = nn.Parameter(torch.full((N, 1), 3.0, device=DEV))
        g._scaling = nn.Parameter(scale[:, None].expand(N, 3).contiguous().to(DEV))
        g._rotation = nn.Parameter(
            torch.tensor([1., 0., 0., 0.], device=DEV).expand(N, 4).contiguous())
        g.max_radii2D = torch.zeros(N, device=DEV)
    g.save_ply(_os.path.join(out_dir, "gs_fill.ply"))
    torch.save(lvl, _os.path.join(out_dir, "cell_level.pt"))
    torch.save(torch.ones(N, dtype=torch.bool), _os.path.join(out_dir, "is_interior.pt"))
    torch.save({"coarse_dx": float(dx), "fine_dx": float(hf)},
               _os.path.join(out_dir, "lattice.pt"))
    if nrm is not None:
        torch.save(nrm.cpu(), _os.path.join(out_dir, "cell_normal.pt"))
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("out_dir")
    ap.add_argument("--radius", type=float, default=0.7065)
    ap.add_argument("--radii", type=float, nargs=3, default=None,
                    help="ellipsoid semi-axes, which is how a generated shape is made to match "
                         "the object it is textured from")
    ap.add_argument("--tube", type=float, default=0.18)
    ap.add_argument("--dx", type=float, default=0.0118)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--skin", type=float, default=None)
    ap.add_argument("--axis", type=int, default=1)
    a = ap.parse_args()

    kw = {"radius": a.radius}
    if a.shape == "torus":
        kw.update(tube=a.tube, axis=a.axis)
    if a.shape == "ellipsoid":
        kw = {"radii": a.radii or [a.radius] * 3}
    coarse, fine, hf = build(a.shape, a.dx, a.refine, a.skin, **kw)
    nrm = normals(a.shape, torch.cat([coarse, fine]).double(), hf * 0.25, **kw).float()
    n = write(a.out_dir, coarse, fine, a.dx, hf, nrm=nrm)
    print(f"  {a.shape}: {len(coarse):,} coarse cells at {a.dx:.5f} + {len(fine):,} fine at "
          f"{hf:.5f} = {n:,}")
    p = torch.cat([coarse, fine]).cpu().numpy()
    print(f"  extent {(p.max(0) - p.min(0)).round(4).tolist()}")
    print(f"  -> {a.out_dir}")


if __name__ == "__main__":
    main()
