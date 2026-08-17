"""Take a downloaded mesh (obj/stl/ply/glb) straight to the two-level voxel lattice.

This replaces three steps at once -- the shell generator, the ray-cast interior filler and
the voxeliser -- and removes the assumption that ties them together. Each of those steps
was written for a shell that came from a scan, and each recovers something the mesh
already states outright:

  what is inside      the filler ray-casts against splats and its occupancy falls off
                      toward the surface, 69% at r/R 0.3-0.45 down to 33% at 0.7-0.8, which
                      is why a separate pass had to go back and close the band underneath
                      the skin. A closed mesh answers inside-or-outside exactly.
  where the skin is   inferred from a radial density profile, which assumes the object is
                      roughly round and reports a single radius for the whole surface. The
                      mesh's own triangles are the surface, at any shape.
  the vertical axis   inferred from principal components, which on a round object is noise.

Dropping those inferences is what admits irregular shapes: nothing here refers to a radius,
a centre or an axis, so a handle, a concavity or a stem is filled as readily as a ball.

The two levels are kept, because they were measured to matter: quantising the surface at
the interior's cell size costs 29.5 dB against 35.9 dB at half it, and the lattice shows
through as banding. So the surface is rasterised at the fine size and the interior filled
at twice it.

Output is exactly what voxelize.py writes -- gs_fill.ply, cell_level.pt, lattice.pt,
is_interior.pt -- so init_from_ref.py and train_voxel.py take it unchanged.
"""
import sys, os, argparse
sys.path.append("/home/gino/project/FruitNinja_clean")
sys.path.append("/home/gino/project/FruitNinja_clean/gaussian-splatting")
os.chdir("/home/gino/project/FruitNinja_clean")

import numpy as np
import torch
from torch import nn
from scipy import ndimage
import trimesh
from scene.gaussian_model import GaussianModel

DEV = "cuda:0"
C0 = 0.28209479177387814


def load_mesh(path):
    """Load and concatenate whatever the file holds into one triangle soup."""
    m = trimesh.load(path, force="mesh", process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    return m


def occupancy(mesh, dx, origin, dims):
    """Solid occupancy on a regular grid: rasterise the triangles, then fill the inside.

    Filling by flood from outside rather than by testing each cell against the mesh. A
    per-cell containment test needs a watertight mesh and a winding-number query per cell,
    which downloaded assets often fail and which costs minutes at these grid sizes. What a
    rasterised surface guarantees is that the shell separates inside from outside, so
    whatever the outside flood cannot reach is interior -- true for any shape, including
    one with handles or concavities, and it needs no mesh repair.
    """
    surf = np.zeros(dims, dtype=bool)
    # Sample each triangle densely enough that no cell is skipped: a point every half cell
    # along the longest edge leaves no gap a diagonal can slip through.
    v = mesh.vertices[mesh.faces]                     # (F, 3, 3)
    e = np.linalg.norm(v[:, [1, 2, 0]] - v, axis=2).max(1)
    n_sub = np.clip(np.ceil(e / (dx * 0.5)).astype(int), 1, 64)
    for k in np.unique(n_sub):
        sel = v[n_sub == k]
        # barycentric grid on the reference triangle
        a, b = np.meshgrid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1), indexing="ij")
        keep = (a + b) <= 1.0
        w = np.stack([a[keep], b[keep], 1 - a[keep] - b[keep]], 1)      # (S, 3)
        pts = np.einsum("sj,fjc->fsc", w, sel).reshape(-1, 3)
        idx = np.floor((pts - origin) / dx).astype(np.int64)
        np.clip(idx, 0, np.array(dims) - 1, out=idx)
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # pad by one so the flood always has an outside to start from
    p = np.pad(surf, 1)
    free = ndimage.label(~p)[0]
    outside = free == free[0, 0, 0]
    solid = ~outside[1:-1, 1:-1, 1:-1]                 # surface plus everything enclosed
    return surf, solid


def main(mesh_path, out_dir, cells=600000, refine=2, diameter=1.413, grey=0.5):
    os.makedirs(out_dir, exist_ok=True)
    mesh = load_mesh(mesh_path)
    ext = mesh.vertices.max(0) - mesh.vertices.min(0)
    mesh.vertices = (mesh.vertices - mesh.vertices.mean(0)) * (diameter / ext.max())
    ext = mesh.vertices.max(0) - mesh.vertices.min(0)
    print(f"{mesh_path}: {len(mesh.faces):,} faces  watertight={mesh.is_watertight}  "
          f"extent {[round(float(v),3) for v in ext]}")

    # Cell size from a budget on the solid, not from the mesh's own tessellation: a
    # downloaded asset's triangle density says how it was authored, not how finely the
    # object needs to be represented. Volume is known, so one pass lands close.
    vol = float(abs(mesh.volume)) if mesh.is_volume else float(np.prod(ext)) * 0.5
    coarse = (vol / cells) ** (1 / 3)
    fine = coarse / refine

    pad = 2
    origin = mesh.vertices.min(0) - pad * fine
    dims_f = tuple(int(np.ceil((mesh.vertices.max(0)[i] - origin[i]) / fine)) + pad
                   for i in range(3))
    surf_f, solid_f = occupancy(mesh, fine, origin, dims_f)
    dims_c = tuple(int(np.ceil((mesh.vertices.max(0)[i] - origin[i]) / coarse)) + pad
                   for i in range(3))
    _, solid_c = occupancy(mesh, coarse, origin, dims_c)
    print(f"  coarse dx {coarse:.6f}  fine dx {fine:.6f}   "
          f"solid {int(solid_c.sum()):,} coarse / surface {int(surf_f.sum()):,} fine")

    # skin = the fine cells on the surface; interior = coarse cells the skin does not cover
    skin_idx = np.stack(np.nonzero(surf_f), 1)
    skin_pos = origin + (skin_idx + 0.5) * fine
    int_idx = np.stack(np.nonzero(solid_c & ~ndimage.binary_dilation(
        _downsample(surf_f, refine, dims_c))), 1)
    int_pos = origin + (int_idx + 0.5) * coarse
    print(f"  interior {len(int_pos):,} + skin {len(skin_pos):,} = "
          f"{len(int_pos)+len(skin_pos):,} cells")

    # Colour the skin from the mesh where it carries one, so an asset that arrives textured
    # keeps its appearance; anything without one starts flat and is painted from the
    # reference photographs by the next step, exactly as the generated shell was.
    skin_rgb = _surface_colour(mesh, skin_pos, grey)
    pos = np.concatenate([int_pos, skin_pos])
    rgb = np.concatenate([np.full((len(int_pos), 3), grey, np.float32), skin_rgb])
    lvl = torch.cat([torch.zeros(len(int_pos), dtype=torch.uint8),
                     torch.ones(len(skin_pos), dtype=torch.uint8)])

    p = torch.as_tensor(pos, dtype=torch.float32, device=DEV)
    c = torch.as_tensor(rgb, dtype=torch.float32, device=DEV).clamp(0, 1)
    n = p.shape[0]
    scale = torch.where(lvl.to(DEV)[:, None] == 0, coarse * 0.5, fine * 0.5).expand(-1, 3)

    g = GaussianModel(0)
    with torch.no_grad():
        g._xyz = nn.Parameter(p.contiguous())
        g._features_dc = nn.Parameter(((c - 0.5) / C0).unsqueeze(1).contiguous())
        g._features_rest = nn.Parameter(torch.zeros(n, 0, 3, device=DEV))
        g._opacity = nn.Parameter(torch.full((n, 1), 3.0, device=DEV))
        g._scaling = nn.Parameter(torch.log(scale.contiguous().float()))
        g._rotation = nn.Parameter(
            torch.tensor([1., 0., 0., 0.], device=DEV).expand(n, 4).contiguous())
        g.max_radii2D = torch.zeros(n, device=DEV)
        g.trained = torch.zeros(n, dtype=torch.bool)
        g.is_interior = torch.ones(n, dtype=torch.bool)

    g.save_ply(os.path.join(out_dir, "gs_fill.ply"))
    torch.save(torch.ones(n, dtype=torch.bool), os.path.join(out_dir, "is_interior.pt"))
    torch.save(lvl, os.path.join(out_dir, "cell_level.pt"))
    # R and skin_frac are what the round-object path measured; here the mesh defines the
    # surface directly, so they are recorded for the consumers that read them and nothing
    # downstream infers a shape from them.
    r = float(np.linalg.norm(pos - pos.mean(0), axis=1).max())
    # The axis everything downstream slices across. For a round object no direction is
    # distinguished and the renderer's up is as good as any; for an elongated one the
    # object names its own -- a loaf is sliced across its length, not across its width, and
    # its cross-sections are only alike in that direction. Take the longest principal axis
    # when the object is clearly elongated, the world up when it is not.
    dv = mesh.vertices - mesh.vertices.mean(0)
    ev, evec = np.linalg.eigh((dv.T @ dv) / len(dv))
    # Long or short, depending on which way the object is not round. A loaf is prolate and
    # is sliced across its length; a torus or a disc is oblate and its meaningful axis is
    # the short one it is symmetric about -- taking the longest there would pick an
    # arbitrary direction in the plane the object lies in. Comparing the two adjacent
    # eigenvalue ratios says which case it is without naming either shape.
    # Measured as lengths along the principal axes, not as second moments. A rounded box's
    # moments are not proportional to its sides: the loaf is 1.41 by 1.00, plainly prolate,
    # and its moment ratio is 0.86 -- inside any threshold that also has to call a sphere
    # round. The extent is the quantity the classification is actually about.
    proj = dv @ evec
    ext_pa = proj.max(0) - proj.min(0)          # along [short, middle, long]
    r_lo = float(ext_pa[0] / ext_pa[1])
    r_hi = float(ext_pa[1] / ext_pa[2])
    if r_hi < 0.80:
        up_ax = torch.as_tensor(evec[:, 2] / np.linalg.norm(evec[:, 2]), dtype=torch.float32)
        kind = f"prolate, long axis (mid/long {r_hi:.2f})"
    elif r_lo < 0.80:
        up_ax = torch.as_tensor(evec[:, 0] / np.linalg.norm(evec[:, 0]), dtype=torch.float32)
        kind = f"oblate, symmetry axis (short/mid {r_lo:.2f})"
    else:
        up_ax = torch.tensor([0., -1., 0.])
        kind = f"round, world up ({r_lo:.2f}, {r_hi:.2f})"
    print(f"  slicing axis {[round(float(v),3) for v in up_ax]}  ({kind})")
    torch.save({"coarse_dx": coarse, "fine_dx": fine,
                "origin": torch.as_tensor(origin, dtype=torch.float32),
                "refine": refine, "up": up_ax,
                "skin_frac": 0.0, "R": r, "source_mesh": mesh_path},
               os.path.join(out_dir, "lattice.pt"))
    print(f"  -> {out_dir}")


def _downsample(fine_grid, refine, dims_c):
    """Which coarse cells contain at least one occupied fine cell."""
    idx = np.stack(np.nonzero(fine_grid), 1) // refine
    out = np.zeros(dims_c, dtype=bool)
    np.clip(idx, 0, np.array(dims_c) - 1, out=idx)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def _surface_colour(mesh, pts, grey):
    """Nearest-face colour from the mesh's own visual, or flat grey if it has none."""
    vis = getattr(mesh, "visual", None)
    try:
        vc = np.asarray(vis.to_color().vertex_colors)[:, :3].astype(np.float32) / 255.
        if vc.std() < 1e-4:
            raise ValueError("uniform")
    except Exception:
        return np.full((len(pts), 3), grey, np.float32)
    _, _, fid = trimesh.proximity.closest_point(mesh, pts)
    return vc[mesh.faces[fid]].mean(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--cells", type=int, default=600000)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--diameter", type=float, default=1.413)
    a = ap.parse_args()
    main(a.mesh, a.out_dir, cells=a.cells, refine=a.refine, diameter=a.diameter)
