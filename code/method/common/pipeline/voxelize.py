"""Build the two-level lattice for any object, deriving every parameter from the model.

The orange went through internal_filling.py, which needs a hand-written config: a
boundary box, a rotation, a vertical axis, a grid resolution. That config exists only for
the orange, so nothing else could enter the pipeline. Everything in it is recoverable
from the point cloud itself:

  vertical axis   the direction of least spatial variance -- a fruit is widest across its
                  equator, narrowest along the stem axis
  cell size       from the primitives' own nearest-neighbour spacing, so the lattice
                  matches the detail the reconstruction actually carries
  two levels      the skin needs cells half the interior's size; quantising the skin at
                  the interior's spacing costs 29.5 dB and the lattice shows through as
                  banding, halving it reaches 35.9 dB and is visually clean

The split between skin and interior is the radius past which the object is hollow, found
from the radial density profile rather than assumed.
"""
import os as _os
# The repository root, so the same file runs here and on the remote box. It was written three
# times in every script -- two sys.path entries and a chdir -- and each one silently pinned the
# script to one machine: on the remote the chdir landed somewhere else and a relative source
# path then could not be found, which surfaces as "no such file" for a file that is plainly
# there.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

import torch, numpy as np
from torch import nn
from scene.gaussian_model import GaussianModel

DEV = "cuda:0"
C0 = 0.28209479177387814


def vertical_axis(x):
    """The axis the object is sliced along -- shape-derived, or the world's if it has none.

    Least-variance PCA on a ball returns a confident-looking direction built from sampling
    noise: the generated sphere gave [1, 0, 0] here while make_config, applying its own
    near-spherical fallback, wrote [0, -1, 0] into the physics config. The two are
    orthogonal, so the interior pattern was extruded along one axis and every rendered and
    supervised section cut across the other -- an extrusion seen end-on, which is the
    diagonal band that appeared across the orange and the watermelon instead of segments.
    Classifying by extent along the principal axes, the same way the mesh path does, keeps
    the two in step.
    """
    d = x - x.mean(0)
    ev, evec = torch.linalg.eigh((d.T @ d) / d.shape[0])
    proj = d @ evec
    ext = (proj.max(0).values - proj.min(0).values)
    r_lo = float(ext[0] / ext[1])
    r_hi = float(ext[1] / ext[2])
    if r_hi < 0.80:
        up = evec[:, 2]
    elif r_lo < 0.80:
        up = evec[:, 0]
    else:
        up = torch.tensor([0., -1., 0.], device=x.device)
    return up / up.norm()


def nn_spacing(x, n=4000, chunk=150000):
    s = x[torch.randperm(x.shape[0], device=DEV)[:n]]
    best = torch.full((s.shape[0],), 1e9, device=DEV)
    with torch.no_grad():
        for j in range(0, x.shape[0], chunk):
            dm = torch.cdist(s, x[j:j + chunk])
            dm[dm < 1e-9] = 1e9
            best = torch.minimum(best, dm.min(1).values)
            del dm
    return float(best.median())


def skin_radius(x, up, nb=40):
    """Radius past which the primitives form a shell rather than fill a volume."""
    c = x.mean(0)
    d = x - c
    z = d @ up
    rho = (d - z[:, None] * up).norm(dim=1)
    r = d.norm(dim=1)
    R = float(r.quantile(0.995))
    edges = torch.linspace(0, 1, nb + 1, device=DEV) * R
    dens = []
    for i in range(nb):
        m = (r >= edges[i]) & (r < edges[i + 1])
        vol = (4 / 3) * np.pi * (float(edges[i + 1]) ** 3 - float(edges[i]) ** 3)
        dens.append(float(m.sum()) / max(vol, 1e-9))
    dens = torch.tensor(dens, device=DEV)
    core = dens[nb // 4:nb // 2].mean()
    # the skin is where density first rises well above the interior's own level
    hit = (dens > core * 1.8).nonzero()
    hit = hit[hit[:, 0] > nb // 2]
    frac = float(edges[int(hit[0, 0])] / R) if hit.numel() else 0.85
    return min(max(frac, 0.70), 0.95), R


def merge(xyz, sh, op, sc, ro, dx, origin):
    """One primitive per occupied cell, at the cell centre."""
    idx = ((xyz - origin) / dx).round().long()
    key = (idx[:, 0] * 100003 + idx[:, 1]) * 100019 + idx[:, 2]
    uk, inv = torch.unique(key, return_inverse=True)
    C = uk.shape[0]
    w = op.reshape(-1, 1).clamp_min(1e-6)
    accs = torch.zeros(C, sh.shape[1], device=DEV).index_add_(0, inv, sh * w)
    accp = torch.zeros(C, 3, device=DEV).index_add_(0, inv, xyz * w)
    accw = torch.zeros(C, 1, device=DEV).index_add_(0, inv, w)
    cnt = torch.zeros(C, 1, device=DEV).index_add_(0, inv, torch.ones_like(w))
    pos = origin + ((accp / accw - origin) / dx).round() * dx
    order = torch.argsort(op.reshape(-1), descending=True)
    rep = torch.zeros(C, dtype=torch.long, device=DEV)
    rep[inv[order]] = order
    return pos, accs / accw, accw / cnt, sc[rep], ro[rep]


def main(ply, out_dir, refine=2, target_cells=0, coarse_dx=0.0, skin_frac=0.0):
    """coarse_dx: use this cell size instead of inferring one from the input's spacing.

    Inferring is for an input that is a splat cloud. When the shell was generated rather
    than scanned the lattice is already known exactly -- the generator writes the shell at
    the fine size and the interior filler works on a grid at twice it, which is the two
    levels this function otherwise has to guess at. Guessing on top of that guessed once
    too often: nearest-neighbour spacing on the mixed cloud reports the interior's size,
    doubling it lands a level too coarse, and the lattice came out at 102k cells with the
    shell quantised to half the resolution it was written at.
    """
    os.makedirs(out_dir, exist_ok=True)
    g = GaussianModel(0)
    g.load_ply_zero_sh(ply)
    xyz = g.get_xyz.detach().to(DEV)
    sh = g._features_dc.detach().to(DEV).squeeze(1)
    op = g.get_opacity.detach().to(DEV)
    sc = g._scaling.detach().to(DEV)
    ro = g._rotation.detach().to(DEV)

    up = vertical_axis(xyz)
    coarse = coarse_dx or nn_spacing(xyz) * 2.0     # interior needs no more than this
    if target_cells and not coarse_dx:
        # Size the lattice to a cell budget instead of to the input's own density, so
        # objects reconstructed at different resolutions come out comparable. Cell count
        # scales as dx^-3, so one pass of the ratio lands close enough.
        probe = ((xyz - xyz.min(0).values) / coarse).round().long()
        got = torch.unique((probe[:, 0] * 100003 + probe[:, 1]) * 100019 + probe[:, 2]).shape[0]
        coarse = coarse * (got / target_cells) ** (1 / 3)
        print(f"  cell budget {target_cells:,}: dx {nn_spacing(xyz)*2:.6f} -> {coarse:.6f}")
    fine = coarse / refine
    frac, R = skin_radius(xyz, up)
    if skin_frac:
        # Where the fine level starts, when the caller knows. `skin_radius` infers it from
        # where the density rises and then clamps to 0.95, which is right for a scan and
        # wrong for a shell written on purpose: a skin three cells thick at 0.0037 begins at
        # 0.984, and quantising everything from 0.95 outwards at the fine size turns the
        # interior fill under it into two million cells that carry no more information than
        # the coarse ones they replace.
        print(f"  skin boundary overridden: {frac:.3f} -> {skin_frac:.3f}")
        frac = skin_frac
    c = xyz.mean(0)
    rn = (xyz - c).norm(dim=1) / R
    skin = rn > frac
    # Snap the origin to the coarse grid.
    #
    # Cells are found by rounding (x - origin) / dx, so the origin decides the phase of both
    # lattices. Taking the raw minimum takes it from whichever primitive happens to be furthest
    # out, and that is a skin point on the fine grid: the interior points, one coarse cell
    # apart, then land on half-integers, round() sends them to the nearest even, and pairs
    # merge. Feeding 833,401 interior points produced 185,295 cells -- a factor of 4.5 lost to
    # a half-cell offset, silently, since a merge is what this function is for.
    origin = (xyz.min(0).values / coarse).floor() * coarse
    print(f"{ply}")
    print(f"  primitives {xyz.shape[0]:,}   vertical axis "
          f"{[round(float(v),3) for v in up]}")
    print(f"  coarse dx {coarse:.6f}   fine dx {fine:.6f}   skin starts at r/R {frac:.2f}")

    p0, s0, o0, c0, r0 = merge(xyz[~skin], sh[~skin], op[~skin], sc[~skin], ro[~skin],
                               coarse, origin)
    p1, s1, o1, c1, r1 = merge(xyz[skin], sh[skin], op[skin], sc[skin], ro[skin],
                               fine, origin)
    pos = torch.cat([p0, p1])
    lvl = torch.cat([torch.zeros(p0.shape[0], dtype=torch.uint8),
                     torch.ones(p1.shape[0], dtype=torch.uint8)])
    # every cell gets a Gaussian the size of its own cell, or it renders as a dot screen
    scale = torch.where(lvl.to(DEV)[:, None] == 0, coarse * 0.5, fine * 0.5).expand(-1, 3)

    with torch.no_grad():
        g._xyz = nn.Parameter(pos.contiguous())
        g._features_dc = nn.Parameter(torch.cat([s0, s1]).unsqueeze(1).contiguous())
        g._opacity = nn.Parameter(torch.logit(torch.cat([o0, o1]).clamp(1e-6, 1 - 1e-6)))
        g._scaling = nn.Parameter(torch.log(scale.contiguous()))
        g._rotation = nn.Parameter(torch.cat([r0, r1]).contiguous())
        g._features_rest = nn.Parameter(torch.zeros(pos.shape[0], 0, 3, device=DEV))
        g.max_radii2D = torch.zeros(pos.shape[0], device=DEV)
        g.trained = torch.zeros(pos.shape[0], dtype=torch.bool)
        g.is_interior = torch.ones(pos.shape[0], dtype=torch.bool)

    g.save_ply(os.path.join(out_dir, "gs_fill.ply"))
    torch.save(torch.ones(pos.shape[0], dtype=torch.bool),
               os.path.join(out_dir, "is_interior.pt"))
    torch.save(lvl, os.path.join(out_dir, "cell_level.pt"))
    torch.save({"coarse_dx": coarse, "fine_dx": fine, "origin": origin.cpu(),
                "refine": refine, "up": up.cpu(), "skin_frac": frac, "R": R},
               os.path.join(out_dir, "lattice.pt"))
    print(f"  interior {p0.shape[0]:,} + skin {p1.shape[0]:,} = {pos.shape[0]:,} cells "
          f"({xyz.shape[0]/pos.shape[0]:.2f}x fewer)  -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         refine=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
         target_cells=int(sys.argv[4]) if len(sys.argv) > 4 else 0,
         coarse_dx=float(sys.argv[5]) if len(sys.argv) > 5 else 0.0,
         skin_frac=float(sys.argv[6]) if len(sys.argv) > 6 else 0.0)
