"""binding.gif -- one piece of the orange bending, with its surface carried by the particles.

`binding.py` proves the property that matters analytically: the weights sum to one, so any affine
map is followed exactly. What it cannot show is the case the property does not cover, which is
the only interesting one -- a deformation that is not affine, where the surface has to be *close*
rather than exact and the question becomes how close, and whether it stays attached to the same
material while it gets there.

So the piece sways sideways by an amount quadratic in the depth below the cut face, and narrows
by an amount linear in it, growing and relaxing over the loop. Nothing about the surface is
deformed directly. The particles move; `SurfaceBinding.apply` is the whole of what happens to the
surface.

A twist about the cut normal was the first thing tried and it is the wrong demonstration here:
the piece is half an orange and that axis is very nearly its axis of symmetry, so most of the
motion maps the shape onto itself and the picture barely changes. The bend was chosen because it
moves the silhouette.

Two things are drawn on purpose.

  a material checker    The light and dark squares are a checkerboard in the *rest* frame -- in
                        depth below the cut and in azimuth about the cut normal -- so a square is
                        a set of surface points, fixed once and never recomputed. If the surface
                        slid over the material the squares would stay put while the object moved.
                        They lean and stretch instead, which is what following the material looks
                        like.

  the error, per frame  `b.apply(bend(particles))` against `bend(b.rest)`: where the binding puts
                        the surface against where the same field would put it. That is the
                        measurement `binding.py`'s self-test makes for one fixed bend, reported
                        here for every amplitude the loop passes through, in units of h_f.

The bind-time residual is a separate number and is shown as well, because it is the part that is
paid once and does not move: a vertex that did not already sit at its particles' weighted centre
was moved there when the binding was built.

    python method/common/demo/binding_gif.py LATTICE PLY CFG DEMO OUT.gif [frames]
"""
import os as _os

_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys
import time

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import subdivide as sd                       # noqa: E402
from method.common.cube.binding import SurfaceBinding                # noqa: E402
from method.common.cube.globalovox import boundary_mesh              # noqa: E402
from method.common.cube.physics import CollisionIndex, particles_to_pieces   # noqa: E402
from method.common.demo import gifcam                                # noqa: E402

BEND_MAX = 0.70          # sideways displacement at the far end, as a fraction of the piece's reach
TAPER_MAX = 0.45         # how much the far end narrows at the same moment


def bend(p, axis, side, centre, reach, amp):
    """A quadratic sway across `side` plus a linear taper, both keyed to depth along `axis`.

    Deliberately not affine, and not close to affine: the sway is quadratic in the depth and the
    taper multiplies the cross-section by a function of it, so neither is a map the binding
    reproduces for free. A rotation or a shear would be followed to machine precision and would
    measure nothing -- `binding.py`'s self-test already establishes that, exactly, and this is
    the case it leaves open.

    It is the *particles* this is applied to. The surface only ever moves through `apply`.
    """
    q = np.asarray(p, np.float64) - centre
    s = q @ axis
    perp = q - s[:, None] * axis[None, :]
    k_bend = BEND_MAX * amp / max(reach, 1e-12)
    k_taper = TAPER_MAX * amp / max(reach, 1e-12)
    perp = perp * (1.0 - k_taper * np.abs(s))[:, None]
    return centre + s[:, None] * axis[None, :] + perp + side[None, :] * (k_bend * s * s)[:, None]


def rigid_from(p0, p1):
    """The best rigid map taking one particle cloud to another -- the alternative to binding.

    Section 8.3 offers carrying each piece by a single transform, and the text's objection is that
    it is correct only while nothing bends. That objection is worth *showing* rather than
    asserting, so the same deformation is carried both ways in the same frame: this is the honest
    best a per-piece rigid transform can do, a Kabsch fit to the piece's own particles, not a
    weakened version of it.
    """
    c0, c1 = p0.mean(0), p1.mean(0)
    H = (p0 - c0).T @ (p1 - c1)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, c1 - R @ c0


def err_colour(e, hf, hi=6.0):
    """Grey where the surface is on its material, red where it has left it."""
    t = np.clip(e / (hi * hf), 0.0, 1.0)[:, None]
    return (1.0 - t) * np.array([0.62, 0.62, 0.60]) + t * np.array([0.86, 0.12, 0.10])


def bary(k):
    """Barycentric sample points inside a triangle, on a k x k grid clipped to the simplex."""
    u = (np.arange(k) + 0.5) / k
    bu, bv = np.meshgrid(u, u, indexing="ij")
    keep = (bu + bv) <= 1.0
    bu, bv = bu[keep], bv[keep]
    return np.stack([1.0 - bu - bv, bu, bv], 1)          # (S, 3)


def main(lattice_dir, ply, cfg, demo, out_gif, n_frames=72, size=440, ss=2, k_samp=3,
         radius_scale=1.05):
    from scipy.spatial import cKDTree
    from method.common.cube.occupancy import close_and_fill, to_grid

    lat = gifcam.load_lattice(lattice_dir, ply)
    xyz, rgb, lvl, hc, hf = lat["xyz"], lat["rgb"], lat["level"], lat["hc"], lat["hf"]
    org = xyz[lvl == 0].min(0)
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occg, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occg, 1).nonzero().numpy() + coords.min(0) - 1

    n = np.array([0.13, 0.97, -0.21])
    n = n / np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n)
    r = sd.cut(solid, hc, n, d, hf)
    print(f"  {len(solid):,} solid cells -> {len(r['leaf']):,} leaves to level {r['top']}, "
          f"{r['K']} pieces "
          f"{[int((r['piece'] == k).sum()) for k in range(r['K'])]}")

    ix = CollisionIndex(r, hc, org=org, plane=(n, d))
    pid_part, stray = particles_to_pieces(xyz, ix)
    who = int(np.bincount(pid_part[pid_part >= 0]).argmax())
    parts = xyz[pid_part == who]
    print(f"  {len(xyz):,} particles labelled ({stray:,} took the nearest leaf); "
          f"piece {who} keeps {len(parts):,}")

    # the same piece's cells at the fine spacing, and its boundary -- the cut face included
    off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
    allf = np.unique(np.concatenate([(solid[:, None, :] * 2 + off[None]).reshape(-1, 3),
                                     np.floor((xyz - org) / hf).astype(np.int64)]), axis=0)
    pid_cell, stray_c = particles_to_pieces((allf + 0.5) * hf + org, ix)
    cells = allf[pid_cell == who]
    print(f"  {len(allf):,} fine cells, {len(cells):,} in piece {who} "
          f"({stray_c:,} cells took the nearest leaf)")

    V, F = boundary_mesh(cells, hf, org)
    print(f"  piece boundary: {len(F):,} triangles, {len(V):,} merged vertices")

    t0 = time.time()
    b = SurfaceBinding(V, parts)
    print(f"  bound {len(V):,} vertices to {len(parts):,} particles, k = {b.k}  "
          f"[{time.time() - t0:.1f}s]")
    print(f"      bind-time residual: mean {b.residual.mean():.6f} = "
          f"{b.residual.mean() / hf:.3f} h_f, 95th {np.percentile(b.residual, 95) / hf:.3f} h_f, "
          f"max {b.residual.max() / hf:.3f} h_f")

    rest = b.rest
    centre = parts.mean(0)
    cam = gifcam.Cam(ply, cfg, demo, az=0.0, el=15.0, radius_scale=radius_scale)

    # The bend goes across the view, so that it can be seen. That is the one presentation choice
    # in the deformation, and it is taken from the camera rather than guessed: the camera's own
    # right-hand axis, brought back into lattice space and made perpendicular to the cut normal.
    right_lat = cam.R.T @ cam.w2c[:3, :3][:, 0]
    side = right_lat - (right_lat @ n) * n
    side = side / np.linalg.norm(side)
    q = rest - centre
    hgt = q @ n
    reach = float(np.abs(hgt).max())
    print(f"  bend axis {n.round(3)} (the cut normal), across {side.round(3)}, "
          f"reach {reach:.4f} = {reach / hf:.0f} h_f")

    # a checkerboard fixed in the rest frame: depth along the axis, azimuth about it
    e1 = np.eye(3)[int(np.argmin(np.abs(n)))]
    u1 = np.cross(n, e1)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(n, u1)
    ang = np.arctan2(q @ u2, q @ u1)
    band = (np.floor(hgt / (12.0 * hf)).astype(np.int64)
            + np.floor(ang / (np.pi / 7.0)).astype(np.int64)) % 2
    base = rgb[cKDTree(xyz).query(rest, k=1)[1]]
    vcol = base * np.where(band, 1.0, 0.80)[:, None]

    W = bary(k_samp)
    print(f"  {len(F):,} triangles x {len(W)} barycentric samples = "
          f"{len(F) * len(W):,} surface samples a frame")
    scol = np.einsum("tvc,sv->tsc", vcol[F], W).reshape(-1, 3)

    frames = []
    for i in range(n_frames):
        t = i / n_frames
        amp = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
        parts_now = bend(parts, n, side, centre, reach, amp)
        truth = bend(rest, n, side, centre, reach, amp)   # where the material actually is

        # the two ways of carrying the surface, from the same particles in the same frame
        R, tr = rigid_from(parts, parts_now)
        ways = [("one rigid transform for the piece", rest @ R.T + tr),
                ("bound to the particles", b.apply(parts_now))]

        panels = []
        for title, moved in ways:
            e = np.linalg.norm(moved - truth, axis=1)
            tv = moved[F]
            pts = np.einsum("tvc,sv->tsc", tv, W).reshape(-1, 3)
            fn = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
            fn /= np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-30)
            # Coloured by how far the surface has left its material, on one shared scale, so the
            # two panels are directly comparable and the reader does not have to read a number to
            # see which one works. The peel's own colour would be prettier and would hide it.
            vc = np.einsum("tvc,sv->tsc", err_colour(e, hf)[F], W).reshape(-1, 3)
            col = gifcam.shade(vc, np.repeat(fn, len(W), axis=0)).astype(np.float32)
            img, drawn = gifcam.splat(cam, pts, col, size, ss=ss)
            panels.append(np.asarray(gifcam.caption(
                img, [title, f"off its material by {e.mean() / hf:.2f} h_f, worst "
                             f"{e.max() / hf:.1f}"], size=15, band=True)))
            if title.startswith("bound") and i % 9 == 0:
                print(f"  frame {i:>3}/{n_frames}  sway {BEND_MAX * amp * reach / hf:>5.1f} h_f  "
                      f"bound {e.mean() / hf:.4f} h_f mean  {drawn:,} drawn")

        frames.append(np.concatenate(panels, axis=1))

    gifcam.write_gif(out_gif, frames, duration=70, colors=160)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 72)
