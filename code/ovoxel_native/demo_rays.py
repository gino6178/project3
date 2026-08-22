"""First hit, both ways: where the ray lands and which way the surface faces there.

Ray tracing and collision are the same measurement twice. A tracer needs the hit point and the
normal to build the reflected ray; a contact solver needs the hit point and the normal to build the
contact frame. So one set of rays is cast at the object and both representations answer:

The cube grid marched here is the FINE one, the same cell size the dual grid's vertices live in.
Marching the coarse occupancy instead put the cube surface a median of 2.9 cells outside the real
one, which is a resolution difference and would have flattered nothing but the argument.

  cube occupancy   march the grid until a cell is solid. The hit is on the face that was crossed,
                   so its position is quantised to a cell boundary and its normal is one of six
                   axis vectors. This is what a voxel tracer returns, and it is what a voxel
                   collider hands the solver.

  dual grid        the same rays against the actual surface. The hit is wherever the triangle is,
                   and the normal is interpolated across it.

The camera is orthographic and built here rather than taken from the dataset, because both paths
need the ray geometry explicitly and an orthographic frame makes every ray parallel and every
comparison per-pixel.

Two things come out. A mirror reflection of an analytic sky, which is the tracer's answer, and the
contact normal along a line across the object, which is the solver's -- with the angle between
successive contacts, the quantity that becomes a jolt in a simulation.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RES = int(os.environ.get("RES", "512"))
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
hc, hf = st["hc"], st["hf"]
FINE = os.environ.get("RAY_FINE", "1") == "1"
org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
lo = torch.as_tensor(st["idx_lo"], dtype=torch.float32, device=dev)
G = st["idx3"].shape
# the fine occupancy: every coarse cell is 2x2x2 of them, which is what the dual grid is built on
_off = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev)
_fine = (st["solid"][:, None, :] * 2 + _off[None]).reshape(-1, 3)
_flo = _fine.min(0).values
_fg = tuple((_fine.max(0).values - _flo + 1).tolist())
OCC = torch.zeros(_fg, dtype=torch.bool, device=dev)
OCC[(_fine - _flo)[:, 0], (_fine - _flo)[:, 1], (_fine - _flo)[:, 2]] = True
ctr = (st["solid"].float().mean(0) + 0.5) * hc + org
rad = float(((st["solid"].float().max(0).values - st["solid"].float().min(0).values) * hc).max()) / 2

d = torch.tensor([0.55, 0.35, 0.76], device=dev); d = d / d.norm()
up = torch.tensor([0., 1., 0.], device=dev)
r = torch.cross(up, d, dim=0); r = r / r.norm()
u = torch.cross(d, r, dim=0)
S = rad * 1.15
g = (torch.arange(RES, device=dev).float() + 0.5) / RES * 2 - 1
gy, gx = torch.meshgrid(g, g, indexing="ij")
origin = ctr[None, None] + gx[..., None] * S * r + gy[..., None] * S * u - d * rad * 3
print(f"{OBJ}: {RES}x{RES} parallel rays, step {hf / 2:.5f} ({hf / 2 / hc:.2f} of a coarse cell)")


def march(step_scale=0.5):
    """The voxel tracer: step along the ray until the cell is solid, and take the face crossed."""
    t = torch.zeros(RES, RES, device=dev)
    hit = torch.zeros(RES, RES, dtype=torch.bool, device=dev)
    prev = torch.full((RES, RES, 3), -1, dtype=torch.long, device=dev)
    nrm = torch.zeros(RES, RES, 3, device=dev)
    step = hf * step_scale
    n_steps = int(rad * 6 / step)
    for _ in range(n_steps):
        p = origin + (t[..., None] + step) * d
        cell = hf if FINE else hc
        base = _flo.float() if FINE else lo
        shape = _fg if FINE else G
        ijk = torch.floor((p - org) / cell - base).long()
        ok = ((ijk >= 0) & (ijk < torch.tensor(shape, device=dev))).all(-1)
        cl = ijk.clamp(min=torch.zeros(3, dtype=torch.long, device=dev),
                       max=torch.tensor([q - 1 for q in shape], device=dev))
        solid = ok & (OCC[cl[..., 0], cl[..., 1], cl[..., 2]] if FINE
                      else st["idx3"][cl[..., 0], cl[..., 1], cl[..., 2]] >= 0)
        new = solid & ~hit
        if bool(new.any()):
            ch = (ijk != prev) & new[..., None] & (prev >= 0)
            ax = ch.float().argmax(-1)
            f = torch.zeros(RES, RES, 3, device=dev)
            f.scatter_(-1, ax[..., None], 1.0)
            nrm[new] = (f * -torch.sign(d)[None, None])[new]
            t[new] = (t + step)[new]
            hit |= new
        prev = torch.where(ok[..., None], ijk, prev)
        t = torch.where(hit, t, t + step)
        if bool(hit.all()):
            break
    nrm = torch.where(nrm.norm(dim=-1, keepdim=True) > 0, nrm, -d[None, None])
    return hit, t, nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def raster():
    """The same rays against the real surface: an orthographic rasterise is a parallel ray cast."""
    mv, mf, _ = ON.surface_mesh(st)
    Ft = mf.contiguous().int()
    q = mv - ctr[None]
    x = (q @ r) / S; y = (q @ u) / S; z = (q @ d) / (rad * 4)
    clip = torch.stack([x, y, z, torch.ones_like(x)], -1)[None]
    rast, _ = dr.rasterize(glctx, clip.contiguous(), Ft, resolution=[RES, RES])
    tri = mv[Ft.long()]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    fn = fn / fn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    # weld by position first. The training mesh splits its vertices, so accumulating by index
    # leaves every triangle with its own normal and the surface stays faceted -- measured, the
    # normal then turned 15 degrees between neighbouring pixels on a smooth rind.
    # Weld by position before averaging. The training mesh does not share vertices between
    # triangles, so accumulating by index leaves every triangle with its own normal and the
    # surface stays faceted -- which measured as a 15-degree turn between neighbouring pixels on a
    # smooth rind. The tolerance is reported rather than chosen quietly: it is a fraction of a fine
    # cell, and the count of welded vertices says whether the surface came out connected.
    tol = float(os.environ.get("RAY_WELD", "0.25")) * hf
    key = torch.unique((mv / tol).round().long(), dim=0, return_inverse=True)[1]
    acc = torch.zeros(int(key.max()) + 1, 3, device=dev)
    acc.index_add_(0, key[Ft.long().reshape(-1)], fn.repeat_interleave(3, 0))
    vn = acc[key]
    vn = vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    print(f"  welded at {tol / hf:g} of a fine cell: {len(mv):,} vertices -> "
          f"{int(key.max()) + 1:,} positions, {len(mv) / (int(key.max()) + 1):.1f} triangles' "
          f"corners at each")
    nn, _ = dr.interpolate(vn[None].contiguous(), rast, Ft)
    pp, _ = dr.interpolate(mv[None].contiguous(), rast, Ft)
    hit = rast[0, ..., 3] > 0
    n = nn[0]
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    n = torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)
    t = ((pp[0] - origin) * d).sum(-1)
    return hit, t, n


def sky(v):
    """An analytic environment: a sun, a gradient, and a ground, so a reflection has something to
    reflect."""
    up_ = v[..., 1:2].clamp(-1, 1)
    s = torch.tensor([0.3, 0.8, 0.5], device=dev); s = s / s.norm()
    sun = ((v * s).sum(-1, keepdim=True).clamp(0, 1) ** 64) * 3
    skyc = torch.tensor([0.35, 0.55, 0.9], device=dev) * (0.4 + 0.6 * up_.clamp(0, 1))
    gnd = torch.tensor([0.35, 0.28, 0.22], device=dev).expand_as(skyc)
    return (torch.where(up_ > 0, skyc, gnd) + sun).clamp(0, 1)


def mirror(hit, n):
    v = d[None, None].expand_as(n)
    refl = v - 2 * (v * n).sum(-1, keepdim=True) * n
    img = sky(refl)
    return torch.where(hit[..., None], img, torch.ones_like(img)).permute(2, 0, 1)


h_c, t_c, n_c = march()
h_d, t_d, n_d = raster()
both = h_c & h_d
# grazing rays at the silhouette hit one representation and skim the other, and a handful of them
# dominate any mean. The median is what the body of the surface does.
_e = (t_c - t_d)[both].abs()
print(f"  rays: {int(h_c.sum()):,} hit the cubes, {int(h_d.sum()):,} the surface, "
      f"{int(both.sum()):,} both")
print(f"  first-hit depth: median {float(_e.median()) / hc:.3f} cells, "
      f"90th percentile {float(_e.quantile(0.9)) / hc:.3f}, largest {float(_e.max()) / hc:.1f}")
print(f"  first-hit depth: the two agree to {float((t_c - t_d)[both].abs().mean()):.5f} "
      f"({float((t_c - t_d)[both].abs().mean()) / hc:.2f} of a coarse cell), "
      f"largest {float((t_c - t_d)[both].abs().max()) / hc:.2f} cells")
ang = torch.rad2deg(torch.arccos((n_c * n_d).sum(-1).clamp(-1, 1)))
print(f"  contact normal: the cube face is {float(ang[both].mean()):.1f} deg from the surface's "
      f"own normal on average, worst {float(ang[both].max()):.1f} deg")
print(f"  distinct normals among the hits: cubes {len(torch.unique((n_c[both] * 1e4).round().long(), dim=0))}, "
      f"dual grid {len(torch.unique((n_d[both] * 1e4).round().long(), dim=0)):,}")

# The fruit is roughly a ball, so the direction from its centre to the contact is what a contact
# normal should follow. The dual grid's normals also carry the peel's own relief, which is real
# surface and not error; the cube's cannot carry anything at all.
_p_c = origin + t_c[..., None] * d
_p_d = origin + t_d[..., None] * d
for _tag, _p, _n in (("cubes", _p_c, n_c), ("dual grid", _p_d, n_d)):
    _r = _p - ctr[None, None]
    _r = _r / _r.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    _a = torch.rad2deg(torch.arccos((_n * _r).sum(-1).clamp(-1, 1)))[both]
    print(f"  contact normal against the radial direction, {_tag}: median "
          f"{float(_a.median()):.1f} deg, 90th percentile {float(_a.quantile(0.9)):.1f}")

row = int(RES * 0.5)
sel = both[row]
if int(sel.sum()) > 8:
    a_c = torch.rad2deg(torch.arctan2(n_c[row, :, 0], n_c[row, :, 2]))
    a_d = torch.rad2deg(torch.arctan2(n_d[row, :, 0], n_d[row, :, 2]))
    wrap = lambda x: (x + 180) % 360 - 180        # the angle is periodic; a step from +179 to
    j_c = wrap(a_c[1:] - a_c[:-1]).abs()[sel[1:] & sel[:-1]]      # -179 is 2 degrees, not 358
    j_d = wrap(a_d[1:] - a_d[:-1]).abs()[sel[1:] & sel[:-1]]
    print(f"  along one line across the object, the contact normal turns by "
          f"{float(j_c.median()):.2f} deg between neighbouring contacts for cubes "
          f"(90th pct {float(j_c.quantile(0.9)):.0f}) and {float(j_d.median()):.2f} deg for the "
          f"dual grid (90th pct {float(j_d.quantile(0.9)):.2f})")
    # the same line drawn, because a staircase is easier to recognise than to summarise
    # plotted as the angle away from radial, which runs 0..180 and cannot wrap; plotting the
    # normal's own bearing put spikes in the curve wherever it crossed +-180 and those were an
    # artefact of the plot rather than of the surface
    def _dev(pp, nn):
        rr = pp[row] - ctr[None]
        rr = rr / rr.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.rad2deg(torch.arccos((nn[row] * rr).sum(-1).clamp(-1, 1))).cpu().numpy()

    xs = np.arange(RES)[sel.cpu().numpy()]
    ca, da = _dev(_p_c, n_c)[sel.cpu().numpy()], _dev(_p_d, n_d)[sel.cpu().numpy()]
    Hh, Ww = 300, 900
    plot = np.ones((Hh, Ww, 3), np.uint8) * 255
    x0, x1 = xs.min(), xs.max()
    for series, col in ((ca, (200, 60, 40)), (da, (40, 90, 190))):
        px = ((xs - x0) / max(x1 - x0, 1) * (Ww - 20) + 10).astype(int)
        py = ((1 - np.clip(series, 0, 90) / 90) * (Hh - 20) + 10).astype(int)
        for k in range(len(px) - 1):
            n_seg = max(abs(int(py[k + 1]) - int(py[k])), abs(int(px[k + 1]) - int(px[k])), 1)
            for q in range(n_seg + 1):
                yy = int(py[k] + (py[k + 1] - py[k]) * q / n_seg)
                xx = int(px[k] + (px[k + 1] - px[k]) * q / n_seg)
                if 0 <= yy < Hh and 0 <= xx < Ww:
                    plot[yy, xx] = col
    Image.fromarray(plot).save(f"{W}/demo_contact_{OBJ}.png")
    print(f"  PLOT demo_contact_{OBJ}.png  (how far the contact normal is from radial along one "
          f"line, 0 at the top and 90 deg at the bottom; red the cube collider, blue the dual grid)")

sheet = torch.cat([mirror(h_d, n_d), mirror(h_c, n_c)], -1).clamp(0, 1).permute(1, 2, 0)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/demo_rays_{OBJ}.jpg",
                                                                   quality=94)
print(f"SHEET demo_rays_{OBJ}.jpg  (a mirror reflection: dual grid left, voxel tracer right)")
