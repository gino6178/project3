"""Cubes never get smoother, however many of them there are. Measured, at three resolutions.

The dual grid puts one vertex inside each boundary cell and training moves it, so the surface is
placed with sub-cell precision. A cube occupancy can only put a face on a cell boundary, and every
face is axis-aligned -- so between two neighbouring faces the normal either does not change or
turns by 90 degrees, whatever the cell size is.

That is the claim worth measuring, because it is the one that does not go away with resolution:
refining a cube grid makes the steps smaller but leaves the normals as wrong as they were, and
normals are what lighting, contact and any gradient of the surface are computed from. The silhouette
does converge, at 1/N; the shading does not converge at all.

Both surfaces are shaded here with one directional light and no colour, so what is being compared is
geometry and nothing else.
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
RES = int(os.environ.get("RES", "1024"))
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
mvp = torch.as_tensor(C["e_mvp"][0], dtype=torch.float32, device=dev)
hc, hf = st["hc"], st["hf"]
solid = st["solid"].cpu().numpy()


def cubes_at(f):
    """The occupancy as cube faces, at f times the fine cell size."""
    h = hf * f
    q = np.floor(((solid + 0.5) * hc + st["org"] - st["org"]) / h).astype(np.int64)
    if f <= 1:                      # every coarse cell is 2x2x2 fine cells
        off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
        q = (solid[:, None, :] * 2 + off[None]).reshape(-1, 3)
    q = np.unique(q, axis=0)
    V, F = ON.boundary_mesh(q, h, st["org"])
    return (torch.as_tensor(V, dtype=torch.float32, device=dev),
            torch.as_tensor(F, dtype=torch.int32, device=dev), len(q))


def facts(V, F):
    """How many directions the surface's normals take, and how far each is from its neighbourhood.

    The mean angle between neighbouring faces was tried first and it is the wrong measurement: a
    cube surface is mostly flat, so most of its edges are interior to a face and contribute zero,
    and its mean came out LOWER than the dual grid's. What is actually wrong with a cube surface is
    not that neighbouring faces disagree often -- it is that the normal can only ever be one of six
    vectors, so a curved surface is represented by a staircase whose every face points the wrong
    way. Both numbers below say that, and neither improves when the cells are made smaller.
    """
    tri = V[F.long()]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)
    distinct = len(torch.unique((n * 1e4).round().long(), dim=0))
    # each face against the average normal of the faces sharing its vertices
    vn = torch.zeros_like(V)
    vn.index_add_(0, F.long().reshape(-1), n.repeat_interleave(3, 0))
    vn = vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    local = vn[F.long()].mean(1)
    local = local / local.norm(dim=1, keepdim=True).clamp(min=1e-12)
    off = float(torch.rad2deg(torch.arccos((n * local).sum(1).clamp(-1, 1))).mean())
    return distinct, off


def dihedral(V, F):
    """Mean angle between the normals of triangles sharing an edge, in degrees."""
    tri = V[F.long()]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)
    e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0).long()
    k = torch.stack([e.min(1).values, e.max(1).values], 1)
    k = k[:, 0] * (len(V) + 1) + k[:, 1]
    o = k.argsort()
    k, fid = k[o], torch.arange(len(F), device=V.device).repeat(3)[o]
    s = k[1:] == k[:-1]
    cos = (n[fid[:-1][s]] * n[fid[1:][s]]).sum(1).clamp(-1, 1)
    return float(torch.rad2deg(torch.arccos(cos)).mean())


# a headlight, from the camera this view was taken with, tilted a little so that a curved surface
# shows a gradient rather than one flat tone. A fixed world-space direction lit the far side and
# every panel came out black.
_v = mvp[:3, 2] + 0.35 * mvp[:3, 1]
LIGHT = _v / _v.norm()


def shade(V, F):
    """One directional light on the geometry alone: no colour, no texture, nothing else."""
    tri = V[F.long()]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vn = torch.zeros_like(V)
    vn.index_add_(0, F.long().reshape(-1), n.repeat_interleave(3, 0))
    vn = vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    d = vn @ LIGHT
    if float(d.mean()) < 0:
        d = -d
    lam = (0.15 + 0.85 * d.clamp(0, 1))[:, None].expand(-1, 3).contiguous()
    ph = (torch.cat([V, torch.ones_like(V[:, :1])], 1) @ mvp)[None]
    Ft = F.contiguous()
    rast, _ = dr.rasterize(glctx, ph, Ft, resolution=[RES, RES])
    img, _ = dr.interpolate(lam[None], rast, Ft)
    img = dr.antialias(img.contiguous(), rast, ph, Ft)
    a = (rast[..., 3:4] > 0).float()
    return (img * a + 1.0 * (1 - a))[0].permute(2, 0, 1), a[0].permute(2, 0, 1)


mv, mf, _ = ON.surface_mesh(st)
rows, names = [], []
im, al = shade(mv, mf.int())
rows.append(im); names.append("dual grid")
print(f"{OBJ}: cell {hc:.5f} coarse, {hf:.5f} fine")
print(f"  {'surface':<26}{'cells':>10}{'faces':>11}{'normal directions':>19}"
      f"{'off its neighbourhood':>23}")
_d, _o = facts(mv, mf.int())
print(f"  {'dual grid at the fine cell':<26}{len(st['coords']):>10,}{len(mf):>11,}{_d:>19,}"
      f"{_o:>20.1f} deg")
for f, tag in ((1, "cubes at the fine cell"), (2, "cubes at the coarse cell"),
               (4, "cubes at twice the coarse")):
    V, F, nq = cubes_at(f)
    im, _ = shade(V, F)
    rows.append(im); names.append(tag)
    _d, _o = facts(V, F)
    print(f"  {tag:<26}{nq:>10,}{len(F):>11,}{_d:>19,}{_o:>20.1f} deg")

sheet = torch.cat(rows, -1).clamp(0, 1).permute(1, 2, 0)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/demo_normals_{OBJ}.jpg",
                                                                   quality=94)
ys, xs = (al[0] > 0.5).nonzero(as_tuple=True)
cy, x1, S = int(ys.float().mean()), int(xs.max()), 220
y0, x0 = max(cy - S // 2, 0), max(min(x1 - S + 40, RES - S), 0)
z = torch.cat([r[:, y0:y0 + S, x0:x0 + S] for r in rows], -1)
Image.fromarray((z.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) \
    .resize((z.shape[-1] * 3, S * 3), Image.NEAREST).save(f"{W}/demo_normals_zoom_{OBJ}.jpg", quality=94)
print(f"SHEET demo_normals_{OBJ}.jpg and demo_normals_zoom_{OBJ}.jpg  ({', '.join(names)})")
