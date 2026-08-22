"""The skin, drawn from the dual grid and drawn as cubes, from the same camera.

A cube occupancy can only put a surface on a cell boundary: every face is axis-aligned and every
vertex sits on the lattice, so a curved skin becomes a staircase whose step is one cell. The dual
grid puts one vertex inside each boundary cell and lets training move it, so the same skin is a
smooth surface at the same cell count -- the resolution is the same, the placement is not.

Two numbers say it without needing the picture. The isoperimetric ratio of the silhouette,
perimeter over the square root of area, which is 3.545 for a circle and rises with every step; and
the angle between neighbouring face normals, which is 0 on a smooth surface and 90 degrees at every
edge of a staircase.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RUN = os.environ.get("DEMO_RUN", "cu10_orange_sp")
RES = int(os.environ.get("RES", "1024"))
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
p = torch.load(f"{W}/{RUN}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
if "dec_i" in p:
    di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
    di.load_state_dict(p["dec_i"]); ds = anchor.ColourDecoder(len(st["surf_rgb"]),
                                                              init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()

mvp = torch.as_tensor(C["e_mvp"][0], dtype=torch.float32, device=dev)


def cube_surface():
    """The same solid cells, with their own faces as the surface."""
    solid = st["solid"].cpu().numpy()
    V, F = ON.boundary_mesh(solid, st["hc"], st["org"])
    Vt = torch.as_tensor(V, dtype=torch.float32, device=dev)
    Ft = torch.as_tensor(F, dtype=torch.int32, device=dev)
    # colour it from the same field, so only the geometry differs
    c = ON.sample_interior(st, Vt).clamp(0, 1)
    return Vt, Ft, c


def draw(V, F, col):
    ph = (torch.cat([V, torch.ones_like(V[:, :1])], 1) @ mvp)[None]
    rast, _ = dr.rasterize(glctx, ph, F.contiguous(), resolution=[RES, RES])
    img, _ = dr.interpolate(col[None].contiguous(), rast, F.contiguous())
    img = dr.antialias(img.contiguous(), rast, ph, F.contiguous())
    alpha = (rast[..., 3:4] > 0).float()
    return (img * alpha + 1.0 * (1 - alpha))[0].permute(2, 0, 1), alpha[0].permute(2, 0, 1)


def ragged(mask):
    m = mask[0] > 0.5
    area = float(m.sum())
    pad = torch.nn.functional.pad(m[None, None].float(), (1, 1, 1, 1), value=0)
    inner = pad.unfold(2, 3, 1).unfold(3, 3, 1).amin((-1, -2))[0, 0] > 0.5
    return area, float((m & ~inner).sum()) / max(area, 1) ** 0.5


def sharpness(V, F):
    """Mean angle between the normals of triangles that share an edge, in degrees."""
    tri = V[F.long()]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)
    e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0).long()
    key = torch.stack([e.min(1).values, e.max(1).values], 1)
    key = key[:, 0] * (len(V) + 1) + key[:, 1]
    order = key.argsort()
    key, fid = key[order], (torch.arange(len(F), device=V.device).repeat(3))[order]
    same = key[1:] == key[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    cos = (n[a] * n[b]).sum(1).clamp(-1, 1)
    return float(torch.rad2deg(torch.arccos(cos)).mean())


mv, mf, mc = ON.surface_mesh(st)
img_d, al_d = draw(mv, mf.int(), mc.clamp(0, 1))
Vc, Fc, cc = cube_surface()
img_c, al_c = draw(Vc, Fc, cc)
ad, rd = ragged(al_d); ac, rc = ragged(al_c)
print(f"{OBJ}: dual grid {len(mv):,} vertices / {len(mf):,} faces, "
      f"cube surface {len(Vc):,} / {len(Fc):,}")
print(f"  silhouette raggedness: dual grid {rd:.3f}, cubes {rc:.3f}  (a circle is 3.545)")
print(f"  angle between neighbouring faces: dual grid {sharpness(mv, mf.int()):.1f} deg, "
      f"cubes {sharpness(Vc, Fc):.1f} deg")

sheet = torch.cat([img_d, img_c], -1).clamp(0, 1).permute(1, 2, 0)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/demo_shell_{OBJ}.jpg",
                                                                   quality=94)
ys, xs = (al_d[0] > 0.5).nonzero(as_tuple=True)
cy = int(ys.float().mean()); x1 = int(xs.max())
S = 200
y0, x0 = max(cy - S // 2, 0), max(min(x1 - S + 30, RES - S), 0)
z = torch.cat([img_d[:, y0:y0 + S, x0:x0 + S], img_c[:, y0:y0 + S, x0:x0 + S]], -1)
Image.fromarray((z.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) \
    .resize((z.shape[-1] * 3, S * 3), Image.NEAREST).save(f"{W}/demo_shell_zoom_{OBJ}.jpg", quality=94)
print(f"SHEET demo_shell_{OBJ}.jpg and demo_shell_zoom_{OBJ}.jpg  (dual grid left, cubes right)")
