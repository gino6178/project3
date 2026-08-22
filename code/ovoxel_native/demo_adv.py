"""Where the representation is actually ahead, measured three ways.

The first attempt at this compared our cut face against a dense grid sliced at the SAME cell size,
and found no difference at all: 0.0% of the cut area, outline ratios equal to three digits. That is
correct and it is not a defect -- `cut_polygons` intersects the plane with each solid cube, so the
outline of the union IS the occupancy boundary, the same one a nearest-cell test returns. Any claim
about smoother oblique cuts has to be dropped.

These three are what survive.

  1. equal memory       a dense grid must store the empty cells too. This object's bounding box is
                        117x127x121 and only 43% of it is solid, so at the same budget the dense
                        grid's cells are 1.3x wider in each direction. Sliced at THAT size it is a
                        mosaic, and this is the honest comparison rather than handing the dense grid
                        2.3x the memory.

  2. the slab           the paper's pipeline does not draw a plane. `plane_filter` splats every
                        primitive within surf_dis of it, which is 2.82 coarse cells here, so its cut
                        face is an integral over a slab 5.63 cells thick. `render_section` can do
                        the same thing on purpose, and the difference is what a zero-thickness cut
                        buys.

  3. the surface        the dual grid's vertices are placed inside their cells and optimised there,
                        so the skin can sit anywhere; a cube occupancy can only put it on a cell
                        boundary. This one is about the object's outside, not its cut.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
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
    di.load_state_dict(p["dec_i"])
    ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()

hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
d = float(C["h_planes"][(H_LO + H_HI) // 2, 3])
G = st["idx3"].shape
n_solid = len(st["solid"])
fill = n_solid / float(np.prod(G))
scale = fill ** (-1 / 3.0)
print(f"{OBJ}: {n_solid:,} solid cells in a {G[0]}x{G[1]}x{G[2]} box ({100 * fill:.1f}% full), so "
      f"a dense grid of the same size has cells {scale:.2f}x wider")


def grad(a, m=None):
    if m is None:
        m = torch.ones_like(a[:1])
    gx = (a[..., 1:] - a[..., :-1]).abs() * m[..., 1:] * m[..., :-1]
    gy = (a[..., 1:, :] - a[..., :-1, :]).abs() * m[..., 1:, :] * m[..., :-1, :]
    n = (m[..., 1:] * m[..., :-1]).sum().clamp(min=1) + (m[..., 1:, :] * m[..., :-1, :]).sum().clamp(min=1)
    return float((gx.sum() + gy.sum()) / n / a.shape[0])


def coarse_slice(f, res):
    """The cut face a dense grid of `f` times the cell size returns: one colour per its own cell."""
    hc = st["hc"] * f
    org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
    lo = torch.as_tensor(st["idx_lo"], dtype=torch.float32, device=dev) * st["hc"] + org
    # the object resampled onto the coarser grid, by taking the cell each coarse centre falls in
    Gc = [max(int(np.ceil(g / f)), 1) for g in G]
    gi = torch.meshgrid(*[torch.arange(x, device=dev) for x in Gc], indexing="ij")
    ctr = (torch.stack(gi, -1).float() + 0.5) * hc + lo
    ij = torch.round((ctr - org) / st["hc"] - 0.5 -
                     torch.as_tensor(st["idx_lo"], dtype=torch.float32, device=dev)).long()
    ok = ((ij >= 0) & (ij < torch.tensor(G, device=dev))).all(-1)
    ij = ij.clamp(min=torch.zeros(3, dtype=torch.long, device=dev),
                  max=torch.tensor([g - 1 for g in G], device=dev))
    row = st["idx3"][ij[..., 0], ij[..., 1], ij[..., 2]].long()
    solid = ok & (row >= 0)
    col = torch.zeros(tuple(Gc) + (3,), device=dev)
    col[solid] = st["interior"][row[solid].clamp(min=0)]
    # now slice that coarse volume the way a dense grid is sliced
    im, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, res, exterior=False)
    pos = _positions(res)
    ijc = torch.round((pos - lo) / hc - 0.5).long()
    okc = ((ijc >= 0) & (ijc < torch.tensor(Gc, device=dev))).all(-1)
    ijc = ijc.clamp(min=torch.zeros(3, dtype=torch.long, device=dev),
                    max=torch.tensor([g - 1 for g in Gc], device=dev))
    out = torch.ones(res, res, 3, device=dev)
    hit = okc & (al[0] > 0.5) & solid[ijc[..., 0], ijc[..., 1], ijc[..., 2]]
    out[hit] = col[ijc[..., 0], ijc[..., 1], ijc[..., 2]][hit]
    return out.permute(2, 0, 1), al


def _positions(res):
    """World position at every pixel of the cut face, from the polygons themselves."""
    P, T, _ = ON.cut_polygons(st, hn, d, device=dev)
    ph = (torch.cat([P, torch.ones_like(P[:, :1])], 1) @ hmvp)[None]
    Ft = T.int().contiguous()
    rast, _ = dr.rasterize(glctx, ph, Ft, resolution=[res, res])
    pos, _ = dr.interpolate(P[None].contiguous(), rast, Ft)
    return pos[0]


ours, al = ON.render_section(st, glctx, hmvp, hn, d, RES, exterior=False)[:2]
same, _ = coarse_slice(1.0, RES)
eqmem, _ = coarse_slice(scale, RES)
slab, al_s, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES, exterior=False,
                                     thickness=0.03322, n_sub=7)
m = (al > 0.5).float()
print(f"  cut face gradient: ours {grad(ours, m):.4f}, dense grid at the same cell size "
      f"{grad(same, m):.4f}, dense grid at equal memory {grad(eqmem, m):.4f}, "
      f"the pipeline's 5.63-cell slab {grad(slab, m):.4f}")
for name, x in (("same cell size", same), ("equal memory", eqmem), ("the slab", slab)):
    print(f"    against {name}: {float(((ours - x).abs() * m).sum() / m.sum() / 3):.4f} per pixel")

sheet = torch.cat([ours, same, eqmem, slab], -1).clamp(0, 1).permute(1, 2, 0)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/demo_adv_{OBJ}.jpg", quality=94)
print(f"SHEET demo_adv_{OBJ}.jpg  (ours | dense at the same cell size | dense at equal memory | "
      f"the pipeline's slab)")

ys, xs = (m[0] > 0.5).nonzero(as_tuple=True)
cy, cx = int(ys.float().mean()), int(xs.float().mean())
r = int(0.3 * (float(ys.max()) - float(ys.min())))
S = 150
y0, x0 = max(cy + r - S // 2, 0), max(cx - S // 2, 0)
z = torch.cat([x[:, y0:y0 + S, x0:x0 + S] for x in (ours, same, eqmem, slab)], -1)
Image.fromarray((z.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) \
    .resize((z.shape[-1] * 4, S * 4), Image.NEAREST).save(f"{W}/demo_adv_zoom_{OBJ}.jpg", quality=94)
print(f"SHEET demo_adv_zoom_{OBJ}.jpg  (the same patch at 4x, same order)")
