"""An oblique cut, drawn twice: as the closed-form polygon, and the way a voxel grid is sliced.

Both use the same field, the same plane and the same camera. The only difference is how the cut
face is obtained.

  O-Voxel      the plane against the occupancy gives a polygon per cell, exact to the cell's
               corners; the polygons are rasterised, so the outline is a continuous line and the
               colour is the trilinear field sampled at the fragment.

  voxel slice  the plane is a quad over the object's bounding box; every pixel of it looks up the
               cell its own position falls in. A pixel is opaque if that cell is solid. This is
               what slicing a dense grid gives, and its outline can only ever be a boundary between
               whole cells.

Measured first, before claiming anything: the two OUTLINES are the same. At 0, 30 and 45 degrees
they disagree about 0.0% of the cut area and their isoperimetric ratios match to three digits. That
is not a bug in either -- `cut_polygons` intersects the plane with each solid CUBE, so the outline
of the union is the occupancy boundary at cell resolution, which is exactly what a nearest-cell test
returns. The dual grid smooths the object's SURFACE; it has nothing to do with the outline of a cut.

What does differ is inside the outline, and it is the reason the closed-form polygon exists: the
polygon is rasterised and its colour comes from the field at each fragment, so a cell contributes a
gradient across its own cross-section, while a nearest-cell slice fills that whole cross-section
with one colour. On an oblique plane a cell's cross-section is a hexagon several pixels across, so
this is the difference between flesh and a mosaic of it.
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
STATE = os.environ.get("STATE", f"{W}/state_{OBJ}.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_{OBJ}_bal.npz")
RES = int(os.environ.get("RES", "1024"))
ANGLES = [float(a) for a in os.environ.get("DEMO_ANGLES", "0,30,45").split(",")]
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)

p = torch.load(f"{W}/{RUN}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
if "dec_i" in p:
    di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
    di.load_state_dict(p["dec_i"])
    ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()
else:
    st["interior"] = p["interior"].to(dev); st["surf_rgb"] = p["surf_rgb"].to(dev)
print(f"{OBJ} from {RUN}: {len(st['interior']):,} interior cells, cell size {st['hc']:.5f}")

hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
axis = np.asarray(C["h_planes"][0, :3], float); axis /= np.linalg.norm(axis)
hd = C["h_planes"][:, 3]
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
d_mid = float(hd[(H_LO + H_HI) // 2])
a0 = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
u = np.cross(axis, a0); u /= np.linalg.norm(u)


def tilted(deg):
    t = np.deg2rad(deg)
    n = np.cos(t) * axis + np.sin(t) * u
    return n / np.linalg.norm(n)


def voxel_slice(n, d, res):
    """The same plane, sampled the way a dense grid is sampled: one lookup per pixel, nearest cell."""
    hc = st["hc"]
    org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
    lo = torch.as_tensor(st["idx_lo"], dtype=torch.float32, device=dev)
    G = st["idx3"].shape
    nt = torch.as_tensor(n, dtype=torch.float32, device=dev)
    cen = (st["solid"].float().mean(0) + 0.5) * hc + org
    cen = cen - nt * (cen @ nt + d)      # the renderer's plane is n.x + d = 0, not n.x = d
    ext = float(((st["solid"].float().max(0).values - st["solid"].float().min(0).values) * hc).max())
    w = torch.as_tensor(u, dtype=torch.float32, device=dev)
    w = w - nt * (w @ nt); w = w / w.norm()
    v = torch.cross(nt, w, dim=0)
    quad = torch.stack([cen + s * ext * 0.75 * w + t * ext * 0.75 * v
                        for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=dev)
    ph = (torch.cat([quad, torch.ones_like(quad[:, :1])], 1) @ hmvp)[None]
    rast, _ = dr.rasterize(glctx, ph, tri, resolution=[res, res])
    pos, _ = dr.interpolate(quad[None], rast, tri)
    ijk = torch.round((pos[0] - org) / hc - 0.5 - lo).long()
    ok = ((ijk >= 0) & (ijk < torch.tensor(G, device=dev))).all(-1) & (rast[0, ..., 3] > 0)
    ijk = ijk.clamp(min=torch.zeros(3, dtype=torch.long, device=dev),
                    max=torch.tensor([g - 1 for g in G], device=dev))
    row = st["idx3"][ijk[..., 0], ijk[..., 1], ijk[..., 2]].long()
    solid = ok & (row >= 0)
    img = torch.ones(res, res, 3, device=dev)
    img[solid] = st["interior"][row[solid].clamp(min=0)]
    return img.permute(2, 0, 1), solid.float()[None]


def shape(mask):
    """Area and outline length in pixels, and the ratio that says how ragged the outline is."""
    m = mask[0] > 0.5
    area = float(m.sum())
    b = m & ~(torch.nn.functional.pad(m[None, None].float(), (1, 1, 1, 1), value=0)
              .unfold(2, 3, 1).unfold(3, 3, 1).amin((-1, -2))[0, 0] > 0.5)
    return area, float(b.sum())


rows_ours, rows_vox, notes = [], [], []
for deg in ANGLES:
    n = tilted(deg)
    nt = torch.as_tensor(n, dtype=torch.float32, device=dev)
    ctr = (st["solid"].float().mean(0).cpu().numpy() + 0.5) * st["hc"] + st["org"]
    d = d_mid if deg == 0 else float(-np.dot(n, ctr))    # a plane through the object's centre
    im_o, al_o, _, _ = ON.render_section(st, glctx, hmvp, nt, d, RES, exterior=False)
    im_v, al_v = voxel_slice(n, d, RES)
    ao, po = shape(al_o); av, pv = shape(al_v)
    if min(ao, av) < 16:
        print(f"  {deg:4.0f} deg: empty cut (ours {ao:.0f} px, voxel slice {av:.0f} px)")
        continue
    dis = float(((al_o > 0.5) ^ (al_v > 0.5)).float().sum()) / max(ao, 1)
    notes.append((deg, ao, po / max(ao, 1) ** 0.5, av, pv / max(av, 1) ** 0.5, dis))
    rows_ours.append(im_o); rows_vox.append(im_v)
    print(f"  {deg:4.0f} deg: outline ratio ours {po / ao ** 0.5:.3f}, voxel slice "
          f"{pv / av ** 0.5:.3f} (a circle is 3.545); they disagree about "
          f"{100 * dis:.1f}% of the cut area")


def grid(rs):
    return torch.cat([r.clamp(0, 1) for r in rs], -1).permute(1, 2, 0).cpu().numpy()


sheet = np.concatenate([grid(rows_ours), grid(rows_vox)], 0)
Image.fromarray((sheet * 255).astype(np.uint8)).save(f"{W}/demo_oblique_{OBJ}.jpg", quality=94)
print(f"SHEET demo_oblique_{OBJ}.jpg  (O-Voxel above, voxel slice below; "
      f"{', '.join(f'{a:g} deg' for a in ANGLES)})")

# the same patch of flesh in both, well inside the outline, where the mosaic shows
zo, zv, S = [], [], 128
for im_o, im_v in zip(rows_ours, rows_vox):
    m = (im_o.mean(0) < 0.98)
    ys, xs = m.nonzero(as_tuple=True)
    cy, cx = int(ys.float().mean()), int(xs.float().mean())
    r = int(0.25 * (float(ys.max()) - float(ys.min())))     # a quarter of the way out, all flesh
    y0 = max(min(cy + r, im_o.shape[-2] - S), 0)
    x0 = max(min(cx + r, im_o.shape[-1] - S), 0)
    zo.append(im_o[:, y0:y0 + S, x0:x0 + S])
    zv.append(im_v[:, y0:y0 + S, x0:x0 + S])
    g = lambda a: float((a[..., 1:] - a[..., :-1]).abs().mean() +
                        (a[..., 1:, :] - a[..., :-1, :]).abs().mean()) / 2
    print(f"  the same patch: gradient ours {g(zo[-1]):.4f}, voxel slice {g(zv[-1]):.4f}, "
          f"they differ by {float((zo[-1] - zv[-1]).abs().mean()):.4f} per pixel")
zs = np.concatenate([grid(zo), grid(zv)], 0)
Image.fromarray((zs * 255).astype(np.uint8)).resize((zs.shape[1] * 3, zs.shape[0] * 3), Image.NEAREST) \
    .save(f"{W}/demo_rim_{OBJ}.jpg", quality=94)
print(f"SHEET demo_rim_{OBJ}.jpg  (one patch of flesh at 3x, same order)")
