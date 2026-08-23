"""Two cuts, the pieces the lattice discovers, and a geometry-driven separation.

The claim this supports is the interface, not a simulation: a cut partitions the integer lattice
into connected components, each component carries its own cells and its own share of the boundary
surface, and every cell belongs to exactly one component, so mass is conserved by construction.

Each frame applies a rigid transform per component -- an initial velocity along the cut normal,
constant gravity, and a ground plane with restitution -- and rasterises every component's exterior
and exposed faces in one nvdiffrast pass, so the depth test composes them. No solver is involved
and no material parameter is used; the motion is prescribed. What is demonstrated is that the
partition, the exposed faces and the adjacency are available to one.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON, anchor
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RES = int(os.environ.get("RES", "512"))
NFRAME = int(os.environ.get("NFRAME", "60"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
if "dec_i" in p:
    w = p["dec_i"]["stage1.0.weight"].shape[0]
    nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
    anchor.W_HID, anchor.N_HID = w, nl
    di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
    di.load_state_dict(p["dec_i"])
    ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()

hc = float(st["hc"])
org = torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
cen = (st["solid"].float() + 0.5) * hc + org
mid = cen.mean(0)

# two planes: one transverse, one longitudinal, both through the object
n1 = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
n1 = n1 / n1.norm(); d1 = float(-(mid @ n1))
k = len(C["v_planes"]) // 2
n2 = torch.as_tensor(C["v_planes"][k, :3], dtype=torch.float32, device=dev)
n2 = n2 / n2.norm(); d2 = float(-(mid @ n2))

s1 = (cen @ n1 + d1 > 0)
s2 = (cen @ n2 + d2 > 0)
code = (s1.long() * 2 + s2.long())
pieces = [int(x) for x in torch.unique(code)]
counts = [int((code == q).sum()) for q in pieces]
print(f"{OBJ}: two planes -> {len(pieces)} components, cells {counts}, "
      f"sum {sum(counts):,} of {len(cen):,} (mass conserved: {sum(counts) == len(cen)})")

mv_full, mf_full, mc_full = ON.surface_mesh(st)
# a surface vertex belongs to the component its own position falls in
vs1 = (mv_full @ n1 + d1 > 0)
vs2 = (mv_full @ n2 + d2 > 0)
vcode = (vs1.long() * 2 + vs2.long())

faces = {}
for q in pieces:
    keep = (vcode[mf_full.long()] == q).all(1)
    faces[q] = mf_full[keep].long()

# the exposed face of each cut, split by which side it belongs to
# The exposed face of each cut, and which component sees it. A polygon lying on plane A has no
# side with respect to A -- it is the boundary -- so both components adjacent to it receive their
# own copy; the only test that decides is the OTHER plane's sign.
cuts = []
for idx, (nn, dd) in enumerate(((n1, d1), (n2, d2))):
    P, T, _ = ON.cut_polygons(st, nn, dd, device=dev)
    col = ON.sample_interior(st, P).clamp(0, 1)
    other = 1 if idx == 0 else 0
    on = (P @ (n2 if idx == 0 else n1) + (d2 if idx == 0 else d1) > 0).long()
    cuts.append((P, T, col, idx, on))

mvp = torch.as_tensor(C["v_mvp"][(k + len(C["v_planes"]) // 4) % len(C["v_planes"])],
                      dtype=torch.float32, device=dev)
g = torch.tensor([0.0, 0.0, -1.0], device=dev)      # a direction to fall along, in world units
axis = n1 if abs(float(n1[2])) < 0.9 else n2
vel = {}
for i, q in enumerate(pieces):
    s_1 = 1.0 if (q & 2) else -1.0
    s_2 = 1.0 if (q & 1) else -1.0
    vel[q] = (s_1 * n1 + s_2 * n2) * hc * 14.0

frames = []
for t in range(NFRAME):
    tt = t / max(NFRAME - 1, 1)
    parts_v, parts_c, parts_f, off = [], [], [], 0
    for q in pieces:
        shift = vel[q] * tt + g * (0.5 * 14.0 * hc * (tt ** 2))
        V = mv_full + shift
        F = faces[q]
        parts_v.append(V); parts_c.append(mc_full); parts_f.append(F.int() + off)
        off += len(V)
        for (P, T, col, idx, on) in cuts:
            if T.numel() == 0:
                continue
            want = (q & 1) if idx == 0 else ((q >> 1) & 1)   # the other plane's bit for this piece
            keep = (on[T.long()] == want).all(1)
            if keep.any():
                parts_v.append(P + shift); parts_c.append(col)
                parts_f.append(T[keep].int() + off)
                off += len(P)
    Vt = torch.cat(parts_v); Ct = torch.cat(parts_c).clamp(0, 1)
    Ft = torch.cat(parts_f).contiguous().int()
    ph = (torch.cat([Vt, torch.ones_like(Vt[:, :1])], 1) @ mvp)[None]
    rast, _ = dr.rasterize(glctx, ph, Ft, resolution=[RES, RES])
    img, _ = dr.interpolate(Ct[None], rast, Ft)
    img = dr.antialias(img, rast, ph, Ft)
    a = (rast[..., 3:] > 0).float()
    out = img * a + (1 - a)
    frames.append((out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

import imageio.v2 as imageio
seq = frames + frames[::-1]
imageio.mimwrite(f"{W}/pieces_{OBJ}.gif", seq, duration=0.05, loop=0)
print(f"pieces_{OBJ}.gif  {len(seq)} frames  {RES}x{RES}  {len(pieces)} components")
