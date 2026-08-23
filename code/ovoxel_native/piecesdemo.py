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
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON, anchor
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RES = int(os.environ.get("RES", "512"))
SS = int(os.environ.get("SS", "2"))   # supersample, then box-filter down
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

# The camera has to see BOTH cuts open. Looking along a plane makes its two components overlap
# exactly, and the first render showed four pieces as two halves. Choose the stored view under
# which each plane's separation direction travels furthest across the image, worst case first.
def _ndc(mvp, x):
    q = torch.cat([x, torch.ones_like(x[:1])])[None] @ mvp
    return q[0, :2] / q[0, 3]


def _sees(mvp):
    o = _ndc(mvp, mid)
    return min(float((_ndc(mvp, mid + n * hc * 10) - o).norm()) for n in (n1, n2))


mvps = [torch.as_tensor(m, dtype=torch.float32, device=dev) for m in C["v_mvp"]]
best = max(range(len(mvps)), key=lambda i: _sees(mvps[i]))
mvp = mvps[best]
print(f"  view {best} of {len(mvps)}: the weaker cut opens by {_sees(mvp):.3f} ndc")

# "down" is whatever falls down IN THE IMAGE. -z in world coordinates put the motion sideways.
o0 = _ndc(mvp, mid)
cand = torch.cat([torch.eye(3, device=dev), -torch.eye(3, device=dev)])
g = cand[max(range(6), key=lambda i: float(o0[1] - _ndc(mvp, mid + cand[i] * hc * 10)[1]))]
print(f"  falling along world {g.tolist()}, which is down in this view")


def _ndcz(mvp, x):
    q = torch.cat([x, torch.ones_like(x[:1])])[None] @ mvp
    return float(q[0, 2] / q[0, 3])


def _rot(axis, ang):
    k = axis / axis.norm()
    K = torch.zeros(3, 3, device=dev)
    K[0, 1], K[0, 2], K[1, 0] = -k[2], k[1], k[2]
    K[1, 2], K[2, 0], K[2, 1] = -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


# Every stored camera sits on the equator, so a transverse cut face is exactly edge-on and its
# flesh reduces to a sliver. Tilt the scene about the screen-horizontal axis until that face turns
# towards the camera; the sign is chosen by which one brings it nearer in depth, not by eye.
_o = _ndc(mvp, mid)
_h = max(range(6), key=lambda i: float((_ndc(mvp, mid + cand[i] * hc * 10) - _o)[0]))
TILT = _rot(cand[_h], torch.tensor(0.42, device=dev))
if _ndcz(mvp, mid + (n1 * hc * 10) @ TILT.T) > _ndcz(mvp, mid + (n1 * hc * 10) @ TILT.T.inverse()):
    TILT = TILT.T
print(f"  tilted 24 degrees about world {cand[_h].tolist()} to open the transverse face")

# Each component is its own rigid body: its own centroid, its own outward velocity, its own spin,
# its own floor. Nothing here is solved -- the trajectories are written down.
body = {}
for j, q in enumerate(pieces):
    c = cen[code == q].mean(0)
    s_1 = 1.0 if (q & 2) else -1.0
    s_2 = 1.0 if (q & 1) else -1.0
    body[q] = dict(c=c,
                   v=(s_1 * n1 * 1.35 + s_2 * n2 * 1.45) * hc * 6.5,
                   w=torch.cross(g, c - mid, dim=0) + n1 * 1e-3,
                   sp=(0.05 + 0.02 * j) * s_1,
                   floor=(13.0 + 0.7 * j) * hc * 10)

# the object is shrunk about its centre so the four have room to travel without leaving the frame
Z = 0.58
RISE = None   # set once the floor is known: start high, land low

RISE = -g * (0.46 * float(np.mean([b["floor"] for b in body.values()])) * Z)

frames = []
for t in range(NFRAME):
    tt = t / max(NFRAME - 1, 1)
    parts_v, parts_c, parts_f, off = [], [], [], 0
    # a written timeline, in three parts: the object is whole, the four separate along the cut
    # normals, then gravity acts. Separation before gravity is what makes the partition legible;
    # a single quadratic from t=0 mixes the two and the pieces only ever look like they slid.
    HOLD, OPEN = 0.14, 0.44
    sep = 0.0 if tt < HOLD else min((tt - HOLD) / (OPEN - HOLD), 1.0)
    sep = sep * sep * (3 - 2 * sep)
    gt = max(tt - OPEN, 0.0) / max(1.0 - OPEN, 1e-6)

    for q in pieces:
        B = body[q]
        fall = B["floor"] * (gt ** 2) * 1.9
        if fall > B["floor"]:      # it lands, and keeps a little of what it arrived with
            fall = B["floor"] - 0.30 * (fall - B["floor"])
            fall = max(fall, B["floor"] * 0.80)
        R = _rot(B["w"], torch.tensor(B["sp"] * gt, device=dev))
        shift = B["v"] * (sep + 0.45 * gt) + g * fall

        def place(X):
            return ((X - B["c"]) @ R.T + B["c"] + shift - mid) @ TILT.T * Z + mid + RISE

        V = place(mv_full)
        F = faces[q]
        parts_v.append(V); parts_c.append(mc_full); parts_f.append(F.int() + off)
        off += len(V)
        for (P, T, col, idx, on) in cuts:
            if T.numel() == 0:
                continue
            want = (q & 1) if idx == 0 else ((q >> 1) & 1)   # the other plane's bit for this piece
            keep = (on[T.long()] == want).all(1)
            if keep.any():
                parts_v.append(place(P)); parts_c.append(col)
                parts_f.append(T[keep].int() + off)
                off += len(P)
    Vt = torch.cat(parts_v); Ct = torch.cat(parts_c).clamp(0, 1)
    Ft = torch.cat(parts_f).contiguous().int()
    ph = (torch.cat([Vt, torch.ones_like(Vt[:, :1])], 1) @ mvp)[None]
    rast, _ = dr.rasterize(glctx, ph, Ft, resolution=[RES * SS, RES * SS])
    img, _ = dr.interpolate(Ct[None], rast, Ft)
    img = dr.antialias(img, rast, ph, Ft)
    a = (rast[..., 3:] > 0).float()
    out = img * a + (1 - a)
    o = out[0].permute(2, 0, 1)[None]
    if SS > 1:
        o = torch.nn.functional.avg_pool2d(o, SS)
    frames.append((o[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

import imageio.v2 as imageio
seq = frames + frames[-2:0:-1]
imageio.mimwrite(f"{W}/pieces_{OBJ}.mp4", seq, fps=24, codec="libx264", quality=8,
                 macro_block_size=1, output_params=["-pix_fmt", "yuv420p"])
# the GIF is quantised to 256 colours, so it is written from the same frames at a smaller size
# where the banding is less visible than it is at full width
small = [np.asarray(Image.fromarray(f).resize((RES * 3 // 4, RES * 3 // 4), Image.LANCZOS))
         for f in seq]
imageio.mimwrite(f"{W}/pieces_{OBJ}.gif", small, duration=0.042, loop=0)
print(f"pieces_{OBJ}.mp4 / .gif  {len(seq)} frames  {RES}x{RES} from {RES*SS}  "
      f"{len(pieces)} components")
