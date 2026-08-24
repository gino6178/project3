"""Draw what the solver produced: one rigid transform per piece, applied to the O-Voxel surface.

The solver moves 120k particles; the surface is 2.0M vertices. Rather than skin every vertex to a
neighbourhood of particles, each piece is summarised by the best rigid transform of its own
particles and the surface it owns is carried by that transform. `dropsim.py` measures what this
discards -- the non-rigid residual, in coarse cells -- and the caption reports it, so the
approximation is stated rather than hidden. It is available because the pieces here separate and
land; it would not be available under large deformation, and that is a different experiment.

A vertex's piece is decided by its own position against the cut planes, not by whichever particle
happens to be nearest: nearest-particle assignment lets vertices beside the cut land on either side
depending on sub-cell geometry, which tears the cut face into a ragged edge. The mapping from sign
region to solver piece is checked against the labelling rather than assumed.
"""
import os, sys
import numpy as np, torch
from PIL import Image
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RES = int(os.environ.get("RES", "560"))
SS = int(os.environ.get("SS", "2"))
dev = "cuda"
glctx = dr.RasterizeCudaContext(device=dev)

P = np.load(f"{W}/drop_prep_{OBJ}.npz")
D = np.load(f"{W}/drop_traj_{OBJ}.npz")
t = lambda k, s=P: torch.from_numpy(s[k]).to(dev)

mv, mf, mc = t("mv").float(), t("mf").int(), t("mc").float().clamp(0, 1)
n1, n2 = t("n1").float(), t("n2").float()
d1, d2 = float(P["d1"]), float(P["d2"])
hc, mid = float(P["hc"]), t("mid").float()

R = t("R", D).float(); T = t("T", D).float(); NV = D["nv"]
stage_f = D["stage_frames"]; stage_a = D["stage_assign"]
x0 = t("x0", D).float()
FRAMES = min(len(NV), int(os.environ.get("RFRAMES", len(NV))))

# sign code of every particle and every surface vertex: 0..3 for the four sign regions
def code(x):
    return ((x @ n1 + d1 > 0).long() * 2 + (x @ n2 + d2 > 0).long())


# A plane that has not arrived yet decides nothing, so a stage's sign code carries only the bits
# of the planes already in the object.
BITS = [0, 2, 3]


def code_at(x, s):
    c = torch.zeros(len(x), dtype=torch.long, device=dev)
    if BITS[s] & 2:
        c = c + (x @ n1 + d1 > 0).long() * 2
    if BITS[s] & 1:
        c = c + (x @ n2 + d2 > 0).long()
    return c


# Which solver piece each sign region became, read off the labelling rather than assumed. If a
# region ever held two components -- a shape where one plane cuts off two separate lobes -- this
# mapping is not a function, and the assertion says so instead of drawing the wrong thing.
region_of, vc_at, face_at = [], [], []
for s in range(len(stage_f)):
    pc = code_at(x0, s).cpu().numpy()
    a_ = stage_a[s]
    m = {}
    for b in range(int(a_.max()) + 1):
        cs = np.unique(pc[a_ == b])
        assert len(cs) == 1, f"stage {s} piece {b} spans sign regions {cs}"
        m[int(cs[0])] = b
    region_of.append(m)
    c = code_at(mv, s)
    ok = (c[mf.long()] == c[mf.long()][:, :1]).all(1)
    vc_at.append(c); face_at.append(mf[ok])
    print(f"  stage {s} at frame {stage_f[s]}: regions {m}, "
          f"{int(ok.sum()):,} of {len(mf):,} faces survive")

cuts = []
for i in (0, 1):
    Pp = t(f"cut{i}_P").float(); Tt = t(f"cut{i}_T").int()
    Cc = t(f"cut{i}_C").float().clamp(0, 1)
    cuts.append((Pp, Tt, Cc, t(f"cut{i}_on").long(), i))

mvps = [torch.as_tensor(m, dtype=torch.float32, device=dev) for m in P["v_mvp"]]


def ndc(mvp, x):
    q = torch.cat([x, torch.ones_like(x[..., :1])], -1) @ mvp
    return q[..., :2] / q[..., 3:4]


def sees(mvp):
    o = ndc(mvp, mid[None])[0]
    return min(float((ndc(mvp, (mid + n * hc * 10)[None])[0] - o).norm()) for n in (n1, n2))


best = max(range(len(mvps)), key=lambda i: sees(mvps[i]))
mvp = mvps[best]
print(f"  view {best} of {len(mvps)}: the weaker cut opens by {sees(mvp):.3f} ndc")


def rot(axis, ang):
    k = axis / axis.norm()
    K = torch.zeros(3, 3, device=dev)
    K[0, 1], K[0, 2], K[1, 0] = -k[2], k[1], k[2]
    K[1, 2], K[2, 0], K[2, 1] = -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + float(np.sin(ang)) * K + float(1 - np.cos(ang)) * (K @ K)


# Every stored camera sits on the equator, which leaves a transverse cut face exactly edge-on and
# its flesh a sliver. Tilt the scene about the screen-horizontal axis; the sign is chosen by which
# one brings that face nearer in depth.
cand = torch.cat([torch.eye(3, device=dev), -torch.eye(3, device=dev)])
o0 = ndc(mvp, mid[None])[0]
hx = max(range(6), key=lambda i: float((ndc(mvp, (mid + cand[i] * hc * 10)[None])[0] - o0)[0]))
TILT = rot(cand[hx], 0.42)


def zof(x):
    q = torch.cat([x, torch.ones_like(x[..., :1])], -1) @ mvp
    return float((q[..., 2] / q[..., 3]).mean())


if zof((mid + (n1 * hc * 10) @ TILT.T)[None]) > zof((mid + (n1 * hc * 10) @ TILT.T.inverse())[None]):
    TILT = TILT.T


def tilted(X, Rb, Tb):
    return ((X @ Rb.T + Tb) - mid) @ TILT.T + mid


# One camera for the whole run, fitted to the whole run: the corners of every piece's bounding box
# are projected at every frame it exists, and the scene is scaled about the centre of that swept
# volume until it fits. Nothing is tracked -- the camera is still and the object falls through it.
corners = []
for s in range(len(stage_f)):
    f1 = stage_f[s + 1] if s + 1 < len(stage_f) else FRAMES
    for reg, b in region_of[s].items():
        # a sample of the piece's own surface, not its bounding box: the corners of a box
        # around a round object project a fifth further out than the object does, and fitting
        # to them shrinks the scene by that much for no reason
        v = mv[vc_at[s] == reg]
        g = v[torch.randperm(len(v), device=dev)[:1500]]
        for f in range(int(stage_f[s]), int(f1)):
            corners.append(tilted(g, R[f, b], T[f, b]))
corners = torch.cat(corners)
PIV = corners.mean(0)
q = ndc(mvp, (corners - PIV) + PIV)
lo, hi = q.min(0).values, q.max(0).values
Z = float(1.80 / float((hi - lo).max()))
print(f"  the run sweeps {[round(float(x), 2) for x in (hi - lo)]} of the frame's 2.0 "
      f"in ndc; scene scaled by {Z:.3f} about its centre to fit")

frames = []
for f in range(FRAMES):
    s = int(np.searchsorted(stage_f, f, side="right") - 1)
    parts_v, parts_c, parts_f, off = [], [], [], 0
    for reg, b in region_of[s].items():
        Rb, Tb = R[f, b], T[f, b]

        def place(X):
            return (tilted(X, Rb, Tb) - PIV) * Z + PIV

        F = face_at[s]
        sel = F[(vc_at[s][F.long()] == reg).all(1)]
        parts_v.append(place(mv)); parts_c.append(mc); parts_f.append(sel.int() + off)
        off += len(mv)
        for (Pp, Tt, Cc, on, i) in cuts:
            if s < i + 1:            # this plane has not arrived yet
                continue
            other_live = (s >= (2 if i == 0 else 1))
            if other_live:
                want = (reg & 1) if i == 0 else ((reg >> 1) & 1)
                keep = (on[Tt.long()] == want).all(1)
            else:                    # nothing else decides it: the whole face is this piece's
                keep = torch.ones(len(Tt), dtype=torch.bool, device=dev)
            if keep.any():
                parts_v.append(place(Pp)); parts_c.append(Cc)
                parts_f.append(Tt[keep].int() + off)
            off += len(Pp)
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
    if f % 20 == 0:
        print(f"    frame {f}: stage {s}, {len(region_of[s])} pieces, {len(Ft):,} triangles")

import imageio.v2 as imageio
imageio.mimwrite(f"{W}/drop_{OBJ}.mp4", frames, fps=24, codec="libx264", quality=8,
                 macro_block_size=1, output_params=["-pix_fmt", "yuv420p"])
small = [np.asarray(Image.fromarray(x).resize((RES * 3 // 4, RES * 3 // 4), Image.LANCZOS))
         for x in frames]
imageio.mimwrite(f"{W}/drop_{OBJ}.gif", small, duration=0.042, loop=0)
r = D["resid"]
print(f"drop_{OBJ}.mp4  {len(frames)} frames  {RES}x{RES} from {RES*SS}  "
      f"non-rigid residual {r.min():.3f} to {r.max():.3f} coarse cells")
