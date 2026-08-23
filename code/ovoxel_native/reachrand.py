"""If the planes are drawn afresh each step instead of sitting at fixed depths, how far do they get?

The fixed schedule supervises the same 26 cut faces for the whole run, and measured, those faces
touch about half the interior; the rest of the cells never receive a gradient and keep whatever
they were initialised to. Drawing the planes at random spends the same number of renders per step
but spreads them, so the union grows with the number of steps rather than staying put.

This measures the union alone -- pure geometry, no fitting -- because that is what decides whether
the idea is worth training on. What it cannot say is whether a photograph is a fair target for a
plane it was not taken at; that is the next question, and a separate one.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
STATE = os.environ.get("STATE", f"{W}/state_{OBJ}.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_{OBJ}_bal.npz")
RES = int(os.environ.get("RES", "512"))
DRAWS = int(os.environ.get("RR_DRAWS", "400"))
# How the planes are chosen. "families" keeps them on the two lines the photographs lie on -- a
# transverse depth, or an azimuth about the axis. "golden" leaves those lines entirely: the normal
# walks a spherical Fibonacci spiral and the offset advances by the golden ratio, so the sequence
# never clumps the way independent draws do. Clumping is not hypothetical here: 26 independent
# draws reached less of the interior than the 26 placed planes did.
SEQ = os.environ.get("RR_SEQ", "families")
# "jitter" is the pipeline's own scheme rather than an alternative to it: every plane is visited
# once per outer iteration and moved within its own slot by +-JIT slots. The reach numbers this
# file reported first were taken at the FIXED depths, with no jitter at all, so they understate
# what the pipeline actually touches. This measures the real thing, as a function of how wide the
# slot sweep is -- which is the one knob that can close the gaps between the slabs and the wedges
# without ever leaving the orientations the photographs were taken at.
JIT = float(os.environ.get("RR_JIT", "0.5"))
ITERS = int(os.environ.get("RR_ITERS", "325"))
PHI = (1 + 5 ** 0.5) / 2


def plane_basis_np(n):
    """Two unit vectors spanning the plane whose normal is n -- the axis, for the longitudinal
    family, so the planes this returns all contain it."""
    n = n / np.linalg.norm(n)
    a = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def radical(k, base):
    """Van der Corput: every prefix of this sequence is spread, not just the whole of it."""
    f, r = 1.0, 0.0
    while k:
        f /= base
        r += f * (k % base)
        k //= base
    return r
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
NH, NV = H_HI - H_LO, len(vp)

st["interior"] = st["interior"].detach().clone().requires_grad_(True)
N = st["interior"].shape[0]


def touch(mvp, n, d):
    # a plane drawn on the sphere can miss the object altogether, and a cut with no polygons is an
    # empty triangle list the rasteriser refuses. Those planes touch nothing, which is the answer.
    _, _, k = ON.cut_polygons(st, n, float(d), device=dev)
    if k == 0:
        return torch.zeros(N, dtype=torch.bool, device=dev)
    st["interior"].grad = None
    img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(d), RES)
    img.sum().backward()
    g = st["interior"].grad
    return torch.zeros(N, dtype=torch.bool, device=dev) if g is None else (g.abs().sum(1) > 0)


rng = np.random.default_rng(0)
lo, hi = float(hd[H_LO]), float(hd[H_HI - 1])
step = (hi - lo) / max(NH - 1, 1)
# the band the fixed depths tile, plus the half step each end that they stand in the middle of
lo, hi = lo - step / 2, hi + step / 2

c_all = ((st["solid"].float() + 0.5) * st["hc"]
         + torch.as_tensor(st["org"], dtype=torch.float32, device=dev))
ctr = c_all.mean(0).cpu().numpy()
rad = float((c_all - c_all.mean(0)).norm(dim=1).max())

hit = torch.zeros(N, dtype=torch.bool, device=dev)
_kh = _kv = 0
marks = sorted({NH + NV, 50, 100, 200, DRAWS})
print(f"{OBJ}: {N:,} interior cells; fixed schedule is {NH} transverse and {NV} longitudinal")
t0 = time.time()
if SEQ == "jitter":
    marks = sorted({1, 5, 20, 80, ITERS})
    for it in range(ITERS):
        for i in range(NH):
            f = (rng.random() - 0.5) * 2.0 * JIT
            hit |= touch(hmvp, hn, float(hd[H_LO + i]) + (float(hd[1] - hd[0])) * f)
        for j in range(NV):
            f = (rng.random() - 0.5) * 2.0 * JIT
            a = np.pi * (j + f) / NV
            u2, w2 = plane_basis_np(np.asarray(hn.cpu()))
            nv = np.cos(a) * u2 + np.sin(a) * w2
            hit |= touch(torch.as_tensor(vmvp[j]),
                         torch.as_tensor(nv, dtype=torch.float32, device=dev),
                         float(-np.dot(nv, ctr)))
        if (it + 1) in marks:
            print(f"  jitter {JIT:g} slots, after {it + 1:4d} iterations: "
                  f"{100 * hit.float().mean():5.1f}% of the interior   {time.time() - t0:.0f}s",
                  flush=True)
    raise SystemExit

for k in range(DRAWS):
    if SEQ == "cycle":
        # each family keeps its own counter. Feeding the global step to both meant each family saw
        # a strided subsequence, and a subsequence of a low-discrepancy sequence is not itself low
        # discrepancy -- measured, it reached less of the interior than independent draws did.
        # the same two families the photographs lie on, but walked by a formula instead of drawn.
        # The family alternates in the ratio the plane count assigns; the depth advances by a
        # radical inverse and the azimuth by the golden angle, so any prefix is spread and the
        # sequence is reproducible from k alone -- no random state anywhere in the schedule.
        if (k * NH) // (NH + NV) != ((k + 1) * NH) // (NH + NV):
            _kh += 1
            hit |= touch(hmvp, hn, lo + radical(_kh, 2) * (hi - lo))
        else:
            _kv += 1
            a = np.pi * ((_kv * PHI) % 1.0)
            u2, w2 = plane_basis_np(np.asarray(hn.cpu()))
            nv = np.cos(a) * u2 + np.sin(a) * w2
            j = int(round(np.degrees(a) / 180 * NV)) % NV
            d = float(-np.dot(nv, ctr))     # the convention is n.x + d = 0, so a plane through
                                            # the centre has d = -n.c, not +n.c
            hit |= touch(torch.as_tensor(vmvp[j]),
                         torch.as_tensor(nv, dtype=torch.float32, device=dev), d)
        if (k + 1) in marks:
            print(f"  {k + 1:4d} planes from the formula reach {100 * hit.float().mean():5.1f}%"
                  f"   {time.time() - t0:.0f}s", flush=True)
        continue
    if SEQ == "golden":
        # The first version took z as k/N, which orders the spiral from one pole to the other: its
        # first 26 planes were all near the top and reached 0.8% of the interior. A radical inverse
        # gives the same set for the full sequence and a spread set for every prefix of it.
        z = 1 - 2 * radical(k + 1, 2)
        th = 2 * np.pi * ((k * PHI) % 1.0)
        rr = max(1 - z * z, 0.) ** 0.5
        nv = np.array([rr * np.cos(th), rr * np.sin(th), z])
        # and the offset within the object's own extent along THAT normal, so every plane cuts
        proj = c_all.cpu().numpy() @ nv
        d = float(-(proj.min() + radical(k + 1, 3) * (proj.max() - proj.min())))
        # any camera will do for coverage: what is being counted is which cells the plane touches
        hit |= touch(hmvp, torch.as_tensor(nv, dtype=torch.float32, device=dev), d)
        if (k + 1) in marks:
            print(f"  {k + 1:4d} planes on the spiral reach {100 * hit.float().mean():5.1f}%"
                  f"   {time.time() - t0:.0f}s", flush=True)
        continue
    if rng.random() < NH / (NH + NV):
        hit |= touch(hmvp, hn, rng.uniform(lo, hi))            # a depth anywhere in the band
    else:
        j = int(rng.integers(NV))                              # an azimuth between two cameras
        a = rng.random()
        nv = (1 - a) * vp[j, :3] + a * vp[(j + 1) % NV, :3]
        nv = nv / np.linalg.norm(nv)
        d = float(np.dot(nv, vp[j, :3] * vp[j, 3]))
        hit |= touch(torch.as_tensor(vmvp[j]), torch.as_tensor(nv, dtype=torch.float32,
                                                               device=dev), d)
    if (k + 1) in marks:
        print(f"  {k + 1:4d} drawn planes reach {100 * hit.float().mean():5.1f}%"
              f"   {time.time() - t0:.0f}s", flush=True)
