"""The axis each object is a solid of revolution about, found rather than assumed.

The transverse family is defined perpendicular to a stored axis and the longitudinal family through
it, so if that axis is not the object's own the families cut the wrong way. Measured, the apple's
stored axis is 35 degrees off its nearest principal axis and the cake's 31 -- and a principal axis
is not the right target anyway for a nearly spherical fruit, where the principal directions are
decided by noise.

What an apple, an orange, a watermelon, a pomegranate and a doughnut do have is rotational
symmetry: turn one about its stem and the silhouette does not change. So the axis is the direction
about which the occupancy varies least with azimuth, and that is a search over directions with a
number attached rather than a label anyone has to trust. An object with no such symmetry -- the
cake, the loaf -- gives a high residual whatever axis is tried, which is itself the answer.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W = os.path.dirname(os.path.abspath(__file__))
OBJS = [o for o in os.environ.get("FA_OBJS",
        "watermelon_sp,orange_sp,apple1_sp,bread_sp,cake2_sp,pomegranate2_sp,doughnut").split(",")]
NDIR = int(os.environ.get("FA_DIRS", "600"))
NAZ, NZ = 36, 24


def residual(pts, axis):
    """How much the outer radius varies with azimuth, at each height along the axis."""
    a = axis / np.linalg.norm(axis)
    z = pts @ a
    rel = pts - np.outer(z, a)
    u = np.array([0., 0., 1.]) if abs(a[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(a, u); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    th = np.arctan2(rel @ e2, rel @ e1)
    r = np.linalg.norm(rel, axis=1)
    zi = np.clip(((z - z.min()) / max(float(np.ptp(z)), 1e-9) * NZ).astype(int), 0, NZ - 1)
    ai = np.clip(((th + np.pi) / (2 * np.pi) * NAZ).astype(int), 0, NAZ - 1)
    out = np.zeros((NZ, NAZ))
    np.maximum.at(out, (zi, ai), r)
    ok = out.max(1) > 0
    if ok.sum() < 4:
        return 1.0
    rows = out[ok]
    # the spread of the rim radius around each ring, relative to that ring's own radius
    return float(np.mean(rows.std(1) / np.maximum(rows.mean(1), 1e-9)))


def sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    th = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


print(f"  {'object':<18}{'stored':>9}{'best':>9}{'angle between':>15}   best direction")
dirs = sphere(NDIR)
for OBJ in OBJS:
    try:
        st = torch.load(f"{W}/state_{OBJ}.pt", map_location="cpu", weights_only=False)
        C = np.load(f"{W}/cams_{OBJ}_bal.npz")
    except Exception as e:
        print(f"  {OBJ:<18}skipped ({type(e).__name__})"); continue
    pts = st["solid"].float().numpy() * float(st["hc"])
    pts = pts - pts.mean(0)
    if len(pts) > 200000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 200000, replace=False)]
    stored = np.asarray(C["h_planes"][0, :3], float); stored /= np.linalg.norm(stored)
    rs = residual(pts, stored)
    vals = [residual(pts, d) for d in dirs]
    k = int(np.argmin(vals))
    best = dirs[k]
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(stored @ best))))))
    print(f"  {OBJ:<18}{rs:>9.4f}{vals[k]:>9.4f}{ang:>13.0f} deg   {np.round(best, 3)}")
