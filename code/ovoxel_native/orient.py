"""Is each reference the right way up, and the right way round, against the shell?

    TAG=ov2 python orient.py

`_same_topology_map` carries the reference across at the same angle it sits at, so it preserves
orientation exactly: the photograph's up becomes the render's up. Nothing establishes that the two
ups are the same one. The transverse family is phase-aligned, but that aligns the photographs to
EACH OTHER, not to the object; the longitudinal family is not aligned at all.

The test does not need to know what a stem is. The object's shell is pinned from the released
model, so the render's own profile is the object's. Take the reference's profile along each axis,
take the render's, and compare the correlation as they sit against the correlation with the
reference reversed. If reversing correlates better, that axis is the wrong way round.

Reported as a margin: (corr as-is) - (corr reversed). Positive is right way round, and the size
says how confidently -- a symmetric object has no answer and should come out near zero.
"""
import os
import re
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr

sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/rebuild/project3/code/src")
import ovcut
import ovnative as ON
import refsel

W = "/workspace/ovoxel_native"
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
TAG = os.environ.get("TAG", "ov2")
RES = 512
OBJS = [("orange_sp", "orange"), ("watermelon_sp", "watermelon"), ("apple1_sp", "apple"),
        ("bread_sp", "loaf"), ("cake2_sp", "cake"), ("pomegranate2_sp", "pomegranate"),
        ("doughnut", "doughnut")]


def refdir(obj, which):
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    return os.path.join(FN, m.group(1))


def profile(a, axis, n=128):
    """Mean luminance along one axis over the object's own extent, resampled to n samples."""
    L = a.mean(2)
    m = L < 0.97
    ys, xs = np.where(m)
    if len(ys) < 500:
        return None
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    c, msk = L[sl], m[sl]
    p = (c * msk).sum(1 - axis) / np.maximum(msk.sum(1 - axis), 1)
    ok = msk.sum(1 - axis) > 0.15 * msk.shape[1 - axis]
    if ok.sum() < 8:
        return None
    p = p[ok]
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)


def margin(pr, pg):
    if pr is None or pg is None:
        return np.nan
    f = lambda u, v: float(np.corrcoef(u - u.mean(), v - v.mean())[0, 1])
    return f(pr, pg) - f(pr[::-1], pg)


ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device="cuda")
print(f"  {'object':16s} {'family':6s} {'up/down':>9} {'left/right':>11}   verdict")
for obj, label in OBJS:
    st = ovcut.load(obj, TAG)
    C = np.load(f"{W}/cams_{obj}.npz")
    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    NH, NV = H_HI - H_LO, len(C["v_mvp"])
    for fam, which, mvp, pl, idx, n in (
            ("rh", "REF_H", C["h_mvp"], C["h_planes"][H_LO + NH // 2], NH // 2, NH),
            ("rv", "REF_V", C["v_mvp"][NV // 2], C["v_planes"][NV // 2], NV // 2, NV)):
        ref = np.asarray(refsel.as_array(
            (refsel.solved_photo if fam == "rh" else refsel.photo)(refdir(obj, which), idx, n),
            RES), np.float32)
        nn = torch.as_tensor(pl[:3], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            img, _, _, _ = ON.render_section(
                st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"),
                nn, float(pl[3]), RES)
        R = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ud = margin(profile(ref, 0), profile(R, 0))
        lr = margin(profile(ref, 1), profile(R, 1))
        bad = [nm for nm, v in (("up/down", ud), ("left/right", lr))
               if not np.isnan(v) and v < -0.15]
        print(f"  {label:16s} {fam:6s} {ud:9.3f} {lr:11.3f}   "
              + ("reversed: " + ", ".join(bad) if bad else "consistent"))
