"""Does each object's polar axis point along the object?

The transverse family is defined as the planes perpendicular to a stored axis and the longitudinal
family as those containing it, so if that axis is not the object's own -- the apple's core, the
cake's vertical -- then the two families are cutting the wrong way and every figure and score
inherits it. The occupancy gives the object's own principal axes for free: the eigenvectors of the
cell positions' covariance.

Reported as the angle between the stored axis and each principal axis, and the extents along them,
so an object that is nearly spherical (where the principal axis means little) can be told from one
that is clearly elongated (where it means everything).
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON

W = os.path.dirname(os.path.abspath(__file__))
OBJS = [o for o in os.environ.get("AX_OBJS",
        "watermelon_sp,orange_sp,apple1_sp,bread_sp,cake2_sp,pomegranate2_sp,doughnut").split(",")]
dev = "cpu"
print(f"  {'object':<18}{'angle to the 1st':>18}{'2nd':>8}{'3rd':>8}   extents along them")
for OBJ in OBJS:
    try:
        st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
        C = np.load(f"{W}/cams_{OBJ}_bal.npz")
    except Exception as e:
        print(f"  {OBJ:<18}skipped ({type(e).__name__})"); continue
    axis = np.asarray(C["h_planes"][0, :3], float); axis /= np.linalg.norm(axis)
    p = st["solid"].float().numpy() * float(st["hc"])
    p = p - p.mean(0)
    w, V = np.linalg.eigh(p.T @ p / len(p))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    ang = [float(np.degrees(np.arccos(min(1.0, abs(np.dot(axis, V[:, k])))))) for k in range(3)]
    ext = [float(np.sqrt(w[k]) * 2) for k in range(3)]
    print(f"  {OBJ:<18}{ang[0]:>15.1f} deg{ang[1]:>8.1f}{ang[2]:>8.1f}   "
          f"{ext[0]:.3f} {ext[1]:.3f} {ext[2]:.3f}")
