"""The axis each object is actually cut along, against the axis Gino says is its top.

    python upcheck.py

The transverse family is defined by one camera at elevation 90, which is "straight down" in
whatever frame the object's physics config declares. Five of the seven objects share
config/orange_physics.json. If their tops do not all point the same way, that one declaration
cannot be right for all of them, and a plane called transverse is not transverse.

`h_planes` carries the normal in the lattice frame, which is the frame the turntable drew its
arrows in, so the two are directly comparable.
"""
import numpy as np

W = "/workspace/ovoxel_native"
NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]
VECS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
SAID = {"orange_sp": "-y", "watermelon_sp": "+y", "apple1_sp": "+z", "bread_sp": "-y",
        "cake2_sp": "-z", "pomegranate2_sp": "-y", "doughnut": "+z"}
CFG = {"orange_sp": "orange_physics", "watermelon_sp": "watermelon_raw_physics",
       "apple1_sp": "orange_physics", "bread_sp": "orange_physics",
       "cake2_sp": "orange_physics", "pomegranate2_sp": "orange_physics",
       "doughnut": "torus_physics"}


def nearest(v):
    v = v / np.linalg.norm(v)
    i = int(np.argmax(VECS @ v))
    return NAMES[i], float((VECS @ v)[i])


print(f"  {'object':16s} {'config':24s} {'cut along':>10} {'you said top is':>16} "
      f"{'angle between':>14}")
for obj, said in SAID.items():
    C = np.load(f"{W}/cams_{obj}.npz")
    n = np.asarray(C["h_planes"][0, :3], float)
    nm, c = nearest(n)
    u = VECS[NAMES.index(said)]
    ang = np.degrees(np.arccos(np.clip(abs(float(n @ u / np.linalg.norm(n))), 0, 1)))
    mark = "" if ang < 25 else ("   <-- not the same axis" if ang > 60 else "   <-- off")
    print(f"  {obj:16s} {CFG[obj]:24s} {nm:>10} {said:>16} {ang:13.1f}°{mark}")
print("\n  (the angle is to the nearer of the axis and its opposite: a transverse plane does not"
      "\n   care which end of its normal is up, only that the normal is the object's own axis)")
