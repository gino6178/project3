"""The cut, as four states of the real data.

    (a) the lattice and the plane: solid cells, and the band the plane crosses
    (b) one crossed cell: its twelve edges, the roots v_j, and the convex polygon they bound
    (c) the sign field labelled: connected components of the two sides, one colour each
    (d) the exposed face: the same polygons, coloured by the interior field

No box-and-arrow diagram; every panel is the object's own state at that step.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
dev = "cpu"
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
hc = float(st["hc"])
org = np.asarray(st["org"], float)
solid = st["solid"].numpy().astype(np.int64)
cen = (solid + 0.5) * hc + org

n = np.asarray(C["v_planes"][len(C["v_planes"]) // 2, :3], float)
n /= np.linalg.norm(n)
d = float(C["v_planes"][len(C["v_planes"]) // 2, 3])
sd = cen @ n + d
band = np.abs(sd) <= hc * np.sqrt(3) / 2
K = int(band.sum())

# a slab of cells around the plane, projected into the plane's own frame, so the panels are 2D
e1 = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(n, e1)
P = np.stack([cen @ e1, cen @ e2], 1)
near = np.abs(sd) <= hc * 4
fig, ax = plt.subplots(1, 4, figsize=(19, 5.0))

# (a)
ax[0].scatter(P[near, 0], P[near, 1], s=1.1, c="#cfd6dd", linewidths=0)
ax[0].scatter(P[band, 0], P[band, 1], s=1.4, c="#c0392b", linewidths=0)
ax[0].set_title(r"(a)  $\Pi_k$ against the lattice"
                f"\n{len(solid):,} solid cells, {K:,} crossed", fontsize=11)

# (b) one crossed cell, its 12 edges and the roots
i = int(np.where(band)[0][len(np.where(band)[0]) // 2])
c0 = (solid[i] + 0.5) * hc + org
corn = np.array([[a, b, c] for a in (-.5, .5) for b in (-.5, .5) for c in (-.5, .5)]) * hc + c0
edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
roots = []
for a, b in edges:
    sa, sb = corn[a] @ n + d, corn[b] @ n + d
    if sa * sb < 0:
        roots.append(corn[a] + (corn[b] - corn[a]) * (sa / (sa - sb)))
roots = np.array(roots)
# isometric, not the plane's own frame: projected into the plane the cube is degenerate and its
# twelve edges collapse onto four
_iso = np.array([[np.cos(np.radians(30)), -np.cos(np.radians(30)), 0.0],
                 [np.sin(np.radians(30)),  np.sin(np.radians(30)), 1.0]])
q = lambda X: ((X - c0) / hc) @ _iso.T
for a, b in edges:
    E = q(np.stack([corn[a], corn[b]]))
    ax[1].plot(E[:, 0], E[:, 1], c="#9aa5b1", lw=1.1, zorder=1)
ax[1].scatter(*q(corn).T, s=12, c="#9aa5b1", zorder=1)
if len(roots) >= 3:
    R = q(roots)
    o = np.argsort(np.arctan2(R[:, 1] - R[:, 1].mean(), R[:, 0] - R[:, 0].mean()))
    ax[1].add_patch(MPoly(R[o], closed=True, facecolor="#e8b4ae", edgecolor="#c0392b",
                          lw=2.0, zorder=2))
    ax[1].scatter(R[:, 0], R[:, 1], s=44, c="#c0392b", zorder=3)
    for j, (x, y) in enumerate(R):
        ax[1].annotate(rf"$\mathbf{{v}}_{{{j+1}}}$", (x, y), fontsize=9,
                       xytext=(4, 4), textcoords="offset points")
ax[1].set_title(rf"(b)  one $C_i$: 12 edges $e_j$, {len(roots)} roots"
                "\nthe convex polygon they bound", fontsize=11)
ax[1].set_aspect("equal")

# (c) the sign field, components of the two sides
lo = solid.min(0); G = solid.max(0) - lo + 1
occ = np.zeros(tuple(G), np.int8)
occ[solid[:, 0]-lo[0], solid[:, 1]-lo[1], solid[:, 2]-lo[2]] = np.where(sd > 0, 1, 2)
k = int(np.median(solid[:, int(np.argmax(np.abs(n)))]) - lo[int(np.argmax(np.abs(n)))])
ax_i = int(np.argmin(np.abs(n)))
sl = np.take(occ, occ.shape[ax_i] // 2, axis=ax_i)
ax[2].imshow(sl.T, origin="lower", interpolation="nearest",
             cmap=matplotlib.colors.ListedColormap(["white", "#2c6fbb", "#e0a458"]))
ax[2].set_title(r"(c)  $\mathrm{sign}(\mathbf{n}_k^\top\mathbf{x}+d_k)$ labelled"
                "\ntwo components, one slice of the integer lattice", fontsize=11)

# (d) the exposed face, coloured by the interior field
pts = cen[band]
col = ON.sample_interior(st, torch.as_tensor(pts, dtype=torch.float32)).clamp(0, 1).numpy()
Q = np.stack([pts @ e1, pts @ e2], 1)
ax[3].scatter(Q[:, 0], Q[:, 1], s=2.4, c=col, linewidths=0)
ax[3].set_title(r"(d)  the exposed face"
                f"\n{K:,} polygons, coloured by the interior field", fontsize=11)
ax[3].set_aspect("equal")

for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
    for sp in a_.spines.values():
        sp.set_edgecolor("#cfd6dd")
fig.tight_layout()
fig.savefig(f"{W}/cutstates.jpg", dpi=118, bbox_inches="tight")
print(f"cutstates.jpg  {len(solid):,} cells, {K:,} crossed, {len(roots)} roots on the drawn cell")
