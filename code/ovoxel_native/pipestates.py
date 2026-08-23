"""Two figures the specification requires, both drawn from the object's own arrays.

  pipestates.jpg   the pipeline as four states of the data, replacing a box-and-arrow diagram:
                   (a) the released model's points, (b) the coarse occupancy after close-and-fill,
                   (c) the skin at the fine level, (d) the dual vertices offset inside their voxels
  coverage.jpg     how many supervised planes reach each cell, as a scalar field with a
                   perceptually uniform colorbar and its range
"""
import os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
st = torch.load(f"{W}/state_{OBJ}.pt", map_location="cpu", weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_v2.npz")
hc, hf = float(st["hc"]), float(st["hf"])
org = np.asarray(st["org"], float)
solid = st["solid"].numpy().astype(np.int64)
cen = (solid + 0.5) * hc + org
ax_i = int(np.argmax(np.abs(np.asarray(C["h_planes"][0, :3], float))))
u, v = [i for i in range(3) if i != ax_i]
mid = np.median(cen[:, ax_i])
sl = np.abs(cen[:, ax_i] - mid) < hc

# ---------------- pipeline states ----------------
fig, ax = plt.subplots(1, 4, figsize=(19, 5.0))
from plyfile import PlyData
try:
    el = PlyData.read(f"/workspace/rebuild/worktree/build_{OBJ}/lattice/gs_fill.ply").elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    m = np.abs(xyz[:, ax_i] - mid) < hc
    ax[0].scatter(xyz[m][:, u], xyz[m][:, v], s=0.6, c="#7f8c8d", linewidths=0)
    ax[0].set_title(f"(a)  the released model, one slab\n{len(xyz):,} points", fontsize=11)
except Exception as e:
    ax[0].set_title(f"(a)  unavailable: {type(e).__name__}", fontsize=11)

ax[1].scatter(cen[sl][:, u], cen[sl][:, v], s=1.6, c="#2c6fbb", linewidths=0)
ax[1].set_title(f"(b)  coarse occupancy, $h_c$ = {hc:.5f}\n{len(solid):,} cells after close-and-fill",
                fontsize=11)

lvl = st.get("lvl")
dv = st["dual_v"].numpy() + org
md = np.abs(dv[:, ax_i] - mid) < hf * 2
ax[2].scatter(dv[md][:, u], dv[md][:, v], s=1.0, c="#c0392b", linewidths=0)
ax[2].set_title(f"(c)  the skin at $h_f$ = {hf:.5f}\n{len(dv):,} dual vertices", fontsize=11)

frac = (st["dual_v"].numpy() / hf) - np.floor(st["dual_v"].numpy() / hf)
h2 = ax[3].hist2d(frac[:, u], frac[:, v], bins=48, cmap="viridis")
ax[3].set_title(r"(d)  $\mathbf{u}_v$: where the vertex sits in its voxel"
                "\nuniform would be flat; it is not", fontsize=11)
cb = fig.colorbar(h2[3], ax=ax[3], fraction=0.046, pad=0.03)
cb.set_label(f"vertices per bin  [{int(h2[0].min())}, {int(h2[0].max())}]", fontsize=9)
for a_ in ax[:3]:
    a_.set_aspect("equal")
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
    for sp in a_.spines.values():
        sp.set_edgecolor("#cfd6dd")
fig.tight_layout()
fig.savefig(f"{W}/pipestates.jpg", dpi=118, bbox_inches="tight")
print("pipestates.jpg")

# ---------------- coverage as a scalar field ----------------
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hn = np.asarray(C["h_planes"][0, :3], float); hn /= np.linalg.norm(hn)
cnt = np.zeros(len(solid), np.int32)
rng = np.random.default_rng(0)
STEPS = int(os.environ.get("CV_STEPS", "40"))
hd = C["h_planes"][:, 3]
step = float(np.abs(np.diff(hd[H_LO:H_HI])).mean()) if H_HI - H_LO > 1 else hc
r = hc * np.sqrt(3) / 2
for _ in range(STEPS):
    for i in range(H_HI - H_LO):
        d = float(hd[H_LO + i]) + step * (rng.random() - 0.5)
        cnt += (np.abs(cen @ hn + d) <= r)
    for j in range(len(C["v_planes"])):
        n2 = np.asarray(C["v_planes"][j, :3], float); n2 /= np.linalg.norm(n2)
        cnt += (np.abs(cen @ n2 + float(C["v_planes"][j, 3])) <= r)

fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5.0))
for a_, (mask, ttl) in zip(ax2, ((sl, "a slab across the polar axis"),
                                 (np.abs(cen[:, u] - np.median(cen[:, u])) < hc,
                                  "a slab along it"))):
    sc = a_.scatter(cen[mask][:, u if mask is sl else ax_i], cen[mask][:, v],
                    s=2.2, c=cnt[mask], cmap="viridis", linewidths=0,
                    vmin=0, vmax=int(np.percentile(cnt, 99)))
    a_.set_title(ttl, fontsize=11); a_.set_aspect("equal")
    a_.set_xticks([]); a_.set_yticks([])
cb2 = fig2.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
cb2.set_label(f"planes reaching the cell over {STEPS} iterations "
              f"[0, {int(np.percentile(cnt, 99))}], max {int(cnt.max())}", fontsize=9)
fig2.savefig(f"{W}/coverage.jpg", dpi=118, bbox_inches="tight")
print(f"coverage.jpg  reached {(cnt > 0).mean()*100:.1f}% of cells, "
      f"never reached {(cnt == 0).sum():,}, median {int(np.median(cnt))}, max {int(cnt.max())}")
