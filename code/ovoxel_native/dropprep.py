"""What the solver needs from an O-Voxel object, and what the renderer needs to draw it.

`dynamic_cut.py` had to recover a lattice before it could cut one: it sampled pairwise distances
to estimate the spacing, rounded positions onto the inferred grid, and dilated the occupancy by
two cells because internal filling leaves holes that shatter 6-connectivity into thousands of
fragments. None of that applies here. The cells ARE the particles, the spacing is `hc`, and the
occupancy is exact, so connectivity is adjacency on the integer lattice and nothing is estimated.

The cut is defined in the material frame, which is what lets a plane arrive while the object is
already moving: the plane is aimed at the object, not at the room. So the exposed faces are
extracted once, here, in the rest pose, and each is carried by the piece that owns it.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON, anchor

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
dev = "cuda"
ON.FDG = ON._load_ovoxel()

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
solid = st["solid"].long()
cen = (solid.float() + 0.5) * hc + org
mid = cen.mean(0)

n1 = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev); n1 = n1 / n1.norm()
k = len(C["v_planes"]) // 2
n2 = torch.as_tensor(C["v_planes"][k, :3], dtype=torch.float32, device=dev); n2 = n2 / n2.norm()
d1, d2 = float(-(mid @ n1)), float(-(mid @ n2))

# The peel is the boundary of the occupied set, taken at the lattice: a cell whose 3x3x3
# neighbourhood is not full, grown by one cell. That is the same statement the two-level lattice
# makes about where the fine level lives, read off the occupancy rather than off a file.
import torch.nn.functional as Fn
dims = [int(solid[:, i].max()) + 3 for i in range(3)]
occ = torch.zeros(dims, device=dev)
occ[solid[:, 0] + 1, solid[:, 1] + 1, solid[:, 2] + 1] = 1.0
cnt = Fn.avg_pool3d(occ[None, None], 3, 1, 1)[0, 0] * 27.0
shell = (cnt < 26.5).float()
grown = (Fn.max_pool3d(shell[None, None], 5, 1, 2)[0, 0] > 0.5)
peel = grown[solid[:, 0] + 1, solid[:, 1] + 1, solid[:, 2] + 1].cpu().numpy()
print(f"{OBJ}: {len(cen):,} cells, peel {int(peel.sum()):,} ({peel.mean()*100:.1f}%)")

mv, mf, mc = ON.surface_mesh(st)
print(f"  surface mesh {len(mv):,} vertices, {len(mf):,} faces")

out = dict(hc=hc, org=org.cpu().numpy(), solid=solid.cpu().numpy().astype(np.int32),
           n1=n1.cpu().numpy(), n2=n2.cpu().numpy(), d1=d1, d2=d2, mid=mid.cpu().numpy(),
           peel=peel, mv=mv.cpu().numpy(), mf=mf.cpu().numpy().astype(np.int32),
           mc=mc.clamp(0, 1).cpu().numpy(), v_mvp=C["v_mvp"], h_mvp=C["h_mvp"])

# The exposed face of each cut. A polygon lying on plane A has no side with respect to A -- it is
# the boundary -- so both components adjacent to it get their own copy, and the only test that
# decides is the OTHER plane's sign.
for idx, (nn, dd) in enumerate(((n1, d1), (n2, d2))):
    P, T, _ = ON.cut_polygons(st, nn, dd, device=dev)
    col = ON.sample_interior(st, P).clamp(0, 1)
    on = (P @ (n2 if idx == 0 else n1) + (d2 if idx == 0 else d1) > 0)
    out[f"cut{idx}_P"] = P.cpu().numpy()
    out[f"cut{idx}_T"] = T.cpu().numpy().astype(np.int32)
    out[f"cut{idx}_C"] = col.cpu().numpy()
    out[f"cut{idx}_on"] = on.cpu().numpy()
    print(f"  cut {idx}: {len(P):,} polygon vertices, {len(T):,} triangles")

np.savez(f"{W}/drop_prep_{OBJ}.npz", **out)
print(f"drop_prep_{OBJ}.npz written")
