"""The field's variation in cylindrical directions, not lattice ones.

The earlier measurement split the neighbour pairs into "along the polar axis" and "across it", and
across mixes two very different directions: radial, in which a fruit really does change, and
azimuthal, in which a solid of revolution does not change at all. The prior weights both of them
four times harder than the axial direction, because it weights LATTICE axes and the two lattice
axes across the pole carry both.

If the azimuthal variation has been flattened, the field is close to a function of radius alone --
and a function of radius alone, cut through the axis at any angle, is a set of vertical stripes.
This measures the three directions separately, from the cells, with no render involved.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor

W = os.path.dirname(os.path.abspath(__file__))
OBJS = [o for o in os.environ.get("CV_OBJS", "watermelon_sp,orange_sp").split(",") if o]
RUN = os.environ.get("CV_RUN", "s_rs")
dev = "cuda"
ON.FDG = ON._load_ovoxel()

for OBJ in OBJS:
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    C = np.load(f"{W}/cams_{OBJ}_bal.npz")
    axis = np.asarray(C["h_planes"][0, :3], float); axis /= np.linalg.norm(axis)
    ax_t = torch.as_tensor(axis, dtype=torch.float32, device=dev)
    p = torch.load(f"{W}/{RUN}_{OBJ}/params.pt", map_location=dev)
    if "dec_i" in p:
        w = p["dec_i"]["stage1.0.weight"].shape[0]
        n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
        anchor.W_HID, anchor.N_HID = w, n
        di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
        di.load_state_dict(p["dec_i"])
        with torch.no_grad():
            col = di()
    else:
        col = p["interior"].to(dev)

    solid, idx3 = st["solid"], st["idx3"]
    lo = torch.as_tensor(st["idx_lo"], dtype=torch.long, device=dev)
    G = torch.tensor(idx3.shape, device=dev)
    cen = (solid.float() + 0.5) * st["hc"] + torch.as_tensor(st["org"], dtype=torch.float32,
                                                             device=dev)
    ctr = cen.mean(0)
    rel = cen - ctr[None]
    z = (rel @ ax_t)[:, None] * ax_t[None]
    rad_v = rel - z
    r = rad_v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    rhat = rad_v / r
    that = torch.cross(ax_t[None].expand_as(rhat), rhat, dim=1)

    tot = {"radial": [0., 0], "azimuthal": [0., 0], "axial": [0., 0]}
    for ax in range(3):
        step = torch.zeros(3, dtype=torch.long, device=dev); step[ax] = 1
        q = solid + step - lo
        ok = ((q >= 0) & (q < G)).all(1)
        qq = q.clamp(min=torch.zeros(3, dtype=torch.long, device=dev), max=G - 1)
        j = idx3[qq[:, 0], qq[:, 1], qq[:, 2]].long()
        m = ok & (j >= 0)
        i0 = torch.arange(len(solid), device=dev)[m]
        j0 = j[m]
        d = (col[i0] - col[j0]).abs().mean(1)
        # which cylindrical direction this pair points along, at its own midpoint
        e = torch.zeros(3, device=dev); e[ax] = 1.0
        c_r = (e[None] * rhat[i0]).sum(1).abs()
        c_t = (e[None] * that[i0]).sum(1).abs()
        c_z = (e[None] * ax_t[None]).sum(1).abs().expand_as(c_r)
        which = torch.stack([c_r, c_t, c_z], 1).argmax(1)
        for k, name in enumerate(("radial", "azimuthal", "axial")):
            sel = which == k
            if int(sel.sum()):
                tot[name][0] += float(d[sel].sum()); tot[name][1] += int(sel.sum())
    print(f"\n{OBJ} ({RUN}): mean |difference| between adjacent cells, by direction")
    base = tot["radial"][0] / max(tot["radial"][1], 1)
    for name in ("radial", "azimuthal", "axial"):
        s, c = tot[name]
        v = s / max(c, 1)
        print(f"  {name:<11}{v:.4f}   ({c:,} pairs, {v / base:.2f} of the radial)")
