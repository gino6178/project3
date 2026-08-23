"""Is the field extruded along the axis?

A longitudinal render shows vertical stripes on every plane of that family, supervised or not, and
on the watermelon those stripes are not anatomy -- a watermelon has no radial membranes to see edge
on. The suspicion is that the field varies with radius and hardly at all along the axis: each
transverse photograph paints a disc, successive depths paint similar discs, and what is left is a
structure extruded along z, which a longitudinal cut sees as vertical lines.

Measured directly on the cells, with no render in the way: the colour difference between cells
adjacent ALONG the axis, against between cells adjacent across it. An isotropic interior gives a
ratio near 1. A field extruded along the axis gives much less than 1.

The photographs cannot be compared to this directly -- they are 2D -- so the control is the
transverse family's own structure: if the ratio is low for the watermelon and near 1 for an object
whose interior really is radial, the number is measuring anatomy rather than an artefact.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor

W = os.path.dirname(os.path.abspath(__file__))
OBJS = [o for o in os.environ.get("EX_OBJS", "watermelon_sp,orange_sp").split(",") if o]
RUN = os.environ.get("EX_RUN", "s_rs")
dev = "cuda"
ON.FDG = ON._load_ovoxel()

for OBJ in OBJS:
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    C = np.load(f"{W}/cams_{OBJ}_bal.npz")
    axis = np.asarray(C["h_planes"][0, :3], float)
    axis = axis / np.linalg.norm(axis)
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
    out = {}
    for ax in range(3):
        step = torch.zeros(3, dtype=torch.long, device=dev)
        step[ax] = 1
        q = solid + step - lo
        ok = ((q >= 0) & (q < G)).all(1)
        qq = q.clamp(min=torch.zeros(3, dtype=torch.long, device=dev), max=G - 1)
        j = idx3[qq[:, 0], qq[:, 1], qq[:, 2]].long()
        m = ok & (j >= 0)
        out[ax] = float((col[torch.arange(len(solid), device=dev)[m]] - col[j[m]]).abs().mean())
    # the lattice axis closest to the polar axis, and the two across it
    a_ax = int(np.argmax(np.abs(axis)))
    across = [k for k in range(3) if k != a_ax]
    print(f"{OBJ} ({RUN}): polar axis is closest to lattice axis {a_ax} "
          f"(direction {np.round(axis, 3)})")
    print(f"  difference along the axis      {out[a_ax]:.4f}")
    print(f"  difference across it           {0.5 * (out[across[0]] + out[across[1]]):.4f}"
          f"   (axes {across[0]} {out[across[0]]:.4f}, {across[1]} {out[across[1]]:.4f})")
    print(f"  ratio along / across           {out[a_ax] / (0.5 * (out[across[0]] + out[across[1]])):.3f}"
          f"   (1.0 = no preferred direction, below 1 = extruded along the axis)")
