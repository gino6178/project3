"""Does the object overflow the camera frame it is drawn in?

A flat edge inside a panel is not evidence that nothing was clipped: the figure rescales each panel
to a common size, which moves a frame-edge cut inland.  What settles it is whether the render
touches the border before any rescaling.
"""
import sys
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
import ovnative as ON, nvdiffrast.torch as dr
W, dev = "/workspace/ovoxel_native", "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
RES = 384
print(f"{'object':16s} {'family':12s} {'cut on the border':>18s} {'outside on the border':>22s}")
for OBJ in ("watermelon_sp", "orange_sp", "apple1_sp", "bread_sp", "cake2_sp",
            "pomegranate2_sp", "doughnut"):
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    conf = open(f"/workspace/rebuild/project3/code/objects/{OBJ}.conf").read()
    C = np.load(f"{W}/cams_{OBJ}{'_up' if chr(10)+'UP_AXIS=' in conf else '_bal'}.npz")
    p = torch.load(f"{W}/s_rs_{OBJ}/params.pt", map_location=dev)
    st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
    for fam in ("h", "v"):
        n_pl = (int(C["h_hi"][0]) - int(C["h_lo"][0])) if fam == "h" else len(C["v_planes"])
        hit_c = hit_e = 0
        for i in range(n_pl):
            if fam == "h":
                mv = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
                nn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
                dd = float(C["h_planes"][int(C["h_lo"][0]) + i, 3])
            else:
                mv = torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev)
                nn = torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32, device=dev)
                dd = float(C["v_planes"][i, 3])
            with torch.no_grad():
                _, ac, _, _ = ON.render_section(st, glctx, mv, nn, dd, RES, exterior=False)
                ae = ON.render_exterior(st, glctx, mv, RES)[1]
            for a, which in ((ac, "c"), (ae, "e")):
                m = (a[0] > 0.5).cpu().numpy()
                if m.sum() and (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()):
                    if which == "c":
                        hit_c += 1
                    else:
                        hit_e += 1
        print(f"{OBJ:16s} {('transverse' if fam=='h' else 'longitudinal'):12s} "
              f"{hit_c:8d} of {n_pl:<7d} {hit_e:12d} of {n_pl:<7d}")
