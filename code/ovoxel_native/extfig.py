"""The exterior of each arm, from the six views that supervised it.

Every figure so far cuts the object. The skin is what route 1 takes from the released model and
what route 2 projects from six photographs, and neither has been drawn, so this draws it: no
plane, the whole dual surface, from the cameras the training used.
"""
import glob, json, os, sys
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/workspace/ovoxel_native")
import ovnative as ON
import nvdiffrast.torch as dr

W = "/workspace/ovoxel_native"; OUT = W + "/out"
dev = "cuda"
ARMS = [("decoder, exterior pinned", "r1_pin"),
        ("decoder, flat interior, pinned", "r1flat_pin"),
        ("decoder, full parity", "r1_pin_full"),
        ("route 2, full parity", "r2_pin_full")]
cams = np.load(W + "/cams_ext.npz") if os.path.exists(W + "/cams_ext.npz") else None
glctx = dr.RasterizeCudaContext(device=dev)

ON.FDG = ON._load_ovoxel()
base = {}
rows = []
for label, arm in ARMS:
    p = f"{W}/{arm}/params.pt"
    if not os.path.exists(p): print("skip", arm); continue
    route = "2" if arm.startswith("r2") else "1"
    lat = ("/workspace/rebuild/worktree/build_orange_r2/skin" if route == "2"
           else "/workspace/rebuild/worktree/build_orange/lattice")
    if route not in base:
        base[route] = ON.build(lat, verbose=False)
    st = dict(base[route])
    sd = torch.load(p, map_location=dev)
    for k, v in sd.items():
        if k in st and torch.is_tensor(st[k]): st[k] = v.to(dev)
    if "dec_s" in sd or "feat_s" in sd:
        import anchor
        st = anchor.materialise(st, sd, dev) if hasattr(anchor, "materialise") else st
    imgs = []
    for az in (0, 60, 120, 180, 240, 300):
        mvp = ON.mvp_for(az, 15.0, dev) if hasattr(ON, "mvp_for") else None
        if mvp is None: break
        img, al, _, _ = ON.render_exterior(st, glctx, mvp, 384)
        imgs.append(img.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())
    if imgs: rows.append((label, imgs))
print("rows:", [r[0] for r in rows])
if rows:
    n = len(rows[0][1])
    fig, ax = plt.subplots(len(rows), n, figsize=(2.0 * n, 2.15 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (label, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            ax[r, c].imshow(im); ax[r, c].set_axis_off()
        ax[r, 0].set_title(label, fontsize=8, loc="left")
    fig.suptitle("the exterior, no plane: the dual surface from six azimuths", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + "/exterior_arms.png", dpi=140)
    print("->", OUT + "/exterior_arms.png")
