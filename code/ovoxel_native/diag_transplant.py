"""Decompose the gap: put the EXISTING pipeline's trained appearance into the O-Voxel-native
representation and render it with the O-Voxel-native renderer.

Whatever is left between that and the pipeline's own number is the representation and the
renderer; whatever is between it and our trained number is our training.
"""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import ovnative as ON
import nvdiffrast.torch as dr
from plyfile import PlyData
from scipy.spatial import cKDTree

LAT = "/workspace/rebuild/worktree/build_orange/lattice"
TRAINED = "/workspace/rebuild/worktree/orange/orange_demo_epoch_199.ply"
OUT = "/workspace/ovoxel_native/diag_transplant"
dev = "cuda"; C0 = 0.28209479177387814

st = torch.load("/workspace/ovoxel_native/state_orange.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
el = PlyData.read(TRAINED).elements[0]
xyz_t = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
lat_xyz = st["xyz"]; lvl = st["lvl"]
print(f"trained model {len(xyz_t):,} rows, lattice {len(lat_xyz):,} rows, "
      f"positions identical: {np.allclose(xyz_t, lat_xyz)}")
dc = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
rest = [p.name for p in el.properties if p.name.startswith("f_rest_")]
print(f"  {len(rest)} higher-band coefficients")
rgb0 = np.clip(dc * C0 + 0.5, 0, 1)

hc, org = st["hc"], st["org"]
solid = st["solid"].cpu().numpy()
idx_lo = st["idx_lo"]; idx3 = st["idx3"].cpu().numpy()
coarse_raw = np.floor((lat_xyz[lvl == 0] - org) / hc).astype(np.int64)
have = idx3[coarse_raw[:, 0] - idx_lo[0], coarse_raw[:, 1] - idx_lo[1], coarse_raw[:, 2] - idx_lo[2]]
ok = have >= 0

def make(rgb, tag):
    interior = np.full((len(solid), 3), 0.5, np.float32)
    interior[have[ok]] = rgb[lvl == 0][ok]
    seeded = np.zeros(len(solid), bool); seeded[have[ok]] = True
    c_all = (solid + 0.5) * hc + org
    t2 = cKDTree(c_all[seeded]); interior[~seeded] = interior[seeded][t2.query(c_all[~seeded], k=1)[1]]
    skin = lvl == 1
    tr = cKDTree(lat_xyz[skin])
    surf = rgb[skin][tr.query(st["dual_pos"], k=1)[1]]
    st["interior"] = torch.as_tensor(interior, device=dev)
    st["surf_rgb"] = torch.as_tensor(surf, dtype=torch.float32, device=dev)
    return tag

Cm = np.load("/workspace/ovoxel_native/cams_orange.npz")
def plane_in_pos_frame(i):
    a = Cm[f"c{i}_affine"]; A, t = a[:3], a[3]
    nt = Cm[f"c{i}_plane_t"][:3]; dt = float(Cm[f"c{i}_plane_t"][3])
    npos = np.linalg.solve(A, nt); dpos = dt - float(t @ npos)
    s = np.linalg.norm(npos); return npos / s, dpos / s
EV = [plane_in_pos_frame(i) for i in range(6)]
MVP = torch.as_tensor(Cm["c0_fp"], dtype=torch.float32, device=dev)
NRM = torch.as_tensor(EV[0][0], dtype=torch.float32, device=dev)
glctx = dr.RasterizeCudaContext(device=dev)

import realism, glob
refs = realism._paths("/workspace/rebuild/worktree/secref_orraw_hsep")

def render_all(folder):
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        for i, (n, d) in enumerate(EV):
            img, _, _, _ = ON.render_section(st, glctx, MVP, NRM, float(d), 512)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            cv2.imwrite(f"{folder}/rh{i}_init_0.png", (a[:, :, ::-1] * 255).astype(np.uint8))
    return realism._dreamsim(refs, sorted(glob.glob(folder + "/rh*_init_0.png")), dev)

make(rgb0, "dc")
print(f"  the pipeline's trained appearance, f_dc only, in our representation and renderer: "
      f"{render_all(OUT + '/dc'):.4f}")

# and with the higher bands evaluated at this camera's view direction, which is what
# random_cuts feeds the Gaussian rasteriser (FULL_SH=1)
if rest:
    n_rest = len(rest) // 3
    deg = int(round(((n_rest + 1) ** 0.5) - 1))
    fr = np.stack([np.stack([el[f"f_rest_{c*n_rest+j}"] for j in range(n_rest)], 1)
                   for c in range(3)], 2).astype(np.float32)          # (N, n_rest, 3)
    cam_c = Cm["c0_center"]
    dirs = lat_xyz - cam_c[None]
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    sh = [0.4886025119029199 * -y, 0.4886025119029199 * z, 0.4886025119029199 * -x]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
        sh += [1.0925484305920792*xy, -1.0925484305920792*yz, 0.31539156525252005*(2*zz-xx-yy),
               -1.0925484305920792*xz, 0.5462742152960396*(xx-yy)]
    if deg >= 3:
        sh += [-0.5900435899266435*y*(3*xx-yy), 2.890611442640554*xy*z,
               -0.4570457994644658*y*(4*zz-xx-yy), 0.3731763325901154*z*(2*zz-3*xx-3*yy),
               -0.4570457994644658*x*(4*zz-xx-yy), 1.445305721320277*z*(xx-yy),
               -0.5900435899266435*x*(xx-3*yy)]
    S = np.stack(sh[:n_rest], 1)                                       # (N, n_rest)
    rgb_sh = np.clip(dc * C0 + 0.5 + np.einsum("nk,nkc->nc", S, fr), 0, 1).astype(np.float32)
    make(rgb_sh, "sh")
    print(f"  same, with the {len(rest)} higher bands evaluated at this camera: "
          f"{render_all(OUT + '/sh'):.4f}")
print("DIAG_OK")
