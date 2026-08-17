"""The collapsed surface, seen. One column per tolerance, so the trade is visible and not only
tabulated.

An adaptive dual grid has no uniform mesh to hand to `flexible_dual_grid_to_mesh`, so each node
is drawn as the patch it stands for: a square of its own cell's size, lying in the plane the
quadric fitted, sampled densely enough that a coarse node is not a sparser thing than a fine one.
The normal comes from the quadric itself -- A = sum n n^T, so its dominant eigenvector is the
direction the planes in that node agree on -- which means nothing here is estimated twice.

This is a splat and says so. It shows whether the geometry and the colour survive a collapse; it
is not the renderer, and the gaps a naive point splat leaves are exactly what the per-node
footprint removes.

    python method/common/cube/qef_views.py OVOX.npz PLY CFG DEMO OUT.png [tau ...]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import qef                                  # noqa: E402

DEV = "cuda:0"


def node_samples(r, A6, base_h):
    """Each node as a square patch of its own size, in the plane its quadric agrees on.

    The sample count scales with the node's area, so a level-L node is drawn with the same
    density as a level-0 one and a coarse region does not merely look sparser.
    """
    n = len(r["pos"])
    # the collapse's own oriented normal, which A cannot supply: n n^T loses the sign, and a
    # cull based on an unsigned eigenvector removes about half the front of the object
    nrm = r["normal"]

    a = np.zeros_like(nrm)
    pick = np.argmin(np.abs(nrm), axis=1)
    a[np.arange(n), pick] = 1.0
    u = np.cross(nrm, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-30)
    v = np.cross(nrm, u)

    out_p, out_i = [], []
    for L in sorted({int(x) for x in r["level"]}):
        m = np.nonzero(r["level"] == L)[0]
        k = int(min(max(2 * (2 ** L), 4), 24))
        t = ((np.arange(k) + 0.5) / k - 0.5)
        gu, gv = np.meshgrid(t, t, indexing="ij")
        gu, gv = gu.ravel(), gv.ravel()
        # sqrt(3), not 1: the patch lies in the fitted plane and the cell is a cube, so a
        # square of the cell's edge does not cover the cell's projection when the plane is
        # oblique to it. The shortfall showed as dark speckle -- the far side of the object seen
        # through gaps that are the splat's and not the surface's.
        hh = r["h"][m][:, None, None] * np.sqrt(3.0)
        p = (r["pos"][m][:, None, :]
             + u[m][:, None, :] * (gu[None, :, None] * hh)
             + v[m][:, None, :] * (gv[None, :, None] * hh))
        out_p.append(p.reshape(-1, 3))
        out_i.append(np.repeat(m, len(gu)))
    return np.concatenate(out_p), np.concatenate(out_i), nrm


def main(npz, ply, cfg, demo, out_png, taus, size=420):
    from scene.gaussian_model import GaussianModel
    from scipy.spatial import cKDTree
    from utils.camera_view_utils import get_camera_view
    from utils.decode_param import decode_param_json
    from utils.render_utils import load_params_from_gs
    from utils.transformation_utils import (apply_inverse_rotations,
                                            generate_rotation_matrices,
                                            get_center_view_worldspace_and_observant_coordinate,
                                            shift2center111, transform2origin,
                                            undoshift2center111, undotransform2origin)

    z = np.load(npz)
    voxel, pos0, rgb0 = z["voxel"].astype(np.int64), z["pos"].astype(np.float64), z["rgb"]
    h, origin = float(z["voxel_size"]), z["origin"].astype(np.float64)
    V, F = z["mesh_v"].astype(np.float64), z["mesh_f"].astype(np.int64)
    A6, b, c, w, gn, _ = qef.quadrics_from_mesh(V, F, pos0, voxel, h, origin)
    tree0 = cKDTree(pos0)

    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(ply)
    par = load_params_from_gs(g, type("P", (), dict(sh_degree=0, compute_cov3D_python=True,
                                                    convert_SHs_python=False, debug=False)))
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(par["pos"])
    tpos = shift2center111(tpos)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)

    xyz_ply = g.get_xyz.detach().cpu().numpy().astype(np.float64)
    wp = world.detach().cpu().numpy().astype(np.float64)
    idx = np.linspace(0, len(wp) - 1, 20000).astype(int)
    A, B = xyz_ply[idx], wp[idx]
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    dsg = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dsg]) @ U.T
    sc = float(S[:2].sum() + dsg * S[2]) / float(((A - ca) ** 2).sum())
    tt = cb - sc * (R @ ca)

    cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                             observant_coordinates=oc, show_hint=False, init_azimuthm=0,
                             init_elevation=15, init_radius=cam_p["init_radius"],
                             move_camera=False, current_frame=0, delta_a=None, delta_e=None,
                             delta_r=None)
    fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
    w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)

    tiles, labels = [], []
    for tau in taus:
        # pos0, like qef.main and qef_mesh do. Without it an untouched node is
        # re-solved from a quadric rebuilt out of the boundary mesh rather than kept
        # at the vertex the library solved from Hermite data, so the tau = 0 tile is
        # not the uniform grid it is captioned as.
        r = qef.collapse(voxel, A6, b, c, w, h, origin, tau, g=gn, pos0=pos0)
        pts, owner, nrm = node_samples(r, None, h)
        col = rgb0[tree0.query(r["pos"], k=1)[1]][owner]
        p = sc * (pts @ R.T) + tt
        hom = np.concatenate([p, np.ones((len(p), 1))], 1)
        clip = hom @ fp
        ok = clip[:, 3] > 1e-6
        nd = clip[:, :2] / clip[:, 3:4]
        px = ((nd[:, 0] + 1) * 0.5 * size).astype(np.int64)
        py = ((1 - (nd[:, 1] + 1) * 0.5) * size).astype(np.int64)
        dep = (hom @ w2c)[:, 2]
        ok &= (px >= 0) & (px < size) & (py >= 0) & (py < size)

        # No back-face culling. It was tried twice and made things worse both times: an
        # unsigned eigenvector culls about half the front, and a summed normal cancels wherever
        # a merged node's children face different ways, so normalising it gives an arbitrary
        # direction and the cull opens holes. The z-buffer already prefers the near surface,
        # which is the whole job; the residual speckle at large tau is this splat's and not the
        # collapse's, and the error table is what measures the collapse.

        img = np.ones((size * size, 3), np.float32)
        o = np.argsort(-dep[ok])
        img[(py[ok] * size + px[ok])[o]] = col[ok][o]
        tiles.append((img.reshape(size, size, 3) * 255).astype(np.uint8))
        labels.append(f"tau {tau:g}   {len(r['pos']):,} nodes "
                      f"({100 * len(r['pos']) / len(voxel):.0f}% of uniform)")
        print(f"  {labels[-1]}, {len(pts):,} samples")

    sheet = Image.new("RGB", (size * len(tiles), size + 22), "white")
    d = ImageDraw.Draw(sheet)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        sheet.paste(Image.fromarray(t), (i * size, 22))
        d.text((i * size + 4, 6), lab, fill="black")
    sheet.save(out_png)
    print(f"  -> {out_png}")


def _requad(r, voxel, A6, tree0):
    """The quadric of each surviving node, taken from the fine ones it stands for."""
    _, j = tree0.query(r["pos"], k=1)
    return A6[j]


if __name__ == "__main__":
    taus = [float(x) for x in sys.argv[6:]] or [0.0, 0.5, 1.0]
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], taus)
