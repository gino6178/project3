"""The O-Voxel exterior, drawn from the six canonical views, beside the model it replaced.

A surface made of dual-grid voxels has no rasteriser in this repository, and it does not need
one to be checked: each active voxel carries a position and a colour, so projecting them through
the same camera the Gaussian renderer uses and keeping the nearest per pixel is a z-buffer over
points. It is not the final renderer -- that is what turning the dual grid back into a mesh is
for -- but it answers the question this step raises, which is whether the surface is in the right
place and the right colour.

    python method/common/cube/ovox_views.py OVOX.npz CFG DEMO OUT.png [size]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch
from PIL import Image

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from plyfile import PlyData                                         # noqa: E402
from scene.gaussian_model import GaussianModel                      # noqa: E402
from utils.camera_view_utils import get_camera_view                 # noqa: E402
from utils.decode_param import decode_param_json                    # noqa: E402
from utils.render_utils import load_params_from_gs                  # noqa: E402
from utils.transformation_utils import *                            # noqa: E402

DEV = "cuda:0"
DIRS = [("up", 0, 90), ("front", 0, 0), ("right", 90, 0),
        ("down", 0, -90), ("back", 180, 0), ("left", 270, 0)]


class P:
    sh_degree = 0
    compute_cov3D_python = True
    convert_SHs_python = False
    debug = False


def main(npz, ply, cfg, demo, out_png, size=512):
    z = np.load(npz)
    pts0, rgb = z["pos"].astype(np.float64), z["rgb"]
    # If the dual grid was turned back into a mesh, draw that instead of the points. A point per
    # voxel leaves the background showing wherever the projected voxel spacing exceeds a pixel;
    # the quads of the dual grid tile, so a rasterised mesh cannot. Each triangle is sampled on
    # a small barycentric grid rather than scan-converted, which is enough because at this
    # resolution a triangle covers a few pixels at most.
    mesh = ("mesh_v" in z.files)
    if mesh:
        MV, MF = z["mesh_v"].astype(np.float64), z["mesh_f"].astype(np.int64)
        k = 6
        u = (np.arange(k) + 0.5) / k
        bu, bv = np.meshgrid(u, u, indexing="ij")
        keep = (bu + bv) <= 1.0
        bu, bv = bu[keep], bv[keep]
        tv = MV[MF]                                            # (T, 3, 3)
        samp = (tv[:, None, 0] * (1 - bu - bv)[None, :, None]
                + tv[:, None, 1] * bu[None, :, None]
                + tv[:, None, 2] * bv[None, :, None]).reshape(-1, 3)
        # each sample takes the colour of the voxel its vertex came from
        cs = np.repeat(rgb[MF[:, 0]], len(bu), axis=0)
        print(f"  mesh: {len(MF):,} triangles -> {len(samp):,} surface samples")
        pts0, rgb = samp, cs

    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(ply)
    par = load_params_from_gs(g, P())
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

    # the same similarity that M6 fits, so the saved points land where the model does
    A = np.stack([g.get_xyz.detach().cpu().numpy()[i] for i in
                  np.linspace(0, len(world) - 1, 20000).astype(int)]).astype(np.float64)
    B = world.detach().cpu().numpy().astype(np.float64)[
        np.linspace(0, len(world) - 1, 20000).astype(int)]
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    sc = float(S[:2].sum() + d * S[2]) / float(((A - ca) ** 2).sum())
    t = cb - sc * (R @ ca)
    pts = sc * (pts0 @ R.T) + t

    tiles = []
    for name, az, el in DIRS:
        cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                 observant_coordinates=oc, show_hint=False, init_azimuthm=az,
                                 init_elevation=el, init_radius=cam_p["init_radius"],
                                 move_camera=False, current_frame=0, delta_a=None,
                                 delta_e=None, delta_r=None)
        fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
        hom = np.concatenate([pts, np.ones((len(pts), 1))], 1)
        clip = hom @ fp
        ok = clip[:, 3] > 1e-6
        ndc = clip[:, :2] / clip[:, 3:4]
        px = ((ndc[:, 0] + 1) * 0.5 * size).astype(np.int64)
        py = ((1 - (ndc[:, 1] + 1) * 0.5) * size).astype(np.int64)
        dep = (hom @ w2c)[:, 2]
        ok &= (px >= 0) & (px < size) & (py >= 0) & (py < size)

        zb = np.full(size * size, np.inf)
        cb_ = np.ones((size * size, 3), np.float32)
        idx = (py[ok] * size + px[ok])
        order = np.argsort(-dep[ok])          # far first, so the nearest wins by overwriting
        zb[idx[order]] = dep[ok][order]
        cb_[idx[order]] = rgb[ok][order]
        tiles.append((cb_.reshape(size, size, 3) * 255).astype(np.uint8))
        print(f"  {name:<6} {int(ok.sum()):,} points, "
              f"{100 * np.isfinite(zb).mean():.1f}% of the frame covered")

    sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    Image.fromarray(sheet).save(out_png)
    print(f"  -> {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 512)
