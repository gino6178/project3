"""Section 11.2's resolution stability: how a cross-section holds up as the output grows.

This is the measurement that decides whether the spec's equation (2) is a real trade or a
rhetorical one. A Gaussian's support is a fixed size in world space, so the number of pixels it
covers grows with the output resolution but the *gaps between* its neighbours grow just as fast:
a primitive set tuned to look solid at 512 has to be either fatter or denser to look solid at
4096, and fatter means blending across material boundaries. A cube's support is its cell, which
tiles, so the same occupancy is solid at any resolution -- there is nothing to tune.

The number is the background-gap ratio: fill the section's silhouette, and report what fraction
of the filled area the renderer left as background. It is measured on the same plane through the
same object for every representation, so the only thing varying is what the interior is made of.

    python method/common/cube/resolution.py LATTICE ANCHOR_CKPT [GS_PLY]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

SIZES = [int(x) for x in _os.environ.get("RES_SIZES", "512,1024,2048,4096").split(",")]
DEV = "cuda:0"


def gap_ratio(hit):
    """What fraction of the section's own area the renderer left empty."""
    from scipy import ndimage
    ys, xs = np.nonzero(hit)
    if not len(xs):
        return float("nan"), 0.0
    sub = hit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    solid = ndimage.binary_fill_holes(sub)
    area = float(solid.sum())
    return 100.0 * float((solid & ~sub).sum()) / max(area, 1.0), area


def cube_series(lattice_dir, anchor_ckpt, frac=0.5, fill_pt=None):
    from method.common.cube.slice_render import (apply_fill, build_decoder, cube_slice,
                                                 load_cubes)
    xyz, feat, h, lvl, lat, ck = load_cubes(lattice_dir, anchor_ckpt)
    if fill_pt and _os.path.exists(fill_pt):
        xyz, feat, h, lvl = apply_fill(xyz, feat, h, lvl, fill_pt)
    pre, mlp, mlp_s = build_decoder(ck, feat.shape[1])
    with torch.no_grad():
        raw = feat if pre is None else pre(feat)
        c_dim = next(m for m in mlp if hasattr(m, "in_features")).in_features
        cf = raw[:, 11:11 + c_dim] if raw.shape[1] >= 11 + c_dim else raw[:, :c_dim]
        out = mlp(cf)
        if mlp_s is not None:
            out = torch.where((lvl == 1)[:, None], mlp_s(cf), out)
        rgb = torch.sigmoid(out).clamp(0, 1)

    c = xyz.mean(0)
    lo, hi = float((xyz - c)[:, 1].min()), float((xyz - c)[:, 1].max())
    d = lo + (hi - lo) * frac
    outp = {}
    for s in SIZES:
        _, hit = cube_slice(xyz, rgb, h, lvl, (0., 1., 0.), -(float(c[1]) + d), size=s)
        outp[s] = gap_ratio(hit)
    return outp, int(xyz.shape[0])


def gs_series(ply, cfg, demo, frac=0.5):
    """The same plane, drawn by the Gaussian renderer at each resolution."""
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    from utils.camera_view_utils import get_camera_view
    from utils.decode_param import decode_param_json
    from utils.render_utils import convert_SH, initialize_resterize, load_params_from_gs
    from utils.transformation_utils import *
    from method.common.eval.random_cuts import (P, generate_plane_center,
                                                interpolate_along_camera_direction,
                                                plane_filter)

    n_rest = len([q.name for q in PlyData.read(ply).elements[0].properties
                  if q.name.startswith("f_rest_")])
    deg = int(round(((n_rest / 3 + 1) ** 0.5) - 1)) if n_rest else 0
    g = GaussianModel(deg)
    if n_rest:
        g.load_ply(ply)
        g.active_sh_degree = deg
    else:
        g.load_ply_zero_sh(ply)
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    par = load_params_from_gs(g, P())
    pos0, cov0 = par["pos"], par["cov3D_precomp"]
    sp, op, shs = par["screen_points"], par["opacity"], par["shs"]
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(pos0)
    tpos = shift2center111(tpos)
    cov0 = apply_cov_rotations(cov0, rot_m)
    cov0 = so * so * cov0
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    bg = torch.tensor([1., 1., 1.], device=DEV)

    cam, raw = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                               observant_coordinates=oc, show_hint=False, init_azimuthm=0,
                               init_elevation=90, init_radius=cam_p["init_radius"],
                               move_camera=False, current_frame=0, delta_a=None,
                               delta_e=None, delta_r=None)
    _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
    c = centers[int(frac * (len(centers) - 1))]
    plane = generate_plane_center(raw, c)
    mask, mask_suf = plane_filter(plane, tpos, raw, surf_dis=float(avg) / 2,
                                  include_double=True)
    pos = apply_inverse_rotations(undotransform2origin(undoshift2center111(tpos), so, om),
                                  rot_m)
    cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
    col = convert_SH(shs[mask_suf], cam, g, pos[mask_suf], None)

    outp = {}
    for s in SIZES:
        rast = initialize_resterize(cam, g, P(), bg, image_height=s, image_width=s)
        img, _, _, alp = rast(means3D=pos[mask_suf], means2D=sp[mask_suf], shs=None,
                              colors_precomp=col, opacities=op[mask_suf], scales=None,
                              rotations=None, cov3D_precomp=cov[mask_suf])
        a = img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()
        hit = np.abs(a - 1.0).max(2) > 0.02
        outp[s] = gap_ratio(hit)
    return outp, int(mask_suf.sum())


def _table(rows):
    print(f"  {'representation':<30}{'primitives':>12}   "
          + "".join(f"{s:>10}" for s in SIZES))
    for name, n, ser in rows:
        cells = "".join(f"{ser[s][0]:>9.2f}%" if np.isfinite(ser[s][0]) else f"{'-':>10}"
                        for s in SIZES)
        print(f"  {name:<30}{n:>12,}   {cells}")
    print("  background-gap ratio: what fraction of the section's own area is left unpainted")


if __name__ == "__main__":
    lattice, ckpt = sys.argv[1], sys.argv[2]
    rows = []
    ser, n = cube_series(lattice, ckpt)
    rows.append(("Cube + O-Voxel (ours-v1)", n, ser))
    if len(sys.argv) > 3:
        from method.common.cube.composite import _cfg_for
        cfg, demo = _cfg_for(lattice)
        for label, ply in [(x.split("=")[0], x.split("=")[1]) for x in sys.argv[3:]]:
            ser, n = gs_series(ply, cfg, demo)
            rows.append((label, n, ser))
    _table(rows)
