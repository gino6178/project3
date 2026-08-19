"""The interior, initialised from the cross-section photographs themselves.

`skin_project` puts the six exterior views onto the cells they see. This is the same
construction one axis in: the transverse and longitudinal photographs are swept through the
volume and each interior cell takes the colour the photograph of its own depth shows at its own
position. Nothing here comes from a reconstruction; the only inputs are the shape and the
photographs, which is the whole premise -- a scanner gives a shell, and the photographs are what
fills it.

Why this exists. Starting the interior flat asks training to invent every structure from a
gradient, and on six of the seven objects it does. On the pomegranate it does not: measured, its
interior moves 0.185 away from flat but ends with a spatial spread of 0.030 against the orange's
0.133, which is a volume that learned the average colour of a pomegranate and none of its arils.
The supervision is not missing and the gradient is not missing; the optimisation converges to the
mean. Handing it the photographs' own structure as a starting point is not a hint from outside
the problem, because the photographs are the supervision.

    python section_project.py LATTICE_DIR CFG DEMO REF_H REF_V OUT_DIR

The mapping is the one `skin_project` uses, because two ways of putting a photograph on a cell
would eventually disagree: a cell is projected through the cutting camera's own matrix, its
offset from the section's centre is divided by the silhouette radius that section's own cells
span, and that gives the pixel. The photograph for a plane is chosen by equation (27)'s solved
assignment and phase where a solve exists, falling back to equation (11), exactly as
`sds_demo` chooses it during training.
"""
import json
import os as _os
import sys

import cv2
import numpy as np
import torch

_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
sys.path.append(_FN_ROOT)
sys.path.append(_os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting"))
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_FN_ROOT)

from cross_section import generate_plane_center, interpolate_along_camera_direction  # noqa
from scene.gaussian_model import GaussianModel                          # noqa: E402
from utils.camera_view_utils import get_camera_view                     # noqa: E402
from utils.decode_param import decode_param_json                        # noqa: E402
from utils.render_utils import load_params_from_gs                   # noqa: E402
from utils.transformation_utils import *                                # noqa: E402

import sds_demo                                                          # noqa: E402

DEV = "cuda:0"
C0 = 0.28209479177387814


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def photo_for(spec, idx, n, size):
    """The photograph training would use for plane idx of n, as an array."""
    sds_demo._PLANE["idx"], sds_demo._PLANE["n"] = idx, n
    im = (sds_demo._solved_photo(spec) if sds_demo.REF_PHASE_MODE == "solve"
          else sds_demo._photo(spec))
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.
    return cv2.resize(a, (size, size))


def sweep(world, cam_of, spec, n_planes, size, tag):
    """One family: colour every cell from the photograph of the plane nearest to it.

    Returns (colour, weight). The weight is a hat over the plane spacing, so a cell between two
    planes takes a mixture rather than a side, which is equation (14)'s idea applied in space
    instead of in the plane index.
    """
    N = world.shape[0]
    col = torch.zeros(N, 3, device=DEV)
    wsum = torch.zeros(N, 1, device=DEV)
    for j in range(n_planes):
        cam, plane, centre_ndc, r_sil, ndc, dist = cam_of(j)
        img = torch.from_numpy(photo_for(spec, j, n_planes, size)).to(DEV)
        uv = (ndc - centre_ndc[None]) / r_sil
        px = ((uv[:, 0] * .5 + .5) * (size - 1)).round().long().clamp(0, size - 1)
        py = ((uv[:, 1] * .5 + .5) * (size - 1)).round().long().clamp(0, size - 1)
        c = img[py, px]
        # white is background in every reference set; a cell that lands there learns nothing
        keep = (c.mean(1) < 0.96) & (uv.abs().max(1).values <= 1.0)
        w = (1.0 - dist).clamp(min=0.0) * keep.float()
        col += w[:, None] * c
        wsum += w[:, None]
    got = (wsum[:, 0] > 1e-6)
    print(f"    {tag}: {int(got.sum()):,} of {N:,} cells reached by {n_planes} planes")
    return col, wsum


def main(lattice_dir, cfg, demo, ref_h, ref_v, out_dir, size=512):
    _os.makedirs(out_dir, exist_ok=True)
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(_os.path.join(lattice_dir, "gs_fill.ply"))
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).to(DEV)
    par = load_params_from_gs(g, P())
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(par["pos"])
    tpos = shift2center111(tpos)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    N = world.shape[0]
    lvl = lvl[:N]
    hom = torch.cat([world, torch.ones(N, 1, device=DEV)], 1)

    H_LO = int(_os.environ.get("H_LO", "4")); H_HI = int(_os.environ.get("H_HI", "20"))
    NV = int(_os.environ.get("N_VPLANES", "10"))

    def maker(az, el, n_planes, along_axis):
        def cam_of(j):
            cam, raw = get_camera_view(demo, default_camera_index=-1,
                                       center_view_world_space=vc, observant_coordinates=oc,
                                       show_hint=False, init_azimuthm=az(j), init_elevation=el,
                                       init_radius=cam_p["init_radius"], move_camera=False,
                                       current_frame=0, delta_a=None, delta_e=None, delta_r=None)
            _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
            c = centers[along_axis(j)]
            # generate_plane_center returns [a, b, c, d] and the plane lives in the
            # transformed frame, which is the frame plane_filter measures against; the camera
            # matrix works in the untransformed one. Measuring the distance in the wrong frame
            # is a rotation's worth of error and it is silent, so both are named here.
            plane = generate_plane_center(raw, c)
            nrm = torch.tensor(plane[:3], dtype=torch.float32, device=DEV)
            d = (tpos.to(DEV) @ nrm + float(plane[3])).abs() / max(float(avg), 1e-6)
            clip = hom @ cam.full_proj_transform
            ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-6)
            near = d < 0.5
            if near.sum() < 16:
                near = d < 1.0
            cen = ndc[near].mean(0)
            r = (ndc[near] - cen[None]).abs().max().clamp_min(1e-6)
            return cam, plane, cen, r, ndc, d
        return cam_of

    print(f"  {N:,} cells, {int((lvl == 0).sum()):,} interior")
    ch, wh = sweep(world, maker(lambda j: 0.0, float(_os.environ.get("CUT_EL", "-90")),
                                H_HI - H_LO, lambda j: H_LO + j),
                   ref_h, H_HI - H_LO, size, "transverse")
    cv_, wv = sweep(world, maker(lambda j: (180.0 / NV) * j, 0.0, NV, lambda j: 12),
                    ref_v, NV, size, "longitudinal")

    col = (ch + cv_) / (wh + wv).clamp(min=1e-6)
    got = ((wh + wv)[:, 0] > 1e-6) & (lvl == 0)
    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
    rgb[got] = col[got]
    print(f"  interior initialised from the photographs: {int(got.sum()):,} cells, "
          f"mean {rgb[lvl == 0].mean(0).detach().cpu().numpy().round(3)}, "
          f"spread {float(rgb[lvl == 0].std(0).mean()):.4f}")

    with torch.no_grad():
        g._features_dc = torch.nn.Parameter(((rgb - 0.5) / C0).unsqueeze(1).contiguous())
    g.trained = torch.zeros(N, dtype=torch.bool)
    g.is_interior = torch.ones(N, dtype=torch.bool)
    g.save_ply(_os.path.join(out_dir, "gs_fill.ply"))
    for f in ("cell_level.pt", "lattice.pt", "is_interior.pt", "cell_normal.pt"):
        s = _os.path.join(lattice_dir, f)
        if _os.path.exists(s):
            torch.save(torch.load(s), _os.path.join(out_dir, f))
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
