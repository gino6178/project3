"""A knife walking through the object, one frame per depth.

The still gallery shows that individual cuts look right. What it cannot show is that they agree
with each other: twelve plausible pictures could be twelve unrelated pictures. Sweeping the plane
continuously answers that in the only way a viewer will accept -- the pith has to stay where it
was between one frame and the next, and any structure that lives on the supervised planes rather
than in the volume flickers as the plane passes between them.

Two sweeps, because the two families of planes are supervised separately and the question of
whether they describe one object is exactly the question this animation can settle.

    python report/sweep_video.py MODEL.ply cfg demo OUT_DIR [n_frames]
"""
import os as _os
# The repository root, so the same file runs here and on the remote box. It was written three
# times in every script -- two sys.path entries and a chdir -- and each one silently pinned the
# script to one machine: on the remote the chdir landed somewhere else and a relative source
# path then could not be found, which surfaces as "no such file" for a file that is plainly
# there.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

from cross_section import (generate_plane_center, interpolate_along_camera_direction,  # noqa
                           plane_filter)
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.camera_view_utils import get_camera_view                  # noqa: E402
from utils.decode_param import decode_param_json                     # noqa: E402
from utils.render_utils import (convert_SH, initialize_resterize,     # noqa: E402
                                load_params_from_gs)
from utils.transformation_utils import *                             # noqa: E402

DEV = "cuda:0"


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def main(ply, cfg, demo, out_dir, n=72, size=512):
    os.makedirs(out_dir, exist_ok=True)
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(ply)
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

    def shot(az, el, frac, tag, i):
        cam, raw = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                   observant_coordinates=oc, show_hint=False,
                                   init_azimuthm=az, init_elevation=el,
                                   init_radius=cam_p["init_radius"], move_camera=False,
                                   current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        # The slab has to be the thickness training used, not one derived from however many
        # positions the sweep happens to sample. Taking `avg` from 96 positions made the slab
        # four times thinner than the trained one; too few cells then cover the frame, the white
        # background reads through the flesh, and a model measured at 0.58% white pixels renders
        # at 19.2%. The sweep is a visualisation, and a visualisation that changes the answer is
        # a measurement error.
        _, _, centres, avg = interpolate_along_camera_direction(raw, tpos, 24)
        c = centres[0] + (centres[-1] - centres[0]) * frac
        plane = generate_plane_center(raw, c)
        mask, mask_suf = plane_filter(plane, tpos, raw, surf_dis=float(avg) / 2,
                                      include_double=True)
        pos = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(tpos), so, om), rot_m)
        cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
        col = convert_SH(shs[mask_suf], cam, g, pos[mask_suf], None)
        rast = initialize_resterize(cam, g, P(), bg, image_height=size, image_width=size)
        img, _, _, _ = rast(means3D=pos[mask_suf], means2D=sp[mask_suf], shs=None,
                            colors_precomp=col, opacities=op[mask_suf], scales=None,
                            rotations=None, cov3D_precomp=cov[mask_suf])
        a = img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{i:03d}.png"),
                    (a[:, :, ::-1] * 255).astype(np.uint8))

    # The plane walks from one side to the other and back, so the loop has no jump.
    for i, f in enumerate(np.concatenate([np.linspace(0.10, 0.90, n // 2),
                                          np.linspace(0.90, 0.10, n // 2)])):
        shot(0.0, -89.0, float(f), "t", i)
    # And turns a full circle about the axis, which for a longitudinal plane is the same plane
    # every 180 degrees, so half a turn is the whole story.
    for i, a in enumerate(np.linspace(0, 180, n)):
        shot(float(a), 0.0, 0.5, "l", i)
    print(f"  -> {out_dir}  ({n} transverse, {n} longitudinal)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         n=int(sys.argv[5]) if len(sys.argv) > 5 else 72)
