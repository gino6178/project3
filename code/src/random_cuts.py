"""Render cuts at heights and angles the training never supervised.

Every interior number so far was read off the planes the run itself was trained on, and the
trainer's own comment records what that is worth: the anchor model scored +0.688 on a
supervised cross-section and +0.274 on an independent cut of the same object. The paper
evaluates on random cuts for the same reason -- horizontal ones at random depths, vertical ones
at random angles about the axis -- so this renders those.

The camera and the cutting are the trainer's own: `generate_plane_center` off the raw camera
for a horizontal cut, and an azimuth the trainer never used for a vertical one. Only the
choice of plane differs from what training saw.

    python random_cuts.py MODEL.ply cfg demo out_dir 12
"""
import os as _os
# The repository root, so the same file runs here and on the remote box. It was written three
# times in every script -- two sys.path entries and a chdir -- and each one silently pinned the
# script to one machine: on the remote the chdir landed somewhere else and a relative source
# path then could not be found, which surfaces as "no such file" for a file that is plainly
# there.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.append(_FN_ROOT)
sys.path.append(_os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting"))
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

# A renderer may be substituted, and nothing else about a run changes when one is.
#
# The point of this evaluation is the cuts: which planes, at which depths, from which cameras,
# scored against which photographs. Asking whether another representation draws the same cuts as
# well means changing what draws them and not one line more -- and reproducing the plane sequence
# in a second file is exactly how that stops being true, since it shares a seed until someone
# edits one of them. So the sequence stays here and the drawing is a hook: set RENDER_HOOK to a
# callable taking (cam, plane, mask, mask_suf, size) and returning an HxWx3 float image.
RENDER_HOOK = [None]


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def build_renderer(ply, cfg, demo, size=512, n_depth=24):
    """The trainer's own camera, model and cutting, as a function of (azimuth, elevation, depth).

    Factored out of main because a figure that renders at depths it chooses -- the continuity
    sweep of code/figures/blend_fig.py is the one that needed it -- must cut exactly the way the
    evaluation does, and the only way to guarantee that is to call the same code rather than a
    copy of it.
    """
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    # Load at the degree the file actually carries. `load_ply_zero_sh` reads only f_dc and
    # discards every higher band, and our models carry 24 f_rest coefficients whose mean
    # magnitude is 0.078 -- the per-voxel directional appearance. Measuring them with that
    # thrown away, against a released model that has no higher bands to throw away, is not a
    # comparison. It is also why the offline renders looked washed out beside the training
    # ones and why a visibly better model scored a worse FID.
    from plyfile import PlyData as _P
    _n = len([q.name for q in _P.read(ply).elements[0].properties
              if q.name.startswith("f_rest_")])
    _deg = int(round(((_n / 3 + 1) ** 0.5) - 1)) if _n else 0
    g = GaussianModel(_deg)
    if os.environ.get("FULL_SH", "1") == "1" and _n:
        g.load_ply(ply)
        g.active_sh_degree = _deg
        print(f"  loaded with {_n} higher-band coefficients (degree {_deg})")
    else:
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

    def render(az, el, depth_frac):
        cam, raw = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                   observant_coordinates=oc, show_hint=False,
                                   init_azimuthm=az, init_elevation=el,
                                   init_radius=cam_p["init_radius"], move_camera=False,
                                   current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        # n_depth is the trainer's own 24 by default, and the depth fraction indexes into it.
        # A sweep finer than this resolves nothing: two fractions that round to one centre give
        # the identical image, so a continuity read-out over such a sweep measures the
        # quantisation of this list and not the volume. blend_fig.py raises it for that reason.
        _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, n_depth)
        avg = float(avg)
        c = centers[int(round(depth_frac * (len(centers) - 1)))]
        plane = generate_plane_center(raw, c)
        mask, mask_suf = plane_filter(plane, tpos, raw, surf_dis=avg / 2, include_double=True)
        pos = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(tpos), so, om), rot_m)
        cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
        if RENDER_HOOK[0] is not None:
            # tpos as well as pos, because the plane is built in the transformed frame that
            # plane_filter measures against and the renderer works in the untransformed one. A
            # hook that assumed they were the same would cut in the wrong place, by a rotation.
            return RENDER_HOOK[0](cam, plane, mask, mask_suf, size, tpos, pos)
        col = convert_SH(shs[mask_suf], cam, g, pos[mask_suf], None)
        rast = initialize_resterize(cam, g, P(), bg, image_height=size, image_width=size)
        img, _, _, _ = rast(means3D=pos[mask_suf], means2D=sp[mask_suf], shs=None,
                            colors_precomp=col, opacities=op[mask_suf], scales=None,
                            rotations=None, cov3D_precomp=cov[mask_suf])
        return img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()

    return render


def sweep(ply, cfg, demo, out_dir, depths, az=0.0, el=None, size=512, n_depth=24):
    """Render one transverse section per depth fraction, in the order given.

    n_depth must be at least as large as the sweep is dense, or neighbouring depths collapse
    onto one plane and the sweep is a step function of its own indexing.
    """
    os.makedirs(out_dir, exist_ok=True)
    render = build_renderer(ply, cfg, demo, size, n_depth)
    el = float(os.environ.get("CUT_EL", "-90")) if el is None else el
    out = []
    for i, f in enumerate(depths):
        a = render(az, el, float(f))
        q = os.path.join(out_dir, f"d{i:03d}.png")
        cv2.imwrite(q, (a[:, :, ::-1] * 255).astype(np.uint8))
        out.append(q)
    print(f"  -> {out_dir}  ({len(out)} sections over depth {depths[0]:.3f}..{depths[-1]:.3f})")
    return out


def main(ply, cfg, demo, out_dir, n=12, size=512, seed=7):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    render = build_renderer(ply, cfg, demo, size)

    # Horizontal cuts at depths the trainer never used. It supervises centers[4:20] of 24, so
    # the held-out depths are the outer eighths, plus the fractional positions between its
    # samples that the jitter of +-0.5 of a step cannot reach.
    for i in range(n // 2):
        # The outer eighths are the depths the trainer never samples, so they are the honest
        # held-out set -- but on a round fruit they are shallow caps, and the photographs are
        # all of the middle. Comparing a cap to a full slice measures the framing, not the
        # interior, so HELDOUT_BAND opens the middle instead: still off the trained grid,
        # since the jitter is +-0.5 of a step and these fall between samples, but the same
        # kind of picture as the reference set.
        _lo, _hi = (float(v) for v in os.environ.get("HELDOUT_BAND", "").split(",")) \
            if os.environ.get("HELDOUT_BAND") else (0.0, 0.0)
        f = (random.uniform(_lo, _hi) if _hi > _lo else
             random.choice([random.uniform(0.04, 0.15), random.uniform(0.85, 0.96)]))
        # Not -90. `get_camera_view` builds its look-at from the configured upward axis, and
        # at the pole the view direction is that axis, so the camera's own up vector is
        # degenerate: the released model came back framed from inside itself while its
        # longitudinal cuts were clean. One degree off frames both models the same way.
        # -90 is the pole, and it used to be unusable: `get_camera_view` builds its look-at from
        # the configured upward axis, so at the pole the view direction *is* that axis, the up
        # vector is degenerate and the roll came out of rounding noise -- the released model came
        # back framed from inside itself while its longitudinal cuts were clean. The fix was to
        # give the roll a definite value there rather than to move the camera off the pole, so the
        # default is the pole again and every number in the paper was measured at it. Setting
        # CUT_EL to -89 was the old workaround and still works.
        a = render(0.0, float(os.environ.get('CUT_EL', '-90')), f)
        cv2.imwrite(os.path.join(out_dir, f"rh{i}_init_0.png"),
                    (a[:, :, ::-1] * 255).astype(np.uint8))
    # Vertical cuts at azimuths the trainer does not walk. A cut at a and at a+180 is the same
    # plane, so the comparison is modulo 180; the trained set is `spacing * k` for k < 10, and
    # `spacing` is what the run used -- 12 degrees before this was fixed, 18 after.
    spacing = float(os.environ.get("TRAINED_SPACING", "12"))
    for i in range(n - n // 2):
        def far(a):
            return min(abs(((a - k * spacing) + 90) % 180 - 90) for k in range(10))
        az = random.uniform(0, 180)
        while far(az) < 6:
            az = random.uniform(0, 180)
        a = render(az, 0.0, random.uniform(0.45, 0.55))
        cv2.imwrite(os.path.join(out_dir, f"rv{i}_init_0.png"),
                    (a[:, :, ::-1] * 255).astype(np.uint8))
    print(f"  -> {out_dir}  ({n} held-out cuts)")


if __name__ == "__main__":
    # The image size is a fifth argument rather than only a default, because whether a coverage
    # measurement is real or is an artefact of sampling is answered by varying it, and a default
    # that can only be changed by editing the file is not a parameter anyone will vary.
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         n=int(sys.argv[5]) if len(sys.argv) > 5 else 12,
         size=int(sys.argv[6]) if len(sys.argv) > 6 else 512)
