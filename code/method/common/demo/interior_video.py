"""The object cut into slices along a fixed direction, pulled apart, and orbited.

FruitNinja's interior-texture videos take one cut direction, slice the object all the way through
it, separate the slices along the normal so every section is visible at once, and move the camera.
That is a harder thing to show than a single plane: every section is on screen simultaneously, so
a section that disagrees with its neighbours has nothing to hide behind, and the orbit means each
face is seen at an angle as well as head-on. A texture painted onto the supervised planes would
give itself away in both.

No plane filtering is involved. The object is a filled volume, so a slab's cut face *is* part of
that slab's outer boundary once the slabs are apart; assigning every primitive to a slab by its
coordinate along the cut normal and translating it is the whole operation. Translation leaves the
covariances alone.

    python report/interior_video.py MODEL.ply cfg demo OUT_PREFIX [n_frames]
"""
import os as _os
# The repository root, so the same file runs here and on the remote box. It was written three
# times in every script -- two sys.path entries and a chdir -- and each one silently pinned the
# script to one machine: on the remote the chdir landed somewhere else and a relative source
# path then could not be found, which surfaces as "no such file" for a file that is plainly
# there.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import os
import subprocess
import sys

import cv2
import numpy as np
import torch

sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.camera_view_utils import get_camera_view                  # noqa: E402
from utils.decode_param import decode_param_json                     # noqa: E402
from utils.render_utils import (convert_SH, initialize_resterize,     # noqa: E402
                                load_params_from_gs)
from utils.transformation_utils import *                             # noqa: E402

DEV = "cuda:0"
# The cut direction: the camera whose viewing direction is the slab normal. Elevation 0 makes that
# normal horizontal, so the slabs come apart across the frame rather than up and down it -- which
# is what puts every cut face broadside to a camera that is also near the horizon.
CUT_AZ = float(os.environ.get("CUT_AZ", "0.0"))
CUT_EL = float(os.environ.get("CUT_EL", "0.0"))
N_SLICE = int(os.environ.get("N_SLICE", "11"))
# Two elevations, which is the "two distinct viewing angles" the reference page provides: one
# nearly edge-on, where the stack reads as a stack, and one from above, where every cut face is
# visible at once.
ELEVS = [float(v) for v in os.environ.get("ELEVS", "-6;-34").split(";")]
# Where the camera sits relative to the cut normal, and how far it swings. Not 90 degrees: dead
# perpendicular spreads the stack across the frame but puts every cut face edge-on, so all a
# viewer sees is peel. Around 42 keeps the stack spread and still looks into each face.
AZ_OFF = float(os.environ.get("AZ_OFF", "42"))
SWING = float(os.environ.get("SWING", "11"))
# One swing of one camera is four seconds and answers one question. The clip instead walks a
# sequence of camera regimes over the same open stack, each a (azimuth offset, elevation, radius
# multiplier, fraction of the clip) -- broadside, past the normal so the faces turn away and come
# back, high enough to look down the stack, and in close on three slabs. A section that only holds
# up from the angle it was supervised from has to fail somewhere in that walk.
LEGS = os.environ.get("LEGS", "42,-6,1.00,0.20; 64,-18,0.95,0.18; 34,-30,1.02,0.18; "
                              "52,-48,1.00,0.20; 46,-20,0.80,0.24")
GAP = float(os.environ.get("GAP", "2.0"))        # in slab thicknesses
ZOOM = float(os.environ.get("ZOOM", "2.2"))      # radius multiplier, the stack is longer
FPS = int(os.environ.get("FPS", "24"))


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def main(ply, cfg, demo, out_prefix, n=420, size=640):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
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
    cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)

    def camera(az, el, r):
        return get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                               observant_coordinates=oc, show_hint=False,
                               init_azimuthm=az, init_elevation=el, init_radius=r,
                               move_camera=False, current_frame=0,
                               delta_a=None, delta_e=None, delta_r=None)

    # The slab normal, taken from the cut camera and used only for that.
    _, raw_cut = camera(CUT_AZ, CUT_EL, cam_p["init_radius"])
    nrm = raw_cut["R"] @ np.array([0., 0., 1.])
    nrm = nrm / np.linalg.norm(nrm)
    nt = torch.tensor(nrm, dtype=torch.float32, device=DEV)

    proj = tpos @ nt
    lo, hi = float(proj.min()), float(proj.max())
    thick = (hi - lo) / N_SLICE
    # The slab a primitive belongs to, and how far from the middle slab it is. Membership is by
    # the primitive's own coordinate, which is the same rule the cut uses in the simulator.
    sidx = ((proj - lo) / thick).floor().clamp(0, N_SLICE - 1)
    offs = (sidx - (N_SLICE - 1) / 2.0).unsqueeze(1) * nt.unsqueeze(0)

    legs = []
    for part in LEGS.split(";"):
        a_, e_, r_, f_ = (float(v) for v in part.split(","))
        legs.append((a_, e_, r_, f_))
    tot = sum(l[3] for l in legs)

    def leg_at(u):
        """Camera for a position u in [0,1) of the clip, eased across the leg boundaries."""
        acc = 0.0
        for li, (a_, e_, r_, f_) in enumerate(legs):
            w = f_ / tot
            if u < acc + w or li == len(legs) - 1:
                v = (u - acc) / w
                nxt = legs[(li + 1) % len(legs)]
                # cross-fade the last fifth of a leg into the next, so the camera moves rather
                # than cutting -- a cut would read as a different render, which is the opposite
                # of what this figure is for
                if v > 0.8:
                    k = 0.5 - 0.5 * np.cos(np.pi * (v - 0.8) / 0.2)
                    return tuple(a + (b - a) * k for a, b in zip((a_, e_, r_), nxt[:3]))
                return (a_, e_, r_)
            acc += w
        return legs[-1][:3]

    outs = []
    for vi, el0 in enumerate(ELEVS):
        tmp = f"{out_prefix}_v{vi}_frames"
        os.makedirs(tmp, exist_ok=True)
        for i in range(n):
            t = i / n
            # The stack opens at the start, holds through every camera leg, and closes at the
            # end, so the clip loops without a jump.
            if t < 0.10:
                sfrac = t / 0.10
            elif t > 0.90:
                sfrac = (1.0 - t) / 0.10
            else:
                sfrac = 1.0
            sfrac = 0.5 - 0.5 * np.cos(np.pi * sfrac)
            gap = GAP * thick * float(sfrac)
            moved = tpos + offs * gap
            a_off, e_leg, r_mul = leg_at(t)
            az = CUT_AZ + a_off + SWING * float(np.sin(4 * np.pi * t))
            cam, _ = camera(az, e_leg + el0 - ELEVS[0], cam_p["init_radius"] * ZOOM * r_mul)
            pos = apply_inverse_rotations(
                undotransform2origin(undoshift2center111(moved), so, om), rot_m)
            col = convert_SH(shs, cam, g, pos, None)
            rast = initialize_resterize(cam, g, P(), bg, image_height=size, image_width=size)
            img, _, _, _ = rast(means3D=pos, means2D=sp, shs=None, colors_precomp=col,
                                opacities=op, scales=None, rotations=None, cov3D_precomp=cov)
            a = img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()
            cv2.imwrite(os.path.join(tmp, f"{i:03d}.png"), (a[:, :, ::-1] * 255).astype(np.uint8))
        mp4 = f"{out_prefix}_v{vi}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(tmp, "%03d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "21",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4], check=True)
        outs.append(mp4)
        print(f"  view {vi}: {N_SLICE} slabs of {thick:.4f}, {len(legs)} camera legs over "
              f"{n} frames ({n / FPS:.1f}s), base elevation {el0:.0f}  -> {mp4}")
    return outs


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         n=int(sys.argv[5]) if len(sys.argv) > 5 else 420)
