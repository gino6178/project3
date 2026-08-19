"""Every camera and plane the multi-view supervision needs, taken from the pipeline's own
machinery rather than reconstructed beside it.

Three families, and they are the three the trainer walks:

  transverse    elevation +90 -- the trainer's own transverse camera, which is not the evaluator's:
                random_cuts draws the held-out cuts from -90.  One fixed camera, the 24 depths
                `interpolate_along_camera_direction` returns, of which train_voxel supervises
                centers[H_LO:H_HI] = centers[4:20].  All 24 are dumped so the +-0.5-step jitter can
                be applied in the plane's own coordinate, which is linear in the index.
  longitudinal  elevation 0, azimuth (180/N_VPLANES) * k for k < N_VPLANES = 10, which is the
                18-degree spacing the trainer walks -- a different camera per azimuth, and a plane
                through the middle of each.
  exterior      the six directions cube_or6_prep/dirs.json names, no plane at all.

and the twelve held-out cuts random_cuts.py draws under HELDOUT_BAND=0.30,0.70, so the evaluation
here is the evaluation there.

The plane arrives stated in the transformed frame `plane_filter` measures against; the O-Voxel
state lives in the lattice's own frame.  tpos -> pos is one global similarity, so it is solved once
from the whole point set and every plane is carried across by it.
"""
import os, sys
import numpy as np
import torch

FN_ROOT = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
CODE = "/workspace/rebuild/project3/code/src"
sys.path.insert(0, CODE)
sys.path.append(FN_ROOT)
sys.path.append(os.environ.get("GS_ROOT", FN_ROOT + "/gaussian-splatting"))
os.chdir(FN_ROOT)

from cross_section import (generate_plane_center, interpolate_along_camera_direction)  # noqa
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.camera_view_utils import get_camera_view                  # noqa: E402
from utils.decode_param import decode_param_json                     # noqa: E402
from utils.render_utils import load_params_from_gs                   # noqa: E402
from utils.transformation_utils import *                             # noqa: E402

LAT = os.environ.get("LAT", "build_orange/lattice/gs_fill.ply")
CFG = os.environ.get("CFG", "config/orange_physics.json")
DEMO = os.environ.get("DEMO", "config/sphere_demo")
EXT = os.environ.get("EXT_DIRS", "cube_or6_prep")
OUT = os.environ.get("OUT", "/workspace/ovoxel_native/cams_mv.npz")
N_AZ = int(os.environ.get("N_VPLANES", "10"))
SPACING = 180.0 / N_AZ          # the trainer's own: (180 / N_VPLANES) * i


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


(mat, bc, tp, pre, cam_p) = decode_param_json(CFG)
g = GaussianModel(0)
g.load_ply_zero_sh(LAT)
par = load_params_from_gs(g, P())
pos0 = par["pos"]
rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
up = up / up.norm()
tpos, so, om = transform2origin(pos0)
tpos = shift2center111(tpos)
vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
pos = apply_inverse_rotations(undotransform2origin(undoshift2center111(tpos), so, om), rot_m)

# tpos -> pos, once, exactly: it is a similarity, so four points would do and the whole set
# reports how well that is true.
T = tpos.detach().cpu().double().numpy()
Q = pos.detach().cpu().double().numpy()
A4 = np.concatenate([T, np.ones((len(T), 1))], 1)
M, *_ = np.linalg.lstsq(A4, Q, rcond=None)          # (4,3): pos = [tpos,1] @ M
res = float(np.abs(A4 @ M - Q).max())
print(f"tpos -> pos affine solved over {len(T):,} points, max residual {res:.3e}")

lat_xyz = np.stack([np.asarray(g.get_xyz.detach().cpu())[:, i] for i in range(3)], 1)
print(f"lattice positions match the frame the plane is carried into: "
      f"{np.allclose(Q, lat_xyz.astype(np.float64), atol=1e-5)}")


def to_pos_frame(plane):
    """A plane stated in the tpos frame, in the lattice's frame, with a unit normal."""
    nt = np.asarray(plane, np.float64)[:3]
    dt = float(np.asarray(plane, np.float64)[3])
    n = np.linalg.solve(M[:3], nt)
    d = dt - float(M[3] @ n)
    s = np.linalg.norm(n)
    return n / s, d / s


def cam_at(az, el):
    return get_camera_view(DEMO, default_camera_index=-1, center_view_world_space=vc,
                           observant_coordinates=oc, show_hint=False,
                           init_azimuthm=az, init_elevation=el,
                           init_radius=cam_p["init_radius"], move_camera=False,
                           current_frame=0, delta_a=None, delta_e=None, delta_r=None)


def mvp_of(cam):
    return cam.full_proj_transform.detach().cpu().numpy().astype(np.float32)


rec = {}

# ---- transverse: one camera, 24 depths -------------------------------------------------
cam, raw = cam_at(0.0, 90.0)
_, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
hp = []
for i in range(len(centers)):
    n, d = to_pos_frame(generate_plane_center(raw, centers[i]))
    hp.append(np.concatenate([n, [d]]))
hp = np.stack(hp)
rec["h_mvp"] = mvp_of(cam)
rec["h_planes"] = hp
rec["h_lo"], rec["h_hi"] = np.array([4]), np.array([20])   # centers[4:20], what training sees
print(f"transverse: 1 camera, {len(hp)} depths, d from {hp[:,3].min():+.4f} to {hp[:,3].max():+.4f}, "
      f"normal {np.round(hp[0,:3],4)}, normals identical: {np.allclose(hp[:,:3], hp[0,:3])}")

# ---- longitudinal: one camera per azimuth ----------------------------------------------
vm, vp, vaz = [], [], []
for k in range(N_AZ):
    az = SPACING * k
    cam, raw = cam_at(az, 0.0)
    _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
    c = centers[int(0.5 * (len(centers) - 1))]
    n, d = to_pos_frame(generate_plane_center(raw, c))
    vm.append(mvp_of(cam)); vp.append(np.concatenate([n, [d]])); vaz.append(az)
rec["v_mvp"] = np.stack(vm); rec["v_planes"] = np.stack(vp); rec["v_az"] = np.array(vaz)
print(f"longitudinal: {N_AZ} cameras at {SPACING}-degree spacing, "
      f"normals spread {np.round(np.stack(vp)[:,:3].std(0),4)}")

# ---- exterior: the six directions ------------------------------------------------------
import json
dj = json.load(open(os.path.join(EXT, "dirs.json")))
names = [k for k in dj if os.path.exists(os.path.join(EXT, f"{k}_ref.png"))]
em, ed = [], []
for nm in names:
    az, el = dj[nm]
    cam, raw = cam_at(float(az), float(el))
    em.append(mvp_of(cam)); ed.append([az, el])
rec["e_mvp"] = np.stack(em); rec["e_dir"] = np.array(ed, np.float64)
rec["e_names"] = np.array(names)
rec["e_root"] = np.array([os.path.abspath(EXT)])
print(f"exterior: {len(names)} directions {names}")

# ---- held out: the twelve random_cuts draws --------------------------------------------
import random
random.seed(7)
lo, hi = 0.30, 0.70
ehm, ehp = [], []
for i in range(6):
    f = random.uniform(lo, hi)
    cam, raw = cam_at(0.0, float(os.environ.get("CUT_EL", "-90")))
    _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
    c = centers[int(f * (len(centers) - 1))]
    n, d = to_pos_frame(generate_plane_center(raw, c))
    ehm.append(mvp_of(cam)); ehp.append(np.concatenate([n, [d]]))
evm, evp, evaz = [], [], []
spacing = float(os.environ.get("TRAINED_SPACING", "12"))


def far(a):
    return min(abs(((a - k * spacing) + 90) % 180 - 90) for k in range(10))


for i in range(6):
    az = random.uniform(0, 180)
    while far(az) < 6:
        az = random.uniform(0, 180)
    fr = random.uniform(0.45, 0.55)
    cam, raw = cam_at(az, 0.0)
    _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
    c = centers[int(fr * (len(centers) - 1))]
    n, d = to_pos_frame(generate_plane_center(raw, c))
    evm.append(mvp_of(cam)); evp.append(np.concatenate([n, [d]])); evaz.append(az)
rec["eh_mvp"] = np.stack(ehm); rec["eh_planes"] = np.stack(ehp)
rec["ev_mvp"] = np.stack(evm); rec["ev_planes"] = np.stack(evp); rec["ev_az"] = np.array(evaz)
print(f"held out: 6 transverse d {np.round(np.stack(ehp)[:,3],4)}")
print(f"          6 longitudinal az {np.round(evaz,2)}")

rec["affine"] = M
rec["affine_res"] = np.array([res])
np.savez(OUT, **rec)
print("wrote", OUT)
print("MVCAMS_OK")
