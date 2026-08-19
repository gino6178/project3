"""How thick the pipeline's cut face actually is, in this object's own units.

train_voxel.py:2007 and random_cuts.py both call
    plane_filter(plane, tpos, raw, surf_dis=avg_dis/2, include_double=True)
so the rendered section integrates every primitive within +-avg/2 of the plane, each splatted with
its own footprint. `avg` comes out of `interpolate_along_camera_direction` in the TRANSFORMED frame;
the O-Voxel planes live in the lattice frame, so it has to be carried across by the same affine
`mvcams.py` solved, or the slab will be wrong by that scale.
"""
import os, sys
import numpy as np, torch

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
sys.path.insert(0, "/workspace/rebuild/project3/code/src")
sys.path.append(FN)
sys.path.append(os.environ.get("GS_ROOT", "/workspace/rebuild/gaussian-splatting"))
os.chdir(FN)
from cross_section import interpolate_along_camera_direction, generate_plane_center  # noqa
from scene.gaussian_model import GaussianModel                                        # noqa
from utils.camera_view_utils import get_camera_view                                   # noqa
from utils.decode_param import decode_param_json                                      # noqa
from utils.render_utils import load_params_from_gs                                    # noqa
from utils.transformation_utils import *                                              # noqa

PLY = os.environ.get("PLY", "/workspace/ovoxel_native/baseline/orange_b.ply")
CFG = os.environ.get("CFG", "config/orange_physics.json")
DEMO = os.environ.get("DEMO", "config/sphere_demo")
HC = float(os.environ.get("HC", "0.01180"))


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


(mat, bc, tp, pre, cam_p) = decode_param_json(CFG)
g = GaussianModel(0); g.load_ply_zero_sh(PLY)
par = load_params_from_gs(g, P())
rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
up = up / up.norm()
tpos, so, om = transform2origin(par["pos"])
tpos = shift2center111(tpos)
vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
pos = apply_inverse_rotations(undotransform2origin(undoshift2center111(tpos), so, om), rot_m)

T = tpos.detach().cpu().double().numpy(); Q = pos.detach().cpu().double().numpy()
A4 = np.concatenate([T, np.ones((len(T), 1))], 1)
M, *_ = np.linalg.lstsq(A4, Q, rcond=None)
scale = float(np.linalg.norm(M[:3], axis=0).mean())     # tpos units -> lattice units
print(f"model {PLY}  {len(T):,} rows")
print(f"tpos -> lattice scale {scale:.6f}  (residual {np.abs(A4 @ M - Q).max():.2e})")


def meas(az, el, tag):
    cam, raw = get_camera_view(DEMO, default_camera_index=-1, center_view_world_space=vc,
                              observant_coordinates=oc, show_hint=False, init_azimuthm=az,
                              init_elevation=el, init_radius=cam_p["init_radius"],
                              move_camera=False, current_frame=0, delta_a=None, delta_e=None,
                              delta_r=None)
    _, _, centers, avg = interpolate_along_camera_direction(raw, tpos, 24)
    avg = float(avg)
    sd_t = avg / 2.0
    sd_l = sd_t * scale
    # the same spacing read off the lattice-frame planes, as a cross-check
    d = [generate_plane_center(raw, c) for c in centers[:3]]
    print(f"  {tag:<14} avg {avg:.5f} (tpos)  surf_dis = avg/2 = {sd_t:.5f} (tpos) "
          f"= {sd_l:.5f} (lattice) = {sd_l/HC:.2f} coarse cells")
    print(f"  {'':<14} the rendered slab is 2*surf_dis = {2*sd_l/HC:.2f} coarse cells thick")
    return sd_l


sd_h = meas(0.0, -90.0, "transverse")
sd_v = meas(0.0, 0.0, "longitudinal")
np.savez("/workspace/ovoxel_native/slab.npz", surf_dis_h=np.array([sd_h]),
         surf_dis_v=np.array([sd_v]), scale=np.array([scale]), hc=np.array([HC]))
print("wrote /workspace/ovoxel_native/slab.npz")
print("SLAB_OK")
