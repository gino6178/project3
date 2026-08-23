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
# N_AZ and SPACING are derived from the object further down, not set here; N_VPLANES still
# overrides for anything that needs the old fixed fan.
N_AZ = int(os.environ.get("N_VPLANES", "0"))


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
# Every object's conf points CFG at the orange's physics file, so all seven inherit the orange's
# upward axis.  Four of the released models happen to stand the same way and it is right for them;
# the apple, the cake and the bread are turned about 90 degrees, and the family called transverse
# then cuts along the object instead of across it.  UP_AXIS names the axis for those.
up = torch.tensor([float(x) for x in os.environ["UP_AXIS"].split(",")]
                  if os.environ.get("UP_AXIS") else
                  cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
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


RADIUS = float(cam_p["init_radius"])


def cam_at(az, el):
    return get_camera_view(DEMO, default_camera_index=-1, center_view_world_space=vc,
                           observant_coordinates=oc, show_hint=False,
                           init_azimuthm=az, init_elevation=el,
                           init_radius=RADIUS, move_camera=False,
                           current_frame=0, delta_a=None, delta_e=None, delta_r=None)


def mvp_of(cam):
    return cam.full_proj_transform.detach().cpu().numpy().astype(np.float32)


# ---- far enough back that the object is inside the frame ---------------------------------
# `init_radius` comes from the physics config, and every object's conf names the orange's, so
# every object is viewed from the distance that suited the orange.  The pomegranate is taller
# along its axis than that allows: all 17 of its longitudinal cuts reached the frame's edge, so
# what the loss compared a photograph against was a cut with its base cropped off, at every step
# of training.  The guard is one-sided -- a radius is only ever increased -- so an object that
# already fits keeps the cameras it was trained with, to the bit.
_q = Q[:: max(1, len(Q) // 20000)]
_qh = np.concatenate([_q, np.ones((len(_q), 1))], 1)


def _worst_ndc():
    """How far outside the frame the object reaches, over the directions cameras are taken from."""
    w = 0.0
    for az, el in [(0.0, 90.0), (0.0, -90.0)] + [(45.0 * k, 0.0) for k in range(8)]:
        cl = _qh @ mvp_of(cam_at(az, el)[0]).astype(np.float64)
        ndc = cl[:, :2] / cl[:, 3:4]
        w = max(w, float(np.abs(ndc).max()))
    return w


_r0 = RADIUS
_fill0 = _fill = _worst_ndc()
for _ in range(8):
    if _fill <= 0.92:
        break
    RADIUS *= _fill / 0.88          # apparent size goes as 1/distance, so this is one Newton step
    _fill = _worst_ndc()
print(f"camera distance: {_r0:.4f}"
      + (f" -> {RADIUS:.4f}: the object reached {_fill0:.3f} of the frame and now reaches "
         f"{_fill:.3f}" if RADIUS != _r0 else
         f": the object reaches {_fill0:.3f} of the frame, so it is left alone"))


rec = {}

# ---- how many planes each family gets --------------------------------------------------
# The two families sweep different parameters, so the number each needs is a property of the
# object. A transverse plane moves along the axis and a longitudinal one turns about it, so
# resolving both at the same spacing takes planes in the ratio (extent along the axis) to
# (pi times the radius) -- an axis length against half a circumference, half because a plane and
# its opposite normal are the same cut.
#
# Sixteen and ten was fixed for every object, a ratio of 1.6. For a sphere the right ratio is
# 2R : piR, about 0.64, so the longitudinal family should have MORE planes than the transverse
# one on anything round; 1.6 is right only for something much longer than it is wide. Measured on
# the seven objects here the fixed split over-supervises the transverse family by 1.22x on the
# cake and 7.55x on the doughnut, whose axis is 43 cells against a half-circumference of 203.
#
# The total is held at what it has always been, so every object costs exactly what it did and
# nothing is bought with extra compute. The only constant is that total; the split comes from the
# object.
TOTAL = int(os.environ.get("N_PLANES_TOTAL", "26"))
cam, raw = cam_at(0.0, 90.0)
_, _, _c0, _ = interpolate_along_camera_direction(raw, tpos, 24)
_n0, _ = to_pos_frame(generate_plane_center(raw, _c0[len(_c0) // 2]))
_n0 = np.asarray(_n0, float) / np.linalg.norm(_n0)
# The volume, not the surface. gs_fill.ply carries the coarse cells and the level-1 skin
# together, and the skin is half the spacing and all of it at the rim: on the orange 480,287 of
# 1,162,387 rows, which pulls the mean radius up 13% and moves the split by a plane. What the
# sampling has to cover is the interior, so the skin rows come out. cell_level.pt sits beside the
# ply and is row-aligned with it.
_t = tpos.detach().cpu().numpy() if hasattr(tpos, "detach") else np.asarray(tpos)
_lvlf = os.path.join(os.path.dirname(LAT), "cell_level.pt")
if os.path.isfile(_lvlf):
    _lvl = torch.load(_lvlf).reshape(-1).cpu().numpy()[:len(_t)]
    if (_lvl == 0).sum() > 1000:
        _t = _t[_lvl == 0]
        print(f"shape measured on {len(_t):,} coarse cells, skin excluded")
_p = _t @ _n0
_perp = _t - np.outer(_p, _n0)
_perp = _perp - _perp.mean(0)
_axis_len = float(_p.max() - _p.min())
_r_all = np.linalg.norm(_perp, axis=1)
# The rim, not the average. The rule balances the arc between neighbouring longitudinal planes
# against the transverse spacing, and the structure that has to be resolved -- segment walls,
# seeds, crumb -- is at the rim. The rim is 1.5 to 1.9 times the mean here, so with the mean the
# arc where it matters was 11.8 to 15.0 cells while the rule believed it was arranging 6.8 to 9.2.
# The 90th percentile rather than the maximum, so a single stray cell cannot set the scale.
RIM_PCT = float(os.environ.get("N_PLANES_RIM_PCT", "90"))
_rad = float(np.percentile(_r_all, RIM_PCT)) if RIM_PCT > 0 else float(_r_all.mean())
# The 1.5 the sampler actually lays down. The transverse depths are not N_H samples across the
# axis: `interpolate_along_camera_direction` is asked for M = ceil(1.5 * N_H) of them and the
# middle N_H are supervised, so the spacing in force is L/M, not L/N_H. Balancing L/N_H against
# pi*r/N_V therefore solved for a spacing the code never produces, and every object came out 1.29
# to 1.59 times finer on the transverse side than the balance it had just solved -- the rule
# over-supervised the very family it was written to stop over-supervising. Putting the constant
# into the equation is the whole fix: L/(OVER*N_H) = pi*r/N_V.
OVER = float(os.environ.get("N_PLANES_OVER", "1.5"))
if N_AZ > 0:
    N_H = max(TOTAL - N_AZ, 2)
else:
    N_H = int(np.clip(round(TOTAL * _axis_len / max(_axis_len + OVER * np.pi * _rad, 1e-9)),
                      2, TOTAL - 2))
    N_AZ = TOTAL - N_H
SPACING = 180.0 / N_AZ
_M_pred = int(np.ceil(OVER * N_H))
print(f"shape: axis {_axis_len:.4f}, rim radius (p{RIM_PCT:g}) {_rad:.4f} "
      f"[mean {float(_r_all.mean()):.4f}], half-circumference {np.pi * _rad:.4f}, "
      f"oversampling {OVER:g} -> {N_H} transverse and {N_AZ} longitudinal of {TOTAL}")
print(f"  the two spacings this balances: transverse {_axis_len / max(_M_pred, 1):.3f} cells "
      f"against longitudinal {np.pi * _rad / max(N_AZ, 1):.3f} at the rim")

# ---- transverse: one camera, the supervised band centred in a wider sweep ---------------
# The supervised depths have always been the middle two thirds of the range, which leaves the caps
# out; that is kept, so the only thing changing is how many there are.
# How wide a sweep the N_H supervised depths are drawn from.
#
# The supervised depths have always been the middle N_H of ceil(1.5*N_H), which leaves the caps out:
# 0.62 to 0.65 of the axis is inside the band at every plane count, and the two ends have no
# transverse plane at any setting. That was tolerable while the transverse family had 14 planes and
# reached 91% of the cells anyway; at 9 it reaches 71.8% and 19.6% of the orange never receives a
# gradient, which is visible as coloured speckle in the caps.
#
# SPAN=1.0 takes all M depths, so the band is the whole axis and M = N_H. The old behaviour is
# SPAN=1.5. It is a separate knob from N_PLANES_OVER because they answer different questions --
# OVER is how finely the depths are spaced, SPAN is how much of the object they are spread over --
# and the balance equation needs whichever of the two is in force, so OVER should be set to SPAN.
SPAN = float(os.environ.get("N_PLANES_SPAN", "1.5"))
_M = max(int(np.ceil(N_H * max(SPAN, 1.0))), N_H)
_, _, centers, avg = interpolate_along_camera_direction(raw, tpos, _M)
hp = []
for i in range(len(centers)):
    n, d = to_pos_frame(generate_plane_center(raw, centers[i]))
    hp.append(np.concatenate([n, [d]]))
hp = np.stack(hp)
_lo = (len(hp) - N_H) // 2

# SPAN=0 replaces the fixed middle window with a measured one.
#
# The middle N_H of ceil(1.5*N_H) is not arbitrary and taking all of them instead does not work:
# the outermost depths put the plane past the object, where nothing is cut and nothing is behind it,
# and `render_section` is then an empty rasteriser call -- "tri must have shape [>0, 3]", which is
# how this was found. But the fixed fraction is too conservative: it supervises 0.62 of the axis
# whatever the object, and when the transverse family is small that leaves cells no plane reaches at
# all, 19.6% of the orange at 9 planes.
#
# What the band should be is the part of the axis where a plane has something to cut, which is
# measurable. Sample the depth densely, count the cells within half a coarse cell of the plane, and
# keep the range where that count is at least MINCUT of its own maximum; then spread N_H depths
# evenly across it. The planes stay uniformly spaced and contiguous, which is what mvtrain's
# h_step and h_lo/h_hi require.
if SPAN <= 0:
    MINCUT = float(os.environ.get("N_PLANES_MINCUT", "0.05"))
    _nd = max(8 * N_H, 64)
    _, _, _cd, _ = interpolate_along_camera_direction(raw, tpos, _nd)
    _dd = np.array([to_pos_frame(generate_plane_center(raw, c))[1] for c in _cd], float)
    _nrm = np.asarray(hp[0, :3], float)
    _proj = _t @ _nrm
    # the tolerance is the scan's own resolution, so it needs no cell size and no unit conversion
    _half = float(np.abs(np.diff(_dd)).mean()) / 2.0
    _area = np.array([int(np.sum(np.abs(_proj + d) <= _half)) for d in _dd])
    _ok = np.where(_area >= MINCUT * max(_area.max(), 1))[0]
    if len(_ok) >= 2:
        _d0, _d1 = float(_dd[_ok[0]]), float(_dd[_ok[-1]])
        # The planes TILE the band rather than span it. mvtrain jitters each depth by +-h_step/2,
        # so planes at linspace(d0, d1, N_H) sweep out to d1 + step/2, past the last depth that
        # cuts anything -- and an empty cut with no exterior behind it is an empty rasteriser call,
        # which is how the first version of this died at outer 1. Putting N_H slab centres across
        # the band instead makes each plane's jitter sweep exactly its own slab: the slabs abut,
        # nothing is swept twice and nothing overshoots.
        _s = (_d1 - _d0) / N_H
        hp = np.stack([np.concatenate([_nrm, [_d0 + _s * (i + 0.5)]]) for i in range(N_H)])
        _lo = 0
        print(f"measured band: depths with a cut area at least {MINCUT:g} of the peak run "
              f"{_d0:+.4f} to {_d1:+.4f} ({len(_ok)} of {_nd} sampled), {N_H} planes tiling it "
              f"at step {_s:.4f}, jitter sweeps {_d0 + _s*0.0:+.4f} to {_d1:+.4f}")
    else:
        print(f"measured band: only {len(_ok)} of {_nd} depths cut anything -- keeping the "
              f"fixed window")
rec["h_mvp"] = mvp_of(cam)
rec["h_planes"] = hp
rec["h_lo"], rec["h_hi"] = np.array([_lo]), np.array([_lo + N_H])
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
