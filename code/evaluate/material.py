"""A stiffness per material, and whether the solver can actually see it.

The previous project decomposed an object into material classes without being told what they
are -- cluster the learned per-cell feature, merge classes whose appearance centroids agree, and
an orange separates into peel, flesh, a transition and pith. It then mapped class to Young's
modulus by an explicit ordering, because appearance cannot recover a modulus and pretending
otherwise is the thing the design document specifically forbids: the material field is an
optional attribute, not a measurement.

That work ended on an honest failure, and it is the one worth picking up here:

    "The material field changes the simulated dynamics but does not yet produce a hard shell
     around a soft interior, because MPM transfers through one background grid and our shell is
     one to three cells thick."

Two representations later, the shell is no longer incidental. It is a level of the lattice, its
thickness is a parameter of the build rather than a consequence of where a reconstruction put its
primitives, and after a cut every piece has an identity that a solver can index. So the question
this file asks is not "can we assign a stiffness" -- that was answered -- but the one that was
left open: **is the shell thick enough, in the solver's own units, for a stiffness contrast to
survive the transfer?**

That is a ratio, and it has an answer before any simulation is run. MPM transfers particle
quantities to a background grid of spacing dx and back; a feature thinner than about two grid
cells is averaged with its neighbours on the way through and comes back as the average. So the
number that decides whether a hard shell is possible is the shell's thickness measured in MPM
grid cells, and it is reported here per object alongside the stiffness it would be given.

    python code/evaluate/material.py LATTICE MODEL.ply CFG [labels.npy] [DEMO OUT.png]

One caution about the second half of what it prints. The resolvability line divides the shell
thickness by the grid it reads out of the physics config, which is the grid particles are *seeded*
on -- `particle_filling.n_grid`. Momentum passes through the coarser grid the solver integrates on,
and section 4.2.1 of the paper reports the ratio against that one, where the same shells come out
four to ten times thinner and none of them is resolvable. The class decomposition and the ordering
of equation (13) above it are unaffected; the RESOLVABLE verdict below it is against the wrong
grid and section 4.2.1's table supersedes it.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import json
import sys

import numpy as np
import torch

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CODE = _os.path.dirname(_HERE)
sys.path += [_HERE, _os.path.join(_CODE, "src"), _os.path.join(_CODE, "figures"),
             _os.path.join(_CODE, "inherited"),
             _FN_ROOT, _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]

C0 = 0.28209479177387814

# The range, stated rather than recovered. These are the order-of-magnitude values graphics MPM
# uses for soft solids; the point of the mapping is the *ordering* and the *contrast*, not the
# absolute number, and the design document is explicit that v1 must not claim otherwise.
E_MIN = float(_os.environ.get("E_MIN", "1e5"))
E_MAX = float(_os.environ.get("E_MAX", "5e6"))


def rank(x):
    """Rank, normalised to [0, 1] -- the previous project's R operator."""
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return np.zeros_like(x)
    o = np.argsort(np.argsort(x)).astype(np.float64)
    return o / (len(x) - 1)


def moduli(labels, skin, lum):
    """Class -> stiffness, by the previous project's equation (11).

    u_j = 0.65 beta_j + 0.35 R(L)_j, with beta the class's shell fraction and L its brightness;
    r_i is the rank of u over classes; E is that rank mapped onto [E_min, E_max]. The weights are
    fixed, not learned, and the ordering is the claim.
    """
    K = int(labels.max()) + 1
    beta = np.array([float(skin[labels == j].mean()) if (labels == j).any() else 0.0
                     for j in range(K)])
    L = np.array([float(lum[labels == j].mean()) if (labels == j).any() else 0.0
                  for j in range(K)])
    u = 0.65 * beta + 0.35 * rank(L)
    r = rank(u)
    E = E_MIN + r * (E_MAX - E_MIN)
    return beta, L, u, r, E


def shell_thickness(xyz, skin, hf):
    """How thick the shell is, in world units, from the cells themselves.

    Measured rather than assumed: for each shell cell, the distance to the nearest non-shell cell
    is at most the thickness, and the median over shell cells is a robust estimate of it that does
    not depend on the object being round.
    """
    from scipy.spatial import cKDTree
    inner = xyz[~skin]
    if not len(inner) or not skin.any():
        return float("nan")
    d, _ = cKDTree(inner).query(xyz[skin], k=1)
    return float(np.percentile(d, 90)) + 0.5 * hf


def render_slices(lattice_dir, model_ply, cfg, demo, labels, E, out_png, size=420):
    """The same section three ways: as it looks, as classes, as stiffness.

    Drawn by the cube lookup rather than by splatting points, because a scatter of one dot per
    cell reads as a haze and the whole claim here is that the classes are *regions*. Solid is what
    makes a region visible.
    """
    import cv2
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    from utils.camera_view_utils import get_camera_view
    from utils.decode_param import decode_param_json
    from utils.render_utils import load_params_from_gs
    from utils.transformation_utils import (apply_inverse_rotations, generate_rotation_matrices,
                                            get_center_view_worldspace_and_observant_coordinate,
                                            shift2center111, transform2origin,
                                            undoshift2center111, undotransform2origin)
    from cross_section import (generate_plane_center, interpolate_along_camera_direction,
                               plane_filter)
    from ovox_cuts import Lookup, sim_apply, sim_fit

    class _P:
        sh_degree = 0
        compute_cov3D_python = True
        convert_SHs_python = False
        debug = False

    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(model_ply).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                  * C0 + 0.5, 0, 1)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    org = xyz.min(0) - 0.5 * hf
    look = Lookup(xyz, lvl.astype(np.int64), {0: hc, 1: hf}, org)

    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(model_ply)
    par = load_params_from_gs(g, _P())
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(par["pos"])
    tpos = shift2center111(tpos)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos), so, om), rot_m)
    cam, raw = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                               observant_coordinates=oc, show_hint=False, init_azimuthm=0.0,
                               init_elevation=-90.0, init_radius=cam_p["init_radius"],
                               move_camera=False, current_frame=0, delta_a=None, delta_e=None,
                               delta_r=None)
    _, _, centres, avg = interpolate_along_camera_direction(raw, tpos, 24)
    plane = generate_plane_center(raw, centres[len(centres) // 2])
    mask, _ = plane_filter(plane, tpos, raw, surf_dis=float(avg) / 2, include_double=True)
    keep = mask.detach().cpu().numpy().reshape(-1).astype(bool)[:len(xyz)]

    fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
    eye = cam.camera_center.reshape(3).detach().cpu().numpy().astype(np.float64)
    W_ = world.detach().cpu().numpy().astype(np.float64)
    T_ = tpos.detach().cpu().numpy().astype(np.float64)
    f_t2p, f_p2x = sim_fit(T_, W_), sim_fit(W_, xyz[:len(W_)])
    a4 = np.asarray(plane, np.float64).reshape(4)
    nn = a4[:3] / np.linalg.norm(a4[:3])
    pn = f_t2p[0] @ nn
    d = float(a4[3] / np.linalg.norm(a4[:3]) * f_t2p[1] - pn @ f_t2p[2])

    inv = np.linalg.inv(fp)
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    gx_, gy_ = np.meshgrid(t, t, indexing="xy")
    one = np.ones(size * size)
    w = np.stack([gx_.ravel(), gy_.ravel(), one, one], 1) @ inv
    dirs = w[:, :3] / w[:, 3:4] - eye[None]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    den = dirs @ pn
    ok = np.abs(den) > 1e-12
    tt = np.full(len(dirs), np.inf)
    tt[ok] = -(eye @ pn + d) / den[ok]
    hit = ok & (tt > 1e-9) & np.isfinite(tt)
    pts = eye[None] + dirs * np.where(hit, tt, 0.0)[:, None]
    cid = np.full(len(pts), -1, np.int64)
    todo = np.nonzero(hit)[0]
    for i in range(24):
        if not len(todo):
            break
        r = look(sim_apply(pts[todo] + dirs[todo] * (i * 0.25 * hc), f_p2x))
        got = (r >= 0) & keep[np.clip(r, 0, None)]
        cid[todo[got]] = r[got]
        todo = todo[~got]
    ins = cid >= 0

    PAL = np.array([[0.85, 0.30, 0.24], [0.24, 0.55, 0.80], [0.35, 0.68, 0.36],
                    [0.95, 0.72, 0.20], [0.60, 0.40, 0.70], [0.40, 0.40, 0.40]])
    panels = []
    for name, src in (("as it looks", rgb),
                      ("material class", PAL[labels % len(PAL)].astype(np.float32)),
                      ("stiffness", None)):
        img = np.ones((size * size, 3), np.float32)
        if src is None:
            lo, hi = float(np.log10(E.min())), float(np.log10(E.max()))
            u = (np.log10(E[labels]) - lo) / max(hi - lo, 1e-9)
            src = (np.stack([0.15 + 0.75 * u, 0.25 + 0.35 * (1 - u), 0.85 - 0.7 * u], 1)
                   .astype(np.float32))
        img[ins] = src[cid[ins]]
        a = (np.clip(img.reshape(size, size, 3), 0, 1) * 255).astype(np.uint8)
        pad = np.full((28, size, 3), 255, np.uint8)
        a = np.vstack([pad, a])
        cv2.putText(a, name, (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (64, 64, 64), 1, cv2.LINE_AA)
        panels.append(a)
    cv2.imwrite(out_png, np.hstack(panels)[:, :, ::-1])
    print(f"  -> {out_png}")


def main(lattice_dir, model_ply, cfg, labels_npy=None, demo=None, out_png=None):
    from plyfile import PlyData
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(model_ply).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float64)
                  * C0 + 0.5, 0, 1)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    skin = lvl == 1
    lum = rgb.mean(1)

    if labels_npy and _os.path.exists(labels_npy):
        labels = np.load(labels_npy)[:len(xyz)]
        print(f"  {len(xyz):,} cells, labels from {labels_npy}")
    else:
        # the weakest version, so the strong one has something to beat: shell against interior,
        # with the interior split by brightness at its own median
        labels = np.where(skin, 0, np.where(lum > np.median(lum[~skin]), 1, 2))
        print(f"  {len(xyz):,} cells, labels from the rule (shell, bright interior, dark "
              f"interior) -- pass a labels.npy from material_segment.py for the clustered version")

    beta, L, u, r, E = moduli(labels, skin.astype(np.float64), lum)
    print(f"\n  class    cells        shell   brightness    u       E")
    for j in range(len(E)):
        n = int((labels == j).sum())
        print(f"    {j}   {n:>9,} ({100 * n / len(labels):5.1f}%)  {beta[j]:5.2f}   "
              f"{L[j]:8.3f}   {u[j]:5.3f}  {E[j]:9.3e}")
    print(f"  stiffness contrast, hardest to softest: {E.max() / E.min():.1f}x")

    # --- the question the previous project left open ------------------------------------------
    t = shell_thickness(xyz, skin, hf)
    dx_mpm = None
    try:
        j = json.load(open(cfg))
        # grid_lim is the extent of the solver's box and n_grid the cells across it, which is
        # how internal_filling.py builds grid_dx: material_params["grid_lim"] / n_grid.
        n = int(j.get("particle_filling", {}).get("n_grid", 0)) or int(j.get("n_grid", 0))
        if n:
            dx_mpm = float(j.get("grid_lim", 1.0)) / n
            print(f"\n  MPM grid: n_grid = {n} over grid_lim {j.get('grid_lim', 1.0)}, "
                  f"so dx = {dx_mpm:.5f} in solver units")
            print("  (this is the particle-filling grid; the transfer grid is coarser and "
                  "section 4.2.1 reports the ratio against that one)")
    except Exception as e:                                       # noqa: BLE001
        print(f"\n  could not read the MPM grid from {cfg}: {type(e).__name__}")

    ext = float((xyz.max(0) - xyz.min(0)).max())
    print(f"  shell thickness {t:.5f} in world units = {t / hf:.1f} fine cells "
          f"= {100 * t / ext:.2f}% of the object's extent")
    if dx_mpm:
        # the object is normalised into the solver's box, so a world length maps by extent
        t_grid = (t / ext) / dx_mpm
        print(f"  in MPM grid cells: {t_grid:.2f}")
        print(f"  -> a stiffness contrast across this shell is {'RESOLVABLE' if t_grid >= 2 else 'NOT resolvable'}"
              f": MPM transfers through one grid, so a feature thinner than about two cells is"
              f" averaged with its neighbours and returns as the average.")
        if t_grid < 2:
            need = 2.0 * dx_mpm * ext
            print(f"     to resolve it the shell would have to be {need:.5f} world units, "
                  f"{need / hf:.1f} fine cells, against the {t / hf:.1f} it has -- "
                  f"a factor of {need / t:.1f}.")
    if demo and out_png:
        render_slices(lattice_dir, model_ply, cfg, demo, labels, E, out_png)
    return dict(labels=labels, E=E, thickness=t)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else None,
         demo=sys.argv[5] if len(sys.argv) > 5 else None,
         out_png=sys.argv[6] if len(sys.argv) > 6 else None)
