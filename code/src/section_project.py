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


def photo_for(spec, idx, n):
    """The photograph training would use for plane idx of n, with its own disc.

    The disc matters and getting it wrong is the whole difference between a section and a
    smear. A reference is a hand-framed photograph: its section sits somewhere in the frame at
    some radius, surrounded by white. Mapping the object's silhouette onto the *frame* puts the
    section's edge somewhere inside the object and white everywhere beyond it -- which is
    exactly what the first version of this file did, and it showed up as a shrunken pomegranate
    with grey where no photograph reached. `_disc` is the function sds_demo uses to find that
    circle, so the mapping here and the mapping training uses agree by construction.
    """
    sds_demo._PLANE["idx"], sds_demo._PLANE["n"] = idx, n
    im = (sds_demo._solved_photo(spec) if sds_demo.REF_PHASE_MODE == "solve"
          else sds_demo._photo(spec)).convert("RGB")
    cy, cx, r = sds_demo._disc(im)
    return np.asarray(im, dtype=np.float32) / 255., (cy, cx, max(r, 1e-6))


def sweep(world, cam_of, spec, n_planes, size, tag, rgb0, lvl):
    """One family: colour every cell from the photograph of the plane nearest to it.

    The photograph is placed on the section by `section_match.section_target` -- the same
    function, on the same kind of silhouette, that places it during training. This matters more
    than it looks. That function warps the reference per component along its own ray
    coordinate, so a lobed section is matched shape to shape; a scalar radius, which is what
    this file used first, matches only a circle. Initialising with one mapping and supervising
    with another puts every structure at a radius the loss then asks to move, and training
    spends itself undoing the start it was given.

    The silhouette here is the cells themselves, splatted to pixels, rather than a rasteriser
    render: it is the same set of primitives the renderer would draw, and it keeps this file
    free of the Gaussian pipeline.

    Returns (colour, weight). The weight is a hat over the plane spacing, so a cell between two
    planes takes a mixture rather than a side, which is equation (14)'s idea applied in space
    instead of in the plane index.
    """
    import section_match as sm
    N = world.shape[0]
    col = torch.zeros(N, 3, device=DEV)
    wsum = torch.zeros(N, 1, device=DEV)
    for j in range(n_planes):
        cam, plane, centre_ndc, r_sil, ndc, dist = cam_of(j)
        arr, _ = photo_for(spec, j, n_planes)
        uv = (ndc - centre_ndc[None]) / r_sil
        px = ((uv[:, 0] * .5 + .5) * (size - 1)).round().long().clamp(0, size - 1)
        py = ((uv[:, 1] * .5 + .5) * (size - 1)).round().long().clamp(0, size - 1)
        near = dist < 0.5
        if int(near.sum()) < 64:
            continue
        # The section as this plane's own cells draw it -- and it has to be a *filled*
        # silhouette, not the scatter of pixels the cell centres land on. A rasteriser draws
        # each cell at its own footprint; projecting centres alone leaves a dot screen, whose
        # connected components are all smaller than section_target's minimum and are therefore
        # all skipped, and the target comes back as the render it was handed. Measured: that
        # path returned a grey volume of spread 0.045 where the correct one returns 0.18.
        # Closing the scatter to the footprint the cells actually have is what the renderer
        # does, and it costs one dilation.
        from scipy import ndimage as _nd
        img = torch.ones(size, size, 3, device=DEV)
        cov = torch.zeros(size, size, device=DEV)
        flat = py[near] * size + px[near]
        img.view(-1, 3).index_copy_(0, flat, rgb0[near])
        cov.view(-1).index_fill_(0, flat, 1.0)
        cn = cov.detach().cpu().numpy() > 0.5
        k = max(1, int(round(0.004 * size)))
        cn = _nd.binary_fill_holes(_nd.binary_closing(cn, np.ones((2 * k + 1,) * 2)))
        im_np = img.detach().cpu().numpy()
        # carry each filled pixel a colour, so the render handed to section_target is a section
        # and not a stencil; the mapped components overwrite it anyway
        idx = _nd.distance_transform_edt(~(cov.detach().cpu().numpy() > 0.5),
                                         return_distances=False, return_indices=True)
        im_np = im_np[idx[0], idx[1]]
        im_np[~cn] = 1.0
        cov = torch.from_numpy(cn.astype(np.float32)).to(DEV)
        img = torch.from_numpy(im_np).to(DEV)
        tgt = sm.section_target(img.permute(2, 0, 1), arr, alpha=cov[None])
        t = tgt.permute(1, 2, 0)
        c = t[py, px]
        keep = near & (cov[py, px] > 0.5) & (c.mean(1) < 0.98)
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
    rgb0 = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
    ch, wh = sweep(world, maker(lambda j: 0.0, float(_os.environ.get("CUT_EL", "-90")),
                                H_HI - H_LO, lambda j: H_LO + j),
                   ref_h, H_HI - H_LO, size, "transverse", rgb0, lvl)
    cv_, wv = sweep(world, maker(lambda j: (180.0 / NV) * j, 0.0, NV, lambda j: 12),
                    ref_v, NV, size, "longitudinal", rgb0, lvl)

    col = (ch + cv_) / (wh + wv).clamp(min=1e-6)
    got = ((wh + wv)[:, 0] > 1e-6) & (lvl == 0)
    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
    rgb[got] = col[got]

    # The cells no plane reached. They are at the periphery, where the sweep runs out of
    # photograph before it runs out of object, and leaving them at 0.5 puts grey blocks in a
    # pomegranate -- which is worse than an approximate colour, because grey is not a colour
    # anything in this object has. Each takes its nearest reached neighbour, which is the same
    # rule voxel_smooth_anchors applies during training and is applied here so the starting
    # point is not something training has to undo first.
    miss = (lvl == 0) & ~got
    if int(miss.sum()):
        from scipy.spatial import cKDTree
        src = world[got].detach().cpu().numpy()
        dst = world[miss].detach().cpu().numpy()
        idx = cKDTree(src).query(dst, k=1)[1]
        rgb[miss] = rgb[got][torch.as_tensor(idx, device=DEV)]
        print(f"  {int(miss.sum()):,} cells no plane reached, filled from their nearest "
              f"neighbour that one did")
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
