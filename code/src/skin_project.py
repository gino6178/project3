"""Six views onto the shell, by the camera that made them.

`init_skin_cube` maps a cell to a reference pixel through a cone: take the direction from the
centre, turn it into a fraction of the reference's silhouette radius, and read there. It has to,
because a generated reference has no camera -- a diffusion model was asked for "the side of an
orange" and the result is an orange seen from somewhere. Everything that file says about
`SKIN_REF_R`, the saturating map and the rim light is the cost of not knowing where the picture
was taken from.

When the six views are renders, that cost is unnecessary. The camera is known exactly, so the
texture coordinate of a cell is its projection through that camera, and nothing has to be
inferred: no radius, no cone, no saturating map, no rim compensation. Two adjacent faces then
agree wherever they overlap, because both are exact projections of the same object, and the seams
that a cone mapping leaves -- pale chevrons where `up` meets the sides, bright bands down the
middle of each side view -- have nothing to arise from.

A cell still belongs to the face that sees it most squarely, which is the same rule and the same
reason: an oblique view of a surface is a stretched, dim reading of it. What changes is only how
the reading is addressed.

    python method/common/pipeline/skin_project.py LATTICE CFG DEMO REF_DIR OUT_DIR
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import json
import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]

DEV = "cuda:0"
C0 = 0.28209479177387814


class P:
    sh_degree = 0
    compute_cov3D_python = True
    convert_SHs_python = False
    debug = False


def main(lattice_dir, cfg, demo, ref_dir, out_dir, size=512):
    import shutil
    import cv2
    from scene.gaussian_model import GaussianModel
    from utils.camera_view_utils import get_camera_view
    from utils.decode_param import decode_param_json
    from utils.render_utils import initialize_resterize, load_params_from_gs
    from utils.transformation_utils import (apply_cov_rotations, apply_inverse_cov_rotations,
                                            apply_inverse_rotations,
                                            generate_rotation_matrices,
                                            get_center_view_worldspace_and_observant_coordinate,
                                            shift2center111, transform2origin,
                                            undoshift2center111, undotransform2origin)

    _os.makedirs(out_dir, exist_ok=True)
    dirs = [(n, v[0], v[1]) for n, v in
            json.load(open(_os.path.join(ref_dir, "dirs.json"))).items()]
    refs = {n: cv2.imread(_os.path.join(ref_dir, f"{n}_ref.png"))[:, :, ::-1].astype(np.float32)
            / 255.0 for n, _, _ in dirs}

    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(_os.path.join(lattice_dir, "gs_fill.ply"))
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).to(DEV)
    par = load_params_from_gs(g, P())
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(par["pos"])
    tpos = shift2center111(tpos)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)

    n_all = world.shape[0]
    lvl = lvl[:n_all]
    # What gets painted is what the trainer will pin, from the same function, so the two cannot
    # disagree. `lvl != 0` was a sphere of radius `skin_frac * R`, and on a shape that is not a
    # sphere it leaves part of the actual surface unpainted -- which then gets pinned to whatever
    # colour it happened to carry. Only the level's own cells and the shape's boundary differ
    # here; nothing about the projection changes.
    from occupancy import surface_cells
    lat_dx = float(torch.load(_os.path.join(lattice_dir, "lattice.pt"))["coarse_dx"])
    shell = (lvl != 0) | surface_cells(world, lat_dx)[:n_all]
    centre = world.mean(0)
    # init_skin_cube's normals, not the direction from the centroid. `world - centre` is a normal
    # only for a star-shaped object; on a torus a cell on top of the tube sits a ring radius out
    # from the axis and a tube radius up from it, so the direction from the centre is nearly
    # horizontal and `up` weighs about 0.3 against the side views' larger values. The icing then
    # gets smeared onto the sides -- which is exactly what it did here, and exactly the failure
    # this project already diagnosed once today in paint_skin_seq. That file solved it by taking
    # the normal from the occupancy and blending it with the analytic one where they agree; there
    # is no reason to derive it a second time.
    lat_pt = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    npath = _os.path.join(lattice_dir, "cell_normal.pt")
    if _os.path.exists(npath):
        nrm = torch.load(npath).to(DEV)[:n_all].float()
        print(f"  normals: exact, from the field that generated the shape")
    else:
        import normals as isc
        nrm = isc.surface_normals(world, world, float(lat_pt["coarse_dx"]), centre)
        print(f"  normals: inferred from the occupancy (no cell_normal.pt)")

    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
    # Blended, not winner-takes-all. A hard assignment puts a discontinuity wherever two cones
    # meet, and it shows: pale chevrons where `up` meets the sides, a bright band down the middle
    # of each side view. Blending was the wrong answer for *generated* references -- six pictures
    # of six different oranges have their detail in different places and averaging cancels it --
    # but these six are exact projections of one object, so they already agree where they overlap
    # and the average has nothing to cancel. The weight is how squarely a face sees the cell,
    # raised to a power so it stays local, and the sum is normalised.
    POW = float(_os.environ.get("SKIN_BLEND_POW", "4"))
    # Every view's reading is kept, so a specular highlight can be recognised for what it is: it
    # sits where one camera happens to see the light and moves when the camera does, so among the
    # views that see a cell it is the odd one out in luminance. Rejecting the outlier is general
    # -- no shape, no light position, no material model -- and it is the only thing that removes
    # a highlight without also removing the peel it sits on.
    # Off by default, because measured it costs more than it buys. At a threshold of 0.12 it
    # rejected 186,144 of the orange's 921,320 readings -- 20% -- and the threshold is absolute
    # luminance applied to every view including the one facing the cell squarely, so a cell whose
    # best reading is legitimately bright loses it and takes an oblique, darker, blurrier one
    # instead. Two small highlights were replaced by dark bands and a dark seam down the front.
    # A rejection that only fires with real evidence -- three or more views, and only the single
    # brightest, and only when it is not also the squarest -- would be the version worth having.
    REJ = float(_os.environ.get("SKIN_SPEC_REJECT", "0"))
    cols = torch.zeros(len(dirs), n_all, 3, device=DEV)
    wts = torch.zeros(len(dirs), n_all, device=DEV)
    acc = torch.zeros(n_all, 3, device=DEV)
    wsum = torch.zeros(n_all, device=DEV)
    took = torch.zeros(n_all, dtype=torch.bool, device=DEV)
    best = torch.full((n_all,), -2.0, device=DEV)
    face = torch.full((n_all,), -1, dtype=torch.int8, device=DEV)

    for k, (name, az, el) in enumerate(dirs):
        cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                 observant_coordinates=oc, show_hint=False, init_azimuthm=az,
                                 init_elevation=el, init_radius=cam_p["init_radius"],
                                 move_camera=False, current_frame=0, delta_a=None,
                                 delta_e=None, delta_r=None)
        axis = cam.camera_center.reshape(3).to(DEV) - centre
        axis = axis / axis.norm().clamp_min(1e-9)
        facing = nrm @ axis

        # the reference was reframed to fill its own frame, so the projection has to be put in
        # the same units: measure the object's silhouette in this camera and rescale about its
        # centre. Everything else is the camera's own projection, unaltered.
        hom = torch.cat([world, torch.ones(n_all, 1, device=DEV)], 1)
        clip = hom @ cam.full_proj_transform
        infront = clip[:, 3] > 1e-6
        ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-6)
        c4 = torch.cat([centre, torch.ones(1, device=DEV)])[None] @ cam.full_proj_transform
        ndc_c = (c4[:, :2] / c4[:, 3:4])[0]
        # Per axis, not one radius. The reference was stretched to fill its frame in both
        # directions, so the projection has to be normalised the same way or the two use
        # different conventions -- which is invisible on a sphere, where they agree, and ruinous
        # on the doughnut, whose front silhouette is 1.41 wide by 0.37 tall: the reference gets
        # stretched almost four times in one axis and the projection not at all, and the icing
        # ends up smeared down the sides.
        # One radius, isotropic, to match a reference that kept its aspect. This is only correct
        # because the generated shape has the object's own proportions -- an ellipsoid with its
        # extents, a torus with its tube -- so the two silhouettes are the same shape and there is
        # nothing left to stretch.
        vis = infront & shell & (facing > 0)
        d_ndc = ndc[vis] - ndc_c[None]
        r_sil = torch.stack([d_ndc[:, 0].abs().max(), d_ndc[:, 1].abs().max()]).max().clamp_min(1e-6)
        uv = (ndc - ndc_c[None]) / r_sil

        H, W = refs[name].shape[:2]
        px = ((uv[:, 0] * 0.5 + 0.5) * (W - 1)).round().long().clamp(0, W - 1)
        # No flip on y, because the renderer that wrote these references does not flip:
        # ndc2Pix(v, S) = ((v + 1) S - 1) / 2 on both axes. Flipping one axis and not the
        # other textured every shell with its four side views upside down -- the icing band
        # low on the doughnut's tube with the drips pointing up -- and it survived several
        # readings because `up` and `down` are near-symmetric under a vertical flip.
        py = ((uv[:, 1] * 0.5 + 0.5) * (H - 1)).round().long().clamp(0, H - 1)
        ref = torch.from_numpy(refs[name].copy()).to(DEV)
        col = ref[py, px]

        seen = shell & infront & (facing > 0.0)
        w = torch.where(seen, facing.clamp_min(0.0) ** POW, torch.zeros_like(facing))
        cols[k] = col
        wts[k] = w
        took |= seen
        better = seen & (facing > best)
        best = torch.where(better, facing, best)
        face = torch.where(better, torch.full_like(face, k), face)
        print(f"  {name:<6} az{az:>4} el{el:>4}   sees {int(seen.sum()):>8,} cells   "
              f"silhouette radius {float(r_sil):.4f} in ndc")

    # the median luminance over the views that actually see a cell, and anything far above it is
    # a highlight rather than the surface
    lum = cols.mean(2)
    seen_any = wts > 1e-8
    big = lum.masked_fill(~seen_any, float("nan"))
    med = big.nanmedian(dim=0).values
    keep = seen_any if REJ <= 0 else (seen_any & ~(lum > med[None] + REJ))
    n_rej = int((seen_any & ~keep).sum())
    wts = torch.where(keep, wts, torch.zeros_like(wts))
    acc = (wts[..., None] * cols).sum(0)
    wsum = wts.sum(0)
    print(f"  {n_rej:,} readings rejected as specular (of {int(seen_any.sum()):,})")
    m = wsum > 1e-8
    rgb = torch.where(m[:, None], acc / wsum.clamp_min(1e-8)[:, None], rgb)
    n_shell = int(shell.sum())
    print(f"  shell {n_shell:,} cells, {int(took.sum()):,} took a colour "
          f"({100 * int(took.sum()) / max(n_shell, 1):.1f}%)")
    with torch.no_grad():
        g._features_dc.copy_(((rgb - 0.5) / C0).unsqueeze(1).to(g._features_dc.device))
    g.save_ply(_os.path.join(out_dir, "gs_fill.ply"))
    for f in ("cell_level.pt", "lattice.pt", "is_interior.pt"):
        s = _os.path.join(lattice_dir, f)
        if _os.path.exists(s):
            shutil.copy(s, _os.path.join(out_dir, f))
    torch.save(face.cpu(), _os.path.join(out_dir, "cell_face.pt"))
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
