"""Is enlarging the Gaussians enough? -- the specification's second required baseline.

Section 11.1 asks for four arms, and this is the one the paper's premise stands or falls on:
*only make the Gaussians bigger -- is that sufficient, and does it cause material blending?* The
paper asserts the trade in its first equation,

    rho = sigma / h,   rho << 1 -> holes in the volume,   rho >~ 1 -> blending across boundaries

and then never measures it. Asserting a trade-off is not the same as showing that no setting of
its parameter escapes it, and a reviewer who reads equation (1) will ask for exactly this curve.

Both arms of the trade are measured against the same ground truth, which the cube representation
supplies for free: the specification's own equation (12), a piecewise-constant lookup that reads
the cell containing each pixel and outputs its colour. That answer has no scale parameter, so it
is the reference both failure modes are measured as departures from.

    holes      pixels inside the true section that the Gaussian render left as background
    blending   how much the Gaussian render disagrees with the lookup *at material boundaries*
               beyond how much it disagrees away from them

The second needs the subtraction. A large Gaussian disagrees with a piecewise-constant reference
everywhere, simply by being smooth; what matters is whether it disagrees *more* where two
materials meet, because that is what blending means and that is what the interior is made of.

    python method/common/eval/rho_sweep.py LATTICE MODEL.ply CFG DEMO OUT_DIR
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

DEV = "cuda:0"
C0 = 0.28209479177387814
RHOS = [float(v) for v in _os.environ.get("RHOS", "0.25,0.5,0.75,1.0,1.5,2.0,3.0").split(",")]


class P:
    sh_degree = 0
    compute_cov3D_python = True
    convert_SHs_python = False
    debug = False


def boundary_mask(img, px_per_cell, thresh=0.16):
    """Where two *materials* meet, not where two cells do.

    The reference is piecewise constant per cell, so every cell edge is a step and a plain
    gradient threshold calls 84% of the section a boundary -- which makes the comparison
    meaningless, since there is then nothing to compare it against. Blurring by a cell first
    removes the cell-scale steps and leaves the ones that span several cells, which is what a
    material boundary is: peel to pith, pith to pulp, pulp to seed.
    """
    import cv2
    g = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, px_per_cell))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    m = np.hypot(gx, gy) > thresh
    k = int(max(3, 2 * round(px_per_cell) + 1))
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def main(lattice_dir, model_ply, cfg, demo, out_dir, size=None):
    import cv2
    from plyfile import PlyData
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
    from ovox_cuts import Lookup, sim_apply, sim_fit, sim_inv

    _os.makedirs(out_dir, exist_ok=True)
    # Both arms need a resolution axis, and the specification asks for one anyway (11.2,
    # "resolution stability: background-gap ratio at 512, 1024, 2048, 4096"). Holes are a
    # sub-pixel phenomenon: at 512 a coarse cell is about four pixels across, so even rho = 0.25
    # covers, and the trade the paper asserts is invisible. The two experiments are one.
    SIZES = [int(v) for v in _os.environ.get("SIZES", "512,1024,2048").split(",")]
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
    par = load_params_from_gs(g, P())
    pos0 = par["pos"]
    sp, op, shs = par["screen_points"], par["opacity"], par["shs"]
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(pos0)
    tpos = shift2center111(tpos)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos), so, om), rot_m)

    cam, raw = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                               observant_coordinates=oc, show_hint=False, init_azimuthm=0.0,
                               init_elevation=float(_os.environ.get("CUT_EL", "-90")),
                               init_radius=cam_p["init_radius"], move_camera=False,
                               current_frame=0, delta_a=None, delta_e=None, delta_r=None)

    # one plane through the middle, the same one for every arm
    from cross_section import (generate_plane_center, interpolate_along_camera_direction,
                               plane_filter)
    _, _, centres, avg = interpolate_along_camera_direction(raw, tpos, 24)
    plane = generate_plane_center(raw, centres[len(centres) // 2])
    mask, mask_suf = plane_filter(plane, tpos, raw, surf_dis=float(avg) / 2, include_double=True)

    fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
    w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
    eye = cam.camera_center.reshape(3).detach().cpu().numpy().astype(np.float64)
    W_ = world.detach().cpu().numpy().astype(np.float64)
    T_ = tpos.detach().cpu().numpy().astype(np.float64)
    f_t2p = sim_fit(T_, W_)
    f_p2x = sim_fit(W_, xyz[:len(W_)])
    a4 = np.asarray(plane, np.float64).reshape(4)
    nn = a4[:3] / np.linalg.norm(a4[:3])
    pn = f_t2p[0] @ nn
    d = float(a4[3] / np.linalg.norm(a4[:3]) * f_t2p[1] - pn @ f_t2p[2])
    keep = mask.detach().cpu().numpy().reshape(-1).astype(bool)[:len(xyz)]
    bg = torch.tensor([1., 1., 1.], device=DEV)

    def lookup_reference(size):
        """The specification's equation (12): the cell containing each pixel, and its colour.

        No scale parameter anywhere in it, which is the point -- it is the answer both failure
        modes are measured as departures from.
        """
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
        ref = np.ones((size * size, 3), np.float32)
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
        ref[ins] = rgb[cid[ins]]
        return ref.reshape(size, size, 3), ins.reshape(size, size)

    def gaussian_render(size, rho):
        with torch.no_grad():
            g._scaling.copy_(torch.full_like(g._scaling, float(np.log(rho * hc))))
        par2 = load_params_from_gs(g, P())
        cov0 = apply_cov_rotations(par2["cov3D_precomp"], rot_m) * (so * so)
        cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
        rast = initialize_resterize(cam, g, P(), bg, image_height=size, image_width=size)
        col = (par2["shs"][:, 0, :] * C0 + 0.5).clamp(0, 1)
        img, _, _, _ = rast(means3D=world[mask_suf], means2D=sp[mask_suf], shs=None,
                            colors_precomp=col[mask_suf].contiguous(), opacities=op[mask_suf],
                            scales=None, rotations=None, cov3D_precomp=cov[mask_suf])
        return img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()

    rows = []
    for size in SIZES:
        ref, inside = lookup_reference(size)
        # how many pixels a coarse cell spans, from the section's own extent
        ys, xs = np.nonzero(inside)
        span = max(ys.ptp(), xs.ptp()) + 1
        extent = float(np.abs(xyz[keep] @ pn + d).max() * 2) if keep.any() else 1.0
        px_per_cell = max(1.0, span * hc / max(extent, 1e-9))
        bnd = boundary_mask(ref, px_per_cell) & inside
        far = inside & ~bnd
        cv2.imwrite(_os.path.join(out_dir, f"reference_{size}.png"),
                    (ref[:, :, ::-1] * 255).astype(np.uint8))
        print(f"\n  {size}px: section {inside.sum():,} px, a coarse cell spans "
              f"{px_per_cell:.1f} px, material boundaries {100 * bnd.sum() / inside.sum():.1f}% "
              f"of the section")
        for rho in RHOS:
            a = gaussian_render(size, rho)
            if size == SIZES[0]:
                cv2.imwrite(_os.path.join(out_dir, f"rho_{rho:.2f}_{size}.png"),
                            (a[:, :, ::-1] * 255).astype(np.uint8))
            holes = inside & (a.min(2) > 0.97)
            err = np.abs(a - ref).mean(2)
            e_b = float(err[bnd].mean()) if bnd.any() else float("nan")
            e_f = float(err[far].mean()) if far.any() else float("nan")
            rows.append((size, rho, 100 * holes.sum() / inside.sum(), e_f, e_b, e_b - e_f))
            print(f"    rho {rho:4.2f}  holes {rows[-1][2]:6.3f}%   away {e_f:.4f}  "
                  f"at a boundary {e_b:.4f}   blending excess {rows[-1][5]:+.4f}")

    print("\n  size   rho   holes%    away    at bdy   excess")
    for r in rows:
        print(f"  {r[0]:5d} {r[1]:5.2f} {r[2]:7.3f}  {r[3]:.4f}  {r[4]:.4f}  {r[5]:+.4f}")
    np.savetxt(_os.path.join(out_dir, "sweep.csv"), np.array(rows), delimiter=",",
               header="size,rho,holes_pct,err_away,err_boundary,blending_excess")
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:6])
