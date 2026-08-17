"""M6: the Gaussian exterior and the cut surface in one image, chosen per pixel by depth.

The spec's section 7.1, equations (26)-(27). Two renderers coexist and neither is converted into
the other: the exterior stays the fine Gaussian model it already was, the newly exposed
cross-section is the cut surface, and each pixel takes whichever is nearer. Section 7.2 is the
reason not to turn the cut face back into Gaussians -- a flat face made of alpha-blended
ellipsoids is the coverage-versus-overlap trade the whole spec is trying to leave behind.

The cut surface is rasterised analytically rather than through triangles, and for this
representation that is exact rather than a shortcut. Every cut patch in the first version is
planar by construction (6.1), so a camera ray meets it at one point solved in closed form, and a
pixel is covered exactly when that point lies in a cut leaf belonging to the piece -- which is
the union of M4's polygons, cell for cell. A general O-Voxel surface would need the mesh path;
a planar one does not, and this way there is no rasterisation error to account for.

The convention has to be measured, because the spec warns that the Gaussian renderer's depth may
not be metric and it turns out not to be the obvious thing. `probe_depth_convention` renders the
piece and compares the returned buffer against both candidates computed from the primitives
themselves, so the answer is read off the renderer rather than assumed about it.

    python method/common/cube/composite.py LATTICE [OUT.png]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import cutmesh as cm                        # noqa: E402
from method.common.cube import subdivide as sd                      # noqa: E402

DEV = "cuda:0"


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False
    sh_degree = 0


def _index_rows(query, table):
    """Row of `table` matching each row of `query`, or -1."""
    mn = int(min(query.min(), table.min())) - 2
    span = int(max(query.max(), table.max()) + (-mn) + 3)
    k = sd._pack(table, -mn, span)
    o = np.argsort(k)
    ks, idx = k[o], np.arange(len(table))[o]
    kk = sd._pack(query, -mn, span)
    inr = ((query - mn) >= 0).all(1) & ((query - mn) < span).all(1)
    pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
    out = np.full(len(query), -1, np.int64)
    hit = inr & (ks[pos] == kk)
    out[hit] = idx[pos[hit]]
    return out


def leaf_of(pts, r, h):
    """Which leaf contains each point, or -1. The finest level wins, as in M3's adjacency."""
    out = np.full(len(pts), -1, np.int64)
    for L in sorted({int(x) for x in r["level"]}, reverse=True):
        m = r["level"] == L
        if not m.any():
            continue
        hl = h / (2.0 ** L)
        c = np.floor(pts / hl).astype(np.int64)
        mn = int(min(r["leaf"][m].min(), c.min())) - 2
        span = int(max(r["leaf"][m].max(), c.max()) + (-mn) + 3)
        keys = sd._pack(r["leaf"][m], -mn, span)
        o = np.argsort(keys)
        ks, idx = keys[o], np.nonzero(m)[0][o]
        # in range first: the key is a polynomial in the coordinates, so a point outside the
        # table's range aliases onto a valid key and invents material that is not there
        inr = ((c - mn) >= 0).all(1) & ((c - mn) < span).all(1)
        kk = sd._pack(np.where(inr[:, None], c, mn), -mn, span)
        pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
        got = inr & (ks[pos] == kk) & (out < 0)
        out[got] = idx[pos[got]]
    return out



def fit_similarity(A, B, n=20000):
    """The rotation, scale and translation taking A to B, from corresponding points.

    The lattice is axis aligned in the ply's frame and the renderer works in another one, and
    the map between them is a rotation, a uniform scale and a shift -- assembled in the pipeline
    out of transform2origin, shift2center111 and the rotation matrices. Rather than invert that
    chain by hand, fit it: both point sets are in hand and in the same order.

    Quantising in the wrong frame is not a small error. The rotation alone, with the scale at
    exactly 1, turned one cut of the orange into three pieces, one of them a single cell holding
    a single primitive, because an axis-aligned grid laid over a rotated lattice cuts across its
    cells instead of along them.
    """
    k = min(n, len(A))
    idx = np.linspace(0, len(A) - 1, k).astype(np.int64)
    a, b = A[idx], B[idx]
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float(S[:2].sum() + d * S[2]) / float(((a - ca) ** 2).sum())
    t = cb - s * (R @ ca)
    err = float(np.abs((s * (a @ R.T) + t) - b).max())
    return R, s, t, err


def cut_surface_buffers(cam, size, r, h, n, d, piece, cell_rgb, parent_of_leaf,
                        R, sc, tr, org, fwd, convention):
    """C_O and D_O: the cut face, by ray-plane intersection in the lattice's own frame.

    A similarity maps lines to lines, so the camera ray is carried into the lattice's frame,
    intersected there in closed form, and the hit point carried back to measure depth in
    whichever convention the Gaussian renderer was found to use.
    """
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    eye_w = cam.camera_center.reshape(3).detach().cpu().numpy().astype(np.float64)

    inv = torch.inverse(cam.full_proj_transform).detach().cpu().numpy().astype(np.float64)
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    gx, gy = np.meshgrid(t, t, indexing="xy")
    ndc = np.stack([gx.ravel(), gy.ravel(), np.ones(size * size), np.ones(size * size)], 1)
    w = ndc @ inv
    far = w[:, :3] / w[:, 3:4]
    dir_w = far - eye_w[None]
    dir_w /= np.linalg.norm(dir_w, axis=1, keepdims=True)

    # world -> lattice frame: x_l = R^T (x_w - t) / s
    Rt = R.T
    eye_l = (eye_w - tr) @ Rt.T / sc
    dir_l = dir_w @ Rt.T
    dir_l /= np.linalg.norm(dir_l, axis=1, keepdims=True)

    denom = dir_l @ n
    ok = np.abs(denom) > 1e-12
    tt = np.full(len(dir_l), np.inf)
    tt[ok] = -(eye_l @ n + d) / denom[ok]
    hit = ok & (tt > 0)
    pts = eye_l + dir_l * np.where(np.isfinite(tt), tt, 0.0)[:, None]

    lf = np.full(len(pts), -1, np.int64)
    if hit.any():
        lf[hit] = leaf_of(pts[hit] - org, r, h)
    # Coverage is every cut leaf, not the ones whose centre happens to be on this piece's side.
    # M4 emits each polygon twice, once per piece, precisely because a cut cell's face bounds
    # both of them; filtering by the cell's own label is a different question and gives the
    # wrong answer badly. At the deepest level the crossing window is exactly one centre spacing
    # wide, so it catches one layer of centres and they all fall on the same side -- one piece
    # got the whole cut face and the other got none of it, 12.3% of the frame against 0.0%.
    # Which piece's face is *visible* is then settled by the depth test, not by this mask: from
    # the far side the piece's own body is in front of its cut face and wins.
    cov = hit & (lf >= 0)

    col = np.ones((size * size, 3), np.float32)
    if cov.any():
        col[cov] = cell_rgb[parent_of_leaf[lf[cov]]]

    # back to the render frame, and then into the renderer's own depth convention
    hw = sc * (pts @ R.T) + tr

    # the rays, checked against the projection they were built from: a hit point put back
    # through full_proj_transform must land on the pixel whose ray found it
    if cov.any():
        fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        cl = np.concatenate([hw[cov], np.ones((int(cov.sum()), 1))], 1) @ fp
        nd = cl[:, :2] / cl[:, 3:4]
        pxy = (nd + 1) * 0.5 * size - 0.5
        want = np.stack([np.tile(np.arange(size), size), np.repeat(np.arange(size), size)], 1)
        err = float(np.abs(pxy - want[cov]).max())
        print(f"      rays reproject to their own pixels to within {err:.3f} px")

    if convention == "euclidean":
        dep = np.linalg.norm(hw - eye_w[None], axis=1)
    else:
        w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
        dep = (np.concatenate([hw, np.ones((len(hw), 1))], 1) @ w2c)[:, 2]
    depth = np.where(cov, dep, np.inf)
    return (col.reshape(size, size, 3), depth.reshape(size, size), cov.reshape(size, size))


def gs_buffers(cam, size, world, cov3, sp, op, col, g, bg=(1., 1., 1.)):
    """C_G and D_G, straight from the Gaussian renderer."""
    from utils.render_utils import initialize_resterize
    rast = initialize_resterize(cam, g, P(), torch.tensor(bg, device=DEV),
                                image_height=size, image_width=size)
    img, _, dep, alp = rast(means3D=world, means2D=sp, shs=None,
                            colors_precomp=col.contiguous().clamp(0, 1), opacities=op,
                            scales=None, rotations=None, cov3D_precomp=cov3)
    return (img.permute(1, 2, 0).detach().cpu().numpy(),
            dep.reshape(size, size).detach().cpu().numpy(),
            alp.reshape(size, size).detach().cpu().numpy())


def probe_depth_convention(cam, world, dep, alp, size):
    """Is the renderer's depth Euclidean distance, or view-space z?

    Both are computed from the primitives that land in each pixel and compared against the
    buffer. The spec flags this as something to unify before compositing and does not say which
    it is; measuring beats guessing, and the two differ by up to the field of view's cosine --
    enough to put a cut face behind an exterior it is in front of, near the frame's edge.
    """
    eye = cam.camera_center.reshape(3).to(DEV)
    hom = torch.cat([world, torch.ones(world.shape[0], 1, device=DEV)], 1)
    clip = hom @ cam.full_proj_transform
    front = clip[:, 3] > 1e-6
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-6)
    px = ((ndc[:, 0] + 1) * 0.5 * size).long()
    py = ((ndc[:, 1] + 1) * 0.5 * size).long()
    ok = front & (px >= 0) & (px < size) & (py >= 0) & (py < size)

    euclid = (world - eye).norm(dim=1)
    viewz = (hom @ cam.world_view_transform)[:, 2]     # row-vector convention, as 3DGS stores it

    flat = (py * size + px)[ok]
    t = np.arange(size) - (size - 1) / 2
    rad = np.hypot(*np.meshgrid(t, t, indexing="xy")) / (size / 2)

    out = {}
    for name, val in (("euclidean", euclid[ok]), ("view z", viewz[ok])):
        buf = torch.full((size * size,), float("inf"), device=DEV)
        buf.scatter_reduce_(0, flat, val, reduce="amin", include_self=True)
        b = buf.reshape(size, size).detach().cpu().numpy()
        m = np.isfinite(b) & (alp > 0.5)
        if not m.any():
            out[name] = (float("inf"), float("inf"))
            continue
        # The offset alone cannot decide this. The renderer returns an alpha-weighted depth
        # along the ray, not the nearest surface, so both candidates sit 0.08 to 0.10 behind a
        # minimum over primitive centres and the means are too close to call. What separates
        # them is where the error sits: euclidean and view z differ by 1/cos of the angle off
        # the axis, so reading the buffer in the wrong one leaves an error that grows towards
        # the corners. The slope against radius is the discriminator; the offset is not.
        e = b[m] - dep[m]
        rr = rad[m]
        slope = float(np.polyfit(rr, e, 1)[0])
        out[name] = (float(np.abs(e).mean()), slope)
    return out


def composite(cg, dg, ag, co, do, cov):
    """Equation (27): per pixel, whichever is nearer."""
    dgm = np.where(ag > 0.5, dg, np.inf)
    dom = np.where(cov, do, np.inf)
    take_o = dom < dgm
    out = np.where(take_o[..., None], co, cg)
    both = np.isfinite(dgm) & np.isfinite(dom)
    return out, take_o, both


def ovox_buffers(npz_path, cam, size, R, sc, tr, n, d, body, w2c_np, fp_np, org):
    """The same exterior as an O-Voxel surface: colour, depth and coverage, for one piece.

    The section's claim is that a Gaussian shell is not opaque -- it covers a fifth of the frame
    at alpha above a half -- so the cut face shows through it from the side it should be hidden
    on, and that converting the exterior is what removes it. A figure that shows only the leaky
    version cannot make that claim; this renders the other half of it, from the same camera, so
    the two composites differ in exactly one thing.
    """
    from method.common.eval.ovox_cuts import _splat
    z = np.load(npz_path)
    V = z["mesh_v"].astype(np.float64)
    F = z["mesh_f"].astype(np.int64) if "mesh_f" in z.files else None
    rgb = z["rgb"].astype(np.float32)
    if F is not None:
        k = 3
        t = (np.arange(k) + 0.5) / k
        bu, bv = np.meshgrid(t, t, indexing="ij")
        m = (bu + bv) <= 1.0
        bu, bv = bu[m], bv[m]
        tv = V[F]
        pts = (tv[:, None, 0] * (1 - bu - bv)[None, :, None]
               + tv[:, None, 1] * bu[None, :, None]
               + tv[:, None, 2] * bv[None, :, None]).reshape(-1, 3)
        cols = np.repeat(rgb[F[:, 0]], len(bu), axis=0)
    else:
        pts, cols = V, rgb
    # This piece's share of the surface, by the side its material is on -- the same test the
    # composite's own prediction uses, so the two cannot disagree about which piece this is.
    #
    # In the frame the plane was written in, which is org-relative: `d` comes from cell centres
    # measured as (coords + 1/2) h, and the surface's vertices are in the ply's absolute frame.
    # Skipping the offset does not fail loudly -- it silently keeps almost nothing, and the
    # exterior came out covering 0.3% of the frame against the Gaussian shell's 24%.
    keep = ((pts - org) @ n + d) * body > 0
    return pts, cols, keep


def _ovox_draw(pts, cols, keep, size, R, sc, tr, w2c_np, fp_np):
    from method.common.eval.ovox_cuts import _splat
    img = np.ones((size * size, 3), np.float32)
    zb = np.full(size * size, np.inf)
    _splat(img, zb, sc * (pts[keep] @ R.T) + tr, cols[keep], fp_np, w2c_np, size)
    return (img.reshape(size, size, 3), zb.reshape(size, size), np.isfinite(zb).reshape(size, size))


def main(lattice_dir, out_png=None, size=512, axis=(0.0, 1.0, 0.0), piece=None,
         elevation=20, npz=None):
    import cv2
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    from utils.camera_view_utils import get_camera_view
    from utils.decode_param import decode_param_json
    from utils.render_utils import convert_SH, load_params_from_gs
    from utils.transformation_utils import (apply_cov_rotations, apply_inverse_cov_rotations,
                                            apply_inverse_rotations,
                                            generate_rotation_matrices,
                                            get_center_view_worldspace_and_observant_coordinate,
                                            shift2center111, transform2origin,
                                            undoshift2center111, undotransform2origin)
    from method.common.cube.occupancy import close_and_fill, to_grid
    from scipy.spatial import cKDTree

    cfg, demo = _cfg_for(lattice_dir)
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    ply = _os.path.join(lattice_dir, "gs_fill.ply")

    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(ply)
    par = load_params_from_gs(g, P())
    pos0, cov0 = par["pos"], par["cov3D_precomp"]
    sp, op, shs = par["screen_points"], par["opacity"], par["shs"]
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(pos0)
    tpos = shift2center111(tpos)
    cov0 = apply_cov_rotations(cov0, rot_m)
    cov0 = so * so * cov0
    cov3 = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)

    # The lattice, made solid, cut once -- in the ply's own frame, where it is axis aligned.
    el = PlyData.read(ply).elements[0]
    C0 = 0.28209479177387814
    dc = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                 * C0 + 0.5, 0, 1)
    xyz0 = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)
    keep = (lvl[:len(dc)] == 0).numpy()
    wp = world.detach().cpu().numpy().astype(np.float64)

    R, sc, tr, ferr = fit_similarity(xyz0, wp)
    print(f"  lattice frame -> render frame: scale {sc:.5f}, worst residual {ferr:.2e}")

    # Half a cell, so a centre does not sit on a cell boundary. Same hazard as physics.py and
    # binding.py: floor from the minimum *centre* lets floating point choose the side and, on a
    # lattice whose cells are exactly at (i + 1/2)h, discards 49% of them.
    org = xyz0[keep].min(0) - 0.5 * hc
    coords, first = np.unique(np.floor((xyz0[keep] - org) / hc).astype(np.int64),
                              axis=0, return_index=True)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1

    n = np.asarray(axis, np.float64)
    n = n / np.linalg.norm(n)
    c = (solid + 0.5) * hc
    d = float(-c.mean(0) @ n)
    r = sd.cut(solid, hc, n, d, hf)
    print(f"  {len(solid):,} solid cells, {r['K']} pieces after the cut")

    tree = cKDTree((coords + 0.5) * hc)
    cell_rgb = dc[keep][first]
    lp = (r["leaf"] + 0.5) * (hc / (2.0 ** r["level"].astype(np.float64)))[:, None]
    _, parent_of_leaf = tree.query(lp, k=1)

    # only this piece's exterior primitives, resolved in the frame the leaves live in
    # The exterior is the boundary of the occupancy, at whatever level -- not the refined cells.
    #
    # `cell_level != 0` looks like the skin and is not it. On the orange the refined cells run
    # from r/R 0.808 to 1.000 and the coarse cells reach 0.936, so the shell is thick and uneven
    # and a large part of the outer surface is made of coarse cells that the level test throws
    # away. What that leaves is not a shell but the thick lower part of one: 77% of the refined
    # cells sit in the bottom 40% of the object, so a cut through the centroid handed one piece
    # five times as much "skin" as the other and three views of four had almost nothing to draw.
    #
    # A boundary cell is one with an empty face neighbour, which is the same definition the
    # exterior conversion uses, and it does not care what level the cell is.
    import torch.nn.functional as _F
    # from the *solid*, not the raw quantisation. The stored occupancy is a sponge, so almost
    # every cell in it touches a hole and "the boundary" came out as 83% of the object.
    occ_b, _, _ = to_grid(torch.from_numpy(solid).float(), 1.0)
    inner = (_F.max_pool3d((~occ_b).float()[None, None], 3, 1, 1)[0, 0] > 0.5)
    bset = (occ_b & inner).nonzero().numpy() + solid.min(0) - 1
    ci = np.floor((xyz0 - org) / hc).astype(np.int64)
    kb = _index_rows(ci, bset)
    skin = (kb >= 0) | (lvl[:len(xyz0)] != 0).numpy()
    print(f"  exterior: {int(skin.sum()):,} of {len(xyz0):,} primitives are on the "
          f"occupancy boundary or refined")
    li = leaf_of(xyz0 - org, r, hc)

    # A skin primitive sits on the object's outer boundary and often just outside the coarse
    # grid, which was built from coarse cell centres and therefore stops half a cell short of
    # the surface. Dropping those left the exterior with 11,934 of the orange's 117,000 skin
    # primitives, a shell too sparse to occlude anything -- the cut face then showed through
    # from the side it should have been hidden on, not because the depth test was wrong but
    # because there was nothing in front of it. Outside the grid, the nearest leaf is the
    # answer; the piece boundary is a plane and the skin is at most half a cell beyond it.
    out = skin & (li < 0)
    if out.any():
        lc = (r["leaf"] + 0.5) * (hc / (2.0 ** r["level"].astype(np.float64)))[:, None]
        _, j = cKDTree(lc).query(xyz0[out] - org, k=1)
        li = li.copy()
        li[out] = j
        print(f"  {int(out.sum()):,} skin primitives fell outside the grid and took their "
              f"nearest leaf")
    inside = (li >= 0) & skin
    pid = np.where(inside, r["piece"][np.clip(li, 0, None)], -1)
    sizes = [(int((pid == k).sum()), k) for k in range(r["K"])]
    sizes.sort(reverse=True)
    piece = sizes[0][1] if piece is None else piece
    mine = torch.from_numpy(pid == piece).to(DEV)
    print(f"  pieces by exterior primitives: {[(k, m) for m, k in sizes]}")
    print(f"  rendering piece {piece}: {int(mine.sum()):,} of {len(xyz0):,} primitives")

    az, elv = 0, elevation
    cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                             observant_coordinates=oc, show_hint=False, init_azimuthm=az,
                             init_elevation=elv, init_radius=cam_p["init_radius"],
                             move_camera=False, current_frame=0, delta_a=None, delta_e=None,
                             delta_r=None)
    col = convert_SH(shs, cam, g, world, None)
    cg, dg, ag = gs_buffers(cam, size, world[mine], cov3[mine], sp[mine], op[mine],
                            col[mine], g)

    conv = probe_depth_convention(cam, world[mine], dg, ag, size)
    print("  depth convention (mean |offset|, and how that error grows towards the corners):")
    for k, (m0, sl) in conv.items():
        print(f"      {k:<10} offset {m0:.6f}   radial slope {sl:+.6f}")
    which = min(conv, key=lambda k: abs(conv[k][1]))
    print(f"  -> the renderer returns {which} depth (the flat one)")

    co, do, cov = cut_surface_buffers(cam, size, r, hc, n, d, piece, cell_rgb, parent_of_leaf,
                                      R, sc, tr, org, None, which)
    out, take_o, both = composite(cg, dg, ag, co, do, cov)

    # Which side of the cut the camera is on, so the outcome is predicted rather than observed.
    # The plane lives in the lattice frame and the camera in the render frame, so the normal is
    # carried across; a piece whose material lies on the camera's side shows its cut face, and
    # one whose material lies behind it does not.
    eye_l = ((cam.camera_center.reshape(3).detach().cpu().numpy().astype(np.float64) - tr)
             @ R) / sc
    side_cam = float(np.sign(eye_l @ n + d))
    body = float(np.sign(np.median(r["side"][r["piece"] == piece])))
    faces_us = side_cam != body
    print(f"  camera is on side {side_cam:+.0f} of the cut; piece {piece}'s material is on "
          f"side {body:+.0f} -> its cut face {'faces' if faces_us else 'points away from'} "
          f"the camera")
    print(f"  cut surface covers {100 * cov.mean():.1f}% of the frame, "
          f"exterior {100 * (ag > 0.5).mean():.1f}%")
    print(f"  the cut wins {100 * take_o.mean():.1f}% of the frame, "
          f"{100 * take_o[both].mean():.1f}% of where both are present")
    if both.any():
        print(f"  where both are present, exterior minus cut depth averages "
              f"{(dg[both] - do[both]).mean():+.5f} "
              f"(positive means the cut is nearer and should win)")

    cols = [cg, co, out]
    label = "exterior | cut surface | composite"
    if npz:
        fp_np = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        w2c_np = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
        pts_o, cols_o, keep_o = ovox_buffers(npz, cam, size, R, sc, tr, n, d, body, w2c_np,
                                             fp_np, org)
        # Which half is this piece, settled against the Gaussian render rather than by reading
        # the sign convention off the code twice. The two selections must agree; if the sign is
        # taken the wrong way the panel shows the other half of the object and every number
        # computed from it compares two different pieces.
        gmask = ag > 0.5
        best, overlap = None, -1.0
        for tag, k in (("body side", keep_o), ("the other side", ~keep_o)):
            _, _, cv_try = _ovox_draw(pts_o, cols_o, k, size, R, sc, tr, w2c_np, fp_np)
            ov = float((cv_try & gmask).sum()) / max(float((cv_try | gmask).sum()), 1.0)
            print(f"    O-Voxel exterior taken on {tag}: {100 * ov:.1f}% agreement with the "
                  f"Gaussian render of the same piece")
            if ov > overlap:
                best, overlap = k, ov
        ce, de, cve = _ovox_draw(pts_o, cols_o, best, size, R, sc, tr, w2c_np, fp_np)
        out_o, take_oo, both_o = composite(ce, de, cve.astype(np.float64), co, do, cov)
        # When the cut points away from the camera it should take no pixels at all, so
        # whatever it does take is the leak, and comparing the two exteriors on that number is
        # the measurement the picture is a picture of.
        if not faces_us:
            print(f"  the cut face points away: it leaks through the Gaussian exterior on "
                  f"{int(take_o.sum()):,} pixels and through the O-Voxel one on "
                  f"{int(take_oo.sum()):,}")
        print(f"  O-Voxel exterior covers {100 * cve.mean():.1f}% of the frame against the "
              f"Gaussian shell's {100 * (ag > 0.5).mean():.1f}%")
        print(f"  the cut takes {100 * take_o.mean():.1f}% of the frame through the Gaussian "
              f"exterior and {100 * take_oo.mean():.1f}% through the O-Voxel one")
        cols = [cg, out, ce, out_o]
        label = ("Gaussian exterior | composite with it | O-Voxel exterior | composite with it")
    if out_png:
        panel = np.concatenate(cols, 1)
        cv2.imwrite(out_png, (np.clip(panel, 0, 1)[:, :, ::-1] * 255).astype(np.uint8))
        print(f"  -> {out_png}  ({label})")
    return out


def _cfg_for(lattice_dir):
    """The physics and demo configs that belong to a lattice.

    Read from the object files rather than from a table keyed by directory name. A hard-coded
    table knows about the three directories that existed when it was written and refuses every
    lattice made since -- which is the wrong failure for a pipeline whose whole claim is that
    adding an object is writing objects/<name>.conf and changing nothing else.
    """
    import glob
    import re
    name = lattice_dir.rstrip("/").split("/")[-1]
    best = None
    for conf in sorted(glob.glob(_os.path.join(_FN_ROOT, "method/objects/*.conf"))):
        obj = _os.path.basename(conf)[:-5]
        if obj in name and (best is None or len(obj) > len(best[0])):
            body = open(conf).read()
            cfg = re.search(r"^CFG=(\S+)", body, re.M)
            demo = re.search(r"^DEMO=(\S+)", body, re.M)
            if cfg and demo:
                best = (obj, cfg.group(1), demo.group(1))
    if best is None:
        raise SystemExit(f"no object config matches {lattice_dir}; the name has to contain one "
                         f"of {[_os.path.basename(c)[:-5] for c in sorted(glob.glob(_os.path.join(_FN_ROOT, 'method/objects/*.conf')))]}")
    return best[1], best[2]


if __name__ == "__main__":
    # Both pieces, from above and from below. The cut face of one piece points at the camera and
    # the other's points away, so a composite that is right must show the cut in one case and
    # hide it in the other -- a single view cannot tell a working depth test from a broken one.
    lat = sys.argv[1]
    npz = sys.argv[2] if len(sys.argv) > 2 else None
    for pc in (0, 1):
        for el in (25, -25):
            print(f"--- piece {pc}, elevation {el:+d}")
            main(lat, f"m6_p{pc}_e{el:+d}.png", piece=pc, elevation=el, npz=npz)
