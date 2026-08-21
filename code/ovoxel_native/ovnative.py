"""An O-Voxel dual grid as the representation, from the start.

Three tensors are the model, and nothing else is:

    dual_v    (Ns, 3) float32   where the surface sits inside each active fine voxel,
                                in the fractional convention flexible_dual_grid_to_mesh wants
    split_w   (Ns, 1) float32   the per-voxel weight that decides how each dual quad is split
    surf_rgb  (Ns, 3) float32   the surface's colour, one per dual vertex
    interior  (Nc, 3) float32   the volumetric appearance field: one colour per SOLID COARSE
                                CELL, which is the part TRELLIS.2's own container does not
                                carry, because mesh_to_flexible_dual_grid only ever returns
                                boundary voxels

and these are fixed:

    coords    (Ns, 3) int32     active fine voxels of the dual grid
    inter     (Ns, 3) bool      which of the three edges of each voxel the surface crosses
    solid     (Nc, 3) int64     the occupancy, coarse cells, after close_and_fill
    idx3      (Gx,Gy,Gz) int32  solid -> row of `interior`, -1 where empty

Cutting stays closed form because the occupancy is still a grid: a plane meets a cube in a
convex polygon with a formula, and the polygon's vertices take their colour by trilinear
interpolation of `interior`, which is where the gradient goes.
"""
import glob, importlib.util, os, sys, types
import numpy as np
import torch

TRELLIS = os.environ.get("TRELLIS2_ROOT", "/workspace/rebuild/TRELLIS.2")


def _ext_for_this_python(base):
    """The compiled `_C` in `base` built for the running interpreter, or None.

    `build_ext --inplace` writes into the source tree, so a directory can hold one `_C` per
    interpreter that has ever built there -- and this box now has both a 3.10 and a 3.12. Taking
    the first match fails at import with a version mismatch several frames deep, which is not
    where the reader will look for it.
    """
    import importlib.machinery
    for suf in importlib.machinery.EXTENSION_SUFFIXES:
        hit = glob.glob(os.path.join(base, "_C" + suf))
        if hit:
            return hit[0]
    return None


def _load_ovoxel():
    """o_voxel's submodules by path, skipping the package __init__ (which wants flex_gemm)."""
    cands = [os.path.join(TRELLIS, "o-voxel", "o_voxel")]
    cands += glob.glob(os.path.join(TRELLIS, "o-voxel", "build", "lib.*", "o_voxel"))
    ok = [b for b in cands if _ext_for_this_python(b)
          and os.path.exists(os.path.join(b, "convert", "flexible_dual_grid.py"))]
    if not ok:
        import sys as _s
        raise SystemExit(
            f"no o_voxel _C built for CPython {_s.version_info[0]}.{_s.version_info[1]} under "
            f"{TRELLIS}/o-voxel -- rebuild it with this interpreter (build_ovox.sh); found "
            f"{[os.path.basename(x) for b in cands for x in glob.glob(os.path.join(b, '_C*.so'))]}")
    base = ok[0]
    if "o_voxel" not in sys.modules:
        pkg = types.ModuleType("o_voxel"); pkg.__path__ = [base]; sys.modules["o_voxel"] = pkg
        sub = types.ModuleType("o_voxel.convert"); sub.__path__ = [os.path.join(base, "convert")]
        sys.modules["o_voxel.convert"] = sub
        for nm, p in [("o_voxel._C", _ext_for_this_python(base)),
                      ("o_voxel.convert.flexible_dual_grid",
                       os.path.join(base, "convert", "flexible_dual_grid.py"))]:
            spec = importlib.util.spec_from_file_location(nm, p)
            m = importlib.util.module_from_spec(spec); sys.modules[nm] = m
            spec.loader.exec_module(m)
        pkg._C = sys.modules["o_voxel._C"]
    return sys.modules["o_voxel.convert.flexible_dual_grid"]


FDG = None

# ---------------------------------------------------------------- occupancy -> dual grid

FACE = [
    (np.array([1, 0, 0]), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    (np.array([-1, 0, 0]), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    (np.array([0, 1, 0]), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    (np.array([0, -1, 0]), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    (np.array([0, 0, 1]), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    (np.array([0, 0, -1]), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]


def boundary_mesh(coords, h, origin=None):
    """Every face of an occupied cell whose neighbour is empty.  (globalovox.py's, verbatim.)"""
    origin = np.zeros(3) if origin is None else np.asarray(origin, np.float64)
    c = np.asarray(coords, np.int64)
    mn = c.min(0) - 2
    span = (c.max(0) - mn + 3).astype(np.int64)
    key = ((c[:, 0] - mn[0]) * span[1] + (c[:, 1] - mn[1])) * span[2] + (c[:, 2] - mn[2])
    ks = np.sort(key)
    tris = []
    for d, corners in FACE:
        nb = c + d
        k = ((nb[:, 0] - mn[0]) * span[1] + (nb[:, 1] - mn[1])) * span[2] + (nb[:, 2] - mn[2])
        pos = np.clip(np.searchsorted(ks, k), 0, len(ks) - 1)
        exposed = ks[pos] != k
        if not exposed.any():
            continue
        base = c[exposed].astype(np.float64)
        q = [(base + np.asarray(off, np.float64)) * h + origin for off in corners]
        tris.append(np.stack([q[0], q[1], q[2]], 1))
        tris.append(np.stack([q[0], q[2], q[3]], 1))
    tri = np.concatenate(tris)
    flat = tri.reshape(-1, 3)
    k = np.round(flat / (h * 1e-6)).astype(np.int64)
    _, first, inv = np.unique(k, axis=0, return_index=True, return_inverse=True)
    return flat[first], inv.reshape(-1, 3)


def build(lattice_dir, device="cuda", verbose=True):
    """The orange's lattice -> (occupancy, boundary dual grid, interior field)."""
    global FDG
    FDG = _load_ovoxel()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from occupancy import close_and_fill, to_grid

    lat = torch.load(os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    C0 = 0.28209479177387814
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                  * C0 + 0.5, 0, 1)
    lvl = torch.load(os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)[:len(xyz)].numpy()

    org = xyz.min(0) - 0.5 * hf
    fine = np.floor((xyz - org) / hf).astype(np.int64)
    coarse_raw = np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64)
    coarse_solid = np.unique(coarse_raw, axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coarse_solid).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coarse_solid.min(0) - 1
    if verbose:
        print(f"  occupancy: {len(coarse_solid):,} coarse cells with a particle -> "
              f"{len(solid):,} solid after close_and_fill, h_c {hc:.5f}")

    off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
    allf = np.unique(np.concatenate([(solid[:, None, :] * 2 + off[None]).reshape(-1, 3), fine]),
                     axis=0)
    V, F = boundary_mesh(allf, hf, org)
    if verbose:
        print(f"  boundary at h_f {hf:.5f}: {len(F):,} triangles, {len(V):,} vertices")

    Vt = torch.as_tensor(V, dtype=torch.float32)
    Ft = torch.as_tensor(F, dtype=torch.int32)
    pad = hf * 2
    lo = Vt.min(0).values - pad
    hi = Vt.max(0).values + pad
    aabb = torch.stack([lo, hi])
    vi, dv, inter = FDG.mesh_to_flexible_dual_grid(Vt, Ft, voxel_size=hf, aabb=aabb)
    pos = lo.numpy() + dv.numpy()
    frac = dv.numpy() / hf - vi.numpy()
    if verbose:
        print(f"  dual grid: {len(vi):,} active voxels")

    # surface colour, from the skin cells only (a boundary cell is not always a skin cell)
    skin = lvl == 1
    tree = cKDTree(xyz[skin]); src = rgb[skin]
    surf_rgb = src[tree.query(pos, k=1)[1]]

    # interior colour, one per solid coarse cell, seeded from whatever the lattice holds there
    idx_lo = solid.min(0)
    G = solid.max(0) - idx_lo + 1
    idx3 = -np.ones(tuple(G), np.int32)
    idx3[solid[:, 0] - idx_lo[0], solid[:, 1] - idx_lo[1], solid[:, 2] - idx_lo[2]] = \
        np.arange(len(solid), dtype=np.int32)
    interior = np.full((len(solid), 3), 0.5, np.float32)
    have = idx3[coarse_raw[:, 0] - idx_lo[0], coarse_raw[:, 1] - idx_lo[1],
                coarse_raw[:, 2] - idx_lo[2]]
    ok = have >= 0
    interior[have[ok]] = rgb[lvl == 0][ok]
    seeded = np.zeros(len(solid), bool); seeded[have[ok]] = True
    if (~seeded).any():                       # cells close_and_fill invented, nearest neighbour
        c_all = (solid + 0.5) * hc + org
        t2 = cKDTree(c_all[seeded])
        interior[~seeded] = interior[seeded][t2.query(c_all[~seeded], k=1)[1]]
    if verbose:
        print(f"  interior field: {len(solid):,} cells, {int(seeded.sum()):,} seeded from the "
              f"lattice, {int((~seeded).sum()):,} filled from a neighbour; "
              f"mean RGB {interior.mean(0).round(4)}, spread {interior.std(0).round(4)}")

    D = lambda a, t: torch.as_tensor(a, dtype=t, device=device)
    return dict(
        hc=hc, hf=hf, org=org, aabb=aabb.to(device),
        solid=D(solid, torch.int64), idx3=D(idx3, torch.int32), idx_lo=idx_lo,
        coords=D(vi.numpy(), torch.int32), inter=D(inter.numpy(), torch.bool),
        dual_v=D(frac, torch.float32), split_w=torch.full((len(vi), 1), 0.5, device=device),
        surf_rgb=D(surf_rgb, torch.float32), interior=D(interior, torch.float32),
        dual_pos=pos, xyz=xyz, lvl=lvl,
    )


# ---------------------------------------------------------------- interior field lookup

def sample_interior(st, p):
    """Trilinear interpolation of the per-cell colour field at world points p (M,3).

    Cell k covers [org + k*hc, org + (k+1)*hc], so its centre is org + (k+0.5)*hc.  Empty
    neighbours get zero weight and the rest are renormalised, so the field is defined right up
    to the boundary instead of fading into whatever an absent cell would contribute.
    """
    hc = st["hc"]
    org = torch.as_tensor(st["org"], dtype=torch.float32, device=p.device)
    u = (p - org) / hc - 0.5
    i0 = torch.floor(u)
    w = u - i0
    i0 = i0.long() - torch.as_tensor(st["idx_lo"], dtype=torch.long, device=p.device)
    G = st["idx3"].shape
    out = torch.zeros(len(p), 3, device=p.device, dtype=st["interior"].dtype)
    wsum = torch.zeros(len(p), 1, device=p.device, dtype=st["interior"].dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + torch.tensor([dx, dy, dz], device=p.device)
                inb = ((idx >= 0) & (idx < torch.tensor(G, device=p.device))).all(1)
                cl = idx.clamp(min=torch.zeros(3, dtype=torch.long, device=p.device),
                               max=torch.tensor([g - 1 for g in G], device=p.device))
                row = st["idx3"][cl[:, 0], cl[:, 1], cl[:, 2]].long()
                val = inb & (row >= 0)
                ww = ((w[:, 0] if dx else 1 - w[:, 0]) *
                      (w[:, 1] if dy else 1 - w[:, 1]) *
                      (w[:, 2] if dz else 1 - w[:, 2]))[:, None] * val[:, None].float()
                out = out + ww * st["interior"][row.clamp(min=0)]
                wsum = wsum + ww
    return out / wsum.clamp(min=1e-6)


# ---------------------------------------------------------------- the cut face

_CORN = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.float32)
_EDGE = torch.tensor([(a, b) for a in range(8) for b in range(a + 1, 8)
                      if bin(a ^ b).count("1") == 1], dtype=torch.long)   # 12 cube edges


def cut_polygons(st, n, d, device="cuda"):
    """The polygon the plane n.x + d = 0 makes with every solid cell it crosses.

    Closed form: an edge whose endpoints straddle the plane contributes one point, the points
    of one cell are coplanar and convex, so sorting them by angle about their own centroid
    orders the polygon.  Returns (P, T): points (K*7, 3) -- six ring slots and a centroid per
    cell, padded by repetition -- and triangles (K*6, 3) fanned from the centroid.
    """
    hc, org = st["hc"], torch.as_tensor(st["org"], dtype=torch.float32, device=device)
    corners = _CORN.to(device)
    cellc = st["solid"].float()                                   # (Nc,3)
    s = ((cellc[:, None, :] + corners[None]) * hc + org) @ n + d   # (Nc, 8)
    # `>= 0` on one side and not the other, because a plane can land exactly on a lattice plane and
    # a strict test on both sides then finds nothing at all: the cell behind has max == 0 and the
    # cell in front has min == 0, so neither straddles and the whole cut face disappears. It is not
    # a hypothetical -- the orange's transverse normal is (0, -1, 0) and its supervised plane 10
    # sits at exactly -65 coarse cells, where this returned 0 polygons instead of 60,072 triangles,
    # and `render_section` did not notice because the exterior triangles keep the rasteriser call
    # valid. The frame is then the rind seen from behind, darker and far more saturated than pulp.
    # Five supervised planes across three objects were on such a depth: orange 10, loaf 8,
    # pomegranate 3, 11 and 15. Taking the cell behind (max == 0) rather than the one in front
    # keeps the exposed face on the material that remains, which is the same side every
    # non-degenerate depth already picks. Away from exact coincidence the two tests agree, because
    # `max == 0` to the bit is otherwise measure zero -- verified depth by depth below.
    cross = (s.min(1).values < 0) & (s.max(1).values >= 0)
    S = cellc[cross]                                              # (K,3)
    K = len(S)
    P8 = (S[:, None, :] + corners[None]) * hc + org               # (K,8,3)
    sv = s[cross]                                                 # (K,8)
    e = _EDGE.to(device)
    sa, sb = sv[:, e[:, 0]], sv[:, e[:, 1]]                       # (K,12)
    hit = (sa < 0) != (sb < 0)
    t = (sa / (sa - sb).where((sa - sb).abs() > 1e-20, torch.full_like(sa, 1e-20))).clamp(0, 1)
    pa, pb = P8[:, e[:, 0]], P8[:, e[:, 1]]                       # (K,12,3)
    pts = pa + t[..., None] * (pb - pa)

    cen = (pts * hit[..., None]).sum(1) / hit.sum(1, keepdim=True).clamp(min=1)
    # a basis in the plane
    a0 = torch.tensor([1.0, 0.0, 0.0], device=device)
    if abs(float(n[0])) > 0.9:
        a0 = torch.tensor([0.0, 1.0, 0.0], device=device)
    u = torch.cross(n, a0, dim=0); u = u / u.norm()
    v = torch.cross(n, u, dim=0)
    rel = pts - cen[:, None]
    ang = torch.atan2(rel @ v, rel @ u)
    ang = ang.where(hit, torch.full_like(ang, 1e9))
    order = ang.argsort(1)[:, :6]                                  # (K,6)
    ring = torch.gather(pts, 1, order[..., None].expand(-1, -1, 3))
    okr = torch.gather(hit, 1, order)
    # pad by repeating the last valid slot, so short polygons emit degenerate triangles
    last = okr.float().cumsum(1).argmax(1)
    ring = torch.where(okr[..., None], ring,
                       torch.gather(ring, 1, last[:, None, None].expand(-1, 1, 3)).expand(-1, 6, -1))
    P = torch.cat([ring, cen[:, None]], 1).reshape(-1, 3)          # (K*7,3)
    base = torch.arange(K, device=device)[:, None] * 7
    j = torch.arange(6, device=device)[None]
    T = torch.stack([(base + 6).expand(-1, 6), base + j, base + (j + 1) % 6], -1).reshape(-1, 3)
    return P, T.int(), K


# ---------------------------------------------------------------- rendering one section

def surface_mesh(st, colour=True):
    """flexible_dual_grid_to_mesh, in training mode, twice.

    Once for geometry.  Once with the dual vertices replaced by an encoding of the surface
    colour, which makes `mesh_vertices` come back as the colour of every vertex INCLUDING the
    per-quad midpoints the training-mode split introduces -- the library blends those with the
    same split weights, so the colour and the position stay consistent by construction and
    nothing about the extraction has to be reimplemented.
    """
    hf, aabb = st["hf"], st["aabb"]
    mv, mf = FDG.flexible_dual_grid_to_mesh(st["coords"], st["dual_v"], st["inter"],
                                            st["split_w"], aabb, voxel_size=hf, train=True)
    if not colour:
        return mv, mf, None
    enc = (st["surf_rgb"] - aabb[0].reshape(1, 3)) / hf - st["coords"].float()
    mc, _ = FDG.flexible_dual_grid_to_mesh(st["coords"], enc, st["inter"],
                                           st["split_w"], aabb, voxel_size=hf, train=True)
    return mv, mf, mc


DEFERRED = os.environ.get("CUT_DEFERRED", "0") == "1"


def render_section(st, glctx, mvp, n, d, res, bg=1.0, exterior=True, aa=True,
                   thickness=0.0, n_sub=7):
    """One cross-section: the exposed cut face, plus whatever exterior is behind the plane.

    Both go into a single nvdiffrast pass, so the depth test composes them and the exposed face
    is occluded by nothing it should not be.

    `thickness` is the half-width of a SLAB, in lattice units, and it exists because the pipeline's
    cut face is one and this one was not. `train_voxel.py:2007` and `random_cuts.py` both call
    `plane_filter(..., surf_dis=avg_dis/2, include_double=True)` and splat every primitive inside
    that band, so what they draw is an integral over a slab; `cut_polygons` + `sample_interior` is
    a trilinear sample on a mathematically zero-thickness plane. Those are different physical
    quantities, and the difference is a low-pass filter several cells wide.

    Measured on this orange rather than assumed: avg = 0.04348 in the transformed frame the plane
    is stated in, surf_dis = avg/2 = 0.02174 there, and the transformed-to-lattice scale is
    1.5281, so surf_dis is 0.03322 in lattice units = 2.82 coarse cells, and the slab the pipeline
    integrates is 5.63 cells thick.

    Uniform weighting over `n_sub` sub-planes. That is the right first version here and not merely
    the simplest: OPACITY_FREEZE holds every interior primitive at opacity 1.0 and SCALE_FREEZE
    holds every cell's footprint at the same size, so the pipeline's slab is close to an unweighted
    average over depth. It is still an approximation -- the real thing composites front-to-back
    through overlapping Gaussians -- and it is stated as one.
    """
    if thickness > 0 and n_sub > 1:
        offs = np.linspace(-thickness, thickness, n_sub)
        acc_i = acc_a = None
        K = nf = 0
        for o in offs:
            im, al, k_, nf_ = render_section(st, glctx, mvp, n, d + float(o), res, bg=bg,
                                             exterior=exterior, aa=aa, thickness=0.0)
            acc_i = im if acc_i is None else acc_i + im
            acc_a = al if acc_a is None else acc_a + al
            K, nf = max(K, k_), max(nf, nf_)
        return acc_i / len(offs), acc_a / len(offs), K, nf
    import nvdiffrast.torch as dr
    dev = st["interior"].device
    parts_v, parts_c, parts_f, off = [], [], [], 0
    if exterior:
        mv, mf, mc = surface_mesh(st)
        keep = ((mv @ n + d) < 0)[mf.long()].all(1)
        f = mf[keep].long()
        parts_v.append(mv); parts_c.append(mc); parts_f.append(f.int()); off += len(mv)
    P, T, K = cut_polygons(st, n, d, device=dev)
    C = sample_interior(st, P)
    parts_v.append(P); parts_c.append(C); parts_f.append(T + off)
    Vt = torch.cat(parts_v); Ct = torch.cat(parts_c).clamp(0, 1)
    Ft = torch.cat(parts_f).contiguous().int()

    ph = (torch.cat([Vt, torch.ones_like(Vt[:, :1])], 1) @ mvp)[None]
    rast, _ = dr.rasterize(glctx, ph, Ft, resolution=[res, res])
    if DEFERRED:
        # Sample the field at the fragment, not at the polygon's corners.
        #
        # `sample_interior` is a trilinear sample defined at any point, but it was only ever called
        # at the cut polygon's vertices and the rasteriser interpolated colour between them. A
        # polygon is planar and lies inside one cell, and the trilinear field restricted to a plane
        # is not linear, so a barycentric blend of its corner values is an approximation -- and a
        # different one for each polygon. The same cell cut transversely and longitudinally has two
        # different polygons, hence two different answers at the same point in space.
        #
        # Measured on the orange at 400 points along six intersection lines: the two families differ
        # from each other by 0.082, from the field itself by 0.060 and 0.070. The pixel loss the
        # training minimises is 0.020, so the renderer's own approximation was three times the
        # residual it was being fitted to.
        #
        # Interpolating position instead is exact -- position IS linear over a planar polygon -- so
        # sampling the field at the interpolated position gives every plane through a point the same
        # colour there, by construction rather than by penalty. The exterior keeps vertex colours:
        # its triangles carry `surf_rgb`, which is a per-vertex quantity and not a field.
        flag = torch.zeros(len(Vt), 1, device=dev, dtype=Vt.dtype)
        flag[off:] = 1.0
        attr = torch.cat([Vt, Ct, flag], 1)
        it, _ = dr.interpolate(attr[None], rast, Ft)
        pos, vcol, isc = it[..., :3], it[..., 3:6], it[..., 6:7]
        cut = (isc > 0.5) & (rast[..., 3:4] > 0)
        img = vcol
        if bool(cut.any()):
            idx = cut[..., 0].nonzero(as_tuple=True)
            fc = sample_interior(st, pos[idx]).clamp(0, 1)
            img = img.clone()
            img[idx] = fc
    else:
        img, _ = dr.interpolate(Ct[None], rast, Ft)
    if aa:
        img = dr.antialias(img, rast, ph, Ft)
    alpha = (rast[..., 3:4] > 0).float()
    img = img * alpha + bg * (1 - alpha)
    return img[0].permute(2, 0, 1), alpha[0].permute(2, 0, 1), K, len(Ft)


def render_exterior(st, glctx, mvp, res, bg=1.0, aa=True):
    """The whole dual surface from one camera, with no plane at all.

    The six exterior views are the only supervision that ever sees most of the skin: a transverse
    cut shows one face and a rim, and over any set of planes the rim sweeps a band rather than a
    sphere.  There is no cut here, so `interior` takes no gradient from this and `surf_rgb` and the
    dual geometry take all of it -- which is the same ownership split SEC_SKIP_OUTER states from
    the other side.
    """
    import nvdiffrast.torch as dr
    mv, mf, mc = surface_mesh(st)
    Ft = mf.contiguous().int()
    ph = (torch.cat([mv, torch.ones_like(mv[:, :1])], 1) @ mvp)[None]
    rast, _ = dr.rasterize(glctx, ph, Ft, resolution=[res, res])
    img, _ = dr.interpolate(mc.clamp(0, 1)[None], rast, Ft)
    if aa:
        img = dr.antialias(img, rast, ph, Ft)
    alpha = (rast[..., 3:4] > 0).float()
    img = img * alpha + bg * (1 - alpha)
    return img[0].permute(2, 0, 1), alpha[0].permute(2, 0, 1), 0, len(Ft)
