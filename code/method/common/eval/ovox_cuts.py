"""The held-out cuts, drawn from the representation instead of from the Gaussians.

The baseline's numbers come from `random_cuts`: pick a plane, drop the primitives on the near
side, rasterise the rest. That measures the Gaussian model. Asking whether the cube-and-O-Voxel
representation draws the same cuts as well means changing what draws them and nothing else, so
this does not reproduce the plane sequence -- it registers a renderer with `random_cuts` and lets
that file choose the planes, the depths, the cameras and the order exactly as it does for the
baseline. A second copy of the sequence would agree until one of the two was edited.

What the hook draws:

    the exposed face   a ray-plane intersection into the cube volume, exact because the face is
                       planar, with a pixel covered when the hit lands in an occupied cell.
    the exterior       the O-Voxel surface on the far side of the plane, splatted from points on
                       its own triangles.

and whichever is nearer takes the pixel.

    python method/common/eval/ovox_cuts.py MODEL.ply LATTICE OVOX.npz CFG DEMO OUT_DIR [n]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import subdivide as sd                      # noqa: E402

DEV = "cuda:0"
C0 = 0.28209479177387814


class Lookup:
    """Which cell holds a point, on a lattice that has more than one spacing.

    The lattice is two-level by construction: the interior is filled at h_c and only the skin is
    subdivided to h_f. Indexing all of it at h_f is not a small error -- a coarse cell then owns
    one of the eight fine addresses inside it and the other seven read as empty, so the solid
    reads as a sponge at one eighth density and a ray cast into the middle of the fruit almost
    always misses. That is what held the exposed face to 1% of the frame while the peel drew a
    million samples.

    Each level gets its own table at its own spacing, and the finer one is asked first because
    where the two overlap the fine cells are the ones that replaced the parent.
    """

    def __init__(self, xyz, level, spacing, org):
        self.org, self.tab = org, []
        for lv in sorted(set(int(v) for v in level)):
            rows = np.nonzero(level == lv)[0]
            h = spacing[lv]
            c = np.round((xyz[rows] - org) / h - 0.5).astype(np.int64)
            mn = int(c.min()) - 2
            span = int(c.max() - mn + 3)
            k = _pack(c, -mn, span)
            o = np.argsort(k)
            self.tab.append((h, mn, span, k[o], rows[o]))
        self.tab.sort(key=lambda t: t[0])          # finest first

    def __call__(self, pts):
        out = np.full(len(pts), -1, np.int64)
        todo = np.arange(len(pts))
        for h, mn, span, ks, rows in self.tab:
            if not len(todo):
                break
            p = pts[todo]
            # Rounding rather than flooring. The cells sit on a grid, so (p - org)/h is an exact
            # integer up to floating point at a centre, and `floor` of 3.9999999 is 3: measured on
            # these lattices that loses 27% to 52% of the cells.
            c = np.round((p - self.org) / h - 0.5).astype(np.int64)
            inr = ((c - mn) >= 0).all(1) & ((c - mn) < span).all(1)
            kk = _pack(np.where(inr[:, None], c, mn), -mn, span)
            pos = np.clip(np.searchsorted(ks, kk), 0, max(len(ks) - 1, 0))
            hit = inr & (ks[pos] == kk) if len(ks) else np.zeros(len(p), bool)
            out[todo[hit]] = rows[pos[hit]]
            todo = todo[~hit]
        return out


def _pack(c, off, span):
    return sd._pack(c, off, span)


def sim_fit(A, B, cap=20000):
    """The similarity taking A to B, from rows that already correspond.

    Both frames come out of the same ply row by row, so there is nothing to estimate but the
    transform, and the residual printed below is the check that they really do correspond.
    """
    k = np.linspace(0, len(A) - 1, min(cap, len(A))).astype(np.int64)
    a, b = A[k], B[k]
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, S, Vt = np.linalg.svd(H)
    sgn = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sgn]) @ U.T
    sc = float(S[:2].sum() + sgn * S[2]) / float(((a - ca) ** 2).sum())
    tr = cb - sc * (R @ ca)
    return R, sc, tr, float(np.abs(sc * (a @ R.T) + tr - b).max())


def sim_apply(p, f):
    R, sc, tr = f[0], f[1], f[2]
    return sc * (p @ R.T) + tr


def sim_inv(f):
    R, sc, tr = f[0], f[1], f[2]
    return R.T, 1.0 / sc, -(R.T @ tr) / sc, f[3]


def make_hook(model_ply, lattice_dir, npz_path):
    from plyfile import PlyData

    z = np.load(npz_path)
    V = z["mesh_v"].astype(np.float64) if "mesh_v" in z.files else z["pos"].astype(np.float64)
    F = z["mesh_f"].astype(np.int64) if "mesh_f" in z.files else None
    rgb_ov = z["rgb"].astype(np.float32)

    # The cells come from the model that was trained, not from the lattice it started on. The
    # lattice's own gs_fill.ply carries the flat grey `make_shape` wrote and the skin the six
    # views painted, so rendering the exposed face from it shows an untrained interior -- grey
    # against photographs of pulp. `random_cuts` loads this same file, so the rows the mask
    # indexes are these rows.
    # The appearance model is the one the baseline renders, not its constant term. Our models
    # carry 24 higher-band coefficients whose per-voxel directional part is what gives the pulp
    # its depth; dropping them leaves a flat wash that is visibly paler than the same model
    # rasterised by `random_cuts`, and comparing that to photographs measures the shortcut.
    sh = {}
    try:
        from scene.gaussian_model import GaussianModel
        _nr = len([q.name for q in PlyData.read(model_ply).elements[0].properties
                   if q.name.startswith("f_rest_")])
        if _nr:
            _deg = int(round(((_nr / 3 + 1) ** 0.5) - 1))
            _g = GaussianModel(_deg)
            _g.load_ply(model_ply)
            _g.active_sh_degree = _deg
            sh["g"], sh["shs"] = _g, _g.get_features
            print(f"  cut face shaded with degree {_deg} spherical harmonics")
    except Exception as e:                                   # noqa: BLE001
        print(f"  cut face shaded from f_dc only ({e})")

    el = PlyData.read(model_ply).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    cell_rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                       * C0 + 0.5, 0, 1)
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hf = float(lat["fine_dx"])
    org = xyz.min(0)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()
    lvl = lvl[:len(xyz)].astype(np.int64)
    hc = float(lat["coarse_dx"])
    look = Lookup(xyz, lvl, {0: hc, 1: hf}, org)
    print(f"  lattice {int((lvl == 0).sum()):,} coarse at {hc:.5f} + "
          f"{int((lvl == 1).sum()):,} fine at {hf:.5f}")

    if F is not None:
        k = 3
        t = (np.arange(k) + 0.5) / k
        bu, bv = np.meshgrid(t, t, indexing="ij")
        keep = (bu + bv) <= 1.0
        bu, bv = bu[keep], bv[keep]
        tv = V[F]
        ext_p = (tv[:, None, 0] * (1 - bu - bv)[None, :, None]
                 + tv[:, None, 1] * bu[None, :, None]
                 + tv[:, None, 2] * bv[None, :, None]).reshape(-1, 3)
        ext_c = np.repeat(rgb_ov[F[:, 0]], len(bu), axis=0)
    else:
        ext_p, ext_c = V, rgb_ov
    print(f"  exterior {len(ext_p):,} surface samples")

    def trilinear(p_render, rows):
        """The colour at a point, from the eight cell centres around it.

        `rows` says which cell each point landed in, and that cell's level sets the spacing the
        interpolation is done at -- the lattice is two-level, and interpolating a coarse interior
        at the fine spacing would ask for seven neighbours that do not exist.
        """
        q = sim_apply(p_render, fit["p2x"])
        h = np.where(lvl[rows] == 0, hc, hf)[:, None]
        u = (q - org) / h - 0.5
        i0 = np.floor(u).astype(np.int64)
        t = u - i0
        acc = np.zeros((len(q), 3), np.float32)
        wsum = np.zeros(len(q), np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    dd = np.array([dx, dy, dz])
                    w = np.prod(np.where(dd == 1, t, 1.0 - t), axis=1)
                    r = look(org + (i0 + dd + 0.5) * h)
                    ok = (r >= 0) & (w > 0)
                    acc[ok] += (w[ok, None] * cell_rgb[np.clip(r, 0, None)][ok]).astype(np.float32)
                    wsum[ok] += w[ok]
        out = acc / np.maximum(wsum, 1e-9)[:, None]
        bad = wsum < 1e-9
        out[bad] = cell_rgb[rows[bad]]        # nothing to interpolate from: the cell's own value
        return np.clip(out, 0, 1)

    fit = {}

    # One ray per pixel and one flat colour per cell is nearest-neighbour reconstruction of a
    # piecewise-constant field, and it shows up as noise rather than as blur: measured against the
    # photographs' 13.0e-3 of fine detail, the baseline rasteriser carries 13.8e-3 and this hook
    # carried 24.0e-3 and 30.8e-3 -- and the four FIDs order themselves by exactly that excess.
    # A Gaussian renderer avoids it by construction, because overlapping anisotropic kernels
    # average several primitives into every pixel. Supersampling is the same averaging done
    # honestly: the picture is formed at SS times the resolution and box-filtered down, which
    # changes no data and states no new geometry.
    SS = int(_os.environ.get("OVOX_SS", "2"))
    # Reading one flat colour out of the cell a pixel lands in is nearest-neighbour interpolation,
    # and it is the whole of the remaining gap to the rasteriser. The gap is not capacity: the
    # route-2 orange stores 1,231,312 primitives against the baseline's 877,494, more rather than
    # fewer. It is that a cell stores 14 numbers and this renderer was using 3 of them -- the
    # colour -- while the other eleven, the scale and the rotation, are the primitive's shape, and
    # the shape is a reconstruction filter. Measured on these models sigma/h has median 0.26, so
    # the rasteriser is not blurring across cells either; what it has is anisotropy, median 116 to
    # 1, which is structure inside the cell.
    #
    # Trilinear interpolation recovers the same kind of thing without a Gaussian anywhere: the
    # value at a point is the eight surrounding cell centres weighted by where it falls between
    # them, which is 24 numbers per pixel against the rasteriser's 14, all of them the cube
    # representation's own. Weights of cells that are not there are dropped and the rest
    # renormalised, so the boundary degrades to nearest-neighbour rather than to black.
    INTERP = _os.environ.get("OVOX_INTERP", "1") == "1"
    # Which layers to draw. Splitting the picture is how a defect gets attributed to the surface
    # or to the interior instead of guessed at: `ext` alone is the O-Voxel peel, `face` alone is
    # the exposed section, and whichever one carries the artefact owns it.
    # The face alone, by default, because that is what the baseline draws. It rasterises
    # `mask_suf` -- the slab about the plane -- so its picture is a slice, and a slice's rind is
    # the slab's own skin cells, which the face march already reaches. A separate exterior layer
    # is then not extra fidelity but a second object in the frame: on a longitudinal cut it shows
    # as a thin ring outside the section, offset from it, contributing nothing else. Rendering the
    # layers apart is what showed it -- the face alone was already the baseline's picture, and the
    # exterior alone was the ring and nothing more.
    #
    # `ext` remains available and is what exterior_views uses, because drawing the whole object is
    # a different question from drawing a slice of it.
    LAYERS = _os.environ.get("OVOX_LAYERS", "face").split(",")

    def hook(cam, plane, mask, mask_suf, size, tpos, pos):
        out_size, size = size, size * SS
        fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
        eye_w = cam.camera_center.reshape(3).detach().cpu().numpy().astype(np.float64)

        # Which half survives is not decided here. `random_cuts` built the plane, applied it with
        # `plane_filter` and handed the result in as `mask` -- the very mask the baseline renders
        # -- and it is indexed by ply row, which is the order the cells were read in. Re-deriving
        # it from a sign test is what produced 610 against the baseline's 86 twice: the near peel
        # was kept and the exposed face sat behind the material the cut had removed. There is no
        # sign to get wrong when the answer is an argument.
        #
        # Three frames meet here and only two of them are the same. `tpos` is where the plane is
        # measured, `pos` is where the camera projects, and the cells and the O-Voxel surface are
        # in the ply's. All three are row-aligned through the ply, so each map is fitted once from
        # the points themselves.
        if not fit:
            P_ = pos.detach().cpu().numpy().astype(np.float64)
            fit["pos"] = P_
            T_ = tpos.detach().cpu().numpy().astype(np.float64)
            fit["t2p"] = sim_fit(T_, P_)                    # cut frame  -> render frame
            fit["p2x"] = sim_fit(P_, xyz[:len(P_)])         # render frame -> ply frame
            fit["x2p"] = sim_inv(fit["p2x"])
            # the surface, carried into the render frame once rather than every frame
            fit["ext"] = sim_apply(ext_p, fit["x2p"])
            print(f"  cut->render residual {fit['t2p'][3]:.2e}, "
                  f"render->ply residual {fit['p2x'][3]:.2e}")


        # Two masks, because the two things drawn need different ones, and `random_cuts` already
        # makes both. `mask` is the strict far half and is what the peel is: material that
        # survived. `mask_suf` is that plus the band the plane passes through, and it is what the
        # baseline rasterises -- it has to be, because a cell the plane cuts has its centre on one
        # side or the other and roughly half of them fall on the near side. Testing the face
        # against the strict mask throws away half the cells the face is made of, which is exactly
        # the salt-and-pepper the first working render came out covered in.
        pos_np, ext_pp = fit["pos"], fit["ext"]
        keep_ext = mask.detach().cpu().numpy().reshape(-1).astype(bool)[:len(xyz)]
        keep_face = mask_suf.detach().cpu().numpy().reshape(-1).astype(bool)[:len(xyz)]

        # the plane, carried from the frame it was measured in into the one the rays are cast in
        a4 = np.asarray(plane, np.float64).reshape(4)
        nrm0 = np.linalg.norm(a4[:3])
        R, sc, tr = fit["t2p"][0], fit["t2p"][1], fit["t2p"][2]
        pn = R @ (a4[:3] / nrm0)
        d = float(a4[3] / nrm0 * sc - pn @ tr)

        img = np.ones((size * size, 3), np.float32)
        zbuf = np.full(size * size, np.inf)

        # The exterior is tested against the plane, not looked up in a cell. Its vertices sit on
        # the boundary by construction, so a third of them address a cell just outside the solid
        # and get dropped -- which thins the peel to a broken ring and is an artefact of the test,
        # not of the surface. Which side to keep is still not guessed: it is read off the mask by
        # asking which side the cells the mask kept are actually on, so the sign comes from the
        # same source as before and cannot disagree with it.
        s_keep = float(np.sign(np.median(pos_np[keep_ext] @ pn + d))) or 1.0
        # The baseline does not draw the far half. It rasterises `mask_suf`, which is a slab about
        # the plane, so what it produces is a slice -- and drawing the whole surviving exterior
        # against it compares two different pictures. On a transverse cut of a round fruit the
        # difference is invisible, because the far half's peel is behind the face either way. On a
        # longitudinal cut of the doughnut it is the whole story: the far half's inner surface
        # projects into the hole between the two lobes and fills it with icing, where the baseline
        # correctly shows two separate pieces.
        #
        # The slab's half-width is not chosen here either -- it is measured from the cells the
        # baseline's own band kept, so the two renderers cover the same material by construction.
        sgn = (ext_pp @ pn + d) * s_keep
        band = pos_np[keep_face] @ pn + d
        slab = float(np.abs(band).max()) if keep_face.any() else np.inf
        vis = (sgn > 0) & (sgn < slab)
        if "ext" in LAYERS:
            _splat(img, zbuf, ext_pp[vis], ext_c[vis], fp, w2c, size)

        # the exposed face, by ray-plane intersection: exact, because the face is planar, and a
        # pixel is covered when the hit lands in a cell that survived the cut
        inv = np.linalg.inv(fp)
        t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
        gx, gy = np.meshgrid(t, t, indexing="xy")
        one = np.ones(size * size)
        w = np.stack([gx.ravel(), gy.ravel(), one, one], 1) @ inv
        dirs = w[:, :3] / w[:, 3:4] - eye_w[None]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        den = dirs @ pn
        ok = np.abs(den) > 1e-12
        tt = np.full(len(dirs), np.inf)
        tt[ok] = -(eye_w @ pn + d) / den[ok]
        hit = ok & (tt > 1e-9) & np.isfinite(tt)
        pts = eye_w[None] + dirs * np.where(hit, tt, 0.0)[:, None]
        # What the exposed face is: the first material the ray meets at or beyond the plane. Not
        # a slab of chosen thickness tested against a band of chosen width -- both of those are
        # parameters, and both were wrong here. The band `mask_suf` covers 5.7% of the cells and
        # left 5% of the section uncovered, and what showed through the holes was the far half's
        # own peel, drawn behind but visible where nothing was drawn in front: dark blemishes
        # scattered over the pulp, 8 blobs of 123 pixels in the exterior layer alone. Marching
        # forward until the ray enters the surviving solid has nothing to tune and cannot leave a
        # hole where there is material. Where it does leave one there is genuinely no material,
        # which is worth seeing.
        cid = np.full(len(pts), -1, np.int64)
        todo = np.nonzero(hit)[0] if "face" in LAYERS else np.zeros(0, np.int64)
        step = 0.25 * hc
        for i in range(int(_os.environ.get("OVOX_MARCH", "24"))):
            if not len(todo):
                break
            q = pts[todo] + dirs[todo] * (i * step)
            r = look(sim_apply(q, fit["p2x"]))
            got = (r >= 0) & keep_ext[np.clip(r, 0, None)]
            cid[todo[got]] = r[got]
            # the depth of the hit is where it was found, not where the plane is
            pts[todo[got]] = q[got]
            todo = todo[~got]

        cov = hit & (cid >= 0)
        if cov.any():
            hom = np.concatenate([pts[cov], np.ones((int(cov.sum()), 1))], 1)
            dep = (hom @ w2c)[:, 2]
            nearer = dep < zbuf[cov]
            idx = np.nonzero(cov)[0][nearer]
            rows = cid[cov][nearer]
            if INTERP:
                img[idx] = trilinear(pts[cov][nearer], rows)
            elif sh:
                from utils.render_utils import convert_SH
                img[idx] = convert_SH(sh["shs"][rows], cam, sh["g"], pos[rows], None) \
                    .clamp(0, 1).detach().cpu().numpy()
            else:
                img[idx] = cell_rgb[rows]
            zbuf[idx] = dep[nearer]
        print(f"      cut keeps {100 * keep_ext.mean():.1f}% of cells "
              f"({100 * keep_face.mean():.1f}% with the band); "
              f"exterior {int(vis.sum()):,} samples, face {int(cov.sum()):,} px "
              f"({100 * cov.mean():.1f}% of frame)")
        img = img.reshape(size, size, 3)
        if SS > 1:
            img = img.reshape(out_size, SS, out_size, SS, 3).mean((1, 3))
        return img

    return hook


def _splat(img, zbuf, p, c, fp, w2c, size):
    if not len(p):
        return
    hom = np.concatenate([p, np.ones((len(p), 1))], 1)
    clip = hom @ fp
    ok = clip[:, 3] > 1e-6
    nd = clip[:, :2] / clip[:, 3:4]
    px = ((nd[:, 0] + 1) * 0.5 * size).astype(np.int64)
    # Same convention as the rasteriser this is compared against, which applies no flip on
    # either axis. The flip that used to be here cancelled skin_project's, so the exterior
    # figures looked right for the wrong reason.
    py = ((nd[:, 1] + 1) * 0.5 * size).astype(np.int64)
    dep = (hom @ w2c)[:, 2]
    ok &= (px >= 0) & (px < size) & (py >= 0) & (py < size)
    flat = py[ok] * size + px[ok]
    order = np.argsort(-dep[ok])                 # far first, nearest overwrites
    img[flat[order]] = c[ok][order]
    zbuf[flat[order]] = dep[ok][order]


def main(model_ply, lattice_dir, npz_path, cfg, demo, out_dir, n=12, size=512):
    from method.common.eval import random_cuts as rc
    rc.RENDER_HOOK[0] = make_hook(model_ply, lattice_dir, npz_path)
    # Forward the image size. Whether a coverage number is a property of the representation or an
    # artefact of sampling is answered by varying it, and this entry point could not vary it: it
    # took the renderer's 512 default, so the cube column of a resolution sweep could not be
    # reproduced from the committed code even though the 3DGS columns could.
    rc.main(model_ply, cfg, demo, out_dir, n=n, size=size)


if __name__ == "__main__":
    main(*sys.argv[1:7], n=int(sys.argv[7]) if len(sys.argv) > 7 else 12,
         size=int(sys.argv[8]) if len(sys.argv) > 8 else 512)
