"""Colour the shell from a six-face cube of references, and label each cell with its face.

The single-reference version pasted one image into every view, so anything singular in it was
stamped around the whole sphere -- the orange grew a navel wherever a camera pointed. Six
faces put each feature where it belongs, and the geometry makes the coverage exact: every
direction lies within 54.7 degrees of some face axis, which is where the sampling cone comes
from. A face is used over that cone and no further, so the outer eighteen percent of each
reference, where the generation put its shadow and its background, is never read.

Each face axis is taken from its own reference camera rather than assumed, so the labels and
the images cannot disagree about which way is up.

The face each cell belongs to is written out alongside the colours. It is what made this
work, and it is a fact about the cell rather than about this step -- which side of the object
a cell is on is the same question a cut, a contact test or a material assignment asks.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os, argparse
import os as _os
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)
import torch, numpy as np
from scipy import ndimage
from torch import nn
from PIL import Image
from scene.gaussian_model import GaussianModel
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs
from utils.transformation_utils import *
from utils.camera_view_utils import get_camera_view

DEV = "cuda:0"
C0 = 0.28209479177387814
CE = 35.264
FACES = [("up", 0, 90), ("down", 0, -90), ("front", 0, 0),
         ("right", 90, 0), ("back", 180, 0), ("left", 270, 0)] + \
        [(f"c{i}", 45 + 90 * (i % 4), CE if i < 4 else -CE) for i in range(8)]
# The six faces are the labels; all fourteen directions contribute colour. Accepting out to
# 60 degrees rather than the 54.7 the six alone required means a point is read from three or
# four references instead of one, which is what lets the lighting average out.
FACE_LABELS = 6
COS_CONE = 0.5                          # 60 degrees
# Blend exponent across face boundaries. 8 came from the single-reference version, where a
# high exponent was what kept a watermelon's stripes from averaging into mush -- there every
# view saw the same picture and only the sharpest one should win. Six faces are the opposite
# case: they are different pictures of different parts, and where two meet each is equally
# right, so the blend has to actually happen. At 45 degrees both faces weigh cos45 = 0.707,
# and 0.707^8 = 0.06 is a hard switch, which showed as a seam down the sphere. At 2 the two
# weigh 0.5 each and cross over evenly.
SHARP = float(os.environ.get("SKIN_SHARP", "2"))
# Take the high-frequency detail from the single nearest face instead of averaging it with the
# others. HF_SIGMA is the cut, in pixels of the reference: below it is peel, above it is hue
# and shading. The peel's dimples are a few pixels across in a 512 frame.
HF_FROM_NEAREST = os.environ.get("SKIN_HF_NEAREST", "0") == "1"
HF_SIGMA = float(os.environ.get("SKIN_HF_SIGMA", "4"))
CONE_ZERO = os.environ.get("SKIN_CONE_ZERO", "0") == "1"
# Where the edge of the cone lands in the reference, as a fraction of its silhouette radius.
# The geometry gives 0.866 and that is where the peel has already turned to rim light, so the
# default keeps the old behaviour and anything lower reads further in.
REF_R = float(os.environ.get("SKIN_REF_R", "0.866"))
# How much more a named face weighs than a filler direction. 1.0 treats them alike, which is
# what erased the stem scar and the navel.
NAMED_W = float(os.environ.get("SKIN_NAMED_W", "1.0"))
# The furthest into a reference any cell may sample, as a fraction of the frame
# half-width. The fruit fills about 0.95 of a prepped frame; stay inside its rim.
REL_MAX = float(os.environ.get("SKIN_REL_MAX", "0.93"))
# Divide each reference by its own radial brightness profile before reading it.
FLATTEN = os.environ.get("SKIN_FLATTEN", "0") == "1"
# A cell whose best reference still puts it beyond this fraction of the frame has no
# face that sees it as peel, and takes its colour from its neighbours instead.
REL_POOR = float(os.environ.get("SKIN_REL_POOR", "0.80"))
# Beyond this fraction of the frame a sample is rim, not peel, and carries no weight.
REL_CUT = float(os.environ.get("SKIN_REL_CUT", "1.0"))
# Give each cell a direction-dependent colour instead of one value: 0 off, 1 or 2 the SH
# degree fitted to the (viewing direction, colour) pairs the faces provide.
SH_DEG = int(os.environ.get("SKIN_SH", "0"))
SH_RIDGE = float(os.environ.get("SKIN_SH_RIDGE", "0.05"))
# Weight of the pseudo-observations on the tangent circle, which hold the fit at the mean in
# the directions no face ever looked from.
SH_TANGENT = float(os.environ.get("SKIN_SH_TANGENT", "0.15"))
# Jacobi sweeps that smooth the filled patches after the rounds that created them.
SH_FILL_RELAX = int(os.environ.get("SKIN_FILL_RELAX", "120"))
# Fit every coefficient including the mean, so no photograph is ever averaged with another.
SH_FREE = os.environ.get("SKIN_SH_FREE", "0") == "1"
# The references are renders from these same cameras, so read each cell at its own pixel.
DIRECT = os.environ.get("SKIN_DIRECT", "0") == "1"
# How far inside the fruit a direct sample must land. A splat render fades out over several
# pixels, so its rim is bright without being background, and 0.95 lets that rim through.
DIRECT_FG = float(os.environ.get("SKIN_DIRECT_FG", "0.95"))
# A cell counts as coloured only if its total weight reaches this fraction of a normal
# cell's. Below it the weighted mean is dividing noise by noise.
SERVED_FRAC = float(os.environ.get("SKIN_SERVED_FRAC", "0.02"))
# A separate sampling radius for the named faces. Pulling the sample inward avoids the rim
# light, which is what the filler directions need, and it magnifies whatever is in the middle,
# which is what the poles cannot afford: the stem scar occupies about the inner eighth of the
# `up` reference, and stretching that over a 60 degree cone turns a small sharp calyx into a
# large soft blur. Empty by default, meaning the faces use REF_R like everything else.
_rrn = os.environ.get("SKIN_REF_R_NAMED", "")
REF_R_NAMED = float(_rrn) if _rrn else None


def sh_basis(d, deg):
    """The spherical-harmonic basis at directions `d`, in `eval_sh`'s order and sign.

    Copied term by term from `utils/sh_utils.eval_sh` rather than derived, because the signs on
    the degree-1 terms are not the textbook ones and a fit against a differently signed basis
    renders as its own mirror image.
    """
    C0_, C1_ = 0.28209479177387814, 0.4886025119029199
    C2_ = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
           -1.0925484305920792, 0.5462742152960396]
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    cols = [torch.full_like(x, C0_)]
    if deg > 0:
        cols += [-C1_ * y, C1_ * z, -C1_ * x]
    if deg > 1:
        xx, yy, zz = x * x, y * y, z * z
        cols += [C2_[0] * x * y, C2_[1] * y * z, C2_[2] * (2.0 * zz - xx - yy),
                 C2_[3] * x * z, C2_[4] * (xx - yy)]
    return torch.cat(cols, 1)


def flatten_radial(img, bins=48, gain_max=1.6):
    """Divide a reference by its own radial brightness profile, so it carries albedo not light.

    A photograph of a sphere is darker and paler towards its edge -- the surface turns away
    from the lamp, and what is left there is rim light rather than peel. The projection reads a
    cell at the radius its own direction puts it at, so a cell near the edge of a face's cone
    lands at 0.87 of the reference's radius and takes that rim light as its colour. Where
    several cones end at the same point every one of them does it, and the result is the pale
    marks that have sat on the side views through the reference set being rebuilt, the sampling
    radius being clamped, the cone weighting being changed and the blending being rewritten.

    `SKIN_REF_R` was the standing workaround: read further in, where the light is even. It
    works by never using most of each photograph, so every cell is sampled from too near the
    middle and whatever is in the middle is magnified over the whole cone.

    The shading is a function of radius and nothing else -- the fruit is a sphere, centred and
    lit from the front -- so it can simply be divided out, and then the whole disc is usable.
    What remains is the peel's own variation about the profile, which is what a texture is.
    """
    H, W, _ = img.shape
    yy = torch.arange(H, device=img.device).reshape(-1, 1).float() - (H - 1) / 2
    xx = torch.arange(W, device=img.device).reshape(1, -1).float() - (W - 1) / 2
    r = torch.sqrt(yy ** 2 + xx ** 2) / ((min(H, W) - 1) / 2)
    m = torch.from_numpy(ndimage.binary_erosion(
        (img.min(2).values <= 0.95).cpu().numpy(), iterations=6)).to(img.device)
    b = (r * bins).long().clamp(0, bins - 1)
    # Brightness and colourfulness, separately, with the hue left alone.
    #
    # A per-channel gain corrects the hue as well as the light and turned the sides red and the
    # stem end yellow-green. A luminance-only gain leaves the rim as bright orange diluted with
    # white and merely darkens it, which is mud. What the rim actually is, is brighter *and*
    # less colourful than the peel -- that is what adding white does -- so both have to be
    # undone, and scaling the chroma vector rather than the channels keeps the hue exactly.
    lum = img.mean(2)
    chroma = img - lum.unsqueeze(2)
    colf = chroma.norm(dim=2) / lum.clamp_min(1e-3)
    k = torch.tensor([.25, .5, .25], device=img.device).reshape(1, 1, 3)

    def profile(x):
        p = torch.ones(bins, device=img.device)
        for i in range(bins):
            s = m & (b == i)
            if int(s.sum()) > 32:
                p[i] = x[s].median()
            elif i:
                p[i] = p[i - 1]
        p = torch.nn.functional.conv1d(p.reshape(1, 1, -1), k, padding=1).reshape(-1)
        # Referenced to a mid-radius band, not to the centre: the centre of `up` is the calyx,
        # and normalising a reference against its own feature is how the stem end came back
        # yellow.
        return (p[int(0.15 * bins):int(0.35 * bins)].mean()
                / p.clamp_min(1e-3)).clamp(1.0 / gain_max, gain_max)

    gv = profile(lum)[b]
    gs = profile(colf)[b]
    out = (lum * gv).unsqueeze(2) + chroma * (gs * gv).unsqueeze(2)
    return torch.where(m.unsqueeze(2), out.clamp(0, 1), img)


def _blur(img, sigma):
    """Gaussian blur of an (H,W,3) tensor, separable, done on the GPU it already lives on."""
    r = max(int(3 * sigma), 1)
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma)); k = k / k.sum()
    t = img.permute(2, 0, 1).unsqueeze(0)
    t = torch.nn.functional.conv2d(t, k.view(1, 1, 1, -1).expand(3, 1, 1, -1),
                                   padding=(0, r), groups=3)
    t = torch.nn.functional.conv2d(t, k.view(1, 1, -1, 1).expand(3, 1, -1, 1),
                                   padding=(r, 0), groups=3)
    return t.squeeze(0).permute(1, 2, 0)


class P: convert_SHs_python = False; compute_cov3D_python = True; debug = False


def surface_normals(world, xs, dx, centre,
                    sigma=float(os.environ.get("SKIN_NORMAL_SIGMA", "5.0"))):
    """Which way a shell cell faces, from the lattice's occupancy rather than from the centroid.

    `xs - centre` is the surface normal exactly when the centroid lies inside the material.
    A sphere, an apple and a melon satisfy that; a ring does not -- its centroid sits in the
    hole, 0.3373 away from the nearest cell it has, and every direction from there to the
    surface comes out nearly horizontal. The consequence was not a small error: with a 60
    degree cone, `up` and `down` caught *zero* cells of the doughnut, so two of its six
    references were never read at all and its top was coloured by the upper halves of the
    four side views, which is what the white blotches across the glaze were.

    Occupancy has no such assumption. Bin the cells onto the lattice's own grid, smooth, and
    take the gradient: it points from filled toward empty everywhere on the boundary, which
    is the outward normal for any topology.

    But it is computed on a grid, so on a smooth surface it carries the grid's own steps,
    and replacing the analytic rule with it outright put concentric ridges across the orange
    -- the cone counts did not move at all, so that was the noise and nothing else. Neither
    rule is right everywhere and each says where it fails: where the two agree, the centroid
    ray leaves the material at that cell and the analytic normal is both correct and smooth;
    where they disagree, the centroid is not inside the material along that ray and only the
    gradient means anything. So cross over between them by their own agreement, over the same
    cone this file already uses for the faces. A sphere agrees everywhere and gets exactly
    the old behaviour; a ring's inner wall and its top and bottom disagree and get the
    gradient.
    """
    lo = world.min(0).values
    n = ((world.max(0).values - lo) / dx).long() + 3
    idx = ((world - lo) / dx).long() + 1
    occ = torch.zeros(tuple(int(v) for v in n), device=world.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    occ = torch.from_numpy(
        ndimage.gaussian_filter(occ.cpu().numpy(), sigma=sigma)).to(world.device)
    gx, gy, gz = torch.gradient(occ)
    si = ((xs - lo) / dx).long() + 1
    g = -torch.stack([gx[si[:, 0], si[:, 1], si[:, 2]],
                      gy[si[:, 0], si[:, 1], si[:, 2]],
                      gz[si[:, 0], si[:, 1], si[:, 2]]], 1)     # filled -> empty
    ln = g.norm(dim=1, keepdim=True)
    n_c = xs - centre
    n_c = n_c / n_c.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # a cell the gradient cannot resolve keeps the analytic rule rather than a random direction
    n_g = torch.where(ln > 1e-8, g / ln.clamp_min(1e-12), n_c)

    a = (n_c * n_g).sum(1, keepdim=True)
    w = ((a - COS_CONE) / (1.0 - COS_CONE)).clamp(0.0, 1.0)     # 1 -> analytic, 0 -> gradient
    n = w * n_c + (1.0 - w) * n_g
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    print(f"  法向：{int((w > 0.999).sum()):,} 格用質心解析、"
          f"{int((w < 0.001).sum()):,} 格用佔據梯度、"
          f"{int(((w >= 0.001) & (w <= 0.999)).sum()):,} 格混合"
          f"   (梯度解析成功 {int((ln > 1e-8).sum()):,}/{xs.shape[0]:,})")
    return n


def main(src, cfg, demo, ref_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0); g.load_ply_zero_sh(os.path.join(src, "gs_fill.ply"))
    lvl = torch.load(os.path.join(src, "cell_level.pt")).to(DEV)
    lat = torch.load(os.path.join(src, "lattice.pt"))

    par = load_params_from_gs(g, P())
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(par["pos"]); tpos = shift2center111(tpos)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)

    n_all = min(world.shape[0], lvl.shape[0])
    world, lvl = world[:n_all], lvl[:n_all]
    centre = world.mean(0)
    skin = (lvl == 1).nonzero().squeeze(1)
    xs = world[skin]
    nrm = surface_normals(world, xs, float(lat["coarse_dx"]), centre)

    acc = torch.zeros(xs.shape[0], 3, device=DEV)
    wsum = torch.zeros(xs.shape[0], 1, device=DEV)
    best_d = torch.full((xs.shape[0],), -2.0, device=DEV)
    hf_col = torch.zeros(xs.shape[0], 3, device=DEV)
    best_hf = torch.full((xs.shape[0],), -2.0, device=DEV)
    # SKIN_DEBUG replaces the shell's colour with the quantity named, so it can be rendered and
    # looked at. The pale marks on the side views survive every setting tried on them and are
    # not holes in the geometry -- a flat-coloured shell renders solid -- so the question is
    # which of the three things the loop does put pale colour there, and guessing has cost
    # enough already: `nface` how many cones contain the cell, `wsum` their total weight,
    # `rel` how far out in its reference the cell's own sample fell.
    cnt = torch.zeros(xs.shape[0], 1, device=DEV)
    relmax = torch.zeros(xs.shape[0], device=DEV)
    relmin = torch.full((xs.shape[0],), 9.0, device=DEV)
    # Directional appearance, instead of one colour per cell.
    #
    # With a single colour per primitive the four side references have to be reduced to one
    # value, and that average is where the peel goes: each reference is an independent
    # photograph of the same material, so its dimples sit somewhere else, and averaging four of
    # them cancels the texture while keeping the hue. Every loss written afterwards was trying
    # to recover detail that had already been discarded here.
    #
    # A cell can carry direction instead. Each face that sees it gives a (viewing direction,
    # colour) pair, so fit spherical harmonics to those pairs per cell and let the rasteriser
    # evaluate them: seen from the front the cell returns what `front` photographed, from the
    # right what `right` did, and in between the fit interpolates. Nothing is averaged away
    # where a reference actually looked, and the degree bounds how fast it may vary -- away
    # from every axis the fit falls back towards the mean on its own, which is the right
    # behaviour for a direction no photograph covers.
    #
    # Accumulated as normal equations, one small system per cell, because the number of faces
    # that see a cell varies and a padded design matrix would not fit.
    NB = (SH_DEG + 1) ** 2
    if SH_DEG > 0:
        ata = torch.zeros(xs.shape[0], NB, NB, device=DEV)
        atb = torch.zeros(xs.shape[0], NB, 3, device=DEV)
    face_id = torch.zeros(xs.shape[0], dtype=torch.uint8, device=DEV)

    faces = FACES
    _dj = os.path.join(ref_dir, "dirs.json")
    if os.path.exists(_dj):
        import json as _json
        _d = _json.load(open(_dj))
        faces = [(n, v[0], v[1]) for n, v in _d.items()]
        print(f"  directions from {_dj}: {len(faces)}")
    for fi, (name, az, el) in enumerate(faces):
        p = os.path.join(ref_dir, f"{name}_ref.png")
        if not os.path.exists(p):
            print(f"  missing {p}"); continue
        cam, _ = get_camera_view(demo, default_camera_index=-1,
            center_view_world_space=vc, observant_coordinates=oc, show_hint=False,
            init_azimuthm=az, init_elevation=el, init_radius=cam_p["init_radius"],
            move_camera=False, current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        # the face's axis, read off its own camera
        axis = (cam.camera_center.reshape(3).to(DEV) - centre)
        axis = axis / axis.norm()
        d = nrm @ axis

        img = torch.from_numpy(
            np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.).to(DEV)
        if FLATTEN:
            img = flatten_radial(img)
        H, W, _ = img.shape

        def proj(pts):
            hom = torch.cat([pts, torch.ones(pts.shape[0], 1, device=DEV)], 1)
            clip = hom @ cam.full_proj_transform
            ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-6)
            return torch.stack([(ndc[:, 0] + 1) * 0.5, (ndc[:, 1] + 1) * 0.5], 1), clip[:, 3]

        uv_all, w_all = proj(xs)
        uv_c, _ = proj(centre.reshape(1, 3))
        # radius of the object's silhouette in this render, so the reference -- whose fruit
        # was rescaled to fill its frame -- can be addressed in the same units
        rad = (uv_all - uv_c).norm(dim=1)
        # Read the middle of the reference, not out to its rim.
        #
        # The file says a face is used over its cone "and no further, so the outer eighteen
        # percent of each reference, where the generation put its shadow and its background,
        # is never read". That does not hold. A cell at the edge of a 60 degree cone projects
        # to sin(60) = 0.866 of the silhouette radius, and the reference has already stopped
        # being peel well inside that: measured on `front_ref`, saturation runs 0.962 at the
        # centre, 0.852 at 0.7-0.8 R, 0.602 at 0.8-0.87 R and 0.434 at the rim, because a
        # sphere's edge is rim light and the transition to background. The pale patches are
        # that light: the shell cells carrying them have colour (0.917, 0.727, 0.609) against
        # the rest of the shell's (0.912, 0.521, 0.193), and identical opacity and scale -- so
        # they are not thin, they were painted pale.
        #
        # Mapping the cone edge further in fixes it at the cost of magnifying the peel, since
        # a smaller patch of the reference is stretched over the same cone. It costs nothing
        # in coverage: the cone still covers the same cells, they just address a better part
        # of the picture.
        # rel = (uv - uv_c) / R0, so a *larger* R0 addresses further in. Dividing by REF_R
        # rather than multiplying is the difference between reading the middle of the
        # reference and reading past its edge: the multiply sent every cell outside the frame
        # to be clamped onto the border pixel, and the sphere came back one flat colour with a
        # gradient of 0.0004.
        _rr = REF_R
        if REF_R_NAMED is not None and not name.startswith(("r", "c")):
            _rr = REF_R_NAMED
        R0 = float(rad.quantile(0.995)) * (0.866 / _rr)

        m = (d > COS_CONE) & (w_all > 0)
        if int(m.sum()) == 0:
            continue
        # Where the cell actually appears in this photograph, which is its projection and not
        # its direction. Indexing by the normal's perpendicular component is exact for a
        # convex object and meaningless for anything else: a ring's topmost cells have no
        # perpendicular component at all, so they addressed the centre of the up reference --
        # which for a doughnut is the hole, and they came back white.
        if DIRECT:
            # The reference is a render from this very camera, so the mapping is the identity.
            #
            # Everything below assumes a generated reference: an image of *a* fruit, framed
            # however the generator framed it, which has to be matched to the render by
            # measuring both silhouettes and scaling one onto the other. That scaling is what
            # `R0`, `SKIN_REF_R` and the saturating map are for, and it is why the periphery of
            # each face comes back smooth -- the outer part of the cone is read from further in
            # than it belongs, or cut away entirely.
            #
            # When the references are six renders of a released model taken with these cameras,
            # none of that applies. A cell projects to a pixel and that pixel is its colour.
            # Nothing is estimated, so nothing is lost: the appearance should transfer exactly,
            # which is the whole reason for preferring real views over generated ones.
            rel = (uv_all[m] - torch.tensor([0.5, 0.5], device=DEV)) * 2.0
        else:
            rel = (uv_all[m] - uv_c) / R0                   # -1..1 across the silhouette
        # No sample may leave the fruit.
        #
        # rel is 1 at the edge of the silhouette and the mapping below sends 1 to the edge of
        # the *frame*, but a prepped reference has the fruit filling about 0.95 of the frame
        # and an anti-aliased rim inside that -- so anything above about 0.93 reads the rim or
        # the white background behind it. rel reaches 1.016. The cells that do it are the ones
        # that lie at the far edge of several cones at once, which is a small set of isolated
        # points, and every face contributing to such a point reads its own rim: the result is
        # the pale marks that have sat on the side views through every other change tried on
        # them. `SKIN_REF_R` was compensating for this by pulling the whole map inward, which
        # bought the marks at the cost of sampling every cell from too near its centre.
        #
        # Saturate rather than clamp, so that the bulk of the map is untouched and only the
        # last stretch is bent: identity below t0, asymptotic to REL_MAX above it.
        _r = rel.norm(dim=1, keepdim=True)
        _t0 = 0.85 * REL_MAX
        _sat = torch.where(_r <= _t0, _r,
                           _t0 + (REL_MAX - _t0) * (1 - torch.exp(-(_r - _t0)
                                                                  / (REL_MAX - _t0))))
        rel = rel * (_sat / _r.clamp_min(1e-9))
        # Bilinear, because the shell now out-resolves the photograph.
        #
        # Nearest neighbour was right while a cell covered more than a pixel of the reference.
        # The fine skin reversed that: its outermost layer has 458k cells against the 190k
        # pixels of the reference disc, so 2.4 cells read each source pixel and `round()` gives
        # them all the same value. The photograph's own pixel grid then appears on the fruit,
        # as blocks near the middle of each face and as arcs further out where the lines of
        # constant sampling radius run -- the concentric rings that showed up the moment the
        # lattice was refined, and that survived every change to the cone weighting because
        # they never came from it.
        fx = ((rel[:, 0] * 0.5 + 0.5) * (W - 1)).clamp(0, W - 1)
        fy = ((rel[:, 1] * 0.5 + 0.5) * (H - 1)).clamp(0, H - 1)
        x0, y0 = fx.floor().long(), fy.floor().long()
        x1 = (x0 + 1).clamp(max=W - 1)
        y1 = (y0 + 1).clamp(max=H - 1)
        ax, ay = (fx - x0.float()).unsqueeze(1), (fy - y0.float()).unsqueeze(1)
        col = ((1 - ay) * ((1 - ax) * img[y0, x0] + ax * img[y0, x1])
               + ay * ((1 - ax) * img[y1, x0] + ax * img[y1, x1]))
        px, py = x0, y0                       # the HF path indexes with these
        # Zero at the edge of the cone, not a quarter of the way up it.
        #
        # `d ** SHARP` never reaches zero inside the cone: at the boundary d is COS_CONE, so
        # with SHARP=2 a face still carries 0.25 of the weight there and carries 0 a hair
        # further out. Normalising by the total does not remove that step, it just moves it
        # into the ratio, and the step is a seam -- a visible edge around each face where the
        # colour changes discontinuously. Rescaling d so the cone boundary maps to 0 makes the
        # weight vanish where the face stops being used, which is the only way the six
        # references can meet without a line between them.
        # CONE_ZERO=1 makes the weight vanish at the cone edge instead of stopping at
        # COS_CONE**SHARP. It removes the step, and it also removes the colour: 52.5% of
        # directions lie in exactly one cone, and a cell near that cone's edge then has a
        # total weight near zero, so its colour is whatever the division by wsum leaves. The
        # pale patches at the poles are that. Default off -- the version that produced the
        # three good exteriors on 5 August.
        _dd = d[m].reshape(-1, 1)
        wt = ((((_dd - COS_CONE) / (1.0 - COS_CONE)).clamp(0.0, 1.0) ** SHARP)
              if CONE_ZERO else (_dd ** SHARP))
        # The named faces carry the features; the scattered ones only carry uniformity.
        #
        # Scattering the directions removed the seams and removed the stem scar with them. A
        # seam is a local difference that should not be there and a calyx is a local difference
        # that should, and an average cannot tell them apart: with thirty-two directions the
        # `up` reference, the only one showing the scar, holds about a thirty-second of the
        # weight over the pole, and the pole came back as blank peel.
        #
        # The distinction is available and was not being used. `up`, `down`, `front` and the
        # rest were each prompted for what belongs on them; the filler directions all fall back
        # to the `front` prompt and are, by construction, plain peel. So weight them apart:
        # a named face dominates its own cone, the fillers cover the gaps between.
        if NAMED_W != 1.0 and not name.startswith(("r", "c")):
            wt = wt * NAMED_W
        # A face does not get a vote on a cell it only sees edge-on.
        #
        # The weight so far depends on the angle between the cell and the face axis, which is
        # the right thing for blending but says nothing about whether the *photograph* has
        # anything usable at the place the cell reads. Past about three quarters of the frame
        # a photograph of a sphere is rim -- bright, pale and compressed -- and a cell that
        # every reachable face reads at its rim comes out as an average of rims, which is the
        # pale mark. Cutting the sample rather than the cone leaves each cell to the faces that
        # actually see it as peel, and `relmin` says only a hundred and fifty cells out of six
        # hundred thousand are left with none, which the fill below covers.
        if DIRECT:
            # A face may only colour a cell where that face actually sees fruit.
            #
            # The cone mapping never needed this: it rescaled every read to a fraction of the
            # measured silhouette, so a sample could not leave the fruit by construction. Direct
            # projection has no such guarantee -- a cell whose projection lands on the
            # anti-aliased edge, or just past it where the two silhouettes disagree by a pixel,
            # reads the white background and carries it onto the peel. Four such blobs appeared
            # on the front view, at the points where several cones end together and every one of
            # them is reading its own rim.
            #
            # The reference says where its own fruit is, so ask it: weight zero on background.
            # Sampled the same way the colour is, and used as a weight rather than a test.
            #
            # Testing the rounded pixel while reading a bilinear average of four is a hole
            # exactly one pixel wide, and the fruit's edge is where it opens: the integer pixel
            # is peel, so the sample is accepted, while the average it actually returns is part
            # background. Across several faces that accumulates, and 1.4% of the shell ended up
            # holding (0.93, 0.84, 0.76) -- near-white, in the cell colour, which is why a blue
            # background rendered the streaks unchanged.
            #
            # A mask sampled bilinearly answers what the colour actually is: 1 where all four
            # neighbours are fruit, 0 where none are, and in between it is the fraction of the
            # sample that is fruit -- which is exactly how much that sample deserves to count.
            # The reference's own alpha, when it has one.
            #
            # Coverage cannot be recovered from colour. A rim pixel that is half peel and half
            # background is dilute peel: its darkest channel is around 0.46, so every threshold
            # accepts it as fruit, and what it hands over is (0.93, 0.84, 0.76) -- the warm
            # off-white that 1.4% of the shell was holding. Alpha is the quantity being guessed
            # at, and a render can simply write it out, so it does.
            _ap = os.path.join(ref_dir, f"{name}_alpha.png")
            if os.path.exists(_ap):
                _fg = torch.from_numpy(
                    np.asarray(Image.open(_ap).convert("L"), np.float32) / 255.).to(DEV)
            else:
                _fg = (img.min(2).values <= DIRECT_FG).to(img.dtype)
            _fgw = ((1 - ay.squeeze(1)) * ((1 - ax.squeeze(1)) * _fg[y0, x0] + ax.squeeze(1) * _fg[y0, x1])
                    + ay.squeeze(1) * ((1 - ax.squeeze(1)) * _fg[y1, x0] + ax.squeeze(1) * _fg[y1, x1]))
            wt = wt * (_fgw.reshape(-1, 1) ** 4).to(wt.dtype)
        if REL_CUT < 1.0:
            # Faded, not switched. A step at rel = 0.75 draws a circle on the sphere for every
            # face -- the locus where that face stops contributing -- and while the cells were
            # 1.2 pixels across the circle was lost in the blur. At 0.3 pixels it is a set of
            # concentric rings around each face axis, plainly visible. The weight has to reach
            # zero before the rim, not jump there, so roll it off over the last fifth.
            _rr = rel.norm(dim=1, keepdim=True)
            wt = wt * ((REL_CUT - _rr) / (0.2 * REL_CUT)).clamp(0.0, 1.0)
        idx = m.nonzero().squeeze(1)
        acc.index_add_(0, idx, col * wt)
        wsum.index_add_(0, idx, wt)
        cnt.index_add_(0, idx, torch.ones_like(wt))
        if SH_DEG > 0:
            # the direction the rasteriser will evaluate at: camera to point, normalised,
            # exactly as `convert_SH` builds it
            _dv = xs[idx] - cam.camera_center.reshape(1, 3).to(DEV)
            _dv = _dv / _dv.norm(dim=1, keepdim=True).clamp_min(1e-9)
            _Y = sh_basis(_dv, SH_DEG)                                  # (M, NB)
            _w = wt.reshape(-1, 1)
            ata.index_add_(0, idx, _w.unsqueeze(2) * (_Y.unsqueeze(2) @ _Y.unsqueeze(1)))
            atb.index_add_(0, idx, (_w * _Y).unsqueeze(2) * (col - 0.5).unsqueeze(1))
        relmax[idx] = torch.maximum(relmax[idx], rel.norm(dim=1))
        _rn = rel.norm(dim=1)
        relmin[idx] = torch.minimum(relmin[idx],
                                    torch.where(wt.reshape(-1) > 0, _rn,
                                                torch.full_like(_rn, 9.0)))

        # Colour from many faces, texture from one.
        #
        # Averaging is what removes a seam and it is also what removes the peel, and the two
        # cannot be separated by tuning the weights because they are the same operation. Each
        # reference is an independent sample, so the dimpling sits in a different place in
        # each: with six faces 52.5% of directions lie in a single cone and little averaging
        # happens (gradient 0.141), with fourteen every direction lies in three or four and
        # the textures cancel (0.034) even though every one of the fourteen is individually
        # sharp, 0.154 to 0.176 against the reference's 0.174.
        #
        # Splitting by frequency takes both. The low frequency -- the hue and the shading --
        # is what differs between faces and what has to be blended for the seam to go. The
        # high frequency is the peel, it is equally valid from any face, and it only has to
        # come from one. So blend the blurred colour, and add back the detail of the nearest
        # face alone.
        if HF_FROM_NEAREST:
            lo = _blur(img, HF_SIGMA)[py, px]
            acc[idx] -= col * wt                        # replace the accumulated low part
            acc.index_add_(0, idx, lo * wt)
            # both sides one-dimensional: comparing (M,) against (M,1) broadcasts to (M,M),
            # which at eighty thousand cells asks for 25 GB
            dm = d[m]
            take = dm > best_hf[idx]
            if bool(take.any()):
                sel = idx[take]
                hf_col[sel] = (col - lo)[take]
                best_hf[sel] = dm[take]

        newer = (d > best_d) & (fi < FACE_LABELS)
        best_d = torch.where(newer, d, best_d)
        face_id = torch.where(newer, torch.full_like(face_id, fi), face_id)
        print(f"  {name:<6} 錐內 {int(m.sum()):>7,} 格   軸 "
              f"{[round(float(v),2) for v in axis]}")

    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)[:n_all]
    # Every shell cell, not only the ones some face voted on. With the rim samples cut, a few
    # hundred cells end with no vote at all, and excluding them here left them at the lattice's
    # initial grey -- which is how the white marks the cut removed came back as grey specks in
    # the same four places. They are covered by the fill below instead.
    # "Served" has to mean a real vote, not a non-zero float.
    #
    # `acc / wsum` is a weighted mean and is only meaningful while wsum is a weight. The cone
    # weighting falls smoothly to zero at its edge and the direct mode's foreground guard zeroes
    # more of it, so a cell can come out of the loop holding a total weight of order 1e-9 --
    # and 1e-9 passes a test against 1e-9. What that division returns is numerical noise, and
    # since the accumulator is a colour the noise renders as bright streaks along the cone
    # boundaries, in the cell colours themselves: a blue background renders them identically,
    # so nothing is showing through, the white is genuinely painted on.
    #
    # A threshold relative to what a normal cell collects separates the two cases. Below it a
    # cell has not been seen well enough by anything and belongs to the fill.
    _wq = float(wsum.squeeze(1).median()) if wsum.numel() else 1.0
    _served = wsum.squeeze(1) > max(_wq * SERVED_FRAC, 1e-9)
    got = torch.ones_like(_served)
    wsum = wsum.clamp_min(max(_wq * SERVED_FRAC, 1e-9))
    print(f"  median total weight {_wq:.4f}; {int((~_served).sum()):,} cells "
          f"({100 * float((~_served).float().mean()):.2f}%) below {SERVED_FRAC:g} of it")
    new = rgb.clone()
    skin_col = (acc[got] / wsum[got]).clamp(0, 1)
    _dbg = os.environ.get("SKIN_DEBUG", "")
    if _dbg:
        q = {"nface": cnt.squeeze(1), "wsum": wsum.squeeze(1), "rel": relmax}[_dbg][got]
        lo, hi = float(q.min()), float(q.max())
        print(f"  SKIN_DEBUG {_dbg}: min {lo:.4f}  median {float(q.median()):.4f}  max {hi:.4f}")
        t = ((q - lo) / max(hi - lo, 1e-9)).reshape(-1, 1)
        new[skin[got]] = torch.cat([t, 1.0 - (2 * t - 1).abs(), 1.0 - t], 1)  # blue low, red high
        g._features_dc = ((new - 0.5) / C0).unsqueeze(1).to(g._features_dc.device)
        os.makedirs(out_dir, exist_ok=True)
        g.save_ply(os.path.join(out_dir, "gs_fill.ply"))
        print(f"  -> {out_dir}  ({_dbg} as colour)")
        return
    if HF_FROM_NEAREST:
        skin_col = (skin_col + hf_col[got]).clamp(0, 1)
        print(f"  detail from the nearest face only, cut at sigma {HF_SIGMA}px: "
              f"mean |detail| {float(hf_col[got].abs().mean()):.4f}")

    # Cells no reference sees well, filled from the ones next to them.
    #
    # `relmin` is how far into its *best* reference a cell falls. Almost everywhere some face
    # has it near the middle of the frame, where the peel is peel. At a handful of isolated
    # directions -- the points that lie at the far edge of every cone that reaches them, four
    # of them around the equator on this object -- the best any face manages is its own rim,
    # and rim in a photograph of a sphere is bright, pale and geometrically compressed. Every
    # contributing face reads rim there, so no weighting between them helps: the average of
    # eight rims is a rim, which is what the pale marks are. They survived a rebuilt reference
    # set, a clamped sampling radius, cone-zero weighting on and off, and dividing the light
    # out of the photographs.
    #
    # The condition is geometric and known before the colour is read, so use it as one. A cell
    # whose best view of itself is its rim has no colour of its own to read, and the peel
    # around it does -- so take theirs. The set is a few thousand cells in isolated spots, so
    # the fill is over in a few rounds and nothing else moves.
    if REL_POOR < 1.0:
        poor = ((relmin[got] > REL_POOR) | ~_served[got]).cpu().numpy()
        if poor.any():
            from scipy.spatial import cKDTree
            pts = xs[got].detach().cpu().numpy()
            _, nb = cKDTree(pts).query(pts, k=13)
            nb = nb[:, 1:]
            col_np = skin_col.detach().cpu().numpy()
            _poor0 = poor.copy()          # who was seeded and who was invented
            n0, rounds = int(poor.sum()), 0
            while poor.any() and rounds < 60:
                ok = (~poor)[nb]
                can = poor & (ok.sum(1) > 0)
                if not can.any():
                    break
                w = ok[can].astype(np.float32)[:, :, None]
                col_np[can] = (col_np[nb[can]] * w).sum(1) / w.sum(1)
                poor = poor & ~can
                rounds += 1
            # Relax the filled region, with the cells that had real data held fixed.
            #
            # Filling in rounds, each taking the mean of whatever is already filled, leaves the
            # rounds themselves in the result: every pass is a plateau one ring further in, and
            # at the coarse lattice they were half a pixel wide and invisible while at the fine
            # one they are the concentric rings around each patch -- fourteen fills, fourteen
            # rings. What is wanted is the smooth function that agrees with the data on the
            # boundary, so keep the seeded values and let the interior settle to their average.
            _fix = torch.from_numpy(~_poor0).to(DEV)
            _c = torch.from_numpy(col_np).to(DEV)
            _nb = torch.from_numpy(nb.astype(np.int64)).to(DEV)
            _keep = _c.clone()
            for _ in range(SH_FILL_RELAX):
                _c = _c[_nb].mean(1)
                _c = torch.where(_fix.unsqueeze(1), _keep, _c)
            col_np = _c.cpu().numpy()
            print(f"  {n0:,} cells ({100 * n0 / len(col_np):.2f}%) had no face nearer than "
                  f"{REL_POOR} of its own rim; filled in {rounds} rounds, then relaxed "
                  f"{SH_FILL_RELAX} times")
            skin_col = torch.from_numpy(col_np).to(DEV)

    # A peel has two sides and this only had one. Every cell along a ray took the same colour
    # from the same photograph, so both layers of the shell came out the outside's orange --
    # measured (0.881,0.396,0.149) and (0.878,0.384,0.132) -- and a cut through it showed a
    # rim indistinguishable from the flesh. The photographs say otherwise: the transverse
    # section's outer band is (0.987,0.908,0.643), nearly white, because what a cut exposes is
    # albedo and not peel.
    #
    # Depth is available and was being ignored. Fade each shell cell from the face colour at
    # the surface toward the albedo colour at the inner edge of the shell, over ALBEDO_POW
    # controlling how quickly. Training cannot supply this on its own: the exterior branch
    # sees six views of the whole surface every iteration and the sections see the rim as a
    # thin ring, so the outside wins the shared head and the rim stays orange, which is what
    # sixty iterations did -- (0.967,0.626,0.237) at the start, (0.954,0.615,0.222) at the end.
    _alb = _os.environ.get("ALBEDO_RGB", "")
    if _alb:
        alb = torch.tensor([float(x) for x in _alb.split(",")], device=DEV)
        pw = float(_os.environ.get("ALBEDO_POW", "1.0"))
        rr = (xs[got] - centre).norm(dim=1)
        r_out, r_in = float(rr.max()), float(rr.min())
        # 0 at the outer surface, 1 at the inner edge of the shell
        t = ((r_out - rr) / max(r_out - r_in, 1e-9)).clamp(0, 1).reshape(-1, 1) ** pw
        skin_col = (1 - t) * skin_col + t * alb.reshape(1, 3)
        print(f"  albedo fade to {[round(float(v),3) for v in alb]} over {r_out - r_in:.5f} "
              f"(pow {pw}): surface {[round(float(v),3) for v in skin_col[t.squeeze(1) < 0.2].mean(0)]}"
              f" -> inner {[round(float(v),3) for v in skin_col[t.squeeze(1) > 0.8].mean(0)]}")
    # Smooth the coloured shell in space, which is not the same operation as blending the
    # references in image space and does not have its cost.
    #
    # Blending mixes *content*: at a point between two faces the colour becomes a weighted mean
    # of two pictures, so softening the weights pulls in faces that are wrong for that point
    # and the seam ring goes from 1.66 times the surrounding contrast at SHARP 2 to 1.98 at
    # SHARP 0.5, while the texture goes with it. Softer blending makes the seam worse, and
    # measuring it on the ring where the cones meet rather than over the whole sphere is what
    # made that visible -- a global high-frequency statistic scores a uniformly duller sphere
    # as an improvement.
    #
    # What a seam actually is, is a step in a field that should be continuous. Averaging the
    # field itself over a small neighbourhood turns the step into a ramp a few cells wide and
    # leaves everything away from it alone, because a smooth region is its own average. The
    # radius is the control: at a fraction of a cell it does nothing, and much beyond the width
    # of the peel's dimpling it starts costing the texture the blending was already costing.
    _sm = float(_os.environ.get("SKIN_SMOOTH", "0"))
    if _sm > 0:
        from scipy import ndimage as _nd
        pts = xs[got].detach().cpu().numpy()
        col = skin_col.detach().cpu().numpy()
        dxf = float(lat["coarse_dx"]) * 0.5
        idx = np.round((pts - pts.min(0)) / dxf).astype(np.int64)
        dims = tuple(int(t) for t in idx.max(0) + 1)
        occ = np.zeros(dims, np.float32)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        # normalise by the smoothed occupancy, so a cell at the surface is not dragged toward
        # the empty space outside it
        w = _nd.gaussian_filter(occ, sigma=_sm)
        out = np.empty_like(col)
        for c in range(3):
            vol = np.zeros(dims, np.float32)
            vol[idx[:, 0], idx[:, 1], idx[:, 2]] = col[:, c]
            sm = _nd.gaussian_filter(vol, sigma=_sm)
            out[:, c] = (sm / np.maximum(w, 1e-6))[idx[:, 0], idx[:, 1], idx[:, 2]]
        d0 = float(np.abs(out - col).mean())
        print(f"  voxel smoothing of the shell, sigma {_sm} fine cells: "
              f"mean colour change {d0:.4f}")
        skin_col = torch.from_numpy(np.clip(out, 0, 1)).to(DEV)
    new[skin[got]] = skin_col
    if SH_DEG > 0:
        # Solve each cell's little system. The ridge term is what makes it well posed for the
        # cells only two or three faces reach, and it pulls those towards a flat, direction-
        # independent colour -- which is the honest answer when almost nothing looked there.
        # The DC term is fixed to `skin_col` and the rest is fitted *against* it.
        #
        # Every correction in this file -- the albedo fade through the peel, the neighbour fill
        # where no face sees a cell well, the detail taken from the nearest face -- acts on the
        # mean colour, so the mean has to stay what they made it. But it is not enough to solve
        # for all the coefficients and then swap the DC afterwards: the higher orders were
        # fitted alongside the DC the solve chose, and standing them on a different one leaves
        # whatever the two disagree about as a directional term. That rendered as a spotlight,
        # a hot orange patch facing the camera with the limb gone dark.
        #
        # Constraining it is exact and costs nothing. With k0 known, the remaining coefficients
        # solve A11 k1 = b1 - A10 k0, which is the same normal equations with the first row and
        # column moved to the right-hand side. What is left is a pure residual about the mean.
        # SKIN_SH_FREE: fit every coefficient, mean included, and do not blend at all.
        #
        # The constrained version keeps the file's own mean and gives the fit only the
        # variation about it -- but that mean is still a weighted average of thirty-two
        # photographs, and averaging photographs is what this whole exercise is trying to stop
        # doing. It does not show while a cell spans more than a pixel; on the fine lattice it
        # shows as arcs across the peel, wherever the set of faces contributing to the average
        # changes. The `rel` debug map has the same arcs, and they survived the cone weighting,
        # the sampling cut, the saturating radius map, the fill, the detail term and the normal
        # estimator, because none of those is what draws them.
        #
        # Fitting freely removes the average from the pipeline entirely: from a face's own
        # direction the cell returns that face's photograph, and between directions the fit
        # interpolates. Nothing is mixed, so there is no boundary for a mixture to change at.
        # The cost is that the corrections that act on the mean -- the albedo fade, the
        # neighbour fill -- no longer apply, so this is only usable where every cell has data.
        if SH_FREE:
            _eyef = torch.eye(NB, device=DEV).unsqueeze(0)
            _lamf = SH_RIDGE * ata[got].diagonal(dim1=1, dim2=2).mean(1).clamp_min(
                1e-6).reshape(-1, 1, 1)
            kf = torch.linalg.solve(ata[got] + _lamf * _eyef, atb[got])   # (M, NB, 3)
            skin_col = (kf[:, 0, :] * C0 + 0.5).clamp(0, 1)
            rest = torch.zeros(world.shape[0], NB - 1, 3, device=DEV)
            rest[skin[got]] = kf[:, 1:, :]
            g._features_rest = nn.Parameter(rest.to(g._features_dc.device))
            g.max_sh_degree = SH_DEG
            g.active_sh_degree = SH_DEG
            new[skin[got]] = skin_col
            print(f"  free spherical-harmonic fit, degree {SH_DEG}: no blended mean; "
                  f"mean |directional coefficient| {float(kf[:, 1:, :].abs().mean()):.4f}")
            _skip_constrained = True
        else:
            _skip_constrained = False
        _k0 = ((skin_col - 0.5) / C0).unsqueeze(1)                      # (M, 1, 3)
        # Anchor the directions nothing looked from to the mean.
        #
        # Every face that sees a cell sees it from within sixty degrees of its own axis, so the
        # fit has no data anywhere near the tangent -- and the tangent is exactly where the cell
        # appears when it is on the silhouette. A polynomial asked to extrapolate that far runs
        # away, which rendered as a dark ring right around the limb, worse at degree 2 because
        # it runs quadratically. A ridge damps it only by flattening the fit everywhere, and
        # buys the ring back at the cost of the texture it was fitted for.
        #
        # Pseudo-observations do it without that trade. Around each cell's tangent circle, put
        # a ring of directions whose target colour *is* the mean. Their residual is zero by
        # construction, so they add nothing to the right-hand side and only shape the normal
        # equations: a ridge that acts along the directions no photograph covers and leaves the
        # ones it does alone. Where a face looked, the fit still reproduces what it saw; where
        # none did, the cell shows its average.
        if SH_TANGENT > 0:
            _n = (xs[got] - centre)
            _n = _n / _n.norm(dim=1, keepdim=True).clamp_min(1e-9)
            _u = torch.cross(_n, torch.tensor([0., 0., 1.], device=DEV).expand_as(_n), dim=1)
            _bad = _u.norm(dim=1) < 1e-3
            _u[_bad] = torch.cross(_n[_bad],
                                   torch.tensor([0., 1., 0.], device=DEV).expand_as(_n[_bad]),
                                   dim=1)
            _u = _u / _u.norm(dim=1, keepdim=True).clamp_min(1e-9)
            _v = torch.cross(_n, _u, dim=1)
            _wt = SH_TANGENT * ata[got].diagonal(dim1=1, dim2=2)[:, :1].clamp_min(1e-6)
            for _th in torch.linspace(0, 2 * np.pi, 9, device=DEV)[:-1]:
                _t = torch.cos(_th) * _u + torch.sin(_th) * _v
                _Yt = sh_basis(_t, SH_DEG)
                ata[got] += (_wt.reshape(-1, 1, 1)
                             * (_Yt.unsqueeze(2) @ _Yt.unsqueeze(1)))
        _A = ata[got]
        _eye = torch.eye(NB - 1, device=DEV).unsqueeze(0)
        _lam = SH_RIDGE * _A.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-6).reshape(-1, 1, 1)
        _rhs = atb[got][:, 1:, :] - _A[:, 1:, 0:1] * _k0
        k1 = (torch.zeros_like(_rhs) if _skip_constrained else
              torch.linalg.solve(_A[:, 1:, 1:] + _lam * _eye, _rhs))   # (M, NB-1, 3)
        rest = torch.zeros(world.shape[0], NB - 1, 3, device=DEV)
        if not _skip_constrained:
            rest[skin[got]] = k1
            g._features_rest = nn.Parameter(rest.to(g._features_dc.device))
        g.max_sh_degree = SH_DEG
        g.active_sh_degree = SH_DEG
        _sp = k1.abs().mean()
        print(f"  spherical harmonics degree {SH_DEG}: {NB - 1} directional terms per cell, "
              f"mean |coefficient| {float(_sp):.4f}")
    print(f"\n  shell {skin.shape[0]:,} 格   上到色 {int(got.sum()):,} "
          f"({float(got.float().mean())*100:.1f}%)")
    print(f"  shell 平均 RGB {[round(float(v),3) for v in rgb[skin].mean(0)]} -> "
          f"{[round(float(v),3) for v in new[skin].mean(0)]}")
    for fi, (name, _, _) in enumerate(FACES[:FACE_LABELS]):
        c = int((face_id == fi).sum())
        print(f"    {name:<6} {c:>7,} 格 ({c/max(len(face_id),1)*100:4.1f}%)")

    # per-cell face label for every cell, not only the shell: an interior cell takes the
    # label of the direction it lies in, which is the same question asked of it later.
    all_nrm = world - centre
    all_nrm = all_nrm / all_nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)
    axes = []
    for name, az, el in faces:
        cam, _ = get_camera_view(demo, default_camera_index=-1,
            center_view_world_space=vc, observant_coordinates=oc, show_hint=False,
            init_azimuthm=az, init_elevation=el, init_radius=cam_p["init_radius"],
            move_camera=False, current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        a = cam.camera_center.reshape(3).to(DEV) - centre
        axes.append(a / a.norm())
    all_face = torch.stack([all_nrm @ a for a in axes[:FACE_LABELS]], 1).argmax(1).to(torch.uint8)

    with torch.no_grad():
        g._features_dc = nn.Parameter(((new - 0.5) / C0).unsqueeze(1).contiguous())
        for a in ["_xyz", "_opacity", "_scaling", "_rotation"]:
            setattr(g, a, nn.Parameter(getattr(g, a).detach()[:n_all].contiguous()))
        # Keep the directional terms if there are any. This trimmed every primitive to the
        # active count, and wrote an empty rest tensor while doing it -- which silently undid
        # the whole spherical-harmonic fit two lines before the file was written, so the ply
        # came out with zero `f_rest` fields and rendered as its own mean colour.
        g._features_rest = nn.Parameter(g._features_rest.detach()[:n_all].contiguous()
                                        if SH_DEG > 0 else
                                        torch.zeros(n_all, 0, 3, device=DEV))
        g.max_radii2D = torch.zeros(n_all, device=DEV)
        g.trained = torch.zeros(n_all, dtype=torch.bool)
        g.is_interior = torch.ones(n_all, dtype=torch.bool)
    g.save_ply(os.path.join(out_dir, "gs_fill.ply"))
    torch.save(torch.ones(n_all, dtype=torch.bool), os.path.join(out_dir, "is_interior.pt"))
    torch.save(lvl.cpu(), os.path.join(out_dir, "cell_level.pt"))
    torch.save(lat, os.path.join(out_dir, "lattice.pt"))
    torch.save({"face": all_face.cpu(),
                "names": [n for n, _, _ in faces[:FACE_LABELS]],
                "axes": torch.stack(axes[:FACE_LABELS]).cpu()},
               os.path.join(out_dir, "cell_face.pt"))
    print(f"  -> {out_dir}  (cell_face.pt 已寫入)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("cfg"); ap.add_argument("demo")
    ap.add_argument("ref_dir"); ap.add_argument("out_dir")
    a = ap.parse_args()
    main(a.src, a.cfg, a.demo, a.ref_dir, a.out_dir)
