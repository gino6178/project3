"""Generate the interior *material*, then put it where the geometry says the material is.

Generating a shaped section asks the sampler two questions at once -- what is this made of,
and what shape is the cut -- and it answers the second one from the outline. That is fine
while the outline is uninformative and fails as soon as it is not: a ring's transverse face
is an annulus, an annulus is a doughnut's own outline, and every generation came back as a
doughnut seen from above. Prompting against it moves the failure rather than removing it
(radial fibres one way, a smooth glazed ring the other), and no strength setting helps,
because the outline is in the conditioning image either way.

The shape was never the sampler's to decide. The lattice knows exactly what the cut face
looks like -- it is the occupancy in the plane, and it is right for any topology. So ask the
sampler only for the material, on a frame that the face fills completely and whose boundary
lies outside the image, and put the answer into the silhouette afterwards:

    material   full-frame, no outline anywhere in the conditioning -> nothing to misread
    silhouette from the lattice, so a ring stays a ring and two tube faces stay two
    crust      a band along every edge of the face, in the colour the object's own six
               exterior references show

Nothing here knows which object it is running on, and no step can be defeated by the outline
resembling something.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os, argparse

sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from diffusers import StableDiffusionDepth2ImgPipeline
from scene.gaussian_model import GaussianModel
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs, initialize_resterize, convert_SH
from utils.transformation_utils import *
from utils.camera_view_utils import get_camera_view


DEV = "cuda:0"
# The material must not be a picture of anything. Everything here names an object or a scene.
NEG = ("whole object, uncut object, exterior view, seen from outside, silhouette, outline, "
       "isolated on white, product photo, plate, table, countertop, background, shadow, "
       "watermark, text, logo, border, frame, vignette")


class P: convert_SHs_python = False; compute_cov3D_python = True; debug = False


def _fill_speckle(m, min_frac=0.05):
    """Close the holes thresholding leaves, keep the ones the object actually has.

    binary_fill_holes closes both, and the difference matters: a ring cut through the plane
    of the ring is an annulus, and filling its hole hands the sampler a solid disc to make a
    section out of. That is why the doughnut's section reference came back as a solid lump of
    crumb with no hole -- and why every one of the sixteen transverse planes then had a
    reference it was not homeomorphic to, and had to fall back to the depth mapping. Real
    holes are large: a doughnut's is about a fifth of its outer disc, thresholding speckle is
    a few tenths of a percent.
    """
    filled = ndimage.binary_fill_holes(m)
    holes, n = ndimage.label(filled & ~m)
    if n == 0:
        return filled
    area = max(int(m.sum()), 1)
    out = filled.copy()
    for i in range(1, n + 1):
        h = holes == i
        if int(h.sum()) >= min_frac * area:
            out[h] = False
    return out


def render_face(src, cfg, demo, size=512, az=90.0, el=0.0):
    """Cut the object with the plane the camera faces, and render the exposed face."""
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0); g.load_ply_zero_sh(os.path.join(src, "gs_fill.ply"))
    par = load_params_from_gs(g, P())
    pos0, cov0 = par["pos"], par["cov3D_precomp"]
    sp, op, shs = par["screen_points"], par["opacity"], par["shs"]
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(pos0); tpos = shift2center111(tpos)
    cov0 = apply_cov_rotations(cov0, rot_m); cov0 = so * so * cov0
    cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    cam, _ = get_camera_view(demo, default_camera_index=-1,
        center_view_world_space=vc, observant_coordinates=oc, show_hint=False,
        init_azimuthm=az, init_elevation=el, init_radius=cam_p["init_radius"],
        move_camera=False, current_frame=0, delta_a=None, delta_e=None, delta_r=None)
    c = world.mean(0)
    # the plane is the one the camera looks down, so the face is seen straight on
    nrm = cam.camera_center.reshape(3).to(DEV) - c
    nrm = nrm / nrm.norm()
    # A thin slab at the plane, not the solid half behind it. The half shows the face plus
    # everything under it, which for a ring is the whole arc joining its two tube faces --
    # the largest connected part of that is the arc, and the sampler was handed a bent tube
    # to make a section out of. A slab is the section: cut a ring this way and you see two
    # round dough faces, cut a fruit and you see one disc.
    _lat = torch.load(os.path.join(src, "lattice.pt"))
    thick = 1.5 * float(_lat["coarse_dx"])
    keep = (((world - c) @ nrm).abs() < thick)
    rast = initialize_resterize(cam, g, P(), torch.tensor([1., 1., 1.], device=DEV),
                                image_height=size, image_width=size)
    col = convert_SH(shs[keep], cam, g, world[keep], None)
    img, _, _, _ = rast(means3D=world[keep], means2D=sp[keep], shs=None, colors_precomp=col,
                        opacities=op[keep], scales=None, rotations=None,
                        cov3D_precomp=cov[keep])
    return img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()


def cut_mask(src, cfg, demo, az, el, size=512, pad=0.10, min_frac=0.02):
    """The cut face as the lattice has it: every part of it, at its own relative size."""
    a = render_face(src, cfg, demo, size, az, el)
    bg = np.median(np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)]), 0)
    m = _fill_speckle(np.abs(a - bg).max(2) > 0.06)
    lab, k = ndimage.label(m)
    if k > 1:
        # Keep every real part. Taking only the largest is what reduced a ring's longitudinal
        # face -- two tube faces -- to one, and then filled the frame with it.
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_frac * sizes.max()])
    ys, xs = np.where(m)
    s = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * pad))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    mp = np.pad(m, s)
    cm = mp[cy - s // 2 + s:cy - s // 2 + 2 * s, cx - s // 2 + s:cx - s // 2 + 2 * s]
    return cv2.resize(cm.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0


def exterior_colour(prep_dir, faces=("front", "right", "back", "left")):
    """The object's own crust colour, from the references it was given, not from a constant."""
    px = []
    for f in faces:
        p = os.path.join(prep_dir, f"{f}_ref.png")
        if not os.path.exists(p):
            continue
        a = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        px.append(a.reshape(-1, 3))
    px = np.concatenate(px)
    return np.median(px[px.max(1) < 0.95], 0)


def make_material(pipe, prompt, base, size=512, strength=0.92, seed=1234, steps=50):
    """A full frame of interior material. The face fills it, so there is no outline to read."""
    rng = np.random.default_rng(seed)
    # Start from the object's own colour with grain, not from noise: img2img needs something
    # with the right statistics or it reaches for a subject.
    img = np.clip(base + rng.normal(0, 0.06, (size, size, 3)).astype(np.float32), 0, 1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    # Flat across the frame, with a thin border of background so the depth has a range at all.
    # The pipeline rescales whatever it is given to [-1, 1], so a genuinely constant map is
    # a division by zero -- it returns a black image -- and a nearly constant one has its
    # noise stretched to full depth. A border keeps the interior flat and says nothing about
    # what the object is: the only outline in the conditioning is the frame itself.
    d = torch.full((1, size, size), 1.0, device=DEV)
    b = max(6, size // 64)
    d[0, :b, :] = 0.15; d[0, -b:, :] = 0.15; d[0, :, :b] = 0.15; d[0, :, -b:] = 0.15
    r = pipe(prompt=prompt, image=Image.fromarray((img * 255).astype(np.uint8)), depth_map=d,
             negative_prompt=NEG, strength=float(strength), guidance_scale=9,
             num_inference_steps=steps,
             generator=torch.Generator(pipe.device).manual_seed(int(seed)), return_dict=False)
    out = r[0][0] if isinstance(r, tuple) else r.images[0]
    return np.asarray(out, np.float32) / 255.


def material_from_image(path, size=512, rim_frac=0.06):
    """The interior material of a photographed cut face, and the crust colour at its edge.

    A supplied photograph is worth more than a generated one and costs nothing to accept: the
    sampler here only ever had to invent the material, and a picture of the material answers
    that outright. It does not have to be the cut we are building -- material is material, and
    the shape comes from the lattice either way. So one photograph of a doughnut's transverse
    face also furnishes its longitudinal faces, which no photograph in the set shows.

    Take the object, erode away the crust so the material is not contaminated by the edge, and
    return the largest square that fits inside what is left.
    """
    a = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    bg = np.median(np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)]), 0)
    # _fill_speckle, not binary_fill_holes: a doughnut's hole is a hole, and filling it puts
    # the distance transform's peak inside it, so the "largest square of material" came out
    # straddling the hole and its crust.
    m = _fill_speckle(np.abs(a - bg).max(2) > 0.10)
    lab, k = ndimage.label(m)
    if k > 1:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    r = max(3, int(rim_frac * np.sqrt(m.sum())))
    crust = np.median(a[m & ~ndimage.binary_erosion(m, np.ones((r, r)))], 0)
    # The whole face, rind and all. Eroding the edge away and repainting a fixed nine-pixel
    # band is right for a *generated* material, which is a texture with no edge of its own;
    # a photograph already has one, and an orange's rind and pith are a real fraction of its
    # radius, not a thin line. Stripping them left the reference orange flesh to the very
    # edge while every one of its seven photographs shows a thick pale ring. The mapping is
    # by fraction of the way across the region, so the rind keeps the proportion it has in
    # the photograph.
    print(f"  material from {path}: {int(m.sum()):,} px of face  edge colour {crust.round(3)}")
    return a, m, crust


def _radial_extent(mask, nb_a=1440):
    """Inner and outer radius of the region along each ray from its centroid."""
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    th = np.arctan2(ys - cy, xs - cx)
    r = np.hypot(ys - cy, xs - cx)
    ai = np.clip(((th + np.pi) / (2 * np.pi) * nb_a).astype(np.int64), 0, nb_a - 1)
    r_in = np.full(nb_a, np.inf); r_out = np.zeros(nb_a)
    np.minimum.at(r_in, ai, r)
    np.maximum.at(r_out, ai, r)
    seen = np.isfinite(r_in)
    if not seen.all():                      # rays the region misses: nearest ray that it hits
        idx = ndimage.distance_transform_edt(~seen, return_distances=False,
                                             return_indices=True)[0]
        r_in, r_out = r_in[idx], r_out[idx]
    # a ray crosses a thin region in a couple of pixels; smooth so the parametrisation is
    # not driven by which pixel happened to land on the boundary
    k = np.ones(9) / 9.0
    wrap = lambda v: np.convolve(np.concatenate([v[-8:], v, v[:8]]), k, "same")[8:-8]
    return (cy, cx), wrap(r_in), wrap(r_out)


def _holes_of(mask, min_frac=0.05):
    """Enclosed background big enough to be a hole, counted the way section_match counts it."""
    lab, n = ndimage.label(~mask)
    if n == 0:
        return 0
    border = set(lab[0].tolist()) | set(lab[-1].tolist()) | \
        set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    border.discard(0)
    area = max(int(mask.sum()), 1)
    return sum(1 for i in range(1, n + 1)
               if i not in border and int((lab == i).sum()) >= min_frac * area)


def crop_native(src_img, src_mask, dst_mask):
    """Copy the material at its own resolution, without imposing any correspondence.

    The polar mapping is a correspondence between two regions of the same topology. Between a
    ring and a disc it is not one: the ring's thickness gets stretched over the disc's whole
    radius and comes out as concentric bands, a structure the material does not have. When
    the two do not agree, take a window of the source the size of the target instead -- the
    target components here are small, so this is the source at full resolution and it invents
    nothing.
    """
    ys, xs = np.where(dst_mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    dt = ndimage.distance_transform_edt(src_mask)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)   # deepest inside the material
    H, W = src_mask.shape
    y0 = int(np.clip(cy - h // 2, 0, max(H - h, 0)))
    x0 = int(np.clip(cx - w // 2, 0, max(W - w, 0)))
    win = src_img[y0:y0 + h, x0:x0 + w]
    if win.shape[0] != h or win.shape[1] != w:
        win = cv2.resize(win, (w, h), interpolation=cv2.INTER_CUBIC)
    out = np.ones((*dst_mask.shape, 3), np.float32)
    sub = dst_mask[ys.min():ys.min() + h, xs.min():xs.min() + w]
    tgt = out[ys.min():ys.min() + h, xs.min():xs.min() + w]
    tgt[sub] = win[sub]
    return out


def remap(src_img, src_mask, dst_mask, nb_a=1440):
    """Read the source at the same relative position, one target pixel at a time.

    Cutting the largest square that fits inside the material and scaling it up throws away
    almost all of the resolution when the region is thin: a doughnut's ring admits a square
    of 108 pixels, and blown up to the frame its crumb is a blur. Mapping by coordinates has
    every pixel of the source available -- but only if the mapping is a lookup and not an
    average. Binning the source into (angle, depth) cells and taking each cell's mean cost
    four tenths of the photograph's contrast, 0.147 -> 0.087, and left the cell boundaries
    visible as rectangular patches; the high-frequency energy went *up*, which was those
    seams and not any crumb.

    So parametrise each region by the ray from its centroid and the fraction of the way
    across it, invert that on the source, and sample there bilinearly. Monotonic along every
    ray, so it inverts exactly, and nothing is averaged.
    """
    (scy, scx), sri, sro = _radial_extent(src_mask, nb_a)
    (dcy, dcx), dri, dro = _radial_extent(dst_mask, nb_a)
    ys, xs = np.where(dst_mask)
    th = np.arctan2(ys - dcy, xs - dcx)
    r = np.hypot(ys - dcy, xs - dcx)
    ai = np.clip(((th + np.pi) / (2 * np.pi) * nb_a).astype(np.int64), 0, nb_a - 1)
    s = (r - dri[ai]) / np.maximum(dro[ai] - dri[ai], 1e-6)
    s = np.clip(s, 0.0, 1.0)
    rs = sri[ai] + s * (sro[ai] - sri[ai])
    mx = (scx + rs * np.cos(th)).astype(np.float32)
    my = (scy + rs * np.sin(th)).astype(np.float32)
    # cv2.remap wants both of the destination's dimensions under SHRT_MAX, and a section can
    # easily be more than 32767 pixels, so lay the samples out as a grid rather than a column.
    n, w = mx.size, 512
    h = int(np.ceil(n / w))
    pad = h * w - n
    gx = np.concatenate([mx, np.zeros(pad, np.float32)]).reshape(h, w)
    gy = np.concatenate([my, np.zeros(pad, np.float32)]).reshape(h, w)
    sampled = cv2.remap(src_img, gx, gy, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)[:n]
    out = np.ones((*dst_mask.shape, 3), np.float32)
    out[ys, xs] = sampled
    return np.clip(out, 0, 1)


def fill(mask, material, crust, rim_px=9):
    """Material inside the face, crust along every edge of it, white everywhere else."""
    inner = ndimage.binary_erosion(mask, np.ones((rim_px, rim_px)))
    out = np.ones((*mask.shape, 3), np.float32)
    out[mask] = crust
    out[inner] = material[inner]
    # let the crust meet the crumb rather than step into it
    band = mask & ~inner
    if band.any():
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out[band] = blur[band]
    return np.clip(out, 0, 1)


def main(src, cfg, demo, prep, out_png, prompt, az, el, strength, seed, size=512,
         separate_dir=None,
         material_image=None):
    mask = cut_mask(src, cfg, demo, az, el, size)
    lab, k = ndimage.label(mask)
    print(f"  cut face: {k} part(s), {100 * mask.mean():.1f}% of frame")

    if material_image:
        # A directory or a file, and the same thing either way. The spheres' interiors were
        # built by handing a directory of photographs straight to init_interior_slice while
        # the ring and the loaf came through here, so the two families of object were mapped
        # a different number of times -- once against twice -- for no reason but which script
        # they happened to enter by. Map every photograph onto this cut face and take the
        # per-pixel median of the results: one photograph reduces to itself, and a directory
        # blends after the correspondence rather than before it, so the photographs never
        # have to be registered to each other.
        srcs = sorted(os.path.join(material_image, f)
                      for f in os.listdir(material_image)) \
            if os.path.isdir(material_image) else [material_image]
        inner = mask            # whole face onto whole face; the rind is in the photograph
        mats, crusts = [], []
        for sp in srcs:
            src_img, src_mask, crust = material_from_image(sp, size)
            crusts.append(crust)
            hs = _holes_of(src_mask)
            m1 = np.ones((size, size, 3), np.float32)
            clab, cn = ndimage.label(inner)
            for j in range(1, cn + 1):
                comp = clab == j
                if comp.sum() < 64:
                    continue
                part = remap(src_img, src_mask, comp) if _holes_of(comp) == hs \
                    else crop_native(src_img, src_mask, comp)
                m1[comp] = part[comp]
            mats.append(m1)
        mat = np.median(np.stack(mats), 0).astype(np.float32) if len(mats) > 1 else mats[0]
        crust = np.median(np.stack(crusts), 0)
        print(f"    {len(srcs)} photograph(s) mapped onto the face"
              + (", median-blended after mapping" if len(srcs) > 1 else ""))
        if separate_dir and len(mats) > 1:
            # The blend is right for the initialisation, which wants one plausible interior
            # to extrude, and wrong for the supervision, which has fifty planes and gets one
            # image for all of them. _photo already spreads a directory's files across the
            # planes -- it is the directory holding a single file that collapses it. Measured
            # on the orange: the fifty transverse targets correlate at +1.0000 with a core
            # brightness spread of 0.0003, against 0.0572 across the six photographs they
            # were made from, so fifty planes carry one photograph's worth of information and
            # the only interior that satisfies them is invariant along the axis.
            os.makedirs(separate_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(out_png))[0]
            for i, m1 in enumerate(mats):
                o1 = np.ones_like(m1); o1[mask] = m1[mask]
                cv2.imwrite(os.path.join(separate_dir, f"{stem}_{i:02d}.png"),
                            cv2.cvtColor(o1, cv2.COLOR_BGR2RGB) * 255)
            print(f"    {len(mats)} mapped photographs kept separately -> {separate_dir}")
    else:
        crust = exterior_colour(prep)
        crumb = np.clip(crust * 0.55 + 0.45, 0, 1)
        print(f"  crust {crust.round(3)}  base {crumb.round(3)}")
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-depth", torch_dtype=torch.float16).to(DEV)
        pipe.set_progress_bar_config(disable=True)
        mat = make_material(pipe, prompt, crumb, size, strength, seed)
    cv2.imwrite(out_png.replace(".png", "_material.png"),
                cv2.cvtColor(mat, cv2.COLOR_BGR2RGB) * 255)
    if material_image:
        out = np.ones_like(mat); out[mask] = mat[mask]
    else:
        out = fill(mask, mat, crust)
    cv2.imwrite(out_png, cv2.cvtColor(out, cv2.COLOR_BGR2RGB) * 255)
    print(f"  -> {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("cfg"); ap.add_argument("demo")
    ap.add_argument("prep", help="the object's six exterior references, for the crust colour")
    ap.add_argument("out"); ap.add_argument("prompt")
    ap.add_argument("--az", type=float, default=90.0)
    ap.add_argument("--el", type=float, default=0.0)
    ap.add_argument("--strength", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--separate-dir", dest="separate_dir", default=None,
                    help="also write each photograph mapped onto this face, one file per "
                         "photograph, so the supervision can spread them across its planes "
                         "instead of seeing one blend fifty times")
    ap.add_argument("--material-image", dest="material_image", default=None,
                    help="photograph of the interior material; used instead of generating "
                         "one, and it need not be the cut being built")
    a = ap.parse_args()
    main(a.src, a.cfg, a.demo, a.prep, a.out, a.prompt, a.az, a.el, a.strength, a.seed,
         material_image=a.material_image, separate_dir=a.separate_dir)
