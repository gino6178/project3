"""Six exterior references, one per cube face, instead of one image reused everywhere.

A single reference pasted into every view stamps whatever is in it around the whole sphere.
That is harmless for a repeating pattern and wrong for anything singular: the orange's navel
appeared wherever a camera pointed, and the generation's highlight and drop shadow were
duplicated the same way.

Six faces fix it structurally rather than by wording. Each direction gets its own image, so
a feature lives on the face it belongs to and nowhere else -- and the faces can be asked for
different things, which is what the object actually is. An orange has a stem scar at one
pole, a navel at the other and uniform peel around the equator; a watermelon has a pale
ground spot where it rested. Prompting per face makes the reference more correct, not just
less repetitive.

Lighting is asked to be flat and shadowless, because a highlight belongs to a light source
rather than to the fruit, and anything baked into the reference is baked into the model.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os, json, argparse
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)
import torch, cv2, numpy as np
import torch.nn.functional as F
from PIL import Image
from scene.gaussian_model import GaussianModel
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs, initialize_resterize, convert_SH
from utils.transformation_utils import *
from utils.camera_view_utils import get_camera_view
from diffusers import StableDiffusionDepth2ImgPipeline

DEV = "cuda:0"

# The six cube faces, as (name, azimuth, elevation). +up / -up are the poles; the four
# sides are 90 degrees apart around the equator. Named by where they sit on the object, so
# the labels the voxels carry mean the same thing as the images.
# Six faces cover the sphere, but only just: a point at a face centre is inside exactly one
# reference's cone, and there is then no second opinion about it. Lighting is fixed to the
# image and albedo is fixed to the object, so telling them apart needs a point to appear in
# references taken from different directions -- with one, a shadow is indistinguishable from
# a dark patch of peel and gets baked in, which is what put a dark band across the orange.
#
# Adding the eight cube corners brings the worst-case angle to the nearest direction down
# from 54.7 to 37.4 degrees, so every point falls inside three or four cones. Each direction
# is sampled with its own seed: the prompt and the depth hold the fruit steady while the
# lighting lands somewhere different each time, which is exactly the difference that lets
# the average keep one and drop the other.
# Off by default, because the reason it was added -- averaging the lighting over directions
# sampled independently -- does not survive a shared seed, and a per-face seed changes the
# fruit rather than the light. CUBE_CORNERS=1 turns it on for the other reason, which is
# geometric and unaffected by any of that: six cones leave 52.5% of directions covered by a
# single face, so half the sphere is bounded by a cone edge and every one of those edges is a
# seam. Fourteen directions bring the worst-case angle to the nearest axis from 54.7 to 37.4
# degrees and put every point inside three or four cones, which is what lets the blend be
# continuous rather than merely smooth in the middle.
CORNERS_OFF = os.environ.get("CUBE_CORNERS", "0") != "1"
# Shared, and it stays shared. Reseeding per face does make the four sides differ --
# measured RMS on the orange went from 0.006-0.011 to 0.133-0.188 -- but they stop
# being the same object: the orange came back green with red spots on one face and
# flat vermilion on another, the doughnut blue. The depth buffer pins the silhouette,
# not the material, so the seed is still carrying the appearance. Four samples of one
# peel is the right thing to want and this is not a way to get it.
PER_FACE = os.environ.get("PER_FACE_SEED", "0") == "1"
FACES = [("up",    0,  90),
         ("down",  0, -90),
         ("front", 0,   0),
         ("right", 90,  0),
         ("back",  180, 0),
         ("left",  270, 0),
         ("c0", 45, 35.264),
         ("c1", 135, 35.264),
         ("c2", 225, 35.264),
         ("c3", 315, 35.264),
         ("c4", 45, -35.264),
         ("c5", 135, -35.264),
         ("c6", 225, -35.264),
         ("c7", 315, -35.264),
]
if CORNERS_OFF:
    FACES = FACES[:6]

# Directions scattered instead of on a lattice.
#
# A seam is where one reference stops being used and the next starts, so it lives on a cone
# boundary, and cone boundaries on a fixed set of axes are themselves a fixed pattern: with six
# faces they form a box around the sphere, and the box is what the eye picks out. It is not a
# defect of any one reference and no weighting removes it -- softening the blend widened it,
# fourteen directions moved it, smoothing the field in space cost the peel before it touched
# the seam.
#
# Scattering the directions removes the pattern rather than the boundaries. Each cell still
# sits at some boundary, but neighbouring cells sit at different ones, so there is no line for
# them to lie along. Fibonacci rather than uniform random, because random directions clump and
# a clump is a coarse lattice again.
EXTRA = int(os.environ.get("CUBE_EXTRA", "0"))
if EXTRA > 0:
    import math as _m
    _g = _m.pi * (3.0 - _m.sqrt(5.0))
    for _i in range(EXTRA):
        _y = 1.0 - 2.0 * (_i + 0.5) / EXTRA
        FACES.append((f"r{_i}", _m.degrees(_g * _i) % 360.0, _m.degrees(_m.asin(_y))))

NEG = ("cross section, cut in half, sliced, halved, wedge, segments, pulp, interior, "
       "flesh, seeds, watermark, text, "
       # what the reference must not carry: light and ground belong to the scene, and
       # anything baked in here is baked into every cell this face colours
       "drop shadow, cast shadow, shadow, vignette, dark rim, specular highlight, "
       "glare, reflection, gradient background, grey background")


class P: convert_SHs_python = False; compute_cov3D_python = True; debug = False



def pattern_axis(img, fg_tol=0.06):
    """How much of the pattern's structure lies along one fixed direction.

    A view down a pole is organised about the centre, so the gradient directions spread over
    every angle; a side view has stripes running one way and they pile up. The resultant length
    of the doubled gradient angles is 0 when no direction is preferred and 1 when only one is.

    It is here because a prompt cannot fix the `up` views. The rasteriser's depth buffer is the
    conditioning, and a sphere seen down its axis and seen from the side give the *same* dome --
    there is nothing in the conditioning that distinguishes them, so the sampler falls back on
    whatever it has seen most, which is the side of a fruit. The watermelon's `up` reference came
    back as a side view with the correct wording already in the prompt.

    Measured on the references in the repository the two cases do not overlap: genuine pole views
    score 0.059 (watermelon `down`), 0.098 and 0.104 (doughnut `up`, `down`); side views score
    0.213 to 0.431. POLE_MAX sits in the gap, and is a property of the measurement rather than of
    any object.
    """
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    fg = np.abs(a - 1).max(2) > fg_tol
    gy, gx = np.gradient(a.mean(2))
    m = fg & (np.hypot(gx, gy) > 0.02)
    if m.sum() < 64:
        return 0.0
    th = 2 * np.arctan2(gy[m], gx[m])
    return float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))


POLE_MAX = float(os.environ.get("REF_POLE_MAX", "0.15"))
TRIES = int(os.environ.get("REF_TRIES", "1"))


def main(src, cfg, demo, out_dir, prompts_json, strength=0.95, seed=1234, only=None):
    os.makedirs(out_dir, exist_ok=True)
    prompts = json.load(open(prompts_json))
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

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-depth", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for fi, (name, az, el) in enumerate(FACES):
        if only and name not in only:
            print(f"  {name:<6} kept")
            continue
        cam, _ = get_camera_view(demo, default_camera_index=-1,
            center_view_world_space=vc, observant_coordinates=oc, show_hint=False,
            init_azimuthm=az, init_elevation=el, init_radius=cam_p["init_radius"],
            move_camera=False, current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        rast = initialize_resterize(cam, g, P(), torch.tensor([1., 1., 1.], device=DEV),
                                    image_height=512, image_width=512)
        col = convert_SH(shs, cam, g, world, None)
        img, _, dep, alp = rast(means3D=world, means2D=sp, shs=None, colors_precomp=col,
                                opacities=op, scales=None, rotations=None,
                                cov3D_precomp=cov)
        a = img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()
        cv2.imwrite(f"{out_dir}/{name}_src.png", cv2.cvtColor(a, cv2.COLOR_BGR2RGB) * 255)
        cur = Image.open(f"{out_dir}/{name}_src.png").convert("RGB")

        # The rasteriser already knows how far away every part of the object is, so use that
        # rather than asking an estimator what a picture of a grey shape looks like. On a ball
        # the estimate happens to be right and on anything else it is not: the loaf's sides
        # came back as round loaves and the ring's sides as pink slabs, because a flattened
        # ring silhouette means nothing to the estimator and it invented a shape that did.
        # Inverse depth, larger nearer, which is the convention the sampler expects.
        m = (alp[0] > 0.5)
        z = dep[0].clone()
        inv = torch.zeros_like(z)
        inv[m] = 1.0 / z[m].clamp_min(1e-4)
        if int(m.sum()) > 0:
            lo, hi = inv[m].min(), inv[m].max()
            inv[m] = (inv[m] - lo) / (hi - lo + 1e-8) * 0.85 + 0.15
        d = inv[None].float()
        d = F.interpolate(d.unsqueeze(0), size=(512, 512), mode="bilinear").squeeze(0)

        # A seed per direction. Sharing one was the fix for the loaf, whose side views came
        # back as flat brown slabs on three seeds out of four -- but that was while the
        # conditioning was an estimated depth map, which for a non-spherical silhouette says
        # nothing the sampler can use, so the seed was carrying the whole generation. With
        # the rasteriser's own depth buffer the shape is pinned by the conditioning and the
        # seed only chooses which sample of the peel to draw.
        #
        # Sharing it has its own cost, and on a rotationally symmetric object it is fatal:
        # the four side cameras see identical depth, so identical seed and identical prompt
        # give the identical image. Measured on the orange, the four sides differed by an
        # RMS of 0.006 to 0.011 -- one photograph stamped four times, which puts a four-fold
        # repeat around the peel. A real peel is a stationary texture, so four different
        # samples of it is what the object is; the prompt and the depth keep them agreeing
        # about colour and scale, which is the consistency that matters.
        is_pole = abs(el) >= 89
        best, best_v = None, None
        for t in range(TRIES if is_pole else 1):
            gen = torch.Generator(pipe.device).manual_seed(
                int(seed) + (fi if PER_FACE else 0) + 1000 * t)
            r = pipe(prompt=prompts.get(name, prompts["front"]), image=cur, depth_map=d,
                     negative_prompt=NEG, strength=float(strength), guidance_scale=12,
                     num_inference_steps=50, generator=gen, return_dict=False)
            out = r[0][0] if isinstance(r, tuple) else r.images[0]
            if not is_pole:
                best, best_v = out, None
                break
            v = pattern_axis(out)
            if best_v is None or v < best_v:
                best, best_v = out, v
            print(f"      try {t}: pattern axis {v:.3f}"
                  + ("  accepted" if v <= POLE_MAX else f"  > {POLE_MAX}, redrawing"))
            if v <= POLE_MAX:
                break
        best.save(f"{out_dir}/{name}_ref.png")
        note = "" if best_v is None else f"  pattern axis {best_v:.3f}" + (
            "" if best_v <= POLE_MAX else "  NOT A POLE VIEW -- no sample passed")
        print(f"  {name:<6} az{az:>4} el{el:>4}  -> {name}_ref.png{note}")
    # The directions, next to the images. Three files hard-code a list of face names and each
    # one silently ignores what it does not recognise: cube_prep dropped twenty-six of thirty-
    # two references, and init_skin_cube then used six cones and reported success. A generated
    # set has to carry its own geometry.
    json.dump({n: [az, el] for n, az, el in FACES},
              open(os.path.join(out_dir, "dirs.json"), "w"), indent=1)
    json.dump({n: prompts.get(n, prompts["front"]) for n, _, _ in FACES},
              open(os.path.join(out_dir, "prompts.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("cfg"); ap.add_argument("demo")
    ap.add_argument("out_dir"); ap.add_argument("prompts")
    ap.add_argument("--strength", type=float, default=0.95)
    ap.add_argument("--only", default="", help="comma-separated faces to redraw; "
                                              "the rest are left as they are")
    a = ap.parse_args()
    main(a.src, a.cfg, a.demo, a.out_dir, a.prompts, strength=a.strength,
         only=[x for x in a.only.split(',') if x] or None)
