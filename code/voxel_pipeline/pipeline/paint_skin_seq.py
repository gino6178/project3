"""Paint the shell one view at a time, each view generated on top of what is already there.

Every version of this so far generated N references independently and then blended them onto
the shell. That is the wrong order, and the two failures it produces are the two the literature
names. Blending finished images averages textures whose detail sits in different places, so the
peel cancels: a cell read from three or four references came back at gradient 0.034 against any
single reference's 0.164. And an average is a vote, so a feature only one reference has loses --
seven directions see the north pole, only `up` shows the calyx there (departure from plain peel
0.217 against 0.002 to 0.015), and the calyx went from 0.135 at initialisation to 0.011 after
training. Nothing downstream could recover either; a night of weighting, softmax temperatures,
obliquity terms and per-view loss scales moved them a few points and traded one against the
other, because they all move along the single axis of how much the views share.

TEXTure and Text2Tex do not blend. They paint from one viewpoint at a time, project onto the
texture, and generate the next view *conditioned on what is already painted*, with the render
partitioned into keep, refine and generate. A new view therefore never contradicts an old one --
it is drawn to continue it -- so there is nothing to average and no seam to hide. Text2Tex's
refine state is the other half: a texel painted from a grazing angle is replaced, not averaged,
when a squarer view arrives, which is exactly what the pole needs.

They both assume a UV parameterisation. This does not: the shell's cells are the texture, one
colour each, and the projection from a cell to a pixel is the same one `init_skin_cube` already
uses. So the algorithm carries over unchanged.

  for each direction, best first:
      render the shell as it stands
      generate from that render, at a strength that keeps what is painted
      for every visible cell whose incidence beats its best so far:
          take the colour, and record the new incidence

The last rule is keep/refine in one line: a cell is written by the single view that faces it
most squarely, and a later, squarer view overwrites an earlier, obliquer one. `best` starts
below -1 so the first view to see a cell always writes it.

    python voxel_pipeline/pipeline/paint_skin_seq.py INIT_thick_i config/sphere_physics.json \\
        config/sphere_demo cube_or32_prep config/cube_prompts/orange.json PAINT_or
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

import blend_paint                                                  # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
from diffusers import StableDiffusionDepth2ImgPipeline              # noqa: E402
from scene.gaussian_model import GaussianModel                      # noqa: E402
from utils.camera_view_utils import get_camera_view                 # noqa: E402
from utils.decode_param import decode_param_json                    # noqa: E402
from utils.render_utils import (convert_SH, initialize_resterize,   # noqa: E402
                                load_params_from_gs)
from utils.transformation_utils import *                            # noqa: E402

DEV = "cuda:0"
C0 = 0.28209479177387814
COS_CONE = 0.5                       # 60 degrees, as in init_skin_cube
CONE_LO = float(os.environ.get("PAINT_CONE_LO", "0.30"))   # where the graded mask reaches zero
# Paint wider than is read. Whatever the mask's edge does -- and in latent space it rings, one
# latent cell wide, which is what the teal marks around `up` and the white speckle around
# `down` were -- happens outside the cone the colours are taken from, and a squarer view writes
# those cells later anyway.
HARVEST = float(os.environ.get("PAINT_HARVEST", "0.62"))
FLIP = [True]
REF_R = 0.70                         # where the cone edge lands in the reference
REFINE = float(os.environ.get("PAINT_REFINE", "0"))   # width of the blend band, in cosine
GEN_STRENGTH = float(os.environ.get("PAINT_GEN_STRENGTH", "0.95"))  # for unpainted surface
BLEND = int(os.environ.get("PAINT_BLEND", "1"))       # per-step latent blending
CB_STEPS = int(os.environ.get("PAINT_CB_STEPS", "25"))
GUID = float(os.environ.get("PAINT_GUIDANCE", "12"))
NEG = ("cross section, cut in half, sliced, halved, wedge, segments, pulp, interior, flesh, "
       "seeds, watermark, text, drop shadow, cast shadow, shadow, vignette, dark rim, "
       "specular highlight, glare, reflection, gradient background, grey background")


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def order_directions(dirs):
    """Named faces first, then the fillers, each the furthest from everything placed so far.

    The named faces carry the features and were each prompted for what belongs on them, so they
    should paint before anything can paint over them -- and under the incidence rule nothing
    can, since each owns its own pole or side outright. The fillers then go in an order that
    spreads them out, so each new one lands on the largest remaining gap rather than beside the
    last. That is Text2Tex's next-best-view, computed on directions instead of on coverage,
    which is the same thing for a convex object.
    """
    named = [d for d in dirs if not d[0].startswith(("r", "c"))]
    rest = [d for d in dirs if d[0].startswith(("r", "c"))]
    if not rest:
        return named
    def ax(d):
        a, e = np.radians(d[1]), np.radians(d[2])
        return np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    placed = [ax(d) for d in named] or [ax(rest[0])]
    out = list(named)
    pool = list(rest)
    while pool:
        k = int(np.argmin([max(float(ax(d) @ p) for p in placed) for d in pool]))
        out.append(pool[k])
        placed.append(ax(pool.pop(k)))
    return out


def surface_normals(world, is_shell, centre, res=None):
    """Which way is out, taken from the occupancy rather than from the centre.

    This used to be `world - centre`, which is a normal only for a star-shaped object. On the
    doughnut it is wrong exactly where it matters: a cell on the top of the tube sits a ring
    radius R out from the axis and a tube radius r up from it, so the direction from the centre
    is almost horizontal -- `d_cos = r/sqrt(R^2 + r^2)`, about 0.4 -- and never clears the 60
    degree cone the writing test uses. The `up` and `down` views therefore had nothing to write,
    the top and bottom faces were left to the four side views at grazing incidence, and the
    icing came out on the wrong face.

    The occupancy already says which way is out, and it says it for any topology: smooth the
    indicator volume and the gradient points into the object, so minus the gradient is the
    outward normal. Through the doughnut's hole it points inward toward the axis, which is
    correct and is what the radial version cannot express at all.

    Nothing here is per-object. On a sphere it returns the radial direction to within the grid's
    resolution, so the orange and the watermelon get what they already had; `res` is chosen from
    the cell count rather than set per object. Where the gradient vanishes -- a cell in the
    interior, far from any face -- it falls back to the radial direction, which is only ever
    consulted for cells the shell test has already excluded.
    """
    mn = world.min(0).values
    ext = (world.max(0).values - mn).clamp_min(1e-9)

    def voxelise(res):
        h = ext.max() / (res - 4)
        q = (((world - mn) / h).long() + 2).clamp(min=0, max=res + 1)
        occ = torch.zeros([res + 2] * 3, device=world.device)
        occ[q[:, 0], q[:, 1], q[:, 2]] = 1.0
        return occ, q

    # The grid has to match the cells, not the bounding box. Too fine and the shell voxelises
    # into speckle whose gradient is noise -- on a test sphere that returned normals agreeing
    # with the radial direction at 0.06, which is no direction at all. The rule is the coarsest
    # description that is still watertight: raise the resolution while each occupied voxel still
    # holds two points on average, and stop one step before they start to hold one, because a
    # voxel per cell is where the gaps between cells open up. It reads the spacing off the data,
    # so it is the same rule for a lattice of 800k cells and one of 1.5M.
    n = world.shape[0]
    res = res or 64
    if res == 64:
        while res < 256:
            occ, _ = voxelise(res * 2)
            if float(occ.sum()) > 0.5 * n:
                break
            res *= 2
    occ, q = voxelise(res)
    occ = F.avg_pool3d(F.avg_pool3d(occ[None, None], 5, 1, 2), 5, 1, 2)[0, 0]
    dims = list(occ.shape)

    g = torch.stack(torch.gradient(occ, dim=(0, 1, 2)), -1)          # points into the object
    idx = (q[:, 0].clamp(1, dims[0] - 2), q[:, 1].clamp(1, dims[1] - 2),
           q[:, 2].clamp(1, dims[2] - 2))
    nrm = -g[idx]
    mag = nrm.norm(dim=1, keepdim=True)
    rad = world - centre
    rad = rad / rad.norm(dim=1, keepdim=True).clamp_min(1e-9)
    nrm = torch.where(mag > 1e-6, nrm / mag.clamp_min(1e-9), rad)
    ok = float((nrm[is_shell] * rad[is_shell]).sum(1).mean()) if int(is_shell.sum()) else 0.0
    print(f"  normals from the occupancy at {res}^3   mean agreement with radial {ok:.3f}")
    return nrm


def main(src, cfg, demo, ref_dir, prompts_json, out_dir, strength=0.55, size=512, seed=1234):
    os.makedirs(out_dir, exist_ok=True)
    prompts = json.load(open(prompts_json))
    dirs = [(n, v[0], v[1]) for n, v in json.load(open(os.path.join(ref_dir, "dirs.json"))).items()]
    dirs = order_directions(dirs)
    print(f"  {len(dirs)} directions, painting order: "
          + " ".join(n for n, _, _ in dirs[:10]) + (" ..." if len(dirs) > 10 else ""))

    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    g = GaussianModel(0)
    g.load_ply_zero_sh(os.path.join(src, "gs_fill.ply"))
    lvl = torch.load(os.path.join(src, "cell_level.pt")).to(DEV)
    lat = torch.load(os.path.join(src, "lattice.pt"))

    par = load_params_from_gs(g, P())
    pos0, cov0 = par["pos"], par["cov3D_precomp"]
    sp, op, shs = par["screen_points"], par["opacity"], par["shs"]
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()
    tpos, so, om = transform2origin(pos0)
    tpos = shift2center111(tpos)
    cov0 = apply_cov_rotations(cov0, rot_m)
    cov0 = so * so * cov0
    cov = apply_inverse_cov_rotations(cov0 / (so * so), rot_m)
    world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(tpos.to(DEV)), so, om), rot_m)
    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    bg = torch.tensor([1., 1., 1.], device=DEV)

    n_all = min(world.shape[0], lvl.shape[0])
    K = world.shape[0] // lvl.shape[0]
    centre = world.mean(0)
    is_shell = (lvl.reshape(-1)[:lvl.shape[0]] != 0).repeat_interleave(K)[:world.shape[0]]
    nrm = surface_normals(world, is_shell, centre)

    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
    best = torch.full((world.shape[0],), -2.0, device=DEV)

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-depth", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def write_back():
        # in place, not a rebind. `shs` is `g._features_dc` itself -- `get_features` returns it
        # unwrapped at degree 0 -- and was captured before the loop, so assigning a new tensor
        # to the attribute left the render reading the colours the shell started with. Every
        # view was then generated from the same untouched image, which is the independent
        # generation this file exists to replace, and no amount of strength tuning could have
        # made those views agree with each other.
        with torch.no_grad():
            g._features_dc.copy_(
                ((rgb - 0.5) / C0).unsqueeze(1).to(g._features_dc.device))

    def render(cam, colours, bgc):
        rast = initialize_resterize(cam, g, P(), bgc, image_height=size, image_width=size)
        return rast(means3D=world, means2D=sp, shs=None,
                    colors_precomp=colours.contiguous().clamp(0, 1), opacities=op,
                    scales=None, rotations=None, cov3D_precomp=cov)

    for step, (name, az, el) in enumerate(dirs):
        cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                 observant_coordinates=oc, show_hint=False,
                                 init_azimuthm=az, init_elevation=el,
                                 init_radius=cam_p["init_radius"], move_camera=False,
                                 current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        axis = cam.camera_center.reshape(3).to(DEV) - centre
        axis = axis / axis.norm().clamp_min(1e-9)

        _dc = nrm @ axis
        vis_pre = is_shell & (_dc > best) & (_dc > HARVEST)
        img, _, depth, alpha = render(cam, rgb, bg)
        cur = Image.fromarray(
            (img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

        # The rasteriser's own depth, inverted.
        #
        # Three conditions were tried on the two views that kept coming back as cut fruit. The
        # accumulated opacity is flat over the object, which is what a cut face is, and it
        # produced cross-sections. The monocular estimator fixed `up` and left `down` still
        # cut. The rasteriser also returns the true depth, next to alpha, and it is a dome for
        # a sphere; with it, every view comes back as peel. SD2-depth wants near high and the
        # rasteriser gives distance, so it is inverted inside the silhouette and zero outside,
        # or the background reads as a surface behind the fruit.
        _a = alpha.reshape(1, size, size).float()
        _d = depth.reshape(1, size, size).float()
        _fg = _a > 0.5
        d = torch.zeros_like(_d)
        if bool(_fg.any()):
            _lo, _hi = _d[_fg].min(), _d[_fg].max()
            d[_fg] = (_hi - _d[_fg]) / (_hi - _lo).clamp_min(1e-6)
        d = F.interpolate(d[None], size=(512, 512), mode="bilinear",
                          align_corners=False).squeeze(0)

        gen = torch.Generator(pipe.device).manual_seed(seed + step)
        # Generate where nothing has been painted, refine where something has.
        #
        # TEXTure's three states are generate, refine and keep, and the strength is what
        # separates the first two: a region no view has touched has to be drawn, and one that
        # has only needs sharpening. At a single low strength the untouched regions inherit
        # whatever the blend left there -- the navel came back as a knot of white beads
        # because that is what the initialisation's blurred navel sharpens into. There is no
        # per-pixel strength in depth2img, so the choice is made per view, on how much of what
        # it is about to write it has never seen.
        _fresh = float((best[vis_pre] < -1).float().mean()) if int(vis_pre.sum()) else 1.0
        _st = strength + (GEN_STRENGTH - strength) * _fresh
        _cb = None
        if BLEND:
            # The trimap, rendered rather than reasoned about. Each cell already knows its own
            # state -- never seen, seen worse, seen better -- and the projection from a cell to
            # a pixel is the rasteriser, so painting the states as colours and rendering them
            # gives the mask in image space with the silhouette and the occlusion for free.
            # The weight is graded over the edge of the cone rather than cut at it, for the
            # same reason the latent mask is averaged and not thresholded.
            _w = ((_dc - CONE_LO) / (COS_CONE - CONE_LO)).clamp(0.0, 1.0) * is_shell.float()
            _s = torch.zeros_like(rgb)
            _s[:, 0] = _w * (best < -1).float()            # generate
            _s[:, 1] = _w * (best >= -1).float()           # refine
            _sm, _, _, _ = render(cam, _s, torch.zeros(3, device=DEV))
            g_lat = blend_paint.to_latent_mask(_sm[0][None, None], 512 // 8)
            r_lat = blend_paint.to_latent_mask(_sm[1][None, None], 512 // 8)
            _cb = blend_paint.BlendedPaint(
                pipe, blend_paint.encode(pipe, cur.resize((512, 512))), g_lat, r_lat,
                cb_steps=CB_STEPS, generator=gen)
            m_lat = (g_lat + r_lat).clamp(0, 1)
            # The mask says *where* the model may paint; the strength still says how far it
            # may go, and the two are not interchangeable. Pinning this at GEN_STRENGTH gave
            # the 26 filler views -- which have nothing to generate, only oblique work to
            # square up -- a free redraw of the whole visible face from near-noise every time,
            # and they came back with bruises. `_fresh` is already the generate fraction, so
            # the views that have something to draw get the strength for it and the rest do not.
            print(f"      trimap: generate {float(g_lat.mean()) * 100:.1f}%  "
                  f"refine {float((m_lat * (1 - g_lat)).mean()) * 100:.1f}%  "
                  f"keep {float((1 - m_lat).mean()) * 100:.1f}%  of frame")
        r = pipe(prompt=prompts.get(name, prompts["front"]), image=cur.resize((512, 512)),
                 depth_map=d.half(), negative_prompt=NEG, strength=float(_st),
                 guidance_scale=GUID, num_inference_steps=50, generator=gen, return_dict=False,
                 callback_on_step_end=_cb)
        out = (r[0][0] if isinstance(r, tuple) else r.images[0])
        out.save(os.path.join(out_dir, f"{step:02d}_{name}.png"))
        tex = torch.from_numpy(np.asarray(out.convert("RGB"), np.float32) / 255.).to(DEV)
        if _cb is not None:
            # Composite in pixel space, not in latent space. The kept region comes back from
            # the decoder as an approximation of what went in -- the VAE is not an identity --
            # and the approximation is worst at the mask's edge. Blended Latent Diffusion says
            # to put the original pixels back afterwards, so put them back.
            _mp = F.interpolate(m_lat, size=tex.shape[:2], mode="bilinear",
                                align_corners=False).reshape(*tex.shape[:2], 1)
            _cur_t = torch.from_numpy(np.asarray(cur, np.float32) / 255.).to(DEV)
            tex = _mp * tex + (1 - _mp) * _cur_t
        H, W, _ = tex.shape

        hom = torch.cat([world, torch.ones(world.shape[0], 1, device=DEV)], 1)
        clip = hom @ cam.full_proj_transform
        ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-6)
        uv = torch.stack([(ndc[:, 0] + 1) * 0.5, (ndc[:, 1] + 1) * 0.5], 1)
        uv_c = ((centre.reshape(1, 3) @ cam.full_proj_transform[:3]
                 + cam.full_proj_transform[3]).reshape(-1))
        uv_c = torch.stack([(uv_c[0] / uv_c[3] + 1) * 0.5, (uv_c[1] / uv_c[3] + 1) * 0.5])

        d_cos = nrm @ axis
        # The scale has to come from the silhouette, not from whatever this direction happens
        # to be writing. Taking the 99.5th percentile over the cells about to be written makes
        # the scale depend on the cone: a direction facing a large unpainted region and one
        # touching a sliver get different scales, so the same cell lands on different parts of
        # its reference from different views and the colours disagree. With a free cone at
        # cos > 0.10 that produced white patches over 48.2% of the silhouette. Measure the
        # radius over every shell cell in front of the camera, which is the silhouette itself
        # and is the same for every direction, and restrict writing to the same 60 degree cone
        # `init_skin_cube` uses -- outside it the projection is grazing and the reference has
        # stopped being peel anyway.
        infront = is_shell & (clip[:, 3] > 0) & (d_cos > 0.0)
        vis = infront & (d_cos > best) & (d_cos > HARVEST)
        if int(vis.sum()) == 0:
            print(f"  {step:02d} {name:<6} nothing to write")
            continue
        # The generated image is this view's own render, so a cell's pixel is its own
        # projection. No rescaling.
        #
        # `init_skin_cube` addresses a reference by the fraction of the silhouette radius,
        # because a pre-made reference has the fruit filling its frame and the render does
        # not. Carrying that formula over here addressed an image already framed like the
        # render as though it were not: cells near the silhouette read the image's white
        # corners and cells in the middle read peel, so the shell came out as per-cell noise,
        # which is a flat orange once a million overlapping Gaussians are rendered -- gradient
        # 0.0006 from source images whose own gradient was 0.15.
        px = (uv[vis, 0] * (W - 1)).round().long().clamp(0, W - 1)
        py = ((1.0 - uv[vis, 1]) * (H - 1)).round().long().clamp(0, H - 1)
        if step == 0:
            # The y flip is a convention, so check it rather than assume: sampling the render
            # at each cell's own pixel must return that cell's own colour.
            _src_img = torch.from_numpy(np.asarray(cur, np.float32) / 255.).to(DEV)
            e_flip = float((_src_img[py, px] - rgb[vis]).abs().mean())
            py2 = (uv[vis, 1] * (H - 1)).round().long().clamp(0, H - 1)
            e_same = float((_src_img[py2, px] - rgb[vis]).abs().mean())
            if e_same < e_flip:
                FLIP[0] = False
            print(f"      y-flip check: flipped {e_flip:.4f}  unflipped {e_same:.4f}"
                  f"  -> {'flip' if FLIP[0] else 'no flip'}")
        if not FLIP[0]:
            py = (uv[vis, 1] * (H - 1)).round().long().clamp(0, H - 1)
        # Keep, refine, generate -- the third state was missing and the seams are what it is
        # for. Writing outright wherever this view is squarer leaves a hard boundary between
        # the cells two views wrote, 22.5% of the silhouette in patches. TEXTure's refine
        # region is the fix: near the boundary, where the new view is only slightly squarer,
        # blend rather than replace. `w` is 0 where the two views are equally square and 1
        # where the new one wins by REFINE or more, so a cell deep inside a view's territory
        # is still written outright and only the seam is graded.
        _new = tex[py, px]
        if REFINE > 0:
            w = ((d_cos[vis] - best[vis]) / REFINE).clamp(0.0, 1.0).reshape(-1, 1)
            # a cell no view has touched takes the colour whole, whatever the margin
            w = torch.where((best[vis] < -1).reshape(-1, 1), torch.ones_like(w), w)
            rgb[vis] = (1 - w) * rgb[vis] + w * _new
        else:
            rgb[vis] = _new
        best[vis] = d_cos[vis]
        print(f"  {step:02d} {name:<6} wrote {int(vis.sum()):>7,} cells   "
              f"painted {100 * float((best > -1).float().mean()):.1f}%")

    write_back()
    for f in ("cell_level.pt", "lattice.pt", "is_interior.pt", "cell_face.pt"):
        s = os.path.join(src, f)
        if os.path.exists(s):
            import shutil
            shutil.copy(s, os.path.join(out_dir, f))
    g.save_ply(os.path.join(out_dir, "gs_fill.ply"))
    sh = is_shell
    print(f"  shell mean RGB {[round(float(v), 3) for v in rgb[sh].mean(0)]}   "
          f"unpainted {int((best[sh] <= -1).sum()):,}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("cfg"); ap.add_argument("demo")
    ap.add_argument("ref_dir"); ap.add_argument("prompts"); ap.add_argument("out_dir")
    ap.add_argument("--strength", type=float, default=0.55)
    a = ap.parse_args()
    main(a.src, a.cfg, a.demo, a.ref_dir, a.prompts, a.out_dir, strength=a.strength)
