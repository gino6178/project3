"""The camera, the point splat and the GIF writer the three cube demos share.

Nothing here is new geometry. The projection is `ovox_views.py`'s, copied rather than reinvented
because it is the only place in the repository that gets a cube-space point onto the same pixel
the Gaussian renderer would put it on: the lattice is fitted to the loaded model by a similarity
(Umeyama on 20,000 corresponding rows), and only then does the model's own camera apply. Getting
that fit wrong is silent -- the object still draws, just at the wrong scale, and nothing in the
picture says so.

The renderer is a z-buffer over points, exactly as `ovox_views` argues: a surface made of cells
has no rasteriser here and does not need one to be looked at. Two things are added because an
animation asks more of it than a contact sheet does.

  supersampling      Frames are drawn at `ss` times the output size and box-averaged down. A
                     point splat leaves the background showing wherever the projected spacing
                     exceeds a pixel, and in a still that is speckle; in an animation the speckle
                     moves and reads as noise. Averaging four subpixels turns a missed sample
                     into a shade rather than a hole.

  a shared palette   Every frame is quantised against one palette, taken from the frame with the
                     most colour in it. Per-frame adaptive palettes make the background hunt
                     between two near-whites and the whole picture appears to shimmer.

The captions are drawn from values passed in by the caller, and every caller computes them in the
same loop that produces the frame.
"""
import os as _os

_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

DEV = "cuda:0"
# Resolved, not asserted. The hard-coded Debian path is absent on the machine the renders
# actually run on, and a missing font is not a reason for a figure not to exist: the first of
# these that opens wins, and PIL's own bitmap font is the floor.
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
]


def _fonts():
    import glob as _g
    for a, b in _FONT_CANDIDATES:
        if _os.path.exists(a):
            return a, b if _os.path.exists(b) else a
    got = sorted(_g.glob("/usr/share/fonts/**/*.ttf", recursive=True))
    return (got[0], got[0]) if got else (None, None)


FONT, FONT_B = _fonts()


class _P:
    sh_degree = 0
    compute_cov3D_python = True
    convert_SHs_python = False
    debug = False


class Cam:
    """The model's own camera, plus the similarity that puts lattice space into world space.

    `project` returns pixel coordinates and view-space depth for points given in the same frame
    as `gs_fill.ply`, which is the frame every cube module works in.
    """

    def __init__(self, ply, cfg, demo, az=0.0, el=15.0, radius_scale=1.0):
        from scene.gaussian_model import GaussianModel
        from utils.camera_view_utils import get_camera_view
        from utils.decode_param import decode_param_json
        from utils.render_utils import load_params_from_gs
        from utils.transformation_utils import (apply_inverse_rotations,
                                                generate_rotation_matrices,
                                                get_center_view_worldspace_and_observant_coordinate,
                                                shift2center111, transform2origin,
                                                undoshift2center111, undotransform2origin)

        (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
        g = GaussianModel(0)
        g.load_ply_zero_sh(ply)
        par = load_params_from_gs(g, _P())
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

        xyz_ply = g.get_xyz.detach().cpu().numpy().astype(np.float64)
        wp = world.detach().cpu().numpy().astype(np.float64)
        idx = np.linspace(0, len(wp) - 1, 20000).astype(int)
        A, B = xyz_ply[idx], wp[idx]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, S, Vt = np.linalg.svd(H)
        dsg = np.sign(np.linalg.det(Vt.T @ U.T))
        self.R = Vt.T @ np.diag([1.0, 1.0, dsg]) @ U.T
        self.sc = float(S[:2].sum() + dsg * S[2]) / float(((A - ca) ** 2).sum())
        self.t = cb - self.sc * (self.R @ ca)

        cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                 observant_coordinates=oc, show_hint=False, init_azimuthm=az,
                                 init_elevation=el,
                                 init_radius=cam_p["init_radius"] * float(radius_scale),
                                 move_camera=False, current_frame=0, delta_a=None, delta_e=None,
                                 delta_r=None)
        self.fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        self.w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
        self.xyz = xyz_ply

    def to_world(self, p):
        return self.sc * (np.asarray(p, np.float64) @ self.R.T) + self.t

    def project(self, p, size):
        w = self.to_world(p)
        hom = np.concatenate([w, np.ones((len(w), 1))], 1)
        clip = hom @ self.fp
        ok = clip[:, 3] > 1e-6
        nd = clip[:, :2] / np.where(clip[:, 3:4] == 0, 1e-9, clip[:, 3:4])
        # Both axes the same way, because the rasteriser that made every other picture in this
        # project does the same: ndc2Pix(v, S) = ((v + 1) S - 1) / 2 on x and on y. Flipping y
        # here and not there put every object in these demos upside down -- the pomegranate's
        # calyx pointing at the floor -- against exterior renders of the same model that were
        # the right way up. skin_project.py records the same trap costing four side views.
        px = ((nd[:, 0] + 1) * 0.5 * size).astype(np.int64)
        py = ((nd[:, 1] + 1) * 0.5 * size).astype(np.int64)
        dep = (hom @ self.w2c)[:, 2]
        ok &= (px >= 0) & (px < size) & (py >= 0) & (py < size)
        return px, py, dep, ok


def splat(cam, pts, col, size, ss=2, bg=1.0):
    """Nearest-point-wins, drawn at `ss` times the size and averaged down."""
    S = size * ss
    px, py, dep, ok = cam.project(pts, S)
    img = np.full((S * S, 3), bg, np.float32)
    if ok.any():
        o = np.argsort(-dep[ok])                 # far first, so the nearest wins by overwriting
        img[(py[ok] * S + px[ok])[o]] = np.asarray(col, np.float32)[ok][o]
    img = img.reshape(S, S, 3)
    if ss > 1:
        img = img.reshape(size, ss, size, ss, 3).mean((1, 3))
    return np.clip(img * 255, 0, 255).astype(np.uint8), int(ok.sum())


def shade(rgb, normal, view=(0.35, 0.55, 0.75), ambient=0.62):
    """Lambert on the face normal, so a blocky surface reads as a solid and not as a blob.

    The direction is fixed in lattice space rather than taken from the camera: the camera here is
    the object's own and does not move within a demo, so a light that follows it would be a
    constant.
    """
    v = np.asarray(view, np.float64)
    v = v / np.linalg.norm(v)
    lam = np.abs(np.asarray(normal, np.float64) @ v)
    f = (ambient + (1.0 - ambient) * lam)[:, None]
    return np.clip(np.asarray(rgb, np.float64) * f, 0, 1)


def caption(frame, lines, pad=6, size=17, bold_first=True, colour=(20, 20, 20), band=False):
    """Text on a frame. `lines` are strings the caller has already formatted.

    `band=True` puts the text in a white strip above the picture instead of over it, which is
    what a demo whose subject moves needs: an overlay that reads at rest sits on top of the
    object at the far end of the motion.
    """
    src = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
    h = (size + 4) * len(lines) + 2 * pad if band else 0
    im = src if not band else Image.new("RGB", (src.width, src.height + h), "white")
    if band:
        im.paste(src, (0, h))
    d = ImageDraw.Draw(im)
    y = pad
    for i, s in enumerate(lines):
        want = FONT_B if (bold_first and i == 0) else FONT
        f = ImageFont.truetype(want, size) if want else ImageFont.load_default()
        d.text((pad, y), s, font=f, fill=colour)
        y += size + 4
    return im


def write_frames(path, frames):
    """PNG frames beside `path`, for an encoder that has no palette to quantise to.

    A GIF is at most 256 colours and these pictures are shaded surfaces, so the palette shows as
    grain over every face -- worst on the ones that are nearly one hue. Writing the frames lets
    ffmpeg make an H.264 file at full depth, which is what the rest of the page's motion already
    is.
    """
    d = _os.path.splitext(path)[0] + "_frames"
    _os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        (Image.fromarray(f) if isinstance(f, np.ndarray) else f).save(f"{d}/{i:04d}.png")
    print(f"  -> {d}   {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    return d


def write_gif(path, frames, duration=60, colors=192):
    """One palette for every frame, so nothing shimmers, and a report of what came out."""
    ims = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
    ref = max(ims, key=lambda im: len(im.convert("RGB").getcolors(1 << 22) or [1]))
    pal = ref.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
    q = [im.convert("RGB").quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in ims]
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    q[0].save(path, save_all=True, append_images=q[1:], duration=duration, loop=0,
              optimize=True, disposal=2)
    mb = _os.path.getsize(path) / 2 ** 20
    print(f"  -> {path}   {len(q)} frames, {mb:.2f} MiB")
    return mb


def load_lattice(lattice_dir, colour_from=None):
    """Positions, colours, level and the two spacings -- the four modules' common preamble.

    `colour_from` is the model to take appearance from, when it is not the lattice itself. A
    lattice built by `make_shape` carries flat grey in its interior on purpose -- nothing about
    the fruit is in the geometry -- so a demo that reads colour from it draws every exposed cut
    face grey, which is the one thing these pictures exist to show is not the case. The trained
    model is row-aligned with the lattice it started from, so the substitution is by row and the
    row count is the check.
    """
    from plyfile import PlyData
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    # Positions from the lattice when it is there, and from the trained model when it is not.
    # The two are row-aligned -- that is the invariant `colour_from` already relies on -- so the
    # lattice's own ply is 114 MB of the same coordinates. Letting it be absent is the difference
    # between shipping 375 MB for anyone who wants to redraw these and shipping 1.1 GB.
    src = _os.path.join(lattice_dir, "gs_fill.ply")
    if not _os.path.isfile(src):
        if not (colour_from and _os.path.isfile(colour_from)):
            raise SystemExit(f"neither {src} nor a model to take positions from")
        src = colour_from
        print(f"  positions from {colour_from} (the lattice's own ply is not here)")
    el = PlyData.read(src).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    C0 = 0.28209479177387814
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float64)
                  * C0 + 0.5, 0, 1)
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)[:len(xyz)].numpy()
    if colour_from and _os.path.isfile(colour_from):
        e2 = PlyData.read(colour_from).elements[0]
        if len(e2["x"]) == len(xyz):
            rgb = np.clip(np.stack([e2["f_dc_0"], e2["f_dc_1"], e2["f_dc_2"]], 1)
                          .astype(np.float64) * C0 + 0.5, 0, 1)
            print(f"  appearance from {colour_from}")
        else:
            print(f"  {colour_from} has {len(e2['x']):,} rows against the lattice's "
                  f"{len(xyz):,}; keeping the lattice's own colour")
    return dict(xyz=xyz, rgb=rgb, level=lvl, hc=hc, hf=hf)
