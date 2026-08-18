"""Render a model from the six canonical directions, as one contact sheet.

The evaluation so far has been cross-sections, because that is what the paper is about. But an
appearance that is claimed to be independent of any Gaussian training has to be shown on the
*outside* too -- a shell painted from six views can be right in every section and still be wrong
where the views meet, and a section never looks at a seam.

The six directions are the ones the painting used, so this is the painted shell seen from the
cameras that wrote it plus nothing else. Any object, any lattice directory or ply: the only
thing read is the model and its physics/demo configs.

    python method/common/eval/exterior_views.py MODEL CFG DEMO OUT.png [size]

"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch
from PIL import Image

sys.path += [_FN_ROOT, _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]
_os.chdir(_FN_ROOT)

from plyfile import PlyData                                         # noqa: E402
from scene.gaussian_model import GaussianModel                      # noqa: E402
from utils.camera_view_utils import get_camera_view                 # noqa: E402
from utils.decode_param import decode_param_json                    # noqa: E402
from utils.render_utils import (convert_SH, initialize_resterize,   # noqa: E402
                                load_params_from_gs)
from utils.transformation_utils import *                            # noqa: E402

DEV = "cuda:0"

# The same six the painting used. Named so the sheet can be read without counting.
DIRS = [("up", 0, 90), ("front", 0, 0), ("right", 90, 0),
        ("down", 0, -90), ("back", 180, 0), ("left", 270, 0)]


class P:
    def __init__(self):
        self.sh_degree = 0
        self.compute_cov3D_python = True
        self.convert_SHs_python = False
        self.debug = False


def main(model, cfg, demo, out_png, size=512):
    (mat, bc, tp, pre, cam_p) = decode_param_json(cfg)
    ply = model if model.endswith(".ply") else _os.path.join(model, "gs_fill.ply")
    # At the degree the file carries, for the same reason random_cuts.py does: a trained model
    # keeps its directional appearance in the higher bands, a painted shell has none, and
    # loading both at degree zero would compare one of them against a version of itself.
    n_rest = len([q.name for q in PlyData.read(ply).elements[0].properties
                  if q.name.startswith("f_rest_")])
    deg = int(round(((n_rest / 3 + 1) ** 0.5) - 1)) if n_rest else 0
    g = GaussianModel(deg)
    if n_rest:
        g.load_ply(ply)
        g.active_sh_degree = deg
        print(f"  degree {deg} ({n_rest} higher-band coefficients)")
    else:
        g.load_ply_zero_sh(ply)
        print("  degree 0")

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

    # The same six views off the O-Voxel surface instead of off the Gaussians. The exterior is
    # where the two representations are most directly comparable -- there is no cut, no depth
    # ordering between two layers and no band, just the surface -- so a seam or a hole in the dual
    # grid has nowhere to hide here, which is the point of rendering it.
    surf = None
    tiles = []
    for name, az, el in DIRS:
        cam, _ = get_camera_view(demo, default_camera_index=-1, center_view_world_space=vc,
                                 observant_coordinates=oc, show_hint=False,
                                 init_azimuthm=az, init_elevation=el,
                                 init_radius=cam_p["init_radius"], move_camera=False,
                                 current_frame=0, delta_a=None, delta_e=None, delta_r=None)
        if surf is not None:
            fp = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
            w2c = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
            SS = 2
            im = np.ones((size * SS * size * SS, 3), np.float32)
            zb = np.full(size * SS * size * SS, np.inf)
            _splat(im, zb, surf[0], surf[1], fp, w2c, size * SS)
            a = im.reshape(size * SS, size * SS, 3)
            a = (a.reshape(size, SS, size, SS, 3).mean((1, 3)) * 255).astype(np.uint8)
        else:
            rast = initialize_resterize(cam, g, P(), bg, image_height=size, image_width=size)
            col = convert_SH(shs, cam, g, world, None)
            img, _, _, _ = rast(means3D=world, means2D=sp, shs=None,
                                colors_precomp=col.contiguous().clamp(0, 1), opacities=op,
                                scales=None, rotations=None, cov3D_precomp=cov)
            a = (img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        tiles.append(a)
        print(f"  {name:<6} mean {a.reshape(-1, 3).mean(0).round(1)}")

    sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    Image.fromarray(sheet).save(out_png)
    print(f"  -> {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 512)
