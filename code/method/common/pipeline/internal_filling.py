import sys
import os

# The directory containing THIS script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add project root (for utils)
sys.path.append(script_dir)

# Add gaussian-splatting (for scene, etc)
gs_dir = os.path.join(script_dir, "gaussian-splatting")
sys.path.append(gs_dir)

import argparse
import torch
import numpy as np
import warp as wp
import taichi as ti
from plyfile import PlyData

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs
from utils.transformation_utils import (
    generate_rotation_matrices,
    apply_rotations,
    apply_cov_rotations,
    transform2origin,
    shift2center111,
    undotransform2origin,
    undoshift2center111,
    apply_inverse_rotations,
)
from particle_filling.filling import fill_particles, init_filled_particles2

# Initialize Warp and Taichi
wp.init()
wp.config.verify_cuda = True


def _taichi_pool_gb():
    """Reserve what the fill's Taichi fields need, not a flat 16 GB.

    Taichi takes its pool from the device up front and does not give it back, so the flat
    16 GB left under 1.5 GB for PyTorch and the interior filler's nearest-neighbour pass
    could not allocate a 74 MB tile. What the fill actually holds is the sample pool
    (max_particles_num x 3 floats, 228 MB at the configured 19M) and a handful of
    per-particle and grid_n^3 fields -- under 1 GB together. The pool is sized from the
    config so a bigger grid or sample budget still gets what it asks for.
    """
    cfg_path = None
    for i, a in enumerate(sys.argv):
        if a == "--physics_config" and i + 1 < len(sys.argv):
            cfg_path = sys.argv[i + 1]
    if not (cfg_path and os.path.exists(cfg_path)):
        return 4.0
    import json
    f = json.load(open(cfg_path)).get("particle_filling") or {}
    n = int(f.get("n_grid", 100))
    fields = f.get("max_particles_num", 19_000_000) * 3 * 4 + n ** 3 * 8
    return round(max(2.0, fields / 2 ** 30 * 3.0), 1)      # 3x headroom


ti.init(arch=ti.cuda, device_memory_GB=_taichi_pool_gb())


class PipelineParamsNoparse:
    """Minimal pipeline parameter config (no argparse parsing)."""
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def particle_position_tensor_to_ply(position_tensor, filename):
    """
    Save a (N, 3) tensor of particle positions to a binary .ply file.
    Overwrites the file if it exists.
    """
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy().astype(np.float32)
    num_particles = position.shape[0]

    header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
    with open(filename, "wb") as f:
        f.write(header.encode('utf-8'))
        f.write(position.tobytes())

    print(f"Saved {num_particles} particles to {filename}")


def load_checkpoint(ply_path):
    """Load Gaussian splatting checkpoint from a PLY file directly."""
    gaussians = GaussianModel(sh_degree=0)  # No SHs assumed
    gaussians.load_ply_zero_sh(ply_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .ply file of the Gaussian model.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the new .ply file.")
    parser.add_argument("--physics_config", type=str, required=True, help="Path to physics config JSON file.")
    parser.add_argument("--white_bg", action="store_true", help="Use white background (default is black).")
    parser.add_argument("--debug", action="store_true", help="Enable debug .ply outputs.")
    parser.add_argument("--init_mode", type=str, default="uniform",
                        choices=["uniform", "nearest"],
                        help="Interior colour init: 'uniform' (paper section 3.1.2) or "
                             "'nearest' (the released nearest-neighbour copy).")
    parser.add_argument("--ref_images", type=str, default=None,
                        help="Directory with horizontal/ and vertical/ cross-section photos "
                             "(e.g. ./data_finetune_images/orange) used to initialise interior colour.")
    args = parser.parse_args()

    assert os.path.exists(args.model_path), "Model path does not exist!"
    assert os.path.exists(args.physics_config), "Physics config does not exist!"
    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load configs and checkpoint
    print("Loading configs...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.physics_config)

    print("Loading Gaussians...")
    gaussians = load_checkpoint(args.model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = torch.tensor([1, 1, 1] if args.white_bg else [0, 0, 0], dtype=torch.float32, device="cuda")

    params = load_params_from_gs(gaussians, pipeline)
    pos, cov, screen_points, opacity, shs = (
        params["pos"],
        params["cov3D_precomp"],
        params["screen_points"],
        params["opacity"],
        params["shs"],
    )

    # Filter out low-opacity particles
    mask = opacity[:, 0] > preprocessing_params["opacity_threshold"]
    pos, cov, opacity, screen_points, shs = pos[mask], cov[mask], opacity[mask], screen_points[mask], shs[mask]

    if args.debug:
        os.makedirs("./log", exist_ok=True)
        particle_position_tensor_to_ply(pos, "./log/init_particles.ply")

    # Apply rotation, scale, and shift to canonical coordinates
    rot_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"]
    )
    rotated_pos = apply_rotations(pos, rot_matrices)
    cov = apply_cov_rotations(cov, rot_matrices)

    transformed_pos, scale_origin, mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)
    cov = scale_origin**2 * cov

    if args.debug:
        particle_position_tensor_to_ply(transformed_pos, "./log/transformed_particles.ply")

    # Fill internal points if enabled
    device = "cuda:0"
    if (fill_params := preprocessing_params.get("particle_filling")):
        print("Filling internal particles...")
        filled_pos = fill_particles(
            pos=transformed_pos,
            opacity=opacity,
            cov=cov,
            grid_n=fill_params["n_grid"],
            max_samples=fill_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / fill_params["n_grid"],
            density_thres=fill_params["density_threshold"],
            search_thres=fill_params["search_threshold"],
            max_particles_per_cell=fill_params["max_partciels_per_cell"],
            search_exclude_dir=fill_params["search_exclude_direction"],
            ray_cast_dir=fill_params["ray_cast_direction"],
            boundary=fill_params["boundary"],
            smooth=fill_params["smooth"],
        ).to(device)
    else:
        filled_pos = transformed_pos.to(device)

    init_num = transformed_pos.shape[0]


    if fill_params and fill_params.get("visualize"):
        shs, opacity, scales, rots = init_filled_particles2(
            filled_pos[:init_num], shs, gaussians._rotation, gaussians._scaling, gaussians._opacity, filled_pos[init_num:],
            ref_dir=args.ref_images, init_mode=args.init_mode
        )
    else:
        scales, rots = gaussians._scaling, gaussians._rotation

    # Reverse transform for saving
    new_points = filled_pos[init_num:].clone()
    new_points = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(new_points), scale_origin, mean_pos),
        rot_matrices
    ).detach().cpu()

    def create_colored_cloud(xyz, rgb):
        colors = np.tile(rgb, (xyz.shape[0], 1))
        return BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((xyz.shape[0], 3)))

    green_cloud = create_colored_cloud(new_points, [0, 0.6, 0])  # Greenish RGB for new points

    print("Check saved *.ply files for verification.")
    num_exterior = gaussians.get_xyz.shape[0]
    gaussians.add_points_from_pcd_zero(
        green_cloud,
        p_size=0.008,  # Initial guess for new particle size
        opacity=0.8,   # Initial guess for new particle opacity
        new_features=shs,
        new_scales=scales,
        new_rots=rots,
    )
    # Flag the freshly added interior points so training can pin them to their cells.
    num_total = gaussians.get_xyz.shape[0]
    flag = torch.zeros(num_total, dtype=torch.bool, device=gaussians.get_xyz.device)
    flag[num_exterior:] = True
    gaussians.is_interior = flag
    torch.save(flag.cpu(), os.path.join(args.output_path, "is_interior.pt"))
    print(f"interior flag: {int(flag.sum())} of {num_total} points")
    gaussians.save_ply(os.path.join(args.output_path, "gs_fill.ply"))
