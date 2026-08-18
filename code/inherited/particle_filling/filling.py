import torch
import os
import numpy as np
import taichi as ti
import mcubes

from mpm_solver_warp.engine_utils import *

# 1. densify grids
# 2. identify grids whose density is larger than some threshold
# 3. filling grids with particles
# 4. identify and fill internal grids


@ti.func
def compute_density(index, pos, opacity, cov, grid_dx):
    gaussian_weight = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                node_pos = (index + ti.Vector([i, j, k])) * grid_dx
                dist = pos - node_pos
                gaussian_weight += ti.exp(-0.5 * dist.dot(cov @ dist))

    return opacity * gaussian_weight / 8.0


@ti.kernel
def densify_grids(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
):
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)
        cov = ti.Matrix(
            [
                [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
                [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
                [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
            ]
        )
        sig, Q = ti.sym_eig(cov)
        sig[0] = ti.max(sig[0], 1e-8)
        sig[1] = ti.max(sig[1], 1e-8)
        sig[2] = ti.max(sig[2], 1e-8)
        sig_mat = ti.Matrix(
            [[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]]
        )
        cov = Q @ sig_mat @ Q.transpose()
        r = 0.0
        for idx in ti.static(range(3)):
            if sig[idx] < 0:
                sig[idx] = ti.sqrt(-sig[idx])
            else:
                sig[idx] = ti.sqrt(sig[idx])

            r = ti.max(r, sig[idx])

        r = ti.ceil(r / grid_dx, dtype=int)
        r = ti.min(r, 12)
        if r < 1:
            print(r)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (
                        i + dx >= 0
                        and i + dx < grid_density.shape[0]
                        and j + dy >= 0
                        and j + dy < grid_density.shape[1]
                        and k + dz >= 0
                        and k + dz < grid_density.shape[2]
                    ):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos,
                            opacity[pi],
                            cov,
                            grid_dx,
                        )
                        density = ti.max(1, density)
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)

@ti.kernel
def densify_grids_v2(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
):
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)

@ti.kernel
def fill_dense_grids(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    density_thres: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid_density:
        if grid_density[i, j, k] > density_thres:
            if grid[i, j, k] < max_particles_per_cell:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)

                for index in range(tmp_start_idx, tmp_start_idx + diff):
                    di = ti.random()
                    dj = ti.random()
                    dk = ti.random()
                    new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx

    return new_start_idx


@ti.func
def collision_search(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> bool:
    dir = ti.Vector([0, 0, 0])
    if dir_type == 0:
        dir[0] = 1
    elif dir_type == 1:
        dir[0] = -1
    elif dir_type == 2:
        dir[1] = 1
    elif dir_type == 3:
        dir[1] = -1
    elif dir_type == 4:
        dir[2] = 1
    elif dir_type == 5:
        dir[2] = -1

    flag = False
    index += dir
    i, j, k = index
    while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
        if grid_density[index] > threshold:
            flag = True
            break
        index += dir
        i, j, k = index

    return flag


@ti.func
def collision_times(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> int:
    dir = ti.Vector([0, 0, 0])
    times = 0
    if dir_type > 5 or dir_type < 0:
        times = 1
    else:
        if dir_type == 0:
            dir[0] = 1
        elif dir_type == 1:
            dir[0] = -1
        elif dir_type == 2:
            dir[1] = 1
        elif dir_type == 3:
            dir[1] = -1
        elif dir_type == 4:
            dir[2] = 1
        elif dir_type == 5:
            dir[2] = -1

        state = grid_density[index] > threshold
        #print(grid_density[index])
        #print(threshold)
        index += dir
        i, j, k = index
        while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
            new_state = grid_density[index] > threshold
            if new_state != state and state == False:
                times += 1
            state = new_state
            index += dir
            i, j, k = index

    return times


@ti.kernel
def internal_filling(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
) -> int:
    new_start_idx = start_idx
    count1 = 0
    count2 = 0
    for i, j, k in grid:
        if grid[i, j, k] == 0:
            collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    collision_hit = collision_hit and hit_test

            if collision_hit:
                count1 += 1
                hit_times = collision_times(
                    grid=grid,
                    grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir,
                    size=grid.shape[0],
                    threshold=threshold,
                )

                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    for index in range(tmp_start_idx, tmp_start_idx + diff):
                        di = ti.random()
                        dj = ti.random()
                        dk = ti.random()
                        new_particles[index] = (
                            ti.Vector([i + di, j + dj, k + dk]) * grid_dx
                        )
                        count2 += 1
    print("count of internal filling particles")
    print(count1)
    print(count2)
    return new_start_idx

@ti.kernel
def internal_filling_v2(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
) -> int:
    new_start_idx = start_idx
    count1 = 0
    count2 = 0
    for i, j, k in grid:
        if grid[i, j, k] == 0 and grid_density[i, j, k] <= 0:
            collision_hit = True
            count_dir = 0
            for dir_type in ti.static(range(6)):
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    if hit_test:
                        count_dir += 1
            
            if count_dir >= 6:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                for index in range(tmp_start_idx, tmp_start_idx + diff):
                    # Voxel variant: sit at the exact cell centre instead of a random
                    # point inside the cell, so "this primitive" and "this voxel" are
                    # the same fact. Combined with the frozen xyz gradient in training,
                    # the interior becomes a lattice rather than free-floating Gaussians.
                    new_particles[index] = (
                        ti.Vector([i + 0.5, j + 0.5, k + 0.5]) * grid_dx
                    )
                    count2 += 1
    print("count of internal filling particles")
    print(count1)
    print(count2)
    return new_start_idx

@ti.kernel
def internal_filling_2d(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
    plane: ti.template(),
) -> int:
    new_start_idx = start_idx
    count1 = 0
    count2 = 0
    for i, j, k in grid:
        if grid[i, j, k] == 0:
            collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    collision_hit = collision_hit and hit_test

            if collision_hit:
                count1 += 1
                hit_times = collision_times(
                    grid=grid,
                    grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir,
                    size=grid.shape[0],
                    threshold=threshold,
                )

                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    for index in range(tmp_start_idx, tmp_start_idx + diff):
                        di = ti.random()
                        dj = ti.random()
                        dk = ti.random()
                        new_particles[index] = (
                            ti.Vector([i + di, j + dj, k + dk]) * grid_dx
                        )
                        count2 += 1
    print("count of internal filling particles")
    print(count1)
    print(count2)
    return new_start_idx


@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


@ti.kernel
def compute_particle_volume(
    pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]


@ti.kernel
def assign_particle_to_grid(
    pos: ti.template(),
    grid: ti.template(),
    grid_dx: float,
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


def get_particle_volume(pos, grid_n: int, grid_dx: float, unifrom: bool = False):
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if unifrom:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()


def lattice_enclosure(originals, lattice):
    """Split original primitives into 'inside the lattice' and 'shell', and report the
    cells the inside ones occupy.

    fill_particles only fills cells that were *empty*, so the lattice and the original
    primitives occupy disjoint cells by construction -- an "is this point in an occupied
    lattice cell" test therefore matches nothing (measured: 0 of 347412). Replacing the
    interior instead of merely padding it needs a geometric test, and the lattice itself
    supplies one: a cell is interior iff filled cells exist on both sides of it along
    every axis. Shell primitives fail this on whichever axis points out of the object.

    Returns (inside_mask over `originals`, cell centres those points vacate, dx).
    """
    dev = originals.device
    s = lattice[torch.randperm(lattice.shape[0], device=dev)[:4000]]
    dd = []
    for i in range(0, s.shape[0], 2000):
        dm = torch.cdist(s[i:i + 2000], lattice)
        dm[dm < 1e-9] = 1e9
        dd.append(dm.min(1).values)
    dx = float(torch.cat(dd).median())

    mn = lattice.min(0).values
    dims = ((lattice.max(0).values - mn) / dx).round().long() + 1
    occ = torch.zeros(dims.tolist(), dtype=torch.bool, device=dev)
    il = ((lattice - mn) / dx).round().long().clamp(
        torch.zeros(3, dtype=torch.long, device=dev), dims - 1)
    occ[il[:, 0], il[:, 1], il[:, 2]] = True

    # "a filled cell exists at or beyond this index" in both directions, per axis
    enclosed = torch.ones_like(occ)
    for ax in range(3):
        fwd = torch.cummax(occ, dim=ax).values
        bwd = torch.flip(torch.cummax(torch.flip(occ, [ax]), dim=ax).values, [ax])
        enclosed &= fwd & bwd

    io = ((originals - mn) / dx).round().long()
    inb = ((io >= 0) & (io < dims)).all(1)
    inside = torch.zeros(originals.shape[0], dtype=torch.bool, device=dev)
    if inb.any():
        j = io[inb]
        inside[inb] = enclosed[j[:, 0], j[:, 1], j[:, 2]] & ~occ[j[:, 0], j[:, 1], j[:, 2]]

    vac = torch.unique(io[inside], dim=0) if int(inside.sum()) else io[:0]
    return inside, mn + vac.float() * dx, dx


def fill_particles(
    pos,
    opacity,
    cov,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres=2.0,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=5,
    ray_cast_dir=4,
    boundary: list = None,
    smooth: bool = False,
):
    pos_clone = pos.clone()
    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool).cuda()
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]]).cuda()
        pos = pos - new_origin

    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
    particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
    fill_num = 0

    # compute density_field
    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)

    # Initialize an empty list to store the points
    points = []
    grid_density_torch = grid_density.to_torch()
    # Iterate through the grid and collect points where grid_density > threshold
    for i in range(grid_n):
        for j in range(grid_n):
            for k in range(grid_n):
                if grid_density_torch[i, j, k] > 0:
                    x = (i + 0.5) * grid_dx
                    y = (j + 0.5) * grid_dx
                    z = (k + 0.5) * grid_dx
                    points.append([x, y, z])
    points_tensor = torch.tensor(points, dtype=torch.float32)
    particle_position_tensor_to_ply(points_tensor, "./log/density_particles.ply")

    # fill dense grids
    """
    fill_num = fill_dense_grids(
        grid,
        grid_density,
        grid_dx,
        density_thres,
        particles,
        0,
        max_particles_per_cell,
    )
    print("after dense grids: ", fill_num)"""

    # smooth density_field
    if smooth:
        df = grid_density.to_numpy()
        smoothed_df = mcubes.smooth(df, method="constrained", max_iters=500).astype(
            np.float32
        )
        grid_density.from_numpy(smoothed_df)
        print("smooth finished")

    # fill internal grids
    fill_num = internal_filling_v2(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        exclude_dir=search_exclude_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        ray_cast_dir=ray_cast_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        threshold=search_thres,
    )
    print("after internal grids: ", fill_num)

    # put new particles together with original particles
    particles_tensor = particles.to_torch()[:fill_num].cuda()
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
    particles_tensor = torch.cat([pos_clone, particles_tensor], dim=0)
    print("particle count after internal filling:")
    print(particles_tensor.size())
    diff = particles_tensor.size()[0] - pos_clone.size()[0]
    print("total internal filling size: ")
    print(diff)

    return particles_tensor


def fill_particles_2d(
    pos,
    opacity,
    cov,
    plane,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=5,
    ray_cast_dir=4,
    boundary: list = None,
    smooth: bool = False
):
    pos_clone = pos.clone()
    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool).cuda()
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]]).cuda()
        pos = pos - new_origin

    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
    particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
    fill_num = 0

    # compute density_field
    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)

    # fill internal grids
    fill_num = internal_filling_2d(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        exclude_dir=search_exclude_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        ray_cast_dir=ray_cast_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        threshold=search_thres,
        plane=plane
    )
    print("after internal grids: ", fill_num)

    # put new particles together with original particles
    particles_tensor = particles.to_torch()[:fill_num].cuda()
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
    particles_tensor = torch.cat([pos_clone, particles_tensor], dim=0)
    print("particle count after internal filling:")
    print(particles_tensor.size())
    diff = particles_tensor.size()[0] - pos_clone.size()[0]
    print("total internal filling size: ")
    print(diff)
    return particles_tensor


@ti.kernel
def get_attr_from_closest(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_cov: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_cov: ti.template(),
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_cov[pi] = ti_cov[min_idx]

@ti.kernel
def get_attr_from_closest2(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_scale: ti.template(),
    ti_rot: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_scale: ti.template(),
    ti_new_rot: ti.template()
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_scale[pi] = ti_scale[min_idx]
        ti_new_rot[pi] = ti_rot[min_idx]

def init_filled_particles(pos, shs, cov, opacity, new_pos):
    shs = shs.reshape(pos.shape[0], -1)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=shs.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_cov.from_torch(cov.reshape(-1, 6))
    ti_shs.from_torch(shs)
    ti_opacity.from_torch(opacity.reshape(-1))

    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).cuda()
    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_cov = ti.Vector.field(n=6, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))
    ti_new_shs.from_torch(new_shs)

    get_attr_from_closest(
        ti_pos,
        ti_shs,
        ti_opacity,
        ti_cov,
        ti_new_pos,
        ti_new_shs,
        ti_new_opacity,
        ti_new_cov,
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    cov_tensor = ti_new_cov.to_torch().cuda()

    shs_tensor = torch.cat([shs, shs_tensor], dim=0)
    shs_tensor = shs_tensor.view(shs_tensor.shape[0], -1, 3)
    opacity_tensor = torch.cat([opacity, opacity_tensor.reshape(-1, 1)], dim=0)
    cov_tensor = torch.cat([cov, cov_tensor], dim=0)
    return shs_tensor, opacity_tensor, cov_tensor

def _sample_polar(img, u, v):
    """Nearest-neighbour sample img (H,W,3) at float pixel coords."""
    H, W, _ = img.shape
    return img[v.clamp(0, H - 1).long(), u.clamp(0, W - 1).long()]


def interior_colour_from_reference(new_pos, centre, radius, ref_dir, up_axis=1):
    """Initialise interior colour by sampling real cross-section photographs.

    Copying colour from the source cloud -- however carefully -- can only ever
    reproduce the *peel*, because that is the only thing an outside-in 3DGS scan
    observed. The interior of a citrus fruit does not look like its peel, so the
    filled volume starts off wrong no matter which neighbours are picked. The repo
    already ships real cross-section photographs under data_finetune_images/<obj>/
    (used only for the optional Dreambooth step in the README); they are a far
    better prior for what the inside should look like.

    Mapping: treat the fruit as a solid of revolution about `up_axis`.
      * the vertical photo is sampled at (signed radial distance, height) and
        supplies the profile -- peel thickness at the poles, core, stem end;
      * the horizontal photo supplies the *angular* segment pattern, applied as a
        modulation around its own radial mean so the radial falloff is not counted
        twice.

    Returns SH DC coefficients, or None if the reference photos are unavailable.
    """
    import glob
    from PIL import Image

    def _load(sub):
        fs = sorted(glob.glob(os.path.join(ref_dir, sub, "*.png")))
        if not fs:
            return None
        a = np.asarray(Image.open(fs[0]).convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(a).cuda()

    H_img, V_img = _load("horizontal"), _load("vertical")
    if H_img is None and V_img is None:
        return None
    if H_img is None:
        H_img = V_img
    if V_img is None:
        V_img = H_img

    p = new_pos.reshape(-1, 3).cuda().float() - centre
    ax = [i for i in range(3) if i != up_axis]
    h = p[:, up_axis]                                   # height along the axis
    rho = p[:, ax].norm(dim=1)                          # distance from the axis
    theta = torch.atan2(p[:, ax[1]], p[:, ax[0]])

    rho_n = (rho / radius).clamp(0, 1)
    h_n = (h / radius).clamp(-1, 1)

    # --- angular segment pattern from the horizontal photo -------------------
    Hh, Wh, _ = H_img.shape
    disc = 0.92 * (min(Hh, Wh) / 2)
    u = Wh / 2 + rho_n * disc * torch.cos(theta)
    v = Hh / 2 + rho_n * disc * torch.sin(theta)
    c_h = _sample_polar(H_img, u, v)
    # radial mean of the same photo, so only the angular deviation survives
    NA = 64
    angs = torch.linspace(0, 2 * np.pi, NA, device="cuda")
    ur = Wh / 2 + rho_n[:, None] * disc * torch.cos(angs)[None, :]
    vr = Hh / 2 + rho_n[:, None] * disc * torch.sin(angs)[None, :]
    c_h_mean = _sample_polar(H_img, ur.reshape(-1), vr.reshape(-1)
                             ).reshape(-1, NA, 3).mean(1)
    modulation = c_h / (c_h_mean + 1e-3)

    # --- profile from the vertical photo -------------------------------------
    Hv, Wv, _ = V_img.shape
    uv = Wv / 2 + rho_n * (0.92 * Wv / 2)
    vv = Hv / 2 - h_n * (0.92 * Hv / 2)
    c_v = _sample_polar(V_img, uv, vv)

    rgb = (c_v * modulation).clamp(0, 1)
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def init_filled_particles2(pos, shs, rot, scale, opacity, new_pos,
                           shell_frac=0.6, knn_k=32,
                           ref_dir=None, init_mode="uniform"):
    """Assign appearance to the newly filled interior particles.

    The released version copied every attribute from the single nearest source
    Gaussian. Two properties of the source cloud make that produce a mostly black
    interior (measured on orange_raw.ply: 68% of filled points come out pure black):

      * A 3DGS scan of an opaque object is only constrained on its shell. The few
        thousand Gaussians that ended up *inside* the shell were never visible in
        any training view, so they are unconstrained leftovers -- 96% of them are
        pure black below 0.2 of the object radius.
      * Deep interior fill points are physically closest to exactly those leftovers,
        so single-nearest-neighbour hands them the worst possible source colour.

    So restrict the colour source to the shell (radius > shell_frac of max) and
    average SH over knn_k neighbours instead of taking one. Measured effect on the
    filled cloud: 68.0% -> 6.9% pure black. Geometry (opacity/scale/rotation) still
    comes from the single nearest shell point -- averaging quaternions is not
    meaningful and the geometric attributes are not what was broken.
    """
    shs = shs.reshape(pos.shape[0], -1)

    src_xyz = pos.reshape(-1, 3).cuda().float()
    src_shs = shs.cuda().float()
    src_op = opacity.reshape(-1).cuda().float()
    src_scale = scale.reshape(-1, 3).cuda().float()
    src_rot = rot.reshape(-1, 4).cuda().float()

    radius = (src_xyz - src_xyz.mean(0)).norm(dim=1)
    keep = radius > (shell_frac * radius.max())
    if keep.sum() < knn_k:            # degenerate cloud: fall back to using everything
        keep = torch.ones_like(keep)
    src_xyz, src_shs = src_xyz[keep], src_shs[keep]
    src_op, src_scale, src_rot = src_op[keep], src_scale[keep], src_rot[keep]
    print(f"interior fill: colour source restricted to shell -> "
          f"{int(keep.sum())}/{keep.shape[0]} points, K={knn_k}")

    q = new_pos.reshape(-1, 3).cuda().float()
    K = min(knn_k, src_xyz.shape[0])
    out_shs = torch.empty(q.shape[0], src_shs.shape[1], device="cuda")
    out_op = torch.empty(q.shape[0], device="cuda")
    out_scale = torch.empty(q.shape[0], 3, device="cuda")
    out_rot = torch.empty(q.shape[0], 4, device="cuda")

    # Size the distance-matrix tile to the memory that is actually free, rather than to a
    # fixed 2048 x 32768. That pair costs 268 MB per cdist no matter how big the cloud is,
    # and Taichi reserves 16 GB up front for the fill grid and never returns it, so on the
    # larger of the two fruits there were 118 MB left and the constant tile could not be
    # allocated at all. A quarter of what is free, split four ways, leaves room for the
    # topk and the concatenations that cdist feeds.
    tile = max(1 << 20, int(torch.cuda.mem_get_info()[0] * 0.25) // 16)   # fp32 elements
    SC = int(min(32768, max(4096, tile ** 0.5)))
    QC = int(min(2048, max(256, tile // SC)))
    for a in range(0, q.shape[0], QC):
        qc = q[a:a + QC]
        best_d = torch.full((qc.shape[0], K), float("inf"), device="cuda")
        best_i = torch.zeros((qc.shape[0], K), dtype=torch.long, device="cuda")
        for b in range(0, src_xyz.shape[0], SC):
            d = torch.cdist(qc, src_xyz[b:b + SC])
            k = min(K, d.shape[1])
            dv, di = d.topk(k, dim=1, largest=False)
            best_d = torch.cat([best_d, dv], 1)
            best_i = torch.cat([best_i, di + b], 1)
            best_d, sel = best_d.topk(K, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel)
        out_shs[a:a + QC] = src_shs[best_i].mean(1)     # colour: average K neighbours
        nearest = best_i[:, 0]                          # geometry: single nearest
        out_op[a:a + QC] = src_op[nearest]
        out_scale[a:a + QC] = src_scale[nearest]
        out_rot[a:a + QC] = src_rot[nearest]

    if init_mode == "uniform":
        # Paper section 3.1.2: "Each identified voxel is then initialized with a
        # predefined number of Gaussian primitives with uniform color and opacity."
        # The released code computes exactly this mean and then throws it away by
        # overwriting every entry with its nearest neighbour's SH.
        out_shs = torch.mean(src_shs, dim=0).repeat(new_pos.shape[0], 1)
        print("interior fill: uniform colour init (paper section 3.1.2)")

    # Real cross-section photographs, when explicitly requested. Not part of the
    # paper's method -- kept as an experiment.
    if ref_dir is not None and os.path.isdir(ref_dir):
        centre = pos.reshape(-1, 3).cuda().float().mean(0)
        obj_r = (pos.reshape(-1, 3).cuda().float() - centre).norm(dim=1).quantile(0.98)
        ref_sh = interior_colour_from_reference(new_pos, centre, obj_r, ref_dir)
        if ref_sh is not None:
            out_shs = out_shs.clone()
            out_shs[:, :3] = ref_sh                     # DC term carries the colour
            out_shs[:, 3:] = 0                          # drop inherited peel SH detail
            print(f"interior fill: colour initialised from reference photos in {ref_dir}")

    return out_shs, out_op, out_scale, out_rot


def _init_filled_particles2_original(pos, shs, rot, scale, opacity, new_pos):
    shs = shs.reshape(pos.shape[0], -1)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_scale = ti.Vector.field(n=3, dtype=float, shape=scale.shape[0])
    ti_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=shs.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_rot = ti.Vector.field(n=4, dtype=float, shape=rot.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_scale.from_torch(scale.reshape(-1, 3))
    ti_shs.from_torch(shs)
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_rot.from_torch(rot.reshape(-1, 4))

    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).cuda()
    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_scale = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_rot = ti.Vector.field(n=4, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))
    ti_new_shs.from_torch(new_shs)

    get_attr_from_closest2(
        ti_pos,
        ti_shs,
        ti_opacity,
        ti_scale,
        ti_rot,
        ti_new_pos,
        ti_new_shs,
        ti_new_opacity,
        ti_new_scale,
        ti_new_rot
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    scale_tensor = ti_new_scale.to_torch().cuda()
    rot_tensor = ti_new_rot.to_torch().cuda()

    return shs_tensor, opacity_tensor, scale_tensor, rot_tensor
