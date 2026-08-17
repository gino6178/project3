"""Dynamic topological slicing: cut at runtime, let the system discover the pieces.

The two-body demo hard-coded the answer -- I split the model along a fixed plane and
created exactly two solvers by hand. That is not topology, it is a hard-coded constant.
Here a cut is an event: a plane arrives while the simulation is running, connectivity is
recomputed on the voxel lattice, connected-component labelling reports how many pieces
now exist, and one MPM solver is built per piece with its particles' current positions
and velocities carried over. Cut again and it happens again -- 1 -> 2 -> 4.

Why the lattice is what makes this cheap and exact:
  * connectivity is adjacency on a regular grid, so a cut is "do these two neighbouring
    cells fall on the same side of the plane", answered per cell in O(1);
  * component labelling is then textbook grid labelling, exact and parameter-free.
With free Gaussians neither step is available: there is no adjacency, so you must build a
k-NN graph and pick a distance threshold, and the number of pieces you get depends on
that threshold.

Membership of a Gaussian in a piece follows its nearest lattice particle, and the shell
is skinned to the lattice with weights restricted to its own piece, so a cut severs the
skinning as well as the physics.
"""
import os as _os
# The repository root, so the same file runs here and on the remote box. It was written three
# times in every script -- two sys.path entries and a chdir -- and each one silently pinned the
# script to one machine: on the remote the chdir landed somewhere else and a relative source
# path then could not be found, which surfaces as "no such file" for a file that is plainly
# there.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

import warp as wp
import torch, cv2, numpy as np, time, json
wp.init()

from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from scene.gaussian_model import GaussianModel
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs, initialize_resterize, convert_SH
from utils.transformation_utils import (
    generate_rotation_matrices, apply_cov_rotations, apply_inverse_cov_rotations,
    transform2origin, shift2center111, undotransform2origin, undoshift2center111,
    apply_inverse_rotations, get_center_view_worldspace_and_observant_coordinate)
from utils.camera_view_utils import get_camera_view

DEV = "cuda:0"
GRID_LIM = 3.2
# Resolution across the object, not across the domain, is what matters: the fruit spans
# ~1.0 of a 3.2 domain, so n_grid=80 puts only ~25 cells across it and a 1/6 wedge is a
# few cells thick. Two pieces stayed smooth at that setting; six did not.
N_GRID = int(os.environ.get("N_GRID", "80"))
DX_TARGET = float(os.environ.get("DX_TARGET", "0.04"))
K_SKIN = int(os.environ.get("K_SKIN", "16"))


class P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


# ---------------------------------------------------------------- lattice topology

def build_grid(lat):
    """Recover the lattice: spacing, origin, dimensions, occupancy, cell centres."""
    s = lat[torch.randperm(lat.shape[0], device=DEV)[:4000]]
    dd = []
    with torch.no_grad():
        for i in range(0, s.shape[0], 2000):
            dm = torch.cdist(s[i:i + 2000], lat)
            dm[dm < 1e-9] = 1e9
            dd.append(dm.min(1).values)
    dx = float(torch.cat(dd).median())
    mn = lat.min(0).values
    dims = (((lat.max(0).values - mn) / dx).round().long() + 1)
    idx = ((lat - mn) / dx).round().long().clamp(
        torch.zeros(3, dtype=torch.long, device=DEV), dims - 1)
    occ = torch.zeros(dims.tolist(), dtype=torch.bool, device=DEV)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = torch.meshgrid(*[torch.arange(int(d), device=DEV) for d in dims], indexing="ij")
    centres = mn + torch.stack(g, -1).float() * dx
    return dx, mn, dims, occ, centres, idx


DILATE = 2
# Contact between pieces. `CONTACT_PUSH` is the fraction of the measured penetration depth
# removed per frame -- 1.0 resolves it in one step and jitters, 0 leaves the bodies free to
# interpenetrate, which is what they did.
CONTACT_PUSH = float(os.environ.get('CONTACT_PUSH', '0.5'))
CONTACT_RESTITUTION = float(os.environ.get('CONTACT_RESTITUTION', '0.30'))
# How many query-and-separate passes per contacting pair per frame, and the furthest one pass may
# move a piece. The cap is what keeps a particle from being thrown out of the solver's own grid,
# which is how the per-particle version died.
CONTACT_ITERS = int(os.environ.get('CONTACT_ITERS', '3'))
MAX_SEP_CELLS = float(os.environ.get('MAX_SEP_CELLS', '4'))
MAT = None      # per-particle (E, nu, density), set in main() when MATERIAL is given
MAT_LVL = None  # per-particle level, kept so the squash can be reported per level


def label_components(occ, signs):
    """Connected components of the lattice, with cut planes severing adjacency.

    Two details the naive version got wrong:

    * The lattice is not solid. internal_filling_v2 fills one particle per cell and skips
      cells that already held a primitive, so an occupied cell has on average 18.5 of its
      26 neighbours occupied and plain 6-connectivity shatters the intact fruit into 3141
      fragments. Closing the grid by DILATE cells before labelling puts it back to one
      component; labels are then read at the real cells only.
    * A cut severs adjacency, which is awkward to express as a graph edit. It is instead
      exactly equivalent to labelling each sign region separately -- a component cannot
      span a plane it is on both sides of -- so P planes give 2^P independent labellings.
    """
    from scipy import ndimage
    import torch.nn.functional as Fn
    o = occ.float()[None, None]
    dil = (Fn.max_pool3d(o, 2 * DILATE + 1, 1, DILATE)[0, 0] > 0.5)
    occ_np = occ.cpu().numpy()
    dil_np = dil.cpu().numpy()
    sg_np = [s.cpu().numpy() for s in signs]

    out = np.full(occ_np.shape, -1, dtype=np.int64)
    nxt = 0
    for combo in range(1 << len(sg_np)):
        region = np.ones_like(occ_np)
        for b, s in enumerate(sg_np):
            region &= s if (combo >> b) & 1 else ~s
        m = dil_np & region
        if not m.any():
            continue
        lab_r, k = ndimage.label(m)
        take = occ_np & m
        out[take] = lab_r[take] + nxt
        nxt += k
    return torch.from_numpy(out).to(DEV), nxt


def piece_of_particles(lab, idx, min_size=2000):
    """Component id per lattice particle, compacted to 0..K-1, tiny shards dropped."""
    raw = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    raw = torch.where(raw >= 0, raw, torch.full_like(raw, -1))
    uniq, counts = torch.unique(raw, return_counts=True)
    keep = uniq[counts >= min_size]
    out = torch.full_like(raw, -1)
    for new, u in enumerate(keep.tolist()):
        out[raw == u] = new
    return out, len(keep)


# ---------------------------------------------------------------- solvers

def make_solver(x_local, up, cell_vol, floor_pt_local, grid_lim, n_grid=None,
                E=1.2e6, density=800.0, mat=None):
    """`mat` is (E, nu, density) per particle for this piece, or None for one material.

    The constant pass still runs first: `set_parameters_dict` allocates and fills every field,
    and `finalize_mu_lam` reads E and nu to build the Lame parameters the transfer kernels
    actually use. So the per-particle values have to be written between the two, not after.
    """
    s = MPM_Simulator_WARP(10)
    vol = torch.full((x_local.shape[0],), cell_vol, device=DEV)
    s.load_initial_data_from_torch(x_local, vol, n_grid=n_grid or N_GRID,
                                   grid_lim=grid_lim, device=DEV)
    s.set_parameters_dict({"E": E, "nu": 0.35, "material": "jelly",
                           "density": density, "g": [0.0, 0.0, 0.0]})
    if mat is not None:
        mE, mnu, mrho = mat
        s.set_material_from_torch(E=mE, nu=mnu, density=mrho, device=DEV)
    s.finalize_mu_lam()
    s.set_parameters_dict({"g": ((-up).detach().cpu().numpy() * 9.8).tolist()})
    s.add_surface_collider(tuple(floor_pt_local.detach().cpu().numpy().tolist()),
                           tuple(up.detach().cpu().numpy().tolist()), "slip", 0.25)
    return s


def main(ply, outdir, frames=210, flag_path=None, view_az=105, substeps=30, dt=3e-4):
    # dt has to shrink with the grid. CFL needs dt < dx/sqrt(E/rho); at dx=0.04 that is
    # 1.03e-3 and dt=3e-4 has 3.4x of margin, but refining to dx=0.02 halves the limit to
    # 5.2e-4 and the margin drops to 1.7x -- jelly at E=1.2e6 then diverges into NaN
    # positions and Warp reports an illegal memory access. Substeps rise to match so the
    # simulated time per frame is unchanged.
    substeps = int(os.environ.get("SUBSTEPS", substeps))
    dt = float(os.environ.get("DT", dt))
    os.makedirs(outdir, exist_ok=True)
    (mat, bc, tp, pre, cam_p) = decode_param_json("config/orange_physics_g100.json")

    g = GaussianModel(0); g.load_ply_zero_sh(ply)
    pipeline = P()
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=DEV)
    rot_m = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                       pre["rotation_axis"])
    vc_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda()
    up = up / up.norm()

    par = load_params_from_gs(g, pipeline)
    pos0, cov0 = par["pos"], par["cov3D_precomp"]
    sp, op, shs = par["screen_points"], par["opacity"], par["shs"]
    tpos, so, om = transform2origin(pos0); tpos = shift2center111(tpos)
    cov0 = apply_cov_rotations(cov0, rot_m); cov0 = so * so * cov0
    mpos = tpos.to(DEV).detach()
    cov = cov0 / (so * so); cov = apply_inverse_cov_rotations(cov, rot_m)

    lat_mask = torch.zeros(mpos.shape[0], dtype=torch.bool, device=DEV)
    fl = torch.load(flag_path).to(DEV)
    nn_ = min(mpos.shape[0], fl.shape[0]); lat_mask[:nn_] = fl[:nn_]
    lat0 = mpos[lat_mask].contiguous()                    # rest pose of the lattice
    dx, mn, dims, occ, centres, cell_idx = build_grid(lat0)
    cell_vol = dx ** 3
    print(f"lattice {lat0.shape[0]} particles, grid {tuple(dims.tolist())}, "
          f"dx={dx:.6f}, cell volume={cell_vol:.3e}")

    # Material per cell. MATERIAL names a preset; the field itself is read off the lattice --
    # which level a cell is on, and what colour it decoded to. Unset, every cell gets the one
    # material `make_solver` has always used, and the run is exactly as before.
    global MAT
    MAT = None
    _preset = os.environ.get("MATERIAL", "")
    if _preset:
        from report.material_field import material_field, summarise, _rgb
        _lvl = None
        _lvl_path = os.environ.get("CELL_LEVEL", "")
        if _lvl_path and os.path.exists(_lvl_path):
            _v = torch.load(_lvl_path).reshape(-1).to(DEV).float()
            if _v.shape[0] >= lat_mask.shape[0]:
                _lvl = _v[:lat_mask.shape[0]][lat_mask]
            else:
                print(f"  cell_level has {_v.shape[0]} entries for "
                      f"{lat_mask.shape[0]} primitives; deriving the shell instead")
        if _lvl is None:
            # Without a level file, the shell is the boundary of the occupied set: a cell
            # whose 3x3x3 neighbourhood is not full. That is the same statement the two-level
            # lattice makes, recovered from the grid rather than read from a file.
            import torch.nn.functional as _Fn
            _cnt = _Fn.avg_pool3d(occ.float()[None, None], 3, 1, 1)[0, 0] * 27.0
            _lvl = (_cnt[cell_idx[:, 0], cell_idx[:, 1], cell_idx[:, 2]] < 26.5).float()
        _col = _rgb(g)[lat_mask]
        _lbl = os.environ.get("MATERIAL_LABELS", "")
        if _preset == "classes" and _lbl and os.path.exists(_lbl):
            from report.material_field import material_from_labels
            _cat = os.environ.get("CATEGORY", "fruit")
            print(f"  material from {_lbl}, category '{_cat}':")
            E, nu, rho = material_from_labels(torch.load(_lbl).to(DEV), _col, _lvl, _cat)
        else:
            E, nu, rho = material_field(_col, _lvl, preset=_preset)
        MAT = (E, nu, rho)
        globals()['MAT_LVL'] = _lvl
        print(f"  material field '{_preset}':")
        summarise(E, rho, _lvl)

    # Skinning is computed once, in the rest pose, over the whole lattice. A cut later
    # only masks out the neighbours that ended up in a different piece.
    with torch.no_grad():
        knn_i = torch.empty(mpos.shape[0], K_SKIN, dtype=torch.long, device=DEV)
        knn_d = torch.empty(mpos.shape[0], K_SKIN, device=DEV)
        for s0 in range(0, mpos.shape[0], 4000):
            dm = torch.cdist(mpos[s0:s0 + 4000], lat0)
            dk, ik = torch.topk(dm, K_SKIN, dim=1, largest=False)
            knn_i[s0:s0 + 4000] = ik; knn_d[s0:s0 + 4000] = dk
            del dm, dk, ik
    SIGMA = 0.05

    vc, oc = get_center_view_worldspace_and_observant_coordinate(vc_c, up, rot_m, so, om)
    cam, _ = get_camera_view("config/orange_demo", default_camera_index=-1,
        center_view_world_space=vc, observant_coordinates=oc, show_hint=False,
        init_azimuthm=view_az,
        init_elevation=float(os.environ.get("VIEW_EL", "8")),
        init_radius=cam_p["init_radius"] * float(os.environ.get("VIEW_R", "2.5")),
        move_camera=False, current_frame=0, delta_a=None, delta_e=None, delta_r=None)

    floor_world = lat0.mean(0) - up * 0.55
    a1 = torch.tensor([1.0, 0.0, 0.0], device=DEV); a1 = a1 - (a1 @ up) * up
    a1 = a1 / a1.norm()

    def plane(az):
        th = torch.tensor(float(az) * np.pi / 180.0, device=DEV)
        n = torch.cos(th) * a1 + torch.sin(th) * torch.cross(up, a1, dim=0)
        return n / n.norm()

    # Cut schedule from the environment: "frame:azimuth:offset,..." where offset shifts
    # the plane along its own normal in units of the object radius, so 0 is through the
    # centre and 0.5 shaves off a cap. Two entries at the same frame with a small offset
    # difference carve a slice out of the middle.
    radius = float((lat0 - lat0.mean(0)).norm(dim=1).quantile(0.98))
    spec = os.environ.get("CUT_SPEC", "20:60:0,95:150:0")
    CUTS = []
    for part in spec.split(","):
        fr, az, offs = part.split(":")
        # `az` turns the normal within the horizontal plane, which cuts the object like a
        # knife held upright and cannot produce a slice. Writing the azimuth as "h" takes the
        # normal along the object's own vertical instead, so the plane is horizontal and a pair
        # of them at different offsets carves out a disc -- which is how fruit is actually cut,
        # and what shows the interior of every piece at once as they fall apart.
        n = up / up.norm() if az.strip().lower() == "h" else plane(float(az))
        CUTS.append((int(fr), n, lat0.mean(0) + n * float(offs) * radius))
    print("cut schedule:", [(c[0], round(float(c[1][0]), 2), round(float((c[2]-lat0.mean(0)).norm()/radius), 2))
                            for c in CUTS])

    planes, signs = [], []
    bodies = []          # each: dict(solver, pidx, shift, rest, floor_local)
    lat_x = lat0.clone()                       # current world pose of every particle
    lat_v = torch.zeros_like(lat0)
    assign = torch.zeros(lat0.shape[0], dtype=torch.long, device=DEV)

    def rebuild():
        """Relabel the lattice and rebuild one solver per piece, carrying state over."""
        nonlocal bodies, assign
        t0 = time.time()
        lab, iters = label_components(occ, signs)
        pid, npieces = piece_of_particles(lab, cell_idx)
        torch.cuda.synchronize()
        t_lab = time.time() - t0
        new = []
        for b in range(npieces):
            sel = torch.where(pid == b)[0]
            if sel.numel() < 2000:
                continue
            # Size the domain to the piece instead of using one global box. Every solver
            # was given grid_lim=3.2 while the fruit spans ~1.0, so only ~25 of the 80
            # cells crossed the object and a thin wedge was a handful of cells thick --
            # measured edge wiggle 1.019 for two pieces against 1.066 for six.
            xw = lat_x[sel]
            ext = float((xw.max(0).values - xw.min(0).values).max())
            dfl = max(float(((xw - floor_world) @ up).min()), 0.02)
            L = max((dfl + ext) / 0.75, ext / 0.8) * 1.15
            if os.environ.get('ADAPT_L', '1') != '1':
                L = GRID_LIM
            # Centre the piece and put the floor under it, rather than sliding the piece
            # to sit above a fixed floor: `up` is not a positive axis here, so treating
            # x.up as a height pushed the whole body to [-1.493, 2.003], outside the
            # [0, L] domain, and MPM silently did nothing for every frame.
            shift = torch.tensor([L / 2] * 3, device=DEV) - xw.mean(0)
            xl = xw + shift
            ctr = torch.tensor([L / 2] * 3, device=DEV)
            floor_local = ctr + up * (float(((xl - ctr) @ up).min()) - dfl)
            # Fix the cell size, not the cell count: with n_grid fixed, a smaller piece
            # gets a finer grid and its own CFL limit, so no single dt is stable for all
            # pieces at once (the run reached frame 85 before the thinner quarters
            # diverged). Deriving n_grid from a target dx keeps every solver on the same
            # dt.
            ng = int(max(24, min(160, round(L / DX_TARGET))))
            s = make_solver(xl, up, cell_vol, floor_local, L, n_grid=ng,
                            mat=None if MAT is None
                            else tuple(m[sel].contiguous() for m in MAT))
            s.import_particle_v_from_torch(lat_v[sel].contiguous(), device=DEV)
            _chk = s.export_particle_x_to_torch().to(DEV)
            print(f"    body {len(new)}: L={L:.3f} 粒子範圍 "
                  f"[{float(_chk.min()):.3f},{float(_chk.max()):.3f}] "
                  f"{'OK' if float(_chk.min())>0 and float(_chk.max())<L else '★超出域'}")
            new.append({"solver": s, "pidx": sel, "shift": shift,
                        "rest": lat0[sel].clone(), "L": L})
        # Push the new pieces apart along the cut normal, so a stack of slices opens into a fan
        # instead of settling into a pile with its cut faces hidden. The impulse is a
        # presentation choice and is off unless asked for; it is applied once, at the cut, and
        # the solver has it from there.
        _spread = float(os.environ.get("SPREAD", "0"))
        if _spread > 0 and len(new) > len(bodies):
            _ctr = torch.stack([lat_x[b["pidx"]].mean(0) for b in new])
            _mid = _ctr.mean(0)
            for _b, _bd in enumerate(new):
                _d = _ctr[_b] - _mid
                _n = _d.norm().clamp_min(1e-6)
                _v = _bd["solver"].export_particle_v_to_torch().to(DEV)
                _bd["solver"].import_particle_v_from_torch(
                    (_v + (_d / _n) * _spread).contiguous(), device=DEV)
        bodies = new
        assign = torch.full_like(assign, -1)
        for b, bd in enumerate(bodies):
            assign[bd["pidx"]] = b
        # Freshly separated pieces still share their cut face, so every pair starts with
        # contact disabled: two halves that were one solid a frame ago are touching by
        # construction, and treating that as a collision injects separation every step.
        contact_on.clear()
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                contact_on[(i, j)] = False
        print(f"  -> {len(bodies)} pieces "
              f"({[int(b['pidx'].numel()) for b in bodies]} particles), "
              f"labelling {t_lab*1000:.1f} ms")
        return t_lab

    def bind():
        """Skinning weights and per-Gaussian piece, given the current decomposition.

        A Gaussian's side of the blade is decided by its own position, not by whichever
        lattice particle happens to be nearest. Nearest-particle assignment let shell
        Gaussians next to the cut land on either side depending on sub-cell geometry,
        which tore the cut face into a ragged edge. Scale is clamped to exp(-16) in this
        pipeline, so these primitives are points and the plane test is exact for them --
        the cut face is then planar for the shell and stepped by one cell for the lattice.

        Candidate neighbours are restricted to lattice particles with the same sign
        vector; the connected component is still what defines the piece, so the nearest
        surviving candidate supplies it.
        """
        cand = torch.ones_like(knn_i, dtype=torch.bool)
        for (n, q) in planes:
            sg_g = ((mpos - q) @ n) >= 0
            sg_l = ((lat0 - q) @ n) >= 0
            cand &= (sg_l[knn_i] == sg_g.unsqueeze(1))
        # nearest candidate decides the piece; fall back to the plain nearest if a
        # Gaussian has no neighbour on its own side within K
        big = torch.where(cand, knn_d, torch.full_like(knn_d, 1e9))
        first = big.argmin(1)
        near = knn_i.gather(1, first.unsqueeze(1)).squeeze(1)
        has = cand.any(1)
        near = torch.where(has, near, knn_i[:, 0])
        gb = assign[near]
        same = (assign[knn_i] == gb.unsqueeze(1)) & (cand | ~has.unsqueeze(1))
        w = torch.exp(-(knn_d / SIGMA) ** 2) * same.float()
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        base = (lat0[knn_i] * w.unsqueeze(-1)).sum(1)
        return gb, w, mpos - base

    contact_on = {}

    rebuild_t = rebuild()
    gb, w_skin, off_skin = bind()

    trace, step = [], 0
    for f in range(frames):
        for (cf, n, q) in CUTS:
            if f == cf:
                sg = ((centres - q) @ n) >= 0
                planes.append((n, q)); signs.append(sg)
                print(f"frame {f}: cut plane added")
                rebuild()
                gb, w_skin, off_skin = bind()
                # the blade parts the pieces: push each one off the plane it now borders
                for bd in bodies:
                    side = torch.sign((lat_x[bd["pidx"]].mean(0) - q) @ n)
                    v = bd["solver"].export_particle_v_to_torch().to(DEV)
                    bd["solver"].import_particle_v_from_torch(
                        v + n * side * 0.45, device=DEV)

        for _ in range(substeps):
            step += 1
            for bd in bodies:
                bd["solver"].p2g2p(step, dt, device=DEV)

        # current world pose/velocity of every lattice particle, and domain-following
        for bd in bodies:
            xl = bd["solver"].export_particle_x_to_torch().to(DEV)
            corr = torch.tensor([bd["L"] / 2] * 3, device=DEV) - xl.mean(0)
            corr = corr - (corr @ up) * up
            if corr.norm() > 1e-4:
                bd["solver"].import_particle_x_from_torch(xl + corr, device=DEV)
                bd["shift"] = bd["shift"] + corr
                xl = xl + corr
            lat_x[bd["pidx"]] = xl - bd["shift"]
            lat_v[bd["pidx"]] = bd["solver"].export_particle_v_to_torch().to(DEV)

        # skin: blend within the piece, then rotate the offset with that piece
        skinned = (lat_x[knn_i] * w_skin.unsqueeze(-1)).sum(1)
        for b, bd in enumerate(bodies):
            cur, rest = lat_x[bd["pidx"]], bd["rest"]
            a = rest - rest.mean(0); bb = cur - cur.mean(0)
            u_, _, vt = torch.linalg.svd(a.T @ bb)
            d_ = torch.sign(torch.det(vt.T @ u_.T))
            R = vt.T @ torch.diag(torch.tensor([1., 1., float(d_)], device=DEV)) @ u_.T
            sel = gb == b
            skinned[sel] = skinned[sel] + off_skin[sel] @ R.T

        # pairwise contact between pieces, on the rendered surface
        CELL = 0.045
        nhit_tot = 0
        deep_tot, deep_max = 0, 0.0
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                pi, pj = skinned[gb == i], skinned[gb == j]
                if pi.numel() == 0 or pj.numel() == 0:
                    continue
                bmin = pj.min(0).values - CELL
                bdim = torch.ceil((pj.max(0).values + CELL - bmin) / CELL).long() + 1
                o = torch.zeros(bdim.tolist(), dtype=torch.bool, device=DEV)
                ij = ((pj - bmin) / CELL).long().clamp(
                    torch.zeros(3, dtype=torch.long, device=DEV), bdim - 1)
                o[ij[:, 0], ij[:, 1], ij[:, 2]] = True
                ii = ((pi - bmin) / CELL).long()
                inb = ((ii >= 0) & (ii < bdim)).all(1)
                hit = torch.zeros(pi.shape[0], dtype=torch.bool, device=DEV)
                if inb.any():
                    jj = ii[inb]; hit[inb] = o[jj[:, 0], jj[:, 1], jj[:, 2]]
                nh = int(hit.sum()); nhit_tot += nh
                if nh == 0:
                    contact_on[(i, j)] = True
                if nh > 0 and contact_on.get((i, j), False):
                    # A slice that has been cut off is still one object, and contact must not be
                    # what breaks it. Displacing each penetrating particle individually does
                    # resolve the overlap, but it moves the particles at the contact face and
                    # not the ones behind them, so the piece is thinned exactly where it lands:
                    # the correction and the material model fight each other and the slice comes
                    # apart. It also throws the deepest particles far enough to leave the solver
                    # grid, which is where the run died.
                    #
                    # So the query stays per particle and the response is rigid. The occupied
                    # set is a union of axis-aligned cells, so "how far into you am I, and which
                    # way is out" has an exact answer for every particle -- the vector to the
                    # nearest free cell, all of them from one distance transform of the free
                    # space. Those vectors are then reduced to one translation of the whole
                    # piece, deep enough that the last particle clears, and applied to every
                    # particle equally. A uniform translation adds no strain, so the piece is
                    # the same piece afterwards.
                    #
                    # This is the half of contact a shared MPM background grid would have given
                    # us for free, and which one solver per piece costs; the lattice hands it
                    # back because its occupancy is already a grid.
                    from scipy import ndimage as _nd
                    xa = bodies[i]["solver"].export_particle_x_to_torch().to(DEV)
                    xb = bodies[j]["solver"].export_particle_x_to_torch().to(DEV)
                    ma, mb = xa.shape[0], xb.shape[0]
                    wi, wj = mb / (ma + mb), ma / (ma + mb)
                    nrm, moved = None, False
                    # One query, one rigid translation, repeated: a single translation clears the
                    # overlap it measured, and the second pass measures whatever a piece slid
                    # into on the way. Three is enough here -- the third pass finds nothing left
                    # on every frame of the runs below.
                    for _it in range(CONTACT_ITERS):
                        wa = xa - bodies[i]["shift"]
                        wb = xb - bodies[j]["shift"]
                        bmin = wb.min(0).values - CELL * 2
                        bdim = (torch.ceil((wb.max(0).values + CELL * 2 - bmin) / CELL).long()
                                + 1)
                        if int(bdim.prod()) >= 60_000_000:
                            break
                        ob = torch.zeros(bdim.tolist(), dtype=torch.bool, device=DEV)
                        jb = ((wb - bmin) / CELL).long().clamp(
                            torch.zeros(3, dtype=torch.long, device=DEV), bdim - 1)
                        ob[jb[:, 0], jb[:, 1], jb[:, 2]] = True
                        idx_a = ((wa - bmin) / CELL).long()
                        inb2 = ((idx_a >= 0) & (idx_a < bdim)).all(1)
                        pen = torch.zeros(wa.shape[0], dtype=torch.bool, device=DEV)
                        if inb2.any():
                            q = idx_a[inb2]
                            pen[inb2] = ob[q[:, 0], q[:, 1], q[:, 2]]
                        if not bool(pen.any()):
                            break
                        # nearest free cell for every occupied cell, in one pass
                        _, ind = _nd.distance_transform_edt(ob.cpu().numpy(),
                                                            return_indices=True)
                        ind = torch.from_numpy(np.stack(ind)).to(DEV)   # (3, X, Y, Z)
                        pa = idx_a[pen]
                        tgt = torch.stack([ind[d][pa[:, 0], pa[:, 1], pa[:, 2]]
                                           for d in range(3)], 1).float()
                        out = (tgt - pa.float()) * CELL                 # the way out, in world
                        n_out = out.norm(dim=1, keepdim=True).clamp_min(1e-9)
                        # Depth, not count, is the quantity contact can drive to zero. Two pieces
                        # resting on each other touch, so particles within a cell of the other
                        # body are contact and not a defect; what must go away is anything
                        # *further in* than that. Recorded before the first correction of the
                        # frame, so it is the state the step produced.
                        if _it == 0:
                            dpt = n_out.squeeze(1)
                            deep_tot += int((dpt > CELL).sum())
                            deep_max = max(deep_max, float(dpt.max()) / CELL)
                        # The direction the pair separates in is the mean of the individual exit
                        # directions; the distance is how deep the deepest particle is along it,
                        # at the 99th percentile so one stray particle inside a far part of the
                        # other piece cannot set the step for the whole face.
                        nrm = (out / n_out).mean(0)
                        if nrm.norm() < 1e-8:
                            break
                        nrm = nrm / nrm.norm()
                        depth = (out * nrm).sum(1)
                        d = float(torch.quantile(depth, 0.99)) + CELL
                        d = max(0.0, min(d, CELL * MAX_SEP_CELLS)) * CONTACT_PUSH
                        if d <= 0.0:
                            break
                        delta = nrm * d
                        xa = xa + delta * wi
                        xb = xb - delta * wj
                        moved = True
                    if moved:
                        bodies[i]["solver"].import_particle_x_from_torch(
                            xa.contiguous(), device=DEV)
                        bodies[j]["solver"].import_particle_x_from_torch(
                            xb.contiguous(), device=DEV)
                        # and the same treatment for velocity, or the next step undoes the
                        # translation: take the approaching part of the relative motion out along
                        # the separation direction. Uniform over each body, so the pieces stay
                        # rigid under the correction and only the material model deforms them.
                        va = bodies[i]["solver"].export_particle_v_to_torch().to(DEV)
                        vb = bodies[j]["solver"].export_particle_v_to_torch().to(DEV)
                        vrel = float((va.mean(0) - vb.mean(0)) @ nrm)
                        if vrel < 0:
                            jm = -(1.0 + CONTACT_RESTITUTION) * vrel
                            bodies[i]["solver"].import_particle_v_from_torch(
                                (va + nrm * (jm * wi)).contiguous(), device=DEV)
                            bodies[j]["solver"].import_particle_v_from_torch(
                                (vb - nrm * (jm * wj)).contiguous(), device=DEV)

        world = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(skinned), so, om), rot_m)
        rast = initialize_resterize(cam, g, pipeline, bg, image_height=512, image_width=512)
        col = convert_SH(shs, cam, g, world, None)
        img, _, _, _ = rast(means3D=world, means2D=sp, shs=None, colors_precomp=col,
                            opacities=op, scales=None, rotations=None, cov3D_precomp=cov)
        arr = img.permute(1, 2, 0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(outdir, f"d_{f:03d}.png"),
                    cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) * 255)
        # Squash, per level. A stiffness contrast between shell and interior is invisible in a
        # trajectory -- both fall at g -- and shows only in how much each deforms. Measure the
        # extent along the gravity axis, as a fraction of what it was at rest, separately for
        # the cells the material field called skin and the ones it called interior. A hard
        # shell around a soft interior keeps its own extent and lets the interior lose more.
        _sq = []
        if MAT_LVL is not None:
            for bd in bodies:
                _sel = bd["pidx"]
                _x = lat_x[_sel]
                _l = MAT_LVL[_sel] > 0.5
                for _m in (_l, ~_l):
                    if int(_m.sum()) < 64:
                        _sq.append(float("nan")); continue
                    _now = float((_x[_m] @ up).max() - (_x[_m] @ up).min())
                    _rest = float((bd["rest"][_m] @ up).max() - (bd["rest"][_m] @ up).min())
                    _sq.append(_now / max(_rest, 1e-9))
        trace.append([f, len(bodies), nhit_tot, _sq, deep_tot, round(deep_max, 3)])
        if f % 20 == 0:
            print(f"  frame {f}/{frames}  pieces {len(bodies)}  contacts {nhit_tot}")

    json.dump(trace, open(os.path.join(outdir, "trace.json"), "w"))
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         frames=int(sys.argv[3]) if len(sys.argv) > 3 else 210,
         flag_path=sys.argv[4] if len(sys.argv) > 4 else None)
