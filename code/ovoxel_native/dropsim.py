"""The orange falls, is cut twice in flight, and the pieces land -- solved, not prescribed.

This is `dynamic_cut.py`'s experiment on the O-Voxel representation, and the parts that file spends
most of its length on are the parts that disappear here. It infers the lattice from Gaussian
positions, dilates the occupancy by two cells because internal filling leaves an occupied cell with
only 18.5 of its 26 neighbours and plain connectivity shatters the intact fruit into 3141
fragments, and skins every Gaussian to sixteen lattice particles because a Gaussian has no
adjacency of its own. The cells here are the particles: connectivity is adjacency on the integer
lattice, exact and parameter-free, and the surface is a function of the same cells rather than a
separate cloud that has to be attached to them.

What is kept is the physics and the discipline around it. One MPM solver per discovered piece,
carrying its particles' current positions and velocities; a cut is an event that relabels the
lattice and rebuilds the solvers; the material is the two-rule field, a stiff peel over a softer
interior. The renderer needs no per-frame geometry from this: each piece is summarised by the best
rigid transform of its own particles, and the non-rigid residual that summary discards is measured
and reported rather than assumed small.
"""
import os, sys, time, json
sys.path.insert(0, "/workspace/rebuild/project3/code/inherited")
import numpy as np, torch
import warp as wp
wp.init()
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from scipy import ndimage

DEV = "cuda:0"
W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
FRAMES = int(os.environ.get("FRAMES", "120"))
SUBSTEPS = int(os.environ.get("SUBSTEPS", "180"))
DT = float(os.environ.get("DT", "5e-5"))
NPART = int(os.environ.get("NPART", "120000"))
DX_TARGET = float(os.environ.get("DX_TARGET", "0.04"))
E_INT = float(os.environ.get("E_INT", "1.2e6"))
E_PEEL = float(os.environ.get("E_PEEL", "8.4e6"))
CUT_FRAMES = [int(x) for x in os.environ.get("CUT_FRAMES", "15,38").split(",")]
DROP = float(os.environ.get("DROP", "1.1"))          # floor, in world units below the object
SPREAD = float(os.environ.get("SPREAD", "0.35"))     # the blade parts what it separates

P = np.load(f"{W}/drop_prep_{OBJ}.npz")
hc = float(P["hc"])
solid = torch.from_numpy(P["solid"]).long().to(DEV)
org = torch.from_numpy(P["org"]).float().to(DEV)
cen = (solid.float() + 0.5) * hc + org
peel_all = torch.from_numpy(P["peel"]).bool().to(DEV)
n1 = torch.from_numpy(P["n1"]).float().to(DEV)
n2 = torch.from_numpy(P["n2"]).float().to(DEV)
d1, d2 = float(P["d1"]), float(P["d2"])
mid = torch.from_numpy(P["mid"]).float().to(DEV)
up = n1 / n1.norm()          # the object's own vertical: the transverse plane's normal

# One particle per cell is 770k for a grid that puts 35 cells across the object; the solver wants
# about eight particles per grid cell, so the lattice is thinned by a fixed stride rather than at
# random, which keeps the sampling uniform and keeps the peel's share of it unchanged.
stride = max(1, len(cen) // NPART)
sel0 = torch.arange(0, len(cen), stride, device=DEV)[:NPART]
x0 = cen[sel0].contiguous()
peel = peel_all[sel0]
cell_vol = (hc ** 3) * stride
print(f"{OBJ}: {len(cen):,} cells -> {len(x0):,} particles (stride {stride}), "
      f"peel {int(peel.sum()):,} ({float(peel.float().mean())*100:.1f}%, "
      f"was {float(peel_all.float().mean())*100:.1f}% of the lattice)")

E = torch.where(peel, torch.tensor(E_PEEL, device=DEV), torch.tensor(E_INT, device=DEV))
nu = torch.full_like(E, 0.35)
rho = torch.where(peel, torch.tensor(900.0, device=DEV), torch.tensor(800.0, device=DEV))
print(f"  material: peel E={E_PEEL:.2g} Pa over interior E={E_INT:.2g} Pa, "
      f"a contrast of {E_PEEL/E_INT:.1f}")

# ------------------------------------------------------------------ lattice connectivity
dims = [int(solid[:, i].max()) + 1 for i in range(3)]
occ_np = np.zeros(dims, dtype=bool)
sc = solid.cpu().numpy()
occ_np[sc[:, 0], sc[:, 1], sc[:, 2]] = True
sub = sc[sel0.cpu().numpy()]


def label_components(signs):
    """Connected components, with each cut plane severing adjacency.

    A component cannot span a plane it lies on both sides of, so P planes are exactly P
    independent labellings of the sign regions -- no graph edit, and no dilation, because this
    occupancy has no holes to close.
    """
    out = np.full(occ_np.shape, -1, dtype=np.int64)
    nxt = 0
    for combo in range(1 << len(signs)):
        region = np.ones_like(occ_np)
        for b, s in enumerate(signs):
            region &= s if (combo >> b) & 1 else ~s
        m = occ_np & region
        if not m.any():
            continue
        lab, k = ndimage.label(m)
        out[m] = lab[m] + nxt
        nxt += k
    return out


def pieces_of(signs, min_size=2000):
    lab = label_components(signs)
    raw = lab[sub[:, 0], sub[:, 1], sub[:, 2]]
    uniq, counts = np.unique(raw, return_counts=True)
    keep = [int(u) for u, c in zip(uniq, counts) if u >= 0 and c >= min_size]
    out = np.full(len(raw), -1, dtype=np.int64)
    for new, u in enumerate(keep):
        out[raw == u] = new
    return torch.from_numpy(out).to(DEV), len(keep)


# ------------------------------------------------------------------ solvers
floor_world = mid - up * DROP


def make_solver(xw, vw, mE, mnu, mrho):
    ext = float((xw.max(0).values - xw.min(0).values).max())
    dfl = max(float(((xw - floor_world) @ up).min()), 0.02)
    # The domain is sized to the piece and the floor under it, not to one global box: a quarter
    # given the whole-object domain gets a handful of grid cells across it and turns to mush.
    L = max((dfl + ext) / 0.75, ext / 0.8) * 1.15
    shift = torch.tensor([L / 2] * 3, device=DEV) - xw.mean(0)
    xl = xw + shift
    ctr = torch.tensor([L / 2] * 3, device=DEV)
    floor_local = ctr + up * (float(((xl - ctr) @ up).min()) - dfl)
    # Fix the cell size, not the cell count: with n_grid fixed a smaller piece gets a finer grid
    # and its own CFL limit, and no single dt is stable for all of them at once.
    ng = int(max(24, min(160, round(L / DX_TARGET))))
    s = MPM_Simulator_WARP(10)
    s.load_initial_data_from_torch(xl, torch.full((len(xl),), cell_vol, device=DEV),
                                   n_grid=ng, grid_lim=L, device=DEV)
    s.set_parameters_dict({"E": E_INT, "nu": 0.35, "material": "jelly",
                           "density": 800.0, "g": [0.0, 0.0, 0.0]})
    s.set_material_from_torch(E=mE, nu=mnu, density=mrho, device=DEV)
    s.finalize_mu_lam()
    s.set_parameters_dict({"g": ((-up).cpu().numpy() * 9.8).tolist()})
    s.add_surface_collider(tuple(floor_local.cpu().numpy().tolist()),
                           tuple(up.cpu().numpy().tolist()), "slip", 0.25)
    s.import_particle_v_from_torch(vw.contiguous(), device=DEV)
    return dict(solver=s, shift=shift, L=L, ng=ng)


cur_x = x0.clone()
cur_v = torch.zeros_like(x0)
signs, bodies, assign = [], [], None
stages = []


def rebuild(frame):
    global bodies, assign
    t0 = time.time()
    pid, npc = pieces_of(signs)
    lab_ms = (time.time() - t0) * 1000
    new = []
    for b in range(npc):
        idx = torch.where(pid == b)[0]
        bd = make_solver(cur_x[idx], cur_v[idx], E[idx].contiguous(),
                         nu[idx].contiguous(), rho[idx].contiguous())
        bd["idx"] = idx
        bd["rest"] = x0[idx].clone()
        new.append(bd)
    bodies = new
    assign = pid
    stages.append((frame, pid.cpu().numpy().astype(np.int16)))
    print(f"  frame {frame}: {npc} pieces "
          f"{[int(b['idx'].numel()) for b in bodies]}, labelling {lab_ms:.1f} ms, "
          f"grids {[b['ng'] for b in bodies]}")


rebuild(0)
CUTS = [(CUT_FRAMES[0], "n1", n1, d1), (CUT_FRAMES[1], "n2", n2, d2)]
# cell centres of the whole lattice box, which is where a plane's sign is read
_g = np.stack(np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]], -1).astype(np.float32)
_pc = (_g + 0.5) * hc + P["org"].reshape(1, 1, 1, 3)

Rs = np.zeros((FRAMES, 8, 3, 3), np.float32)
Ts = np.zeros((FRAMES, 8, 3), np.float32)
NV = np.zeros(FRAMES, np.int32)
resid = []
step = 0
t_start = time.time()

for f in range(FRAMES):
    for (cf, nm, nn, dd) in CUTS:
        if f == cf:
            # The cut is stated in the material frame, so a plane aimed at the object still cuts
            # the object it was aimed at after the object has fallen half a diameter.
            signs.append((_pc @ P[nm] + dd) > 0)
            print(f"frame {f}: a plane arrives")
            rebuild(f)
            for bd in bodies:                    # the blade parts what it separates
                side = torch.sign((cur_x[bd["idx"]].mean(0) @ nn + dd))
                v = bd["solver"].export_particle_v_to_torch().to(DEV).detach()
                bd["solver"].import_particle_v_from_torch(
                    (v + nn * side * SPREAD).contiguous(), device=DEV)

    for _ in range(SUBSTEPS):
        step += 1
        for bd in bodies:
            bd["solver"].p2g2p(step, DT, device=DEV)

    for bd in bodies:
        xl = bd["solver"].export_particle_x_to_torch().to(DEV).detach()
        # The domain follows the piece sideways but not downwards: a piece that drifts out of its
        # own box is what the solver reports as an illegal access, and the fall is the one motion
        # the box was sized for, so only the two directions perpendicular to gravity are corrected.
        ctr = torch.tensor([bd["L"] / 2] * 3, device=DEV)
        corr = ctr - xl.mean(0)
        corr = corr - (corr @ up) * up
        if float(corr.norm()) > 1e-4:
            bd["solver"].import_particle_x_from_torch((xl + corr).contiguous(), device=DEV)
            bd["shift"] = bd["shift"] + corr
            xl = xl + corr
        lo, hi = float(xl.min()), float(xl.max())
        if lo < 0.02 * bd["L"] or hi > 0.98 * bd["L"]:
            print(f"    frame {f}: a piece reaches [{lo:.3f},{hi:.3f}] of its "
                  f"[0,{bd['L']:.3f}] domain")
        cur_x[bd["idx"]] = xl - bd["shift"]
        cur_v[bd["idx"]] = bd["solver"].export_particle_v_to_torch().to(DEV).detach()

    NV[f] = len(bodies)
    for b, bd in enumerate(bodies):
        rest, cur = bd["rest"], cur_x[bd["idx"]].detach()
        a = rest - rest.mean(0); c = cur - cur.mean(0)
        u_, _, vt = torch.linalg.svd(a.T @ c)
        dsign = torch.sign(torch.det(vt.T @ u_.T))
        R = vt.T @ torch.diag(torch.tensor([1., 1., float(dsign)], device=DEV)) @ u_.T
        Rs[f, b] = R.cpu().numpy()
        Ts[f, b] = (cur.mean(0) - rest.mean(0) @ R.T).cpu().numpy()
        # what the rigid summary throws away, in coarse cells
        resid.append(float(((a @ R.T - c).norm(dim=1).mean()) / hc))
    if f % 10 == 0 or f == FRAMES - 1:
        h = float(((cur_x.mean(0) - floor_world) @ up))
        print(f"  frame {f:3d}  {len(bodies)} bodies  height {h/hc:6.1f} cells  "
              f"residual {resid[-1]:.3f} cells  {time.time()-t_start:.0f}s")

np.savez(f"{W}/drop_traj_{OBJ}.npz", R=Rs, T=Ts, nv=NV, x0=x0.cpu().numpy(),
         sub=sub.astype(np.int32), sel0=sel0.cpu().numpy().astype(np.int32),
         stage_frames=np.array([s[0] for s in stages], np.int32),
         stage_assign=np.stack([s[1] for s in stages]),
         floor=floor_world.cpu().numpy(), up=up.cpu().numpy(),
         resid=np.array(resid, np.float32))
print(f"drop_traj_{OBJ}.npz: {FRAMES} frames, {len(stages)} stages, "
      f"non-rigid residual {min(resid):.3f} to {max(resid):.3f} coarse cells, "
      f"{time.time()-t_start:.0f}s")
