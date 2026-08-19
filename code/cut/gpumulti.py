"""Several planes at once, on the device, and what a second and a fifth cut cost.

`gpucut` and `gpumesh` do one plane. The specification's multi-cut is the same operator with a
vector where it had a scalar: a leaf is refined while *some* plane still crosses it, it carries
the Q signs of the planes at its centre as a side code, and adjacency survives only between
leaves whose codes agree. Connected components are unchanged. Nothing here is new geometry --
it is the host `multicut.cut` with each pass moved to the device, plus `gpumesh`'s polygons run
once per plane over the leaves that plane crosses.

    python gpumulti.py LATTICE      # agreement with the host, then one, three and five cuts

Every timing is a median of repeats after warm-ups, for the reason `gpumesh` gives.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import statistics
import sys
import time

import numpy as np
import torch

sys.path += [_FN_ROOT]

try:
    from method.common.cube import multicut as mc
    from method.common.cube import cutmesh as cm
    from method.common.cube import gpucut as gc
    from method.common.cube import gpumesh as gm
except ImportError:
    import multicut as mc
    import cutmesh as cm
    import gpucut as gc
    import gpumesh as gm

DEV = gc.DEV
_OFF = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.int64)


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def refine_many(coords, h, planes, h_target, levels_max=8):
    """Kernels A and B, with A reading `any plane crosses this cell` instead of one plane."""
    dev = torch.device(DEV)
    cur = torch.as_tensor(np.asarray(coords), dtype=torch.int64, device=dev)
    par = torch.arange(len(cur), dtype=torch.int64, device=dev)
    off = _OFF.to(dev)
    ns = [torch.as_tensor(np.asarray(n, np.float64), dtype=torch.float64, device=dev)
          for n, _ in planes]
    ds = [float(d) for _, d in planes]

    leaf, lvl, pars = [], [], []
    hl = float(h)
    for L in range(1, levels_max + 1):
        if hl <= h_target:
            break
        m = torch.zeros(len(cur), dtype=torch.bool, device=dev)
        for n, d in zip(ns, ds):
            m |= gc._crossed(cur, hl, n, d)
        if not bool(m.any()):
            break
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        keep = torch.nonzero(~m, as_tuple=False).squeeze(1)
        leaf.append(cur[keep])
        lvl.append(torch.full((len(keep),), L - 1, dtype=torch.int8, device=dev))
        pars.append(par[keep])
        cur = ((cur[idx] * 2)[:, None, :] + off[None]).reshape(-1, 3)
        par = par[idx].repeat_interleave(8)
        hl *= 0.5
    top = int(round(np.log2(float(h) / hl)))
    leaf.append(cur); pars.append(par)
    lvl.append(torch.full((len(cur),), top, dtype=torch.int8, device=dev))
    return torch.cat(leaf), torch.cat(lvl), torch.cat(pars), top


def side_codes(leaf, level, h, planes):
    """The Q signs at each leaf's centre, packed into one integer -- (20) with a vector."""
    dev = leaf.device
    hl = float(h) / (2.0 ** level.to(torch.float64))
    centre = (leaf.to(torch.float64) + 0.5) * hl[:, None]
    code = torch.zeros(len(leaf), dtype=torch.int64, device=dev)
    sgn = torch.zeros((len(leaf), len(planes)), dtype=torch.int8, device=dev)
    for q, (n, d) in enumerate(planes):
        nt = torch.as_tensor(np.asarray(n, np.float64), dtype=torch.float64, device=dev)
        s = torch.sign(centre @ nt + float(d))
        s = torch.where(s == 0, torch.ones_like(s), s)
        sgn[:, q] = s.to(torch.int8)
        code = code * 2 + (s > 0).to(torch.int64)
    return code, sgn, centre


def cut(coords, h, planes, h_target, levels_max=8):
    """Q planes at once: refine, code, join, label. Labelling stays on the host."""
    leaf, lvl, par, top = refine_many(coords, h, planes, h_target, levels_max)
    e = gc.adjacency(leaf, lvl, top)
    code, sgn, centre = side_codes(leaf, lvl, h, planes)
    e_all = e
    if len(e):
        e = e[code[e[:, 0]] == code[e[:, 1]]]
    piece, K = mc.sd.components(len(leaf), e.cpu().numpy())
    return dict(leaf=leaf, level=lvl, parent=par, top=top, code=code, side=sgn,
                edges=e, edges_all=e_all, piece=piece, K=K)


def cut_mesh(coords, h, planes, h_target, levels_max=8):
    """The whole operator: the cut, then one polygon pass per plane."""
    r = cut(coords, h, planes, h_target, levels_max)
    tris = 0
    for n, d in planes:
        nn, _, _ = cm.plane_basis(np.asarray(n, np.float64))
        verts, cnt, cell = gm.polygons(r["leaf"], r["level"], h, nn, d)
        tri, owner = gm.triangulate(verts, cnt)
        if len(tri):
            V, F = gm.merge(tri, tol=h * 1e-6)
            tris += len(F)
    r["tris"] = tris
    return r


def host_cut_mesh(coords, h, planes, h_target, levels_max=8):
    r = mc.cut(coords, h, list(planes), h_target, levels_max)
    tris = 0
    for n, d in planes:
        nn, _, _ = cm.plane_basis(np.asarray(n, np.float64))
        verts, cnt, cell = cm.polygons(r["leaf"], r["level"], h, nn, d)
        tri, owner = cm.triangulate(verts, cnt)
        if len(tri):
            V, F = cm.merge(tri, tol=h * 1e-6)
            tris += len(F)
    r["tris"] = tris
    return r


def _median(fn, warm=2, reps=5):
    for _ in range(warm):
        fn(); _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def make_planes(solid, hc, q, seed=7):
    """Q planes through the object, spread in direction and offset, deterministic."""
    rng = np.random.default_rng(seed)
    ctr = (solid + 0.5).mean(0) * hc
    out = []
    for i in range(q):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        off = (i - (q - 1) / 2.0) * 6.0 * hc
        out.append((n, float(-ctr @ n + off)))
    return out


def main(lattice_dir):
    from plyfile import PlyData
    try:
        from method.common.cube.occupancy import close_and_fill, to_grid
    except ImportError:
        from occupancy import close_and_fill, to_grid
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lv = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    org = xyz[lv == 0].min(0) - 0.5 * hc
    co = np.unique(np.floor((xyz[lv == 0] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(co).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + co.min(0) - 1
    print(f"  {lattice_dir}: {len(solid):,} solid cells, device {DEV}")
    print(f"  {'cuts':>5} {'leaves':>10} {'pieces':>7} {'triangles':>10} "
          f"{'host s':>8} {'device s':>9} {'speedup':>8}")
    rows = []
    for q in (1, 3, 5):
        planes = make_planes(solid, hc, q)
        a = host_cut_mesh(solid, hc, planes, hf)
        b = cut_mesh(solid, hc, planes, hf)
        agree = (a["K"] == b["K"] and len(a["leaf"]) == len(b["leaf"])
                 and a["tris"] == b["tris"])
        th = _median(lambda: host_cut_mesh(solid, hc, planes, hf), warm=1, reps=3)
        td = _median(lambda: cut_mesh(solid, hc, planes, hf))
        print(f"  {q:>5} {len(b['leaf']):>10,} {b['K']:>7} {b['tris']:>10,} "
              f"{th:>8.2f} {td:>9.3f} {th/td:>7.1f}x  "
              f"{'identical' if agree else 'DIFFERENT (host %d/%d/%d)' % (a['K'], len(a['leaf']), a['tris'])}")
        rows.append((q, len(b["leaf"]), b["K"], b["tris"], th, td))
    np.save(_os.path.join(_os.path.dirname(lattice_dir), "multicut_timing.npy"),
            np.array(rows, dtype=object), allow_pickle=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
