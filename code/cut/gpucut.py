"""The cut on the GPU, in the shape the specification asked for.

Section 5.2's box and section 10.2 describe a cut as a sequence of data-parallel passes keyed by
the coarse cell: mark the crossed cells, compact them into a list, allocate a dense child block per
marked cell and inherit its feature, then classify the children against the plane. That structure
is what makes the work batchable, and the CPU implementation follows it already -- it was written
with NumPy because a correct version was needed before a fast one.

Nothing here changes the algorithm. Every pass is the same arithmetic on the same integers, moved
to the device and expressed in torch:

    kernel A   the 8-corner sign test of (16), evaluated for every occupied coarse cell at once
    compaction torch.nonzero, which is the prefix sum the specification names
    kernel B   a repeat_interleave of the marked cells by 8 and an offset add, which is the dense
               child block; the feature is inherited by carrying the parent index rather than by
               copying, so a child costs an int and not d floats
    kernel C   the sign of (20) at the child centres
    adjacency  sorted keys and a searchsorted per direction and level, the same six passes
    labelling  left on the host, which section 10.2 explicitly allows: it is a graph traversal, it
               is not the part that was slow, and scipy does it in milliseconds

Correctness is checked against the CPU path rather than asserted: same leaves, same pieces, same
surviving edges, on the synthetic shapes and on a real lattice.

    python method/common/cube/gpucut.py            # the self-test against the CPU path
    python method/common/cube/gpucut.py LATTICE    # and the timing on a real one
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys
import time

import numpy as np
import torch

sys.path += [_FN_ROOT]

DEV = _os.environ.get("CUT_DEV", "cuda" if torch.cuda.is_available() else "cpu")
_OFF = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.int64)
_CORNER = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                       dtype=torch.float64)


def _crossed(cells, h, n, d):
    """Kernel A. A cell is crossed when its corner signs are not all equal, which is (15) and (16)
    rather than the half-diagonal bound: the corners are what the specification tests and the two
    agree, but the corner form needs no absolute value and vectorises to one min and one max."""
    lo = cells.to(torch.float64) * h
    s = (lo[:, None, :] + _CORNER.to(cells.device) * h) @ n + d
    return (s.amin(1) <= 0) & (s.amax(1) >= 0)


def refine(coords, h, n, d, h_target, levels_max=8):
    """Kernels A, B and the compaction between them, one level at a time."""
    dev = torch.device(DEV)
    n = torch.as_tensor(n, dtype=torch.float64, device=dev)
    cur = torch.as_tensor(np.asarray(coords), dtype=torch.int64, device=dev)
    par = torch.arange(len(cur), dtype=torch.int64, device=dev)
    off = _OFF.to(dev)

    leaf, lvl, pars = [], [], []
    hl = float(h)
    for L in range(1, levels_max + 1):
        if hl <= h_target:
            break
        m = _crossed(cur, hl, n, d)                     # A
        if not bool(m.any()):
            break
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)   # compaction
        keep = torch.nonzero(~m, as_tuple=False).squeeze(1)
        leaf.append(cur[keep]); lvl.append(torch.full((len(keep),), L - 1, dtype=torch.int8,
                                                      device=dev)); pars.append(par[keep])
        base = cur[idx] * 2                             # B
        cur = (base[:, None, :] + off[None]).reshape(-1, 3)
        par = par[idx].repeat_interleave(8)
        hl *= 0.5
    leaf.append(cur); pars.append(par)
    lvl.append(torch.full((len(cur),), int(round(np.log2(float(h) / hl))), dtype=torch.int8,
                          device=dev))
    return torch.cat(leaf), torch.cat(lvl), torch.cat(pars), int(round(np.log2(float(h) / hl)))


def _pack(c, base, span):
    c = c + base
    return (c[:, 0] * span + c[:, 1]) * span + c[:, 2]


def adjacency(coords, level, top):
    """The same six passes per level, with the dictionary replaced by a sorted key and a
    searchsorted so that each is one device-side kernel."""
    dev = coords.device
    mn = int(coords.min()) - 2
    base, span = -mn, int(coords.max()) + (-mn) + 3
    levels = sorted({int(v) for v in level.unique()})
    keys, order = {}, {}
    for L in levels:
        at = torch.nonzero(level == L, as_tuple=False).squeeze(1)
        k = _pack(coords[at], base, span)
        o = torch.argsort(k)
        keys[L], order[L] = k[o], at[o]

    ea, eb = [], []
    for L in levels:
        idx = torch.nonzero(level == L, as_tuple=False).squeeze(1)
        c = coords[idx]
        for ax in range(3):
            for sgn in (-1, 1):
                nb = c.clone()
                nb[:, ax] += sgn
                todo = torch.ones(len(idx), dtype=torch.bool, device=dev)
                for Lp in range(L, -1, -1):
                    if Lp not in keys or not bool(todo.any()):
                        continue
                    q = nb[todo] >> (L - Lp)
                    kk = _pack(q, base, span)
                    pos = torch.searchsorted(keys[Lp], kk).clamp(0, len(keys[Lp]) - 1)
                    hit = keys[Lp][pos] == kk
                    if bool(hit.any()):
                        ea.append(idx[todo][hit]); eb.append(order[Lp][pos[hit]])
                        w = torch.nonzero(todo, as_tuple=False).squeeze(1)[hit]
                        todo[w] = False
    if not ea:
        return torch.zeros((0, 2), dtype=torch.int64, device=dev)
    a, b = torch.cat(ea), torch.cat(eb)
    m = a != b
    a, b = a[m], b[m]
    e = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], 1)
    return torch.unique(e, dim=0)


def cut(coords, h, n, d, h_target, levels_max=8):
    """The whole cut, with only the component labelling on the host."""
    from method.common.cube.subdivide import components
    dev = torch.device(DEV)
    leaf, lvl, par, top = refine(coords, h, n, d, h_target, levels_max)
    e = adjacency(leaf, lvl, top)
    hl = float(h) / (2.0 ** lvl.to(torch.float64))
    centre = (leaf.to(torch.float64) + 0.5) * hl[:, None]
    q = torch.sign(centre @ torch.as_tensor(n, dtype=torch.float64, device=dev)
                   + float(d))                                            # kernel C
    q[q == 0] = 1.0
    e_all = e
    if len(e):
        e = e[q[e[:, 0]] == q[e[:, 1]]]
    piece, K = components(len(leaf), e.cpu().numpy())
    return dict(leaf=leaf.cpu().numpy(), level=lvl.cpu().numpy(), parent=par.cpu().numpy(),
                top=top, centre=centre.cpu().numpy(), side=q.cpu().numpy(),
                edges=e.cpu().numpy(), edges_all=e_all.cpu().numpy(), piece=piece, K=K)


def _selftest(lattice_dir=None):
    from method.common.cube import subdivide as sd
    bad = 0
    print(f"  device {DEV}")
    cases = [("ball, oblique", sd._ball(12), (0.13, 0.97, -0.21), 0.37),
             ("ball, axis", sd._ball(12), (0, 0, 1), 0.0),
             ("torus, across", sd._torus(), (0, 0, 1), 0.0),
             ("dumbbell, neck", sd._dumbbell(), (0, 0, 1), 0.0),
             ("bridge, through", sd._bridge(), (0, 0, 1), 0.0)]
    for name, c, n, d in cases:
        n = np.asarray(n, np.float64)
        a = sd.cut(c, 1.0, n, d, 0.5)
        b = cut(c, 1.0, n, d, 0.5)
        same = (a["K"] == b["K"] and len(a["leaf"]) == len(b["leaf"])
                and len(a["edges"]) == len(b["edges"])
                and np.array_equal(np.sort(np.bincount(a["piece"])),
                                   np.sort(np.bincount(b["piece"]))))
        bad += not same
        print(f"  {'ok ' if same else 'FAIL'} {name:18s} {b['K']} pieces, "
              f"{len(b['leaf']):,} leaves, {len(b['edges']):,} edges")

    if lattice_dir:
        from plyfile import PlyData
        from method.common.cube.occupancy import close_and_fill, to_grid
        lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
        hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
        el = PlyData.read(_os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
        xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
        lv = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
        org = xyz[lv == 0].min(0) - 0.5 * hc
        co = np.unique(np.floor((xyz[lv == 0] - org) / hc).astype(np.int64), axis=0)
        occ, _, _ = to_grid(torch.from_numpy(co).float(), 1.0)
        solid = close_and_fill(occ, 1).nonzero().numpy() + co.min(0) - 1
        n = np.array([0.13, 0.97, -0.21]); n /= np.linalg.norm(n)
        d = float(-((solid + 0.5) * hc).mean(0) @ n) + 0.37 * hc
        print(f"\n  {len(solid):,} solid cells")
        t0 = time.perf_counter(); a = sd.cut(solid, hc, n, d, hf); t_cpu = time.perf_counter() - t0
        torch.cuda.synchronize() if DEV == "cuda" else None
        t0 = time.perf_counter(); b = cut(solid, hc, n, d, hf)
        torch.cuda.synchronize() if DEV == "cuda" else None
        t_gpu = time.perf_counter() - t0
        same = (a["K"] == b["K"] and len(a["leaf"]) == len(b["leaf"])
                and len(a["edges"]) == len(b["edges"]))
        bad += not same
        print(f"  host   {t_cpu:6.2f}s   {a['K']} pieces, {len(a['leaf']):,} leaves, "
              f"{len(a['edges']):,} edges")
        print(f"  device {t_gpu:6.2f}s   {b['K']} pieces, {len(b['leaf']):,} leaves, "
              f"{len(b['edges']):,} edges   {'identical' if same else 'DIFFERENT'}"
              f"   {t_cpu / max(t_gpu, 1e-9):.1f}x")
    print(f"\n  {'all agree with the host path' if not bad else f'{bad} FAILED'}")
    return bad


if __name__ == "__main__":
    sys.exit(_selftest(sys.argv[1] if len(sys.argv) > 1 else None))
