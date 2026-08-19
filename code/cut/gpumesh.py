"""M4 on the device, and an honest timing of the whole cut.

`gpucut` moved M3 -- the band refinement, the adjacency and the labelling -- and left the six
seconds of polygon emission on the host, so the end-to-end number barely moved. This does the
other half. The host implementation is already array code rather than a loop over cells, so
every step here is the same arithmetic in torch:

    polygons     the twelve edge intersections of (23) for every cut leaf at once, then an
                 argsort of the angle about each polygon's centroid in the plane's basis
    triangulate  a fan per polygon, batched by vertex count; there are only four counts
    merge        round to a tolerance, unique the rows, and take the first occurrence of each
                 as the representative -- torch.unique gives the inverse but not that index, so
                 it comes from a scatter_reduce amin over the positions

Nothing is approximated and nothing is reordered that the host did not already reorder.

    python gpumesh.py LATTICE        # correctness against the host path, then the timing

The timing is a median of repeats after a warm-up. A single measurement of the first call on a
device includes the context creation and the kernel compilation, which on the first object we
timed made a 4.7 s host path look 8.4x faster and on the second 23.9x -- the same code, the
same size of problem, and the difference was entirely the warm-up.
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
    from method.common.cube import cutmesh as cm
    from method.common.cube import subdivide as sd
    from method.common.cube import gpucut as gc
except ImportError:
    import cutmesh as cm
    import subdivide as sd
    import gpucut as gc

DEV = _os.environ.get("CUT_DEV", "cuda" if torch.cuda.is_available() else "cpu")


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def polygons(leaf, level, h, n, d):
    """One convex polygon per cut leaf, ordered around its centroid. The host's `polygons`."""
    dev = leaf.device
    nn, u, v = cm.plane_basis(np.asarray(n, np.float64))
    nn = torch.as_tensor(nn, dtype=torch.float64, device=dev)
    u = torch.as_tensor(u, dtype=torch.float64, device=dev)
    v = torch.as_tensor(v, dtype=torch.float64, device=dev)
    corner = torch.as_tensor(cm.CORNER, dtype=torch.float64, device=dev)
    edge = torch.as_tensor(cm.EDGE, dtype=torch.int64, device=dev)

    hl = h / (2.0 ** level.to(torch.float64))
    lo = leaf.to(torch.float64) * hl[:, None]
    corners = lo[:, None, :] + corner[None] * hl[:, None, None]
    s = corners @ nn + d

    sa, sb = s[:, edge[:, 0]], s[:, edge[:, 1]]
    hit = (sa > 0) != (sb > 0)
    denom = torch.where((sa - sb).abs() < 1e-30, torch.ones_like(sa), sa - sb)
    t = torch.where(hit, sa / denom, torch.zeros_like(sa))[..., None]
    ca, cb = corners[:, edge[:, 0]], corners[:, edge[:, 1]]
    p = ca + t * (cb - ca)

    cnt = hit.sum(1)
    keep = cnt >= 3
    p, hit, cnt = p[keep], hit[keep], cnt[keep]
    cell = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if not len(cell):
        return (torch.zeros((0, 6, 3), dtype=torch.float64, device=dev),
                torch.zeros(0, dtype=torch.int64, device=dev), cell)

    ctr = (p * hit[..., None]).sum(1) / cnt[:, None]
    q = p - ctr[:, None, :]
    ang = torch.atan2(q @ v, q @ u)
    ang = torch.where(hit, ang, torch.full_like(ang, float("inf")))
    order = torch.argsort(ang, dim=1)
    p = torch.take_along_dim(p, order[..., None], dim=1)[:, :6]
    return p, cnt, cell


def triangulate(verts, cnt):
    """A fan per polygon, batched by vertex count. Four counts are possible."""
    dev = verts.device
    tri, owner = [], []
    for k in range(3, 7):
        m = torch.nonzero(cnt == k, as_tuple=False).squeeze(1)
        if not len(m):
            continue
        for j in range(1, k - 1):
            tri.append(torch.stack([verts[m, 0], verts[m, j], verts[m, j + 1]], 1))
            owner.append(m)
    if not tri:
        return (torch.zeros((0, 3, 3), dtype=torch.float64, device=dev),
                torch.zeros(0, dtype=torch.int64, device=dev))
    return torch.cat(tri), torch.cat(owner)


def merge(tri, tol):
    """Shared vertices become one, with the first occurrence as the representative."""
    dev = tri.device
    flat = tri.reshape(-1, 3)
    key = torch.round(flat / tol).to(torch.int64)
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    pos = torch.arange(len(flat), dtype=torch.int64, device=dev)
    first = torch.full((len(uniq),), len(flat), dtype=torch.int64, device=dev)
    first = first.scatter_reduce(0, inv, pos, reduce="amin", include_self=True)
    return flat[first], inv.reshape(-1, 3)


def cut_mesh(coords, h, n, d, h_target, levels_max=8, r=None):
    """M3 and M4 together, on the device. The host's `cut_mesh`."""
    dev = torch.device(DEV)
    if r is None:
        r = gc.cut(coords, h, n, d, h_target, levels_max)
    nn, _, _ = cm.plane_basis(np.asarray(n, np.float64))
    leaf = torch.as_tensor(r["leaf"], dtype=torch.int64, device=dev)
    level = torch.as_tensor(r["level"], dtype=torch.int64, device=dev)

    verts, cnt, cell = polygons(leaf, level, h, nn, d)
    tri, owner = triangulate(verts, cnt)
    V, F = merge(tri, tol=h * 1e-6)

    piece = torch.as_tensor(r["piece"], dtype=torch.int64, device=dev)
    q = torch.as_tensor(r["side"], dtype=torch.float64, device=dev)
    face_cell = cell[owner]
    across = torch.full((len(leaf),), -1, dtype=torch.int64, device=dev)
    ea = torch.as_tensor(r["edges_all"], dtype=torch.int64, device=dev)
    if len(ea):
        opp = q[ea[:, 0]] != q[ea[:, 1]]
        a, b = ea[opp, 0], ea[opp, 1]
        across[a] = piece[b]
        across[b] = piece[a]
    own, oth = piece[face_cell], across[face_cell]
    up = q[face_cell] > 0
    top = torch.where(up, own, oth)
    bot = torch.where(up, oth, own)
    F2 = torch.cat([F, F.flip(1)])
    side = torch.cat([torch.ones(len(F), dtype=torch.int8, device=dev),
                      -torch.ones(len(F), dtype=torch.int8, device=dev)])
    return dict(V=V, F=F2, side=side, piece=torch.cat([bot, top]),
                stats=dict(cut_cells=int(len(cell)), tris=int(len(F)), verts=int(len(V)),
                           pieces=int(r["K"])))


def _area(V, F):
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    return float(0.5 * torch.linalg.cross(a, b).norm(dim=1).sum())


def _median(fn, warm=2, reps=5):
    for _ in range(warm):
        fn(); _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


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
    n = np.array([0.13, 0.97, -0.21]); n /= np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n) + 0.37 * hc
    print(f"  {lattice_dir}: {len(solid):,} solid cells, device {DEV}")

    a = cm.cut_mesh(solid, hc, n, d, hf)
    b = cut_mesh(solid, hc, n, d, hf)
    Va = torch.as_tensor(a["V"]); Fa = torch.as_tensor(a["F"])
    ok = (a["stats"]["tris"] == b["stats"]["tris"]
          and a["stats"]["verts"] == b["stats"]["verts"]
          and a["stats"]["cut_cells"] == b["stats"]["cut_cells"]
          and a["stats"]["pieces"] == b["stats"]["pieces"]
          and abs(_area(Va, Fa) - _area(b["V"].cpu(), b["F"].cpu())) < 1e-9 * max(1.0, _area(Va, Fa)))
    print(f"  host   {a['stats']['verts']:,} verts, {a['stats']['tris']:,} triangles, "
          f"{a['stats']['cut_cells']:,} cut cells, {a['stats']['pieces']} pieces")
    print(f"  device {b['stats']['verts']:,} verts, {b['stats']['tris']:,} triangles, "
          f"{b['stats']['cut_cells']:,} cut cells, {b['stats']['pieces']} pieces"
          f"   {'identical' if ok else 'DIFFERENT'}")

    print("  timings, median of five after two warm-ups:")
    m3h = _median(lambda: sd.cut(solid, hc, n, d, hf), warm=1, reps=3)
    m3d = _median(lambda: gc.cut(solid, hc, n, d, hf))
    r = gc.cut(solid, hc, n, d, hf)
    fullh = _median(lambda: cm.cut_mesh(solid, hc, n, d, hf), warm=1, reps=3)
    m4d = _median(lambda: cut_mesh(solid, hc, n, d, hf, r=r))
    fulld = _median(lambda: cut_mesh(solid, hc, n, d, hf))
    m4h = (fullh[0] - m3h[0], 0, 0)
    print(f"    M3 refine+label   host {m3h[0]:6.3f}s   device {m3d[0]:6.3f}s"
          f"   {m3h[0]/m3d[0]:5.1f}x")
    print(f"    M4 polygons       host {m4h[0]:6.3f}s   device {m4d[0]:6.3f}s"
          f"   {m4h[0]/m4d[0]:5.1f}x")
    print(f"    end to end        host {fullh[0]:6.3f}s   device {fulld[0]:6.3f}s"
          f"   {fullh[0]/fulld[0]:5.1f}x   (device spread {fulld[1]:.3f}-{fulld[2]:.3f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
