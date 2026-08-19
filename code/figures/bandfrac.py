"""How much of a lattice one plane crosses, and how that fraction moves with resolution.

Section 4 claims two things about the crossed band of equation (10): that it is a small fraction
of the object, and that it is small for a reason -- a plane meets O(N^(2/3)) of N cells, so the
fraction falls as the lattice is refined rather than being a property of these six objects.

Both are measured here, on each object's own lattice, with `subdivide.crossed` itself rather than
a second copy of the test. The exponent is measured by coarsening the object's own occupancy: a
cell set floor-divided by two is the same object at twice the spacing, so one lattice gives four
resolutions of one shape and the slope of log K against log N is read off them. No lattice is
rebuilt and no shape is approximated.

The plane is through the centroid of the solid cells -- "through the middle" -- and four normals
are taken: the three axes and the body diagonal, which is the worst case, since the band's
thickness along the normal is h times the sum of |n_a| and that sum is largest at (1,1,1).

    python code/figures/bandfrac.py OUT.png orange=build_orange/lattice ...

Writes the figure to OUT.png and prints every number it drew.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path += [_HERE, _os.path.join(_os.path.dirname(_HERE), "src"), _FN_ROOT,
             _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]

from subdivide import crossed, refine                            # equation (10), the only copy

NORMALS = [("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0)),
           ("diagonal", (1.0, 1.0, 1.0))]


def solid_cells(lat):
    """The coarse cells a cut is taken on: the lattice's level-0 cells, closed and filled.

    This is the set `subdivide.cut` is handed, built the way `ovoxel.py`'s own main builds it, so
    the count denominated against here is the count the operator sees and not the primitive count
    of the file.
    """
    import torch
    from plyfile import PlyData
    from occupancy import close_and_fill, to_grid

    meta = torch.load(_os.path.join(lat, "lattice.pt"), map_location="cpu")
    hc = float(meta["coarse_dx"])
    el = PlyData.read(_os.path.join(lat, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(lat, "cell_level.pt"), map_location="cpu").reshape(-1)
    xyz = xyz[(lvl[:len(xyz)] == 0).numpy()]

    raw = np.floor((xyz - xyz.min(0)) / hc).astype(np.int64)
    coords = np.unique(raw, axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    return close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1, hc


def coarsen(coords, k):
    """The same shape at 2^k times the spacing. Floor division is the containing cell."""
    return np.unique(coords >> k, axis=0) if k else coords


def band(coords, h, h_target=None):
    """K and N for a plane through the centroid, one row per normal.

    Two counts, because section 4 needs both and they differ by about a factor of four. `K` is
    what equation (10) returns on the lattice as it is stored, which is the set whose adjacency a
    cut has to rebuild. `K_leaf` is what is left after the operator refines every one of those
    cells to the fine spacing, which is the set that carries a polygon and the one the cut's work
    is actually proportional to. Quoting either without saying which is what makes the two
    numbers look like a disagreement.
    """
    c = (coords + 0.5) * h
    mid = c.mean(0)
    out = []
    for name, n in NORMALS:
        n = np.asarray(n, np.float64)
        n = n / np.linalg.norm(n)
        d = float(-mid @ n)
        K = int(crossed(c, h, n, d).sum())
        kl = 0
        if h_target:
            leaf, lvl, _, _ = refine(coords, h, n, d, h_target)
            hl = h / (2.0 ** lvl.astype(np.float64))
            kl = int(crossed((leaf + 0.5) * hl[:, None], hl, n, d).sum())
        out.append((name, K, kl, len(coords), K / max(len(coords), 1)))
    return out


def main(out, *specs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, series, leaves = [], {}, {}
    for spec in specs:
        obj, _, lat = spec.partition("=")
        coords, hc = solid_cells(lat)
        print(f"{obj}: {len(coords):,} solid coarse cells at h_c = {hc:.6f}")
        for k in range(0, 4):
            cc = coarsen(coords, k)
            if len(cc) < 500:
                break
            h = hc * (2 ** k)
            for name, K, kl, N, f in band(cc, h, h / 2.0 if k == 0 else None):
                rows.append((obj, k, name, K, N, f))
                extra = f"   refined leaves {kl:>8,}  {100*kl/max(N,1):6.2f}%" if kl else ""
                print(f"    2^{k} h_c  {name:<9} K {K:>8,}  N {N:>9,}  K/N {100*f:6.2f}%{extra}")
                series.setdefault((obj, name), []).append((N, K))
                if kl:
                    leaves[(obj, name)] = kl / max(N, 1)

    # the exponent, from each object's own four resolutions
    print("\n  slope of log K against log N, per object and normal (2/3 is the claim)")
    slopes = {}
    for (obj, name), pts in sorted(series.items()):
        if len(pts) < 3:
            continue
        N = np.log(np.array([p[0] for p in pts], float))
        K = np.log(np.array([p[1] for p in pts], float))
        s = float(np.polyfit(N, K, 1)[0])
        slopes[(obj, name)] = s
        print(f"    {obj:<12} {name:<9} {s:.3f}")

    objs = sorted({o for o, _ in series})
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    cmap = plt.get_cmap("tab10")
    for i, obj in enumerate(objs):
        pts = series[(obj, "diagonal")]
        N = np.array([p[0] for p in pts], float)
        K = np.array([p[1] for p in pts], float)
        ax[0].loglog(N, K, "o-", color=cmap(i), label=obj, ms=4, lw=1.3)
    lo = min(min(p[0] for p in v) for v in series.values())
    hi = max(max(p[0] for p in v) for v in series.values())
    xs = np.array([lo, hi], float)
    ax[0].loglog(xs, xs ** (2 / 3) * 2.2, "k--", lw=1, label="$N^{2/3}$")
    ax[0].set_xlabel("$N$, solid cells"); ax[0].set_ylabel("$K$, cells the plane crosses")
    ax[0].set_title("$K$ against $N$, four resolutions each", fontsize=10)
    ax[0].legend(fontsize=7, ncol=2, frameon=False)

    w = 0.34
    a = [100 * next(f for o, k, nm, K, N, f in rows
                    if o == obj and k == 0 and nm == "diagonal") for obj in objs]
    b = [100 * leaves[(obj, "diagonal")] for obj in objs]
    x = np.arange(len(objs))
    ax[1].bar(x - w / 2, a, w, label="cells equation (10) returns", color=cmap(0))
    ax[1].bar(x + w / 2, b, w, label="leaves after refinement to $h_f$", color=cmap(3))
    ax[1].set_xticks(x); ax[1].set_xticklabels(objs, rotation=20, fontsize=8)
    ax[1].set_ylabel("percent of $N$")
    ax[1].set_title("an oblique plane through the middle", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
