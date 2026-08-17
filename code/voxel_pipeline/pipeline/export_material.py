"""Per-cell material labels for a trained model, computed after training rather than during it.

A physics engine wants a discrete label -- this cell is seed, this cell is flesh, this cell is
rind -- and the three differ in colour, so the label can be read off the trained field instead
of being constrained into it. `notes/discrete-inclusions.md` records why that is the right way
round here: the proposal to learn the label needed a volume quota, a total-variation term, an
entropy term and a three-stage anneal, and the measurements went against every premise it
rested on. This does the same job in seconds.

Three materials, found three different ways, because they are three different kinds of thing:

  rind       geometric. The shell is `cell_level != 0` and the lattice already knows it. No
             clustering, no threshold, and in particular no risk of the failure that K=3 over
             the whole model walks straight into: the shell is the darkest thing in the model
             -- darker than the seeds -- so a naive "darkest cluster is the inclusion" over
             all cells labels the rind as seed and finds nothing else.

  inclusion  K-means in LAB over the interior only, taking the darkest centre. It does not
             appear at K=3: the flesh spans a wide red-to-pink gradient that eats the first
             three centres, and only at K=4 does a near-black class separate out.

  matrix     everything else.

The inclusion class is not assumed to exist. Most objects do not have one -- an orange, a loaf
and a doughnut are continuous colour fields, and taking their darkest cluster returns a slice
of a gradient that a hard label would turn into a flat band. So the separation is measured and
reported, and the class is only emitted when the darkest centre stands off from the rest by
more than the clusters' own spread. `--min-gap` is that margin, in units of the pooled
within-cluster standard deviation, and it is printed alongside what was measured so the
decision is visible rather than buried.

Optionally the labels are reorganised spatially before being written. Training leaves the
inclusion as dust -- for the watermelon, 863 pieces with a median size of one cell against a
seed that should be five to seven -- and blurring the lightness over the lattice before taking
the darkest fraction recovers pieces of about the right count. This changes which cells are
labelled, not how many: the quota defaults to the fraction K-means already found, so smoothing
is a pure reorganisation unless a physical estimate is passed explicitly.

    python voxel_pipeline/pipeline/export_material.py <ply> <lattice_dir> [--smooth 0.8]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from scipy import ndimage

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from _cells import cell_labels                                    # noqa: E402

C0 = 0.28209479177387814
MATRIX, INCLUSION, RIND = 0, 1, 2
NAMES = {MATRIX: "matrix", INCLUSION: "inclusion", RIND: "rind"}


def load_colours(ply):
    from plyfile import PlyData
    v = PlyData.read(ply)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.clip(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1
                           ).astype(np.float32) * C0 + 0.5, 0.0, 1.0)
    return xyz, rgb


def to_lab(rgb01):
    return cv2.cvtColor((rgb01.reshape(-1, 1, 3) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def darkest_cluster(lab, k, seed=0):
    """K-means in LAB; return the membership of the darkest centre and how far it stands off.

    The separation is the L* gap between the darkest centre and the next, divided by the
    pooled within-cluster L* spread -- a distance in units of the scatter it has to beat. A
    genuine inclusion is several spreads away; a slice of a gradient is a fraction of one.
    """
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    cv2.setRNGSeed(seed)
    _, lb, ctr = cv2.kmeans(lab, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    lb = lb.ravel()
    order = np.argsort(ctr[:, 0])
    dark, nxt = int(order[0]), int(order[1])
    spread = np.sqrt(np.mean([lab[lb == i, 0].var() for i in range(k) if (lb == i).any()]))
    gap = float(ctr[nxt, 0] - ctr[dark, 0]) / max(spread, 1e-6)
    return lb == dark, gap, ctr, order


def photo_gate(photo_dir, k, min_gap, size=256):
    """Does the same discrete class exist in the photographs the model was trained on?

    The model-side test cannot tell a sparse real material from sparse rendering noise: dark
    speckle in a bread crumb is shading, and in the trained field it is indistinguishable from
    a seed. The supervision can tell them apart, because a material that is really there is a
    cluster in the input photographs too. So run the identical test on the photographs and
    require both to agree.
    """
    # the photograph directories are laid out two ways -- some split into horizontal/ and
    # vertical/, some keep the files at the top -- so recurse rather than assume either
    paths = sorted(p for d, _, fs in os.walk(photo_dir) for f in fs
                   for p in [os.path.join(d, f)]
                   if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not paths:
        return None, None, 0
    px = []
    for p in paths:
        a = cv2.imread(p)
        if a is None:
            continue
        a = cv2.resize(a, (size, size))
        hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.float32)
        fg = ~((hsv[..., 1] / 255.0 < 0.10) & (hsv[..., 2] / 255.0 > 0.95))
        px.append(cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3)[fg.ravel()])
    if not px:
        return None, None, 0
    lab = np.concatenate(px).astype(np.float32)
    sel, gap, _, _ = darkest_cluster(lab, k)
    return gap, float(sel.mean()), len(paths)


def cell_grid(xyz_cells, dx):
    idx = np.round((xyz_cells - xyz_cells.min(0)) / dx).astype(np.int64)
    dims = tuple(int(t) for t in idx.max(0) + 1)
    return idx, dims


def smooth_pick(l_cell, idx, dims, sigma, quota):
    """Blur the lightness over the lattice, then take the darkest `quota` of the cells.

    Cells outside the object are filled brighter than anything inside it, so they never win
    the darkest fraction and never drag a boundary cell's blurred value down. The fill has to
    be on the same scale as `l_cell` -- L* here runs 0-255, and filling with 1.0 makes the
    outside the darkest thing in the volume, which hands the whole quota to the object's
    outer boundary.
    """
    occ = np.zeros(dims, bool)
    vol = np.full(dims, float(l_cell.max()) + 1.0, np.float32)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = l_cell
    sm = ndimage.gaussian_filter(vol, sigma=sigma)
    keep = sm <= np.quantile(sm[occ], quota)
    return (occ & keep)[idx[:, 0], idx[:, 1], idx[:, 2]]


def piece_stats(sel, idx, dims):
    vol = np.zeros(dims, bool)
    vol[idx[sel, 0], idx[sel, 1], idx[sel, 2]] = True
    lab, k = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    if k == 0:
        return 0, 0.0, 0.0
    s = ndimage.sum(vol, lab, range(1, k + 1))
    return k, float(np.median(s)), float((s == 1).mean())


def main(a):
    xyz, rgb = load_colours(a.ply)
    lvl_cell = torch.load(os.path.join(a.lattice, "cell_level.pt"))
    lvl_cell = np.asarray(lvl_cell.cpu() if hasattr(lvl_cell, "cpu") else lvl_cell)
    lvl = cell_labels(len(xyz), lvl_cell, os.path.basename(a.ply))
    K = len(xyz) // len(lvl_cell)
    dx = float(torch.load(os.path.join(a.lattice, "lattice.pt"))["coarse_dx"])

    # per cell, from its K children: mean colour, and the level they all share
    lab_p = to_lab(rgb)
    lab_c = lab_p.reshape(len(lvl_cell), K, 3).mean(1)
    xyz_c = xyz.reshape(len(lvl_cell), K, 3).mean(1)
    interior = lvl_cell == 0
    print(f"{os.path.basename(a.ply)}: {len(xyz):,} primitives, {len(lvl_cell):,} cells, "
          f"K={K}   interior {int(interior.sum()):,}   shell {int((~interior).sum()):,} "
          f"({100 * (~interior).mean():.1f}%)")

    mat = np.full(len(lvl_cell), RIND, np.int8)
    mat[interior] = MATRIX

    inner_lab = lab_c[interior]
    sel, gap, ctr, order = darkest_cluster(inner_lab, a.k)
    ldark, lnext = ctr[order[0], 0], ctr[order[1], 0]
    print(f"  K={a.k} on the interior: darkest centre L*={ldark:.1f} at "
          f"{100 * sel.mean():.2f}% of the interior, next centre L*={lnext:.1f}")
    print(f"  separation {gap:.2f} pooled spreads (need {a.min_gap:.2f} to call it discrete)")

    # Separation alone is not enough, and the orange is why: its interior L* is narrowly
    # spread, so a gradient's quartile boundary clears any reasonable margin while the class
    # it names is 55% of the interior. An inclusion is by definition the minority phase, so
    # also require the darkest cluster to be smaller than an even share of the k clusters --
    # a reference point the clustering itself supplies rather than a number chosen to fit.
    share = float(sel.mean())
    even = 1.0 / a.k
    p_gap = p_share = None
    if a.photos:
        p_gap, p_share, n_ph = photo_gate(a.photos, a.k, a.min_gap)
        if p_gap is None:
            raise SystemExit(f"no photographs under {a.photos}")
        print(f"  the same test on {n_ph} photographs: separation {p_gap:.2f}, "
              f"darkest cluster {100 * p_share:.2f}% of the section")

    if gap < a.min_gap or share >= even:
        why = (f"separation {gap:.2f} below {a.min_gap:.2f}" if gap < a.min_gap else
               f"it is {100 * share:.1f}% of the interior, no smaller than the even "
               f"share of {100 * even:.0f}%, so it is a partition of a continuum rather "
               f"than a minority phase")
    elif p_gap is not None and (p_gap < a.min_gap or p_share >= even):
        why = (f"the model has one but the photographs do not (separation {p_gap:.2f}, "
               f"{100 * p_share:.1f}% of the section). Sparse dark cells that the "
               f"supervision does not contain are shading, not a material")
    else:
        why = None

    if why is not None:
        print(f"  -> no discrete inclusion: {why}. Writing two materials, matrix and rind.")
    else:
        idx, dims = cell_grid(xyz_c[interior], dx)
        if a.smooth > 0:
            quota = a.quota / 100.0 if a.quota > 0 else float(sel.mean())
            before = piece_stats(sel, idx, dims)
            sel = smooth_pick(inner_lab[:, 0], idx, dims, a.smooth, quota)
            after = piece_stats(sel, idx, dims)
            print(f"  smoothing sigma {a.smooth}, darkest {100 * quota:.2f}% of the interior")
            for tag, (n, med, sing) in (("as clustered", before), ("after smoothing", after)):
                print(f"    {tag:<16} {n:>5} pieces   median {med:.0f} cells   "
                      f"singletons {100 * sing:.0f}%")
        else:
            n, med, sing = piece_stats(sel, idx, dims)
            print(f"    {'as clustered':<16} {n:>5} pieces   median {med:.0f} cells   "
                  f"singletons {100 * sing:.0f}%")
        w = np.where(interior)[0]
        mat[w[sel]] = INCLUSION

    out = a.out or os.path.join(a.lattice, "cell_material.pt")
    torch.save(torch.from_numpy(mat), out)
    frac = {n: 100.0 * (mat == c).mean() for c, n in NAMES.items()}
    print("  " + "   ".join(f"{n} {frac[n]:.2f}%" for n in ("inclusion", "matrix", "rind")))
    print(f"  -> {out}  (int8 per cell; expand by K for per-primitive)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("lattice")
    p.add_argument("--out", default="")
    p.add_argument("--k", type=int, default=4,
                   help="clusters over the interior; the inclusion does not separate below 4")
    p.add_argument("--min-gap", type=float, default=1.5,
                   help="how far the darkest centre must stand off, in pooled within-cluster "
                        "spreads, before it is called a discrete material rather than a slice "
                        "of a gradient")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="blur the lightness over the lattice by this sigma in cells before "
                        "picking, which reorganises dust into pieces")
    p.add_argument("--photos", default="",
                   help="directory of the section photographs the model was trained on; "
                        "when given, the same discreteness test must also pass on them, "
                        "which is what separates a real sparse material from shading")
    p.add_argument("--quota", type=float, default=0.0,
                   help="percent of the interior to label as inclusion when smoothing; "
                        "defaults to the fraction K-means found, making it a pure "
                        "reorganisation")
    main(p.parse_args())
