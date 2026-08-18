"""multicut.gif -- three planes through the orange, and the pieces sliding apart and back.

`multicut.py` reports a piece count and a list of sizes. That is the right thing to measure and
the wrong thing to look at, because a count of eight is equally consistent with eight sensible
wedges and with six wedges plus two slivers that happen to clear the threshold. Pulling the
pieces apart along their own outward directions is what shows which of the two it is: every piece
becomes visible on all sides at once, including the faces the cut made.

The labelling is `multicut.cut(..., min_cells=512)` and nothing here touches it. What this file
adds is the two steps between a label and a picture.

  the labels move down a level    Topology is decided on the coarse lattice -- `solid`, after
                                  `close_and_fill` -- while the object's appearance lives at the
                                  fine spacing. So each fine cell takes the piece of the leaf that
                                  contains it, which is the spec's (19) f_child <- f_parent read
                                  in the direction it was written. The handful of fine skin cells
                                  that fall outside the coarse solid have no leaf above them;
                                  they take the piece that dominates their own side code, which
                                  is `multicut.side_codes` applied to them directly rather than a
                                  nearest-neighbour guess.

  a face knows what made it       For each piece, a cell face is drawn when the neighbouring cell
                                  is not in the same piece. If that neighbour is empty the face
                                  was always there and gets the model's own colour; if it is
                                  occupied by another piece the face is new, and it is the cut.
                                  Tinting only the second kind is what makes the cut legible
                                  without inventing an appearance for the peel.

The separation is a translation per piece, along the piece's own centroid direction, with
amplitude (1 - cos 2 pi t) / 2 so the loop closes on itself exactly rather than nearly.

    python method/common/demo/multicut_gif.py LATTICE PLY CFG DEMO OUT.gif [frames]
"""
import os as _os

_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys
import time

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

import multicut as mc                        # noqa: E402
import subdivide as sd                       # noqa: E402
from globalovox import FACE                       # noqa: E402
import gifcam                                # noqa: E402

MIN_CELLS = 512
# eight hues, spaced round the circle, for the cut faces only
HUE = np.array([[0.95, 0.35, 0.25], [0.98, 0.68, 0.20], [0.85, 0.90, 0.25],
                [0.35, 0.85, 0.35], [0.25, 0.85, 0.80], [0.30, 0.55, 0.95],
                [0.65, 0.40, 0.95], [0.95, 0.40, 0.75]])


def _keys(c, mn, span):
    q = c - mn
    return (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]


class Occ:
    """Membership lookup on a set of integer cells: which row, or -1."""

    def __init__(self, cells):
        self.mn = cells.min(0) - 2
        self.span = (cells.max(0) - self.mn + 3).astype(np.int64)
        k = _keys(cells, self.mn, self.span)
        self.o = np.argsort(k)
        self.ks = k[self.o]

    def find(self, q):
        inr = ((q - self.mn) >= 0).all(1) & ((q - self.mn) < self.span).all(1)
        kk = _keys(np.where(inr[:, None], q, self.mn + 1), self.mn, self.span)
        pos = np.clip(np.searchsorted(self.ks, kk), 0, len(self.ks) - 1)
        hit = inr & (self.ks[pos] == kk)
        out = np.full(len(q), -1, np.int64)
        out[hit] = self.o[pos[hit]]
        return out


def piece_of_fine(allf, st, hc, planes, lf=1):
    """A piece label for every fine cell: from the leaf above it, or from its own side code.

    `allf` is indexed on the grid of spacing h_c / 2^lf, and a leaf at level L on the grid of
    spacing h_c / 2^L, so the two are the same integers only when L == lf. Getting that shift
    backwards is not loud: every lookup still succeeds for *some* cells, and the rest fall
    through to the side-code fallback, which produces a plausible eight-piece picture built
    almost entirely out of the fallback. The count of cells that needed it is printed for
    exactly that reason.
    """
    pid = np.full(len(allf), -1, np.int64)
    lvl = st["level"]
    for L in sorted({int(x) for x in lvl}, reverse=True):     # finest first: it is the specific one
        m = lvl == L
        if not m.any():
            continue
        if L >= lf:
            occ = Occ(st["leaf"][m] >> (L - lf))
            j = occ.find(allf)
        else:
            occ = Occ(st["leaf"][m])
            j = occ.find(allf >> (lf - L))
        take = (pid < 0) & (j >= 0)
        pid[take] = st["piece"][np.nonzero(m)[0][j[take]]]
    n_missing = int((pid < 0).sum())
    if n_missing:
        # the fine skin outside the coarse solid: no leaf covers it, so use the side code and
        # the piece that code already stands for
        code_leaf = st["code"]
        code_fine, _, _ = mc.side_codes(allf[pid < 0], np.ones(n_missing, np.int8), hc, planes)
        table = {}
        for cd in np.unique(code_leaf):
            sel = code_leaf == cd
            table[int(cd)] = int(np.bincount(st["piece"][sel]).argmax())
        fallback = int(np.bincount(st["piece"]).argmax())
        pid[pid < 0] = np.array([table.get(int(cd), fallback) for cd in code_fine])
    return pid, n_missing


def faces(allf, pid, K, k_samp=3):
    """Sample points on every face whose neighbour is in a different piece (or in nothing).

    Returns the points, the piece each belongs to, the face normal, whether the face was
    already on the outside of the object, and the cell it belongs to.
    """
    occ = Occ(allf)
    t = (np.arange(k_samp) + 0.5) / k_samp
    ga, gb = [x.ravel() for x in np.meshgrid(t, t, indexing="ij")]
    P, W, N, EXT, CEL = [], [], [], [], []
    for d, corners in FACE:
        j = occ.find(allf + d)
        empty = j < 0
        diff = empty | (pid[np.where(empty, 0, j)] != pid)
        if not diff.any():
            continue
        c = allf[diff].astype(np.float64)
        ax = int(np.nonzero(d)[0][0])
        other = [a for a in range(3) if a != ax]
        base = c.copy()
        base[:, ax] += (1.0 if d[ax] > 0 else 0.0)
        p = np.repeat(base, len(ga), axis=0)
        p[:, other[0]] += np.tile(ga, len(c))
        p[:, other[1]] += np.tile(gb, len(c))
        P.append(p)
        W.append(np.repeat(pid[diff], len(ga)))
        N.append(np.repeat(d.astype(np.float64)[None], len(p), axis=0))
        EXT.append(np.repeat(empty[diff], len(ga)))
        # The cell the face belongs to, not the face's own centre. A centre sits half a cell
        # off the cell it came from, so looking a colour up by it can land on the neighbour --
        # which is an interior cell for every face on the outside. That is what blotched the
        # orange's peel with pith, swapped the apple's skin and flesh, and turned the
        # doughnut magenta.
        CEL.append(np.repeat(allf[diff], len(ga), axis=0))
    return (np.concatenate(P), np.concatenate(W), np.concatenate(N),
            np.concatenate(EXT), np.concatenate(CEL))


def main(lattice_dir, ply, cfg, demo, out_gif, n_frames=72, size=440, ss=2, k_samp=3,
         amp=0.52, radius_scale=1.55):
    from scipy.spatial import cKDTree
    from occupancy import close_and_fill, to_grid

    lat = gifcam.load_lattice(lattice_dir, ply)
    xyz, rgb, lvl, hc, hf = lat["xyz"], lat["rgb"], lat["level"], lat["hc"], lat["hf"]
    # from a corner, not from a centre: floor of a centre sitting exactly on a cell
    # boundary lets floating point choose the side, and on a lattice whose cells are at
    # (i + 1/2)h that discards 49% of them. Offset by half the finest spacing used here.
    org = xyz[lvl == 0].min(0) - 0.5 * hf
    coords = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64), axis=0)
    occg, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occg, 1).nonzero().numpy() + coords.min(0) - 1
    ctr = ((solid + 0.5) * hc).mean(0)
    print(f"  {len(coords):,} coarse cells -> {len(solid):,} after close_and_fill, "
          f"h_c {hc:.5f}, h_f {hf:.5f}")

    def P(v, off=0.0):
        v = np.asarray(v, np.float64)
        v = v / np.linalg.norm(v)
        return (v, float(-ctr @ v) + off * hc)

    planes = [P([0.13, 0.97, -0.21], 0.37), P([0.91, 0.05, 0.41], -0.23),
              P([-0.22, 0.33, 0.92], 0.11)]

    t0 = time.time()
    st = mc.cut(solid, hc, planes, hf, min_cells=MIN_CELLS)
    K = st["K"]
    sizes = [int((st["piece"] == p).sum()) for p in range(K)]
    print(f"  {len(st['leaf']):,} leaves to level {st['top']}, {st['raw_K']} raw pieces -> "
          f"{K} at min_cells={MIN_CELLS}: {sorted(sizes, reverse=True)}  "
          f"[{time.time() - t0:.1f}s]")

    # the fine cells: the solid expressed at h_f, plus the skin's own fine cells
    off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int64)
    fine_skin = np.floor((xyz - org) / hf).astype(np.int64)
    allf = np.unique(np.concatenate([(solid[:, None, :] * 2 + off[None]).reshape(-1, 3),
                                     fine_skin]), axis=0)
    pid, n_miss = piece_of_fine(allf, st, hc, planes)
    print(f"  {len(allf):,} fine cells, {n_miss:,} of them outside every leaf and labelled by "
          f"their side code")

    pts, who, nrm, ext, fcell = faces(allf, pid, K, k_samp=k_samp)
    n_face = len(pts) // (k_samp * k_samp)
    n_ext = int(ext.sum()) // (k_samp ** 2)
    n_cut = n_face - n_ext
    print(f"  {n_face:,} drawn faces ({n_ext:,} were already on the outside, {n_cut:,} are cut) "
          f"-> {len(pts):,} samples")

    pts = pts * hf + org
    # The primitive that occupies the face's own cell, found by the cell index rather than by
    # distance. Both the faces and the model are quantised to the same fine grid, so this is a
    # lookup and not a guess; only a cell the model has no primitive in -- an interior cell of
    # the solid, never on the outside -- falls back to the nearest one.
    def _key(a):
        b = a - lo_cell
        return (b[:, 0] * span[1] + b[:, 1]) * span[2] + b[:, 2]
    prim_cell = np.floor((xyz - org) / hf).astype(np.int64)
    lo_cell = np.minimum(prim_cell.min(0), fcell.min(0))
    span = np.maximum(prim_cell.max(0), fcell.max(0)) - lo_cell + 1
    pk = _key(prim_cell)
    order = np.argsort(pk, kind="stable")
    pk_sorted = pk[order]
    fk = _key(fcell)
    at = np.clip(np.searchsorted(pk_sorted, fk), 0, len(pk_sorted) - 1)
    hit = pk_sorted[at] == fk
    idx = np.where(hit, order[at], 0)
    if not hit.all():
        miss = ~hit
        idx[miss] = cKDTree(xyz).query((fcell[miss] + 0.5) * hf + org, k=1)[1]
    print(f"  face colours: {int(hit.sum()):,} of {len(fk):,} read the primitive in their own "
          f"cell, {int((~hit).sum()):,} fell back to the nearest")
    col = rgb[idx]
    # No per-piece tint. A cut face showing an invented pastel is the one thing this
    # representation is supposed not to do: the colour on an exposed face comes from the volume
    # that was always there, and blending 38% of an arbitrary hue over it hides exactly the claim
    # the picture is meant to support. The pieces are told apart by moving apart and by their
    # shading, which is what tells them apart in a photograph too.
    col = gifcam.shade(col, nrm).astype(np.float32)
    del fcell

    # each piece's own outward direction, from where its leaves are
    cen = np.stack([st["centre"][st["piece"] == p].mean(0) for p in range(K)])
    dirs = cen - st["centre"].mean(0)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-30)
    radius = float(np.linalg.norm(st["centre"] - st["centre"].mean(0), axis=1).max())
    print(f"  outward directions:\n" + "\n".join(
        f"    piece {p}: {sizes[p]:>8,} leaves, dir {dirs[p].round(3)}" for p in range(K)))

    # The demo's own viewpoint. Default for an object whose config axis matches the model:
    # slightly above the horizon. The cake's conf inherits orange_physics.json, whose up axis is
    # not the cake model's own -- its six canonical views show it standing only at elevation 90 --
    # so it is looked at from above instead. This moves the camera and nothing else; the axis the
    # method trained under is untouched.
    cam = gifcam.Cam(ply, cfg, demo, az=float(_os.environ.get("DEMO_AZ", "0")),
                     el=float(_os.environ.get("DEMO_EL", "15")), radius_scale=radius_scale)
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        a = amp * radius * 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
        moved = pts + (dirs[who] * a)
        img, drawn = gifcam.splat(cam, moved, col, size, ss=ss)
        lines = [f"3 planes, {K} pieces   min_cells {MIN_CELLS}",
                 f"{len(st['leaf']):,} leaves to level {st['top']}, {st['raw_K']} raw -> {K}",
                 f"pieces {min(sizes):,} to {max(sizes):,} leaves",
                 f"{n_cut:,} cut faces, {n_ext:,} already outside",
                 f"separation {a / hc:.1f} h_c"]
        # The caption is for reading a diagnostic, not for a page. DEMO_NO_CAPTION drops it so
        # the same command produces the figure and the demo.
        frames.append(np.asarray(img if _os.environ.get("DEMO_NO_CAPTION") == "1"
                                 else gifcam.caption(img, lines, size=15, band=True)))
        if i % 12 == 0:
            print(f"  frame {i:>3}/{n_frames}  separation {a / hc:.1f} h_c, {drawn:,} drawn")

    if out_gif.endswith(".mp4"):
        gifcam.write_frames(out_gif, frames)
    else:
        gifcam.write_gif(out_gif, frames, duration=70, colors=160)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 72,
         size=int(_os.environ.get("DEMO_SIZE", "440")),
         ss=int(_os.environ.get("DEMO_SS", "2")),
         k_samp=int(_os.environ.get("DEMO_KSAMP", "3")))
