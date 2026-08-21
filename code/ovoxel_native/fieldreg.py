"""A spatial prior on the interior field, because this representation has none.

The pipeline's interior is a set of Gaussian primitives whose footprints overlap several coarse
cells, so a cell's colour is averaged with its neighbours' for free and no cell is ever fitted
alone. This representation is piecewise constant per cell with no overlap at all: a cell crossed by
one supervised plane is fitted to one photograph's pixel and nothing couples it to the cell beside
it except the decoder's shared weights, which are a very weak coupling. The longitudinal cuts show
what that produces -- straight full-height columns that ignore the shape of the object, at a scale
the photographs do contain but with a coherence they do not.

`SEC_TV` states the missing coupling as a prior instead of inheriting it from a rendering
primitive: the squared difference between the decoded colours of cells that share a face, summed
over the three axis directions. The pairs are built once from the occupancy.

    SEC_TV      weight; 0 (default) leaves the objective exactly as it was
    SEC_TV_N    how many pairs to sample per step, so the term costs the same on any object

It is a Laplacian and not a total variation on purpose. An L1 difference would preserve edges,
which is the right prior for an image and the wrong one here: the edges this is meant to remove are
exactly the sharp ones, and the structure it must not remove is at a scale of several cells, which
a quadratic penalty attenuates far less than it does a per-cell step.
"""
import os

import numpy as np
import torch

WEIGHT = float(os.environ.get("SEC_TV", "0"))
NPAIR = int(os.environ.get("SEC_TV_N", "200000"))
# Per-axis weights, because the field is not equally constrained in the three directions and it is
# not constrained worst along the one the columns run down.
#
# Measured on the orange rather than assumed. The polar axis is lattice axis 1 there (the transverse
# normal is (0, -1, 0)), and the mean colour step between face-sharing neighbours is 0.0562 along
# axis 1 against 0.0989 and 0.0971 along axes 0 and 2 -- so neighbours agree BEST along the polar
# axis and worst in the plane perpendicular to it. That is the opposite of what "the columns come
# from cells being unconstrained along the axis" predicts, and it has a plain cause: the transverse
# planes are perpendicular to the polar axis, they stack along it, each is supervised by a
# photograph and each is jittered, so that is the direction the data actually pins down. The ratio
# is 1.76 on ov2, 1.68 on ov3 and 1.59 on ovbal, so it is a property of the supervision geometry and
# not of one arm.
#
# The columns therefore run ALONG the polar axis and are made of disagreement ACROSS it, so a prior
# meant to break them belongs on the two perpendicular directions. SEC_TV_AXES sets the three
# lattice-axis weights directly; SEC_TV_POLAR is the shorthand that puts a weight on the two axes
# perpendicular to a given polar axis and 1 on the polar one.
_axes = os.environ.get("SEC_TV_AXES", "")
AXES = [float(x) for x in _axes.split(",")] if _axes else [1.0, 1.0, 1.0]
POLAR_W = float(os.environ.get("SEC_TV_PERP", "0"))


def axis_weights(polar=None):
    """The three lattice-axis weights, from SEC_TV_AXES or from SEC_TV_PERP and the polar axis."""
    if _axes or POLAR_W <= 0 or polar is None:
        return list(AXES)
    import numpy as _np
    a = int(_np.argmax(_np.abs(_np.asarray(polar, float))))
    return [1.0 if i == a else POLAR_W for i in range(3)]


def face_pairs(st, device, polar=None):
    """Row indices of every pair of solid coarse cells that share a face, as (M, 3).

    Built from `idx3`, which already maps a coarse coordinate to its row or -1: shift it by one
    along each axis and keep the positions where both ends are solid. The third column is the axis
    the pair lies along, so the penalty can weight the directions differently -- it used to be
    dropped, and with it went the only thing that distinguishes the direction the data pins down
    from the two it does not.
    """
    idx3 = st["idx3"]
    out = []
    for ax in range(3):
        a = idx3.narrow(ax, 0, idx3.shape[ax] - 1)
        b = idx3.narrow(ax, 1, idx3.shape[ax] - 1)
        m = (a >= 0) & (b >= 0)
        if m.any():
            ab = torch.stack([a[m], b[m]], 1)
            out.append(torch.cat([ab, torch.full_like(ab[:, :1], ax)], 1))
    p = torch.cat(out).long().to(device)
    w = axis_weights(polar)
    n = [int((p[:, 2] == ax).sum()) for ax in range(3)]
    print(f"  field prior: {len(p):,} face-sharing pairs over {len(st['interior']):,} cells "
          f"({100 * len(p) / max(len(st['interior']), 1):.0f}%), weight {WEIGHT}, "
          f"per axis {n} at {[round(x, 3) for x in w]}", flush=True)
    return p


def penalty(colour, pairs, generator=None, polar=None):
    """Mean squared difference across a sample of the face-sharing pairs, weighted by direction."""
    if WEIGHT <= 0 or pairs is None or len(pairs) == 0:
        return colour.new_zeros(())
    if len(pairs) > NPAIR:
        sel = torch.randint(0, len(pairs), (NPAIR,), device=pairs.device, generator=generator)
        pairs = pairs[sel]
    d = colour[pairs[:, 0]] - colour[pairs[:, 1]]
    sq = (d ** 2).sum(-1) if d.dim() > 1 else d ** 2
    if pairs.shape[1] < 3:
        return sq.mean() / (3.0 if d.dim() > 1 else 1.0)
    w = axis_weights(polar)
    if w == [1.0, 1.0, 1.0]:
        return sq.mean() / (3.0 if d.dim() > 1 else 1.0)
    wt = torch.as_tensor(w, dtype=sq.dtype, device=sq.device)[pairs[:, 2]]
    # divided by the mean weight, so turning the anisotropy on does not also change the term's
    # overall size and confound it with SEC_TV itself
    return (sq * wt).mean() / wt.mean() / (3.0 if d.dim() > 1 else 1.0)
