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


def face_pairs(st, device):
    """Row indices of every pair of solid coarse cells that share a face, as (M, 2).

    Built from `idx3`, which already maps a coarse coordinate to its row or -1: shift it by one
    along each axis and keep the positions where both ends are solid.
    """
    idx3 = st["idx3"]
    out = []
    for ax in range(3):
        a = idx3.narrow(ax, 0, idx3.shape[ax] - 1)
        b = idx3.narrow(ax, 1, idx3.shape[ax] - 1)
        m = (a >= 0) & (b >= 0)
        if m.any():
            out.append(torch.stack([a[m], b[m]], 1))
    p = torch.cat(out).long().to(device)
    print(f"  field prior: {len(p):,} face-sharing pairs over {len(st['interior']):,} cells "
          f"({100 * len(p) / max(len(st['interior']), 1):.0f}%), weight {WEIGHT}", flush=True)
    return p


def penalty(colour, pairs, generator=None):
    """Mean squared difference across a sample of the face-sharing pairs."""
    if WEIGHT <= 0 or pairs is None or len(pairs) == 0:
        return colour.new_zeros(())
    if len(pairs) > NPAIR:
        sel = torch.randint(0, len(pairs), (NPAIR,), device=pairs.device, generator=generator)
        pairs = pairs[sel]
    d = colour[pairs[:, 0]] - colour[pairs[:, 1]]
    return (d ** 2).mean()
