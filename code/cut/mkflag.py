"""The lattice flag `dynamic_cut` expects: which primitives are lattice cells.

    python mkflag.py TRAINED.ply OUT.pt

Every primitive in a lattice build is a cell -- both levels are the physical volume -- so the
flag is all true and its only job is to carry the length. It is written rather than assumed
because the demo indexes with it and a wrong length silently truncates the object.
"""
import sys

import torch
from plyfile import PlyData


def main(ply, out):
    n = len(PlyData.read(ply).elements[0])
    torch.save(torch.ones(n, dtype=torch.bool), out)
    print(f"  {n:,} primitives -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
