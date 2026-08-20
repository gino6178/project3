"""Is the motion rigid, or is the material deforming, and does the peel deform less than the flesh?

    python strain.py STATE_DIR LATTICE

For each piece and each pair of dumped frames, the best rigid transform between the two particle
configurations is removed by Procrustes and what remains is reported. A piece that only falls and
spins leaves nothing; a piece that deforms leaves a residual, and the residual is a length that can
be quoted in cells.

The peel and the interior are separated by the lattice's own level tag, so "the stiff part deforms
less" is a comparison between two sets of particles in the same body under the same contact, not
between two runs.
"""
import glob
import os
import sys

import numpy as np
import torch


def procrustes(a, b):
    """Rigid transform taking a onto b, and the per-point residual left after it."""
    ca, cb = a.mean(0), b.mean(0)
    A, B = a - ca, b - cb
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return np.linalg.norm(B - A @ R.T, axis=1)


def main(d, lattice):
    hc = float(torch.load(os.path.join(lattice, "lattice.pt"))["coarse_dx"])
    lvl = torch.load(os.path.join(lattice, "cell_level.pt")).reshape(-1).numpy()
    fs = sorted(glob.glob(os.path.join(d, "state_*.npz")))
    print(f"  {len(fs)} states, coarse spacing {hc:.5f}")
    base = np.load(fs[0])
    # Particles are matched by the global index they carry, not by their piece label. A cut
    # renumbers the pieces, so pairing label 0 in one frame with label 0 in another compares two
    # different sets of particles and returns a residual of tens of cells that is nothing but
    # the mismatch. Measured that way the same run reported 27 cells where it should report 0.2.
    bpos = {int(i): k for k, i in enumerate(base["idx"])}
    for f in fs[1:]:
        cur = np.load(f)
        print(f"  {os.path.basename(fs[0])} -> {os.path.basename(f)}")
        for p in sorted(set(cur["piece"].tolist())):
            m = cur["piece"] == p
            if m.sum() < 500:
                continue
            gi = cur["idx"][m]
            keep = np.array([int(g) in bpos for g in gi])
            if keep.sum() < 500:
                print(f"    piece {p}: no common particles with the first state"); continue
            src = np.array([bpos[int(g)] for g in gi[keep]])
            r = procrustes(base["x"][src], cur["x"][m][keep]) / hc
            gi = gi[keep]
            sk = lvl[gi] == 1 if gi.max() < len(lvl) else np.zeros(len(gi), bool)
            line = (f"    piece {p}: {keep.sum():>9,} particles   "
                    f"non-rigid residual mean {r.mean():5.3f} cells, 95th {np.percentile(r,95):5.3f}")
            if sk.any() and (~sk).any():
                line += (f"   skin {r[sk].mean():5.3f}   interior {r[~sk].mean():5.3f}"
                         f"   ratio {r[~sk].mean()/max(r[sk].mean(),1e-9):4.2f}x")
            print(line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
