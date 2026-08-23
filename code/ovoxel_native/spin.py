"""The axis a solid turns about, found by how well it covers itself when turned.

Rotational self-overlap is a far stronger signal than the azimuthal spread of the rim radius: on a
near-spherical object every direction has a similar rim, but only the true axis leaves the volume
where it was.  The four objects whose stored axis is known to be right are the check -- a method
that cannot recover those has no business correcting the other two.
"""
import sys
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
W = "/workspace/ovoxel_native"

ANG = np.radians([40.0, 80.0, 140.0, 200.0, 260.0, 320.0])


def rodrigues(a, t):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def fib(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - i / n)          # the upper half only; an axis and its negation are one axis
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], 1)


def overlap(cells, axis, occ, lo, shape):
    """Of the solid's cells, the fraction still inside the solid after turning about `axis`."""
    c = cells - cells.mean(0)
    s = 0.0
    for t in ANG:
        q = np.rint(c @ rodrigues(axis, t).T + cells.mean(0)).astype(np.int64) - lo
        ok = ((q >= 0) & (q < shape)).all(1)
        idx = q[ok]
        s += occ[idx[:, 0], idx[:, 1], idx[:, 2]].sum() / len(c)
    return s / len(ANG)


for OBJ in ("watermelon_sp", "orange_sp", "pomegranate2_sp", "doughnut",
            "apple1_sp", "cake2_sp", "bread_sp"):
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location="cpu", weights_only=False)
    C = np.load(f"{W}/cams_{OBJ}_bal.npz")
    cells = st["solid"].cpu().numpy().astype(np.int64)
    lo, hi = cells.min(0), cells.max(0)
    shape = hi - lo + 1
    occ = np.zeros(shape, bool)
    occ[cells[:, 0] - lo[0], cells[:, 1] - lo[1], cells[:, 2] - lo[2]] = True
    if len(cells) > 120000:                      # a fixed budget, so every object costs the same
        cells = cells[np.random.default_rng(0).choice(len(cells), 120000, replace=False)]

    cand = fib(240)
    sc = np.array([overlap(cells, a, occ, lo, shape) for a in cand])
    a0 = cand[sc.argmax()]
    for step in (0.16, 0.06, 0.02):              # refine around the best direction
        loc = a0 + step * np.random.default_rng(1).normal(size=(60, 3))
        loc /= np.linalg.norm(loc, axis=1, keepdims=True)
        s2 = np.array([overlap(cells, a, occ, lo, shape) for a in loc])
        if s2.max() > sc.max():
            a0, sc = loc[s2.argmax()], s2
    stored = np.asarray(C["h_planes"][0, :3], float)
    stored /= np.linalg.norm(stored)
    off = np.degrees(np.arccos(min(1.0, abs(float(a0 @ stored)))))
    print(f"{OBJ:16s} best overlap {sc.max():.3f} (stored axis {overlap(cells, stored, occ, lo, shape):.3f})"
          f"   {off:5.1f} deg apart   axis [{a0[0]:+.3f} {a0[1]:+.3f} {a0[2]:+.3f}]")
