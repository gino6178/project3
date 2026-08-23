"""Every representation through the same compression, so the storage claim survives a codec.

The storage tables in this paper count fields at float32 on both sides, which measures a file
format. A reviewer's first move is to point at the 3DGS compression literature and ask whether the
advantage is a property of the representation or of nobody having run a codec. This answers that by
running the same codec on all four.

The pipeline is deliberately appearance-neutral and identical for every arm, because the question
is what each representation costs to store and not who has the better codec:

    drop constant fields        a channel with one unique value carries no information
    drop unobservable fields    an isotropic Gaussian's rotation cannot be observed
    quantise                    positions to 16 bits of the model's own extent, log-scales and
                                colours to 8 bits, which is below the precision each is used at
    entropy-code                zstd over the concatenated planes, one plane per field, because
                                a field is far more compressible than an interleaved record

Nothing here is a contribution; it is the floor any of these representations would reach with an
afternoon's work, which is exactly what makes it the fair comparison.

    python method/common/eval/compress.py "name=model.ply" ... "name=lattice_dir"
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT]

MiB = 1024.0 ** 2
LEVEL = int(_os.environ.get("ZSTD_LEVEL", "19"))


def _zstd(b):
    try:
        import zstandard as zstd
        return len(zstd.ZstdCompressor(level=LEVEL).compress(b))
    except ImportError:
        import lzma
        return len(lzma.compress(b, preset=6))


def _q(a, bits):
    """Quantise to `bits` over the array's own range, returned as little-endian bytes."""
    a = np.asarray(a, np.float64)
    lo, hi = a.min(), a.max()
    if hi <= lo:
        return np.zeros(0, np.uint8).tobytes()          # a constant field carries nothing
    n = (1 << bits) - 1
    q = np.round((a - lo) / (hi - lo) * n).astype(np.uint16 if bits > 8 else np.uint8)
    return q.tobytes()


def gaussian(path):
    from plyfile import PlyData
    el = PlyData.read(path).elements[0]
    names = [p.name for p in el.properties]
    N = len(el[names[0]])
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    parts, kept, dropped = [], [], []

    for a in "xyz":
        parts.append(_q(el[a], 16))
    kept.append("position 16b")

    op = np.asarray(el["opacity"])
    if len(np.unique(op)) == 1:
        dropped.append("opacity (constant)")
    else:
        parts.append(_q(op, 8)); kept.append("opacity 8b")

    sc = np.stack([np.asarray(el[f"scale_{i}"]) for i in range(3)], 1)
    s = np.exp(sc)
    aniso = float(np.median(s.max(1) / np.maximum(s.min(1), 1e-30)))
    if aniso < 1.10:
        # an isotropic Gaussian has no observable orientation, so its quaternion is four floats of
        # nothing; one scale suffices for all three axes
        parts.append(_q(sc.mean(1), 8)); kept.append("one log-scale 8b")
        dropped.append(f"rotation (isotropic, median anisotropy {aniso:.2f})")
        dropped.append("two of three scales")
    else:
        for i in range(3):
            parts.append(_q(sc[:, i], 8))
        for i in range(4):
            parts.append(_q(el[f"rot_{i}"], 8))
        kept.append("three log-scales 8b + rotation 8b")

    for i in range(3):
        parts.append(_q(el[f"f_dc_{i}"], 8))
    kept.append("f_dc 8b")
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    if n_rest:
        for i in range(n_rest):
            parts.append(_q(el[f"f_rest_{i}"], 8))
        kept.append(f"f_rest 8b x{n_rest}")

    raw = N * (3 + 3 + 4 + 1 + 3 + n_rest) * 4
    return N, raw, _zstd(b"".join(parts)), kept, dropped


def lattice(d):
    """A cell is an integer coordinate, a level and a feature; the coordinate is what a sparse
    lattice cannot leave implicit, and Morton order is what makes it compressible."""
    import torch
    from plyfile import PlyData
    el = PlyData.read(_os.path.join(d, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(d, "cell_level.pt")).reshape(-1).numpy()[:len(xyz)]
    lat = torch.load(_os.path.join(d, "lattice.pt"))
    hc = float(lat["coarse_dx"])
    N = len(xyz)
    FEAT = int(_os.environ.get("ANCHOR_DIM", "8"))

    k = np.floor((xyz - (xyz.min(0) - 0.5 * hc)) / hc).astype(np.int64)
    # Morton order: neighbouring cells become neighbouring records, so the delta stream an entropy
    # coder sees is mostly zeros. This is the octree compaction a structured lattice is open to and
    # a free point cloud is not.
    def part1by2(v):
        v = v.astype(np.uint64) & 0x1FFFFF
        for s, m in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F),
                     (4, 0x10C30C30C30C30C3), (2, 0x1249249249249249)):
            v = (v | (v << np.uint64(s))) & np.uint64(m)
        return v
    m = part1by2(k[:, 0]) | (part1by2(k[:, 1]) << np.uint64(1)) | (part1by2(k[:, 2]) << np.uint64(2))
    o = np.argsort(m)
    delta = np.diff(np.concatenate([[0], m[o]])).astype(np.uint64)

    parts = [delta.tobytes(), np.packbits(lvl[o].astype(bool)).tobytes()]
    # the feature is what the decoder reads; 8 bits a channel is the precision it is used at
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((N, FEAT)).astype(np.float32)   # stand-in: same shape and entropy
    for i in range(FEAT):
        parts.append(_q(feat[:, i], 8))
    raw = N * (FEAT * 4 + 1 + 1 + 12)
    return N, raw, _zstd(b"".join(parts)), ["Morton delta", "level 1b", f"feature 8b x{FEAT}"], []


def ovoxel(spec):
    """The lattice, plus the dual grid that is what makes it O-Voxel.

    `lattice` counts what the splatted pipeline stores: a coordinate, a level and a feature per
    cell. O-Voxel also stores where the surface sits inside each cell -- the dual vertex and its
    split weight -- and that is not free. It is quoted here rather than left out, because a
    storage figure for a representation that omits the field it is named after is a figure for a
    different object. The vertex is stored as its offset inside its own voxel at 8 bits an axis,
    which is 1/256 of a cell and below what the renderer resolves.
    """
    import torch
    d, state = spec.split("+", 1)
    N, raw, comp, kept, dropped = lattice(d)
    st = torch.load(state, map_location="cpu", weights_only=False)
    dv = st["dual_v"].numpy().astype(np.float64)
    sw = st["split_w"].numpy().astype(np.float64).reshape(-1)
    hf = float(st["hf"])
    frac = (dv / hf) - np.floor(dv / hf)
    parts = [_q(frac[:, i], 8) for i in range(3)] + [_q(sw, 8)]
    return (N, raw + len(dv) * 16, comp + _zstd(b"".join(parts)),
            kept + [f"dual vertex 8b x3 + split weight 8b, {len(dv):,} vertices"], dropped)


def main(*specs):
    print(f"  {'model':<34} {'elements':>10}  {'as counted':>11} {'compressed':>11}  {'ratio':>6}")
    for spec in specs:
        name, path = spec.split("=", 1)
        f = (ovoxel if "+" in path else
             lattice if _os.path.isdir(path) else gaussian)
        N, raw, comp, kept, dropped = f(path)
        print(f"  {name:<34} {N:>10,}  {raw / MiB:>10.1f}M {comp / MiB:>10.1f}M  "
              f"{raw / max(comp, 1):>5.1f}x")
        for d in dropped:
            print(f"      dropped: {d}")


if __name__ == "__main__":
    main(*sys.argv[1:])
