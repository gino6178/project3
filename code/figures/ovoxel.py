"""M5: the cut patch becomes an O-Voxel surface, carrying the interior's colour.

The spec's section 6.3. M4 produced an exact planar mesh for the newly exposed cross-section;
this hands it to TRELLIS.2's official `mesh_to_flexible_dual_grid` and writes the interior
appearance onto the result. Nothing is reimplemented -- the spec is explicit that the geometry
encoder is not ours to rewrite -- so this file is the conversion, the attribute transfer and the
tests that say the colour survived.

Only the new patch is converted, never the whole object. That is the spec's engineering boundary
in section 6.3: the original exterior representation stays as it is, so a cut costs a local
surface and not a global remesh.

Loading the library needs a note, because the obvious way does not work. `import o_voxel` runs a
package __init__ that eagerly imports `postprocess`, which wants `flex_gemm` (a CUDA extension
from git) and `nvdiffrast`. Neither is on any path a cut surface takes. Stubbing them turned into
a queue of one missing dependency after another, so instead the compiled kernel and the official
wrapper file are loaded directly, by path, without running the package __init__. The wrapper is
the library's own; only the import ceremony is skipped.

The environment is the other constraint worth stating plainly: the extension is built for
CPython 3.11, so this runs under the `cube_ovoxel` environment rather than `fruitninja`, and it
is not on the remote box at all.

    /home/gino/miniconda3/envs/cube_ovoxel/bin/python method/common/cube/ovoxel.py
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
TRELLIS = _os.environ.get("TRELLIS2_ROOT", "/home/gino/project/TRELLIS.2")

import glob
import importlib.util
import sys
import types

import numpy as np
import torch

sys.path += [_FN_ROOT]

import cutmesh as cm                        # noqa: E402
import subdivide as sd                      # noqa: E402


def load_convert():
    """TRELLIS.2's mesh_to_flexible_dual_grid, without its package's unrelated imports."""
    if "o_voxel.convert.flexible_dual_grid" in sys.modules:
        return sys.modules["o_voxel.convert.flexible_dual_grid"]
    # Both the compiled extension and the wrapper have to be in the same directory, and which
    # directory that is depends on how it was built: a full `setup.py build` fills build/lib.*
    # with both, while `build_ext --inplace` puts only the .so back into the source tree. Asking
    # for the .so alone picked the build directory on the remote box, where the wrapper is not.
    base = glob.glob(_os.path.join(TRELLIS, "o-voxel", "build", "lib.*", "o_voxel"))
    base += [_os.path.join(TRELLIS, "o-voxel", "o_voxel")]
    base = [b for b in base if glob.glob(_os.path.join(b, "_C.*.so"))
            and _os.path.exists(_os.path.join(b, "convert", "flexible_dual_grid.py"))]
    if not base:
        raise SystemExit(f"no built o_voxel under {TRELLIS}/o-voxel -- build it first")
    b = base[0]

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    pkg = types.ModuleType("o_voxel"); pkg.__path__ = [b]
    sys.modules["o_voxel"] = pkg
    sub = types.ModuleType("o_voxel.convert"); sub.__path__ = [_os.path.join(b, "convert")]
    sys.modules["o_voxel.convert"] = sub
    # torch first: the extension links against its libc10
    pkg._C = _load("o_voxel._C", glob.glob(_os.path.join(b, "_C.*.so"))[0])
    return _load("o_voxel.convert.flexible_dual_grid",
                 _os.path.join(b, "convert", "flexible_dual_grid.py"))


def patch_to_o_voxel(V, F, voxel_size, colour=None, pad=None, device="cpu"):
    """One cut patch to a Flexible Dual Grid, with a colour per active voxel.

    `voxel_size` is the spec's "target surface resolution", and h_target is the value that makes
    the new surface match the exterior it meets. `colour` is a callable from points to RGB --
    in the pipeline that is the decoder applied to the coarse cell a point falls in, which is
    (25) composed with (19).

    The AABB is padded by a voxel so the patch's own boundary is inside the grid rather than on
    it; without that the rim voxels lose the edge intersections that define them.
    """
    fdg = load_convert()
    Vt = torch.as_tensor(np.asarray(V), dtype=torch.float32, device=device)
    Ft = torch.as_tensor(np.asarray(F), dtype=torch.int32, device=device)
    pad = voxel_size * 2 if pad is None else pad
    lo = Vt.min(0).values - pad
    hi = Vt.max(0).values + pad
    aabb = torch.stack([lo, hi]).to(device)

    vi, dv, inter = fdg.mesh_to_flexible_dual_grid(Vt, Ft, voxel_size=float(voxel_size),
                                                  aabb=aabb)
    # A dual vertex is where the surface sits inside its voxel, so it is the point to ask the
    # interior about -- not the voxel centre, which for a flat patch is up to half a voxel off
    # the surface and can fall on the wrong side of a material boundary.
    #
    # The two halves of the library disagree about what a dual vertex is, and it has to be
    # measured rather than read off. `mesh_to_flexible_dual_grid` returns it already scaled and
    # relative to aabb[0], so a world position is lo + dual. Its own inverse,
    # `flexible_dual_grid_to_mesh`, instead expects a fraction within the voxel and composes
    # `(coords + dual) * voxel_size + aabb[0]`. Both compositions are wrong for this output --
    # one gave a patch half the width of the cut, the other half again too wide, each with a
    # constant half-voxel offset off a plane every vertex is supposed to lie on. On the cut
    # disc, lo + dual puts |z| at exactly 0 and the extent at 23.95 against the mesh's 24.
    #
    # `frac` is the same vertex in the inverse's convention, so a patch can be handed back to
    # the library without the caller having to know any of this.
    pos = lo.cpu().numpy() + dv.cpu().numpy()
    frac = dv.cpu().numpy() / float(voxel_size) - vi.cpu().numpy()
    rgb = None if colour is None else np.asarray(colour(pos), np.float32)
    return dict(voxel=vi.cpu().numpy(), dual=dv.cpu().numpy(), frac=frac,
                inter=inter.cpu().numpy(), pos=pos, rgb=rgb,
                origin=lo.cpu().numpy(), voxel_size=float(voxel_size))


def cut_to_o_voxel(coords, h, n, d, h_target, colour=None, device="cpu"):
    """M3 and M4, then one O-Voxel patch per piece -- the spec's 6.3 step 6.

    Per piece rather than one patch for the whole plane, because a patch is bound to the piece
    it belongs to: the two halves move independently afterwards, and a surface shared between
    them would have to be split the first time they do.
    """
    m = cm.cut_mesh(coords, h, n, d, h_target)
    out = {}
    for k in range(m["stats"]["pieces"]):
        sel = m["piece"] == k
        if not sel.any():
            continue
        F = m["F"][sel]
        used, F2 = np.unique(F, return_inverse=True)
        out[k] = patch_to_o_voxel(m["V"][used], F2.reshape(-1, 3), h_target,
                                  colour=colour, device=device)
        out[k]["faces"] = int(sel.sum())
    return m, out


def _selftest():
    bad = 0
    h, hf = 1.0, 0.5
    ball = sd._ball(12)

    # a colour field with a known value everywhere, so "the colour arrived" is checkable rather
    # than merely plausible: red runs along x, green along y, blue is constant
    def colour(p):
        q = (p - p.min(0)) / np.ptp(p, 0).clip(1e-9)
        return np.stack([q[:, 0], q[:, 1], np.full(len(q), 0.5)], 1)

    m, patches = cut_to_o_voxel(ball, h, (0, 0, 1), 0.0, hf, colour=colour)
    ok = len(patches) == 2
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} one patch per piece: {len(patches)} patches for "
          f"{m['stats']['pieces']} pieces")

    for k, p in patches.items():
        print(f"      piece {k}: {len(p['voxel']):,} active voxels from {p['faces']:,} faces, "
              f"dual vertices {p['dual'].shape}, intersect flags {p['inter'].shape}")

    p = patches[0]
    # the dual vertices must lie on the cut plane -- the patch is planar, so the grid's idea of
    # where the surface is should agree to well within a voxel
    off = np.abs(p["pos"][:, 2])
    ok = float(off.max()) <= 1e-5
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} dual vertices sit on the cut plane "
          f"(worst {off.max():.2e}, voxel {hf})")
    fr = p["frac"]
    ok2 = bool((fr >= -1e-4).all() and (fr <= 1 + 1e-4).all())
    bad += not ok2
    print(f"  {'ok ' if ok2 else 'FAIL'} each dual vertex lies inside its own voxel in the "
          f"inverse's convention (frac in [{fr.min():.3f}, {fr.max():.3f}])")

    # the colour written to a voxel is the colour of the field where that voxel's surface is
    want = colour(p["pos"])
    err = np.abs(p["rgb"] - want).max()
    ok = err < 1e-6
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} every voxel carries the interior colour at its own "
          f"dual vertex (worst error {err:.2e})")

    # and the colour actually varies, or the check above would pass on a constant
    rng = p["rgb"].max(0) - p["rgb"].min(0)
    ok = rng[0] > 0.9 and rng[1] > 0.9 and rng[2] < 1e-6
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the written colour varies as the field does "
          f"(ranges r {rng[0]:.3f} g {rng[1]:.3f} b {rng[2]:.3f}; expected ~1, ~1, 0)")

    # coverage: the patch's voxels should cover the cut disc, so their extent matches the mesh's
    ext_v = p["pos"][:, :2].max(0) - p["pos"][:, :2].min(0)
    ext_m = m["V"][:, :2].max(0) - m["V"][:, :2].min(0)
    rel = float(np.abs(ext_v - ext_m).max() / ext_m.max())
    ok = rel < 0.05
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} the patch covers the cut face "
          f"(extent {ext_v.round(2)} against the mesh's {ext_m.round(2)})")

    # the two patches are different objects bound to different pieces
    ok = patches[0]["faces"] == patches[1]["faces"]
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'} both pieces carry a patch of their own "
          f"({patches[0]['faces']:,} and {patches[1]['faces']:,} faces)")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_selftest())

    # a real lattice, coloured from what the model actually stores
    from plyfile import PlyData
    from occupancy import close_and_fill, to_grid

    ld = sys.argv[1]
    lat = torch.load(_os.path.join(ld, "lattice.pt"))
    el = PlyData.read(_os.path.join(ld, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    C0 = 0.28209479177387814
    dc = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32) * C0 + 0.5
    lvl = torch.load(_os.path.join(ld, "cell_level.pt")).reshape(-1)
    keep = (lvl[:len(xyz)] == 0).numpy()
    p, pc = xyz[keep], np.clip(dc[keep], 0, 1)
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])

    org = p.min(0)
    raw = np.floor((p - org) / hc).astype(np.int64)
    coords, first = np.unique(raw, axis=0, return_index=True)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1

    # the colour of the coarse cell a point falls in; cells the fill added have no colour of
    # their own, so they take the nearest occupied cell's, which is what (19) does for a child
    from scipy.spatial import cKDTree
    tree = cKDTree((coords + 0.5) * hc)
    cellcol = pc[first]

    def colour(q):
        _, j = tree.query(q - org, k=1)
        return cellcol[j]

    c = (solid + 0.5) * hc
    n = [0.0, 1.0, 0.0]
    d = float(-c.mean(0) @ np.array(n))
    m, patches = cut_to_o_voxel(solid, hc, n, d, hf, colour=lambda q: colour(q))
    print(f"  {len(solid):,} solid cells, h_target {hf:.5f}")
    for k, pt in patches.items():
        print(f"    piece {k}: {pt['faces']:,} cut faces -> {len(pt['voxel']):,} O-Voxels, "
              f"mean RGB {pt['rgb'].mean(0).round(3)}")


def dual_to_mesh(patch, device="cuda"):
    """The dual grid back to a surface, through the library's own inverse.

    This is what a renderer should be given. Splatting one point per active voxel leaves the
    background showing wherever the projected voxel spacing exceeds a pixel -- measured on the
    orange's exterior, 5.6% to 6.7% of the silhouette, in 2,400 to 3,300 separate gaps with a
    median size of one pixel. That is a sampling artefact of the viewer and not a hole in the
    surface, and a rasterised mesh cannot have it: the quads tile.

    `frac` is why patch_to_o_voxel keeps it. The forward call returns dual vertices already
    scaled and relative to the AABB; the inverse wants a fraction within the voxel and composes
    `(coords + dual) * voxel_size + aabb[0]`.
    """
    fdg = load_convert()
    coords = torch.as_tensor(patch["voxel"], dtype=torch.int32, device=device)
    dual = torch.as_tensor(patch["frac"], dtype=torch.float32, device=device)
    inter = torch.as_tensor(patch["inter"], dtype=torch.bool, device=device)
    vs = float(patch["voxel_size"])
    lo = torch.as_tensor(patch["origin"], dtype=torch.float32, device=device)
    hi = lo + torch.as_tensor((patch["voxel"].max(0) + 2) * vs, dtype=torch.float32,
                              device=device)
    V, F = fdg.flexible_dual_grid_to_mesh(coords, dual, inter, None,
                                          torch.stack([lo, hi]), voxel_size=vs)
    return V.detach().cpu().numpy(), F.detach().cpu().numpy()
