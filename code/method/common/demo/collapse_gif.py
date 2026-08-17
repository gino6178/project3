"""collapse.gif -- the exterior dual grid coarsening and refining again as tau sweeps 0 -> 1 -> 0.

`qef_views.py` already draws one frame per tolerance and puts the frames side by side. The sweep
is the same picture with the tolerance as time, and it answers what a contact sheet cannot: where
the collapse spends its nodes. Flat peel goes first and keeps going; the places that hold on to
their nodes as tau rises are the places the surface is actually bending. Nothing here decides
that -- `qef.collapse` does, on the residual of the merged quadric.

Two choices worth stating, because both are call-site choices and neither changes the algorithm:

  `pos0` is passed        A node no level ever merged already has the library's own vertex, and
                          `qef.collapse` puts it back when `pos0` is given. Without it the tau=0
                          frame is not the identity -- every vertex is re-solved from the coarser
                          triangle-based quadric and the surface shifts slightly before any
                          merging has happened. `qef.py` and `qef_mesh.py` both pass it;
                          `qef_views.py` does not, which is why its leftmost tile is not quite the
                          uniform grid. This follows the two that do.

  the sweep is mirrored   Frames are rendered once going up and replayed in reverse coming down,
                          so the loop is exact and costs half as many collapses.

Every number in a caption comes from the collapse that produced the frame: the node count and its
share of the uniform grid from `len(r["pos"])`, and the error from a KD query of those nodes
against the uniform grid's own dual vertices.

    python method/common/demo/collapse_gif.py OVOX.npz PLY CFG DEMO OUT.gif [n_up]
"""
import os as _os

_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import qef                                   # noqa: E402
from method.common.cube import qef_views                             # noqa: E402
from method.common.demo import gifcam                                # noqa: E402


def curve(frame, rows, i, n_uniform, height=78, pad=34):
    """The node count against tau, with a marker at the frame being shown.

    Drawn after the sweep rather than during it, because the axis has to be the same in every
    frame or the curve appears to move when only the marker should.
    """
    W = frame.shape[1]
    im = Image.new("RGB", (W, frame.shape[0] + height), "white")
    im.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(im)
    y0, y1 = frame.shape[0] + 16, frame.shape[0] + height - 16
    x0, x1 = pad, W - pad
    tau = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows], float)
    X = x0 + (tau - tau.min()) / max(tau.ptp(), 1e-9) * (x1 - x0)
    Y = y1 - n / float(n_uniform) * (y1 - y0)
    d.line([(x0, y1), (x1, y1)], fill=(170, 170, 170))
    d.line([(x0, y0), (x0, y1)], fill=(170, 170, 170))
    d.line([tuple(p) for p in np.stack([X, Y], 1)], fill=(200, 90, 20), width=2)
    d.ellipse([X[i] - 4, Y[i] - 4, X[i] + 4, Y[i] + 4], fill=(20, 20, 20))
    f = ImageFont.truetype(gifcam.FONT, 12)
    d.text((x0 - 2, y1 + 2), f"tau {tau.min():g}", font=f, fill=(90, 90, 90))
    d.text((x1 - 32, y1 + 2), f"{tau.max():g}", font=f, fill=(90, 90, 90))
    d.text((x0 + 4, y0 - 14), f"nodes, 0 to {n_uniform:,} (the uniform grid)", font=f,
           fill=(90, 90, 90))
    return np.asarray(im)


def main(npz, ply, cfg, demo, out_gif, n_up=41, size=460, ss=2):
    from scipy.spatial import cKDTree

    z = np.load(npz)
    voxel, pos0, rgb0 = z["voxel"].astype(np.int64), z["pos"].astype(np.float64), z["rgb"]
    h, origin = float(z["voxel_size"]), z["origin"].astype(np.float64)
    if "mesh_v" not in z.files:
        raise SystemExit(f"{npz} carries no mesh; rebuild it on a CUDA device")
    V, F = z["mesh_v"].astype(np.float64), z["mesh_f"].astype(np.int64)
    A6, b, c, w, gn, used = qef.quadrics_from_mesh(V, F, pos0, voxel, h, origin)
    print(f"  {len(voxel):,} active voxels, {len(F):,} triangles, "
          f"{used:,} of them landed in an active voxel, h {h:.5f}")

    tree0 = cKDTree(pos0)
    cam = gifcam.Cam(ply, cfg, demo, az=0.0, el=15.0)

    taus = np.linspace(0.0, 1.0, n_up)
    frames, rows = [], []
    for i, tau in enumerate(taus):
        t0 = time.time()
        r = qef.collapse(voxel, A6, b, c, w, h, origin, float(tau), g=gn, pos0=pos0)
        pts, owner, _ = qef_views.node_samples(r, None, h)
        col = rgb0[tree0.query(r["pos"], k=1)[1]][owner]
        d, _ = tree0.query(r["pos"], k=1)
        img, drawn = gifcam.splat(cam, pts, col, size, ss=ss)
        n = len(r["pos"])
        lines = [f"tau {tau:.3f}",
                 f"{n:,} nodes   {100.0 * n / len(voxel):.1f}% of uniform",
                 f"error {d.mean() / h:.3f} h mean, {np.percentile(d, 95) / h:.3f} h 95th",
                 f"{int(r['level'].max()) + 1} levels in use"]
        frames.append(np.asarray(gifcam.caption(img, lines)))
        rows.append((float(tau), n, float(d.mean() / h)))
        print(f"  tau {tau:.3f}  {n:>9,} nodes  {100.0 * n / len(voxel):>5.1f}%  "
              f"err {d.mean() / h:.3f} h  {len(pts):,} samples, {drawn:,} drawn  "
              f"[{time.time() - t0:.1f}s]")

    frames = [curve(f, rows, i, len(voxel)) for i, f in enumerate(frames)]
    # mirrored: up, then back down without repeating either end
    seq = frames + frames[-2:0:-1]
    gifcam.write_gif(out_gif, seq, duration=90)
    lo = min(rows, key=lambda q: q[1])
    print(f"  uniform grid {len(voxel):,} nodes; fewest at tau {lo[0]:.3f} with {lo[1]:,} "
          f"({100.0 * lo[1] / len(voxel):.1f}%), error {lo[2]:.3f} h")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         int(sys.argv[6]) if len(sys.argv) > 6 else 41)
