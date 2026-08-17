"""The generated interiors, side by side, on terms none of them chose.

Comparing what three methods put inside an object is awkward because each lives in its own frame,
at its own scale, with its own camera conventions, and any comparison routed through one method's
renderer flatters that method. So this routes through none of them.

For each model: take the plane through its own centroid with a chosen normal, keep the primitives
within a slab of half-width `t` of it expressed as a fraction of the model's own extent, and
project them orthographically along that normal onto a square sized to the model's own silhouette.
Every step is stated relative to the object, so the only thing being compared is what the interior
of each looks like at the same relative depth and the same relative slab thickness.

Nearest-first painting, not alpha blending. All three of these models are opaque -- median opacity
1.0000 and 10th percentile 1.0000 for both published ones -- so a cut face is whatever primitive
is nearest, and compositing them any other way would be inventing a translucency none of them has.

    python method/common/eval/slab_compare.py OUT.png "name=model.ply@footprint" ...

The footprint is the primitive's own extent in world units -- sigma for a Gaussian, h for a cell.
Omit it and every primitive is one pixel, which measures how many landed in the slab.
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

C0 = 0.28209479177387814
SIZE = int(_os.environ.get("SLAB_SIZE", "460"))
THICK = float(_os.environ.get("SLAB_THICK", "0.01"))     # of the model's own extent


def slab(path, axis=1, size=SIZE, thick=THICK, footprint=None):
    """`footprint` is each primitive's own extent in world units, or None for one pixel.

    One pixel per primitive measures primitive count, not the representation, and it is unfair in
    the direction that matters here: a cube cell fills h^3 of space by construction and a Gaussian
    at sigma = 1e-7 fills nothing, and painting both as a point throws away exactly that
    difference. Given a footprint each primitive covers the area it actually claims, and the
    unpainted fraction becomes a statement about the representation rather than about how many
    samples happened to land in the slab.
    """
    from plyfile import PlyData
    el = PlyData.read(path).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                  * C0 + 0.5, 0, 1)
    ext = float((xyz.max(0) - xyz.min(0)).max())
    c = xyz.mean(0)
    n = np.zeros(3)
    n[axis] = 1.0
    s = (xyz - c) @ n
    keep = np.abs(s) <= thick * ext
    if not keep.any():
        return None, 0, ext
    # the two axes that are not the normal, so the projection is orthographic and axis-aligned
    o = [a for a in range(3) if a != axis]
    p = (xyz[keep] - c)[:, o]
    # Normalised by the object's own projected radius over the *whole* model, not over the slab
    # and not by the bounding box. Over the slab is wrong because a slab one cell thick misses the
    # cells where the surface curves away, so a coarse representation would be scaled up until its
    # rind left the frame; by the bounding box is wrong because one of these models carries more
    # than the fruit. The whole model's projection is the object, at the same scale for everyone.
    half = 1.05 * float(np.abs((xyz - c)[:, o]).max())
    u = ((p[:, 0] / half) * 0.5 + 0.5) * (size - 1)
    v = (0.5 - (p[:, 1] / half) * 0.5) * (size - 1)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v = u[ok].astype(np.int64), v[ok].astype(np.int64)
    col = rgb[keep][ok]
    dep = np.abs(s[keep][ok])
    img = np.ones((size, size, 3), np.float32)
    # nearest to the plane wins, which is what an opaque cut face is
    order = np.argsort(-dep)
    u, v, col = u[order], v[order], col[order]
    if footprint is None:
        img[v, u] = col
    else:
        import cv2
        r_px = max(0, int(round(0.5 * footprint / half * 0.5 * (size - 1))))
        if r_px <= 0:
            img[v, u] = col
        else:
            for i in range(len(u)):
                cv2.circle(img, (int(u[i]), int(v[i])), r_px,
                           (float(col[i, 0]), float(col[i, 1]), float(col[i, 2])), -1)
    return img, int(keep.sum()), ext


def main(out_png, *specs):
    import cv2
    tiles = []
    for spec in specs:
        parts = spec.split("=", 1)
        name, path = parts[0], parts[1]
        fp = None
        if "@" in path:
            path, fps = path.rsplit("@", 1)
            fp = float(fps)
        img, n, ext = slab(path, footprint=fp)
        if img is None:
            print(f"  {name}: nothing in the slab"); continue
        a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        pad = np.full((30, a.shape[1], 3), 255, np.uint8)
        cv2.putText(pad, name, (6, 21), cv2.FONT_HERSHEY_DUPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
        painted = img.min(2) < 0.97
        # the gaps the design document asks about: holes *inside* the section, not the background.
        # The silhouette is the painted region's convex fill, so an unpainted pixel inside it is a
        # place the representation had nothing to put.
        fill = cv2.morphologyEx(painted.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sil = np.zeros_like(fill)
        if cnts:
            cv2.drawContours(sil, [cv2.convexHull(np.vstack(cnts))], -1, 1, -1)
        inside = sil > 0
        gaps = 100.0 * float((inside & ~painted).sum()) / max(int(inside.sum()), 1)
        sub = np.full((30, a.shape[1], 3), 255, np.uint8)
        cv2.putText(sub, f"{n:,} in the slab", (6, 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
        # The unpainted fraction is deliberately not drawn on the tile any more. It was reported
        # from this figure as cut-face fidelity and that was wrong: a Gaussian is drawn here at
        # radius sigma/2 and no rasteriser draws it that way, so at sigma = 1e-7 the radius rounds
        # to zero and the primitive becomes one pixel. Leaving the number burned into the image
        # meant the figure went on asserting a claim the text had retracted. It is still printed
        # to stdout, where it answers the question this figure does ask -- what a primitive covers.
        if _os.environ.get("SLAB_DRAW_GAPS", "0") == "1":
            cv2.putText(sub, f"{gaps:.1f}% of the section unpainted", (6, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
        tiles.append(np.vstack([pad, a, sub]))
        print(f"  {name:<22} {n:>10,} in the slab, extent {ext:.3f}, "
              f"{gaps:5.1f}% of the section unpainted")
    if tiles:
        cv2.imwrite(out_png, np.hstack(tiles)[:, :, ::-1])
        print(f"  -> {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
