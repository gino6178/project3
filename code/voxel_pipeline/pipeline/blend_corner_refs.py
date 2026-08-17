"""The eight corner references, blended from the six faces rather than sampled.

Fourteen directions is what `init_skin_cube` was written for and six is what it has been given,
so 52.5% of directions fall inside a single cone and every one of those cones is bounded by an
edge where the weight steps -- the seams visible as a pale box around the sphere.

Sampling the eight corners fixes the geometry and breaks something else, which is why they
were turned off: a shared seed returns the same picture for every direction, and a per-face
seed returns a different fruit, measured as RMS 0.006-0.011 rising to 0.133-0.188 with one
face green-and-red and another flat vermilion. Neither is eight more views of one orange.

Blending gives the geometry without the risk. A corner direction lies between exactly three
faces, all three at 54.7 degrees from it, and its reference is their cosine-weighted mean. It
introduces no content -- every pixel comes from images already accepted -- so the fourteen
directions are guaranteed to be the same fruit, which is the property the sampled version
could not offer. What it adds is coverage: the worst-case angle from any direction to the
nearest reference axis falls from 54.7 to 37.4 degrees, and every point is then read from
three or four references instead of one.

Pixel-wise blending is only meaningful for images that register with each other, which is
exactly the case here: `cube_prep` centres each reference on its own silhouette, and a sphere
presents the same outline from every direction.

    python voxel_pipeline/pipeline/blend_corner_refs.py cube_or3_prep cube_or14b_prep
"""
import os
import sys

import numpy as np
from PIL import Image

CE = 35.264
FACES = [("up", 0, 90), ("down", 0, -90), ("front", 0, 0),
         ("right", 90, 0), ("back", 180, 0), ("left", 270, 0)]
CORNERS = [(f"c{i}", 45 + 90 * (i % 4), CE if i < 4 else -CE) for i in range(8)]


def axis(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])


def main(src_dir, out_dir, sharp=2.0):
    os.makedirs(out_dir, exist_ok=True)
    imgs, axes = {}, {}
    for name, az, el in FACES:
        p = os.path.join(src_dir, f"{name}_ref.png")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        imgs[name] = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.
        axes[name] = axis(az, el)
        Image.fromarray((imgs[name] * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{name}_ref.png"))
    shape = next(iter(imgs.values())).shape
    if any(v.shape != shape for v in imgs.values()):
        raise SystemExit("the six references are not the same size")

    for name, az, el in CORNERS:
        u = axis(az, el)
        # the three faces this direction sits between, all at the same angle from it
        d = {n: float(u @ a) for n, a in axes.items()}
        near = sorted(d, key=lambda n: -d[n])[:3]
        w = np.array([max(d[n], 0.0) ** sharp for n in near])
        w = w / w.sum()
        out = sum(wi * imgs[n] for wi, n in zip(w, near))
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{name}_ref.png"))
        print(f"  {name:<4} az{az:>4} el{el:>8.3f}  <- " +
              "  ".join(f"{n} {wi:.3f}" for n, wi in zip(near, w)))
    print(f"-> {out_dir}  ({len(FACES)} faces + {len(CORNERS)} blended corners)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
