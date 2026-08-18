"""Six exterior references taken straight off a model, framed the way the skinner expects.

For an object whose exterior is already reconstructed there is nothing for a generator to add,
and a render of the model is a better reference than a picture of a different fruit. But
`init_skin_cube` addresses a reference in units of the *reference's own silhouette radius* and
assumes the fruit fills the frame -- its own comment says so, "whose fruit was rescaled to fill
its frame". A render does not: framed by the model's camera the fruit spans 0.73 of the frame
against the generated references' 0.97 to 1.00, so every cell near the edge of a cone reads past
the silhouette into the background. That is what the white rim and the arcs across the projected
sphere were.

So the render is cropped to its own silhouette, centred, and scaled to fill the frame. Nothing
about the appearance changes; only the units it is addressed in.

    python method/common/pipeline/ply_six_refs.py PLY CFG DEMO OUT_DIR [size]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import json
import sys

import numpy as np
from PIL import Image

sys.path += [_FN_ROOT, _os.environ.get("GS_ROOT", _FN_ROOT + "/gaussian-splatting")]

FACES = [("up", 0, 90), ("front", 0, 0), ("right", 90, 0),
         ("down", 0, -90), ("back", 180, 0), ("left", 270, 0)]


def fill_frame(img, tol=0.06, margin=1.01):
    """Crop to the silhouette and scale it isotropically, letterboxed.

    Aspect is preserved rather than stretched, and the projection is normalised by one radius to
    match, so the two conventions agree by construction. Stretching each axis to fill the frame
    was self-consistent too, but it makes a reference of a 1.41-by-0.37 silhouette four times too
    tall, and the height at which the icing meets the dough then lands in the wrong place on the
    tube. Keeping the aspect only works when the generated shape has the object's proportions --
    which is why the orange and the watermelon are ellipsoids with their own extents rather than
    spheres, and the doughnut a torus with its own.
    """
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    m = np.abs(a - 1).max(2) > tol
    ys, xs = np.nonzero(m)
    if not len(xs):
        return img
    S = img.size[0]
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    r = 0.5 * max(ys.max() - ys.min(), xs.max() - xs.min()) * margin
    box = tuple(int(round(v)) for v in (cx - r, cy - r, cx + r, cy + r))
    out = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    src = img.crop((max(0, box[0]), max(0, box[1]),
                    min(img.size[0], box[2]), min(img.size[1], box[3])))
    out.paste(src, (max(0, -box[0]), max(0, -box[1])))
    return out.resize((S, S), Image.LANCZOS)


def main(ply, cfg, demo, out_dir, size=512):
    from exterior_views import main as sheet
    _os.makedirs(out_dir, exist_ok=True)
    tmp = _os.path.join(out_dir, "_sheet.png")
    sheet(ply, cfg, demo, tmp, size)
    sh = Image.open(tmp).convert("RGB")
    S = sh.size[1] // 2
    for k, (name, az, el) in enumerate(FACES):
        r, c = divmod(k, 3)
        tile = sh.crop((c * S, r * S, (c + 1) * S, (r + 1) * S))
        fill_frame(tile).save(_os.path.join(out_dir, f"{name}_ref.png"))
    json.dump({n: [az, el] for n, az, el in FACES},
              open(_os.path.join(out_dir, "dirs.json"), "w"), indent=1)
    _os.remove(tmp)
    a = np.asarray(Image.open(_os.path.join(out_dir, "front_ref.png")).convert("RGB"),
                   np.float32) / 255
    m = np.abs(a - 1).max(2) > 0.06
    ys, xs = np.nonzero(m)
    print(f"  six references -> {out_dir}   silhouette now spans "
          f"{max(ys.max() - ys.min(), xs.max() - xs.min()) / a.shape[0]:.2f} of the frame")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 512)
