"""Step 5: the six-object comparison figure.

One held-out plane per object: a photograph of a real cut of that kind, this work, and the released
model the lattice was quantised from. The references come from each object's own EVAL_REF rather
than a hard-coded directory, so the three objects whose photographs were moved off a dark backdrop
show the set that actually supervised and scored them. The renders come from evalw_*, the batch
step 4 produces. Nothing is retrained and nothing is selected: the plane index is the same for
every row and both arms.

    /usr/bin/python3 six/figure.py [out.png]
"""
import glob, os, re, sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.environ.get("FN_ROOT", "/workspace/fn_voxel")
S = 260
OBJ = ["orange", "watermelon", "apple", "pomegranate", "bread", "cake"]
PLANE = "rh3"
COLS = ["a real cut of that kind", "this work", "FruitNinja, released"]


def eval_ref(obj):
    """The reference directory the object is scored against, comment stripped.

    Two confs carry a trailing comment on this line. Taking the whole line as a path made glob
    return nothing, and the blank panel that produced still passed a white-corner check.
    """
    txt = open(os.path.join(ROOT, "method/objects", f"{obj}.conf")).read()
    return re.search(r"^EVAL_REF=(.+)$", txt, re.M).group(1).split("#")[0].strip()


def load(p):
    return Image.open(p).convert("RGB").resize((S, S), Image.LANCZOS)


def first_photo(d):
    ps = [p for p in sorted(glob.glob(os.path.join(ROOT, d, "*.png"))) if "depth" not in p]
    assert ps, f"no references at {d}"
    return load(ps[0]), len(ps)


def cut(run):
    p = os.path.join(ROOT, run, f"{PLANE}_init_0.png")
    if not os.path.exists(p):
        g = sorted(glob.glob(os.path.join(ROOT, run, "rh*_init_0.png")))
        assert g, f"no renders in {run}; run six/eval.sh first"
        p = g[0]
    return load(p)


def label(text, w, scale=3):
    """Draw small and scale up, so the default bitmap font reads at figure size.

    The box is sized to the text, not to the column: at a fixed w // scale the two longest headings
    lost their last characters, and "FruitNinja, released" reached the page as "FruitNinja, release".
    """
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = max(int(probe.textlength(text)) + 4, w // scale)
    small = Image.new("RGB", (box, 13), "white")
    ImageDraw.Draw(small).text((2, 1), text, fill=(20, 20, 20))
    return small.resize((box * scale, 13 * scale), Image.NEAREST)


def main(out):
    W = 3 * S + 2 * 14 + 130 + 90       # the right margin is room for the last heading
    H = len(OBJ) * (S + 12) + 46
    sheet = Image.new("RGB", (W, H), "white")
    for c, name in enumerate(COLS):
        sheet.paste(label(name, S, scale=2), (130 + c * (S + 14), 12))
    for r, obj in enumerate(OBJ):
        d = eval_ref(obj)
        photo, n = first_photo(d)
        y = 46 + r * (S + 12)
        sheet.paste(label(obj, 126, 2), (2, y + S // 2 - 13))
        for c, im in enumerate((photo, cut(f"evalw_{obj}"), cut(f"evalw_{obj}_base"))):
            sheet.paste(im, (130 + c * (S + 14), y))
        a = np.asarray(photo, np.float32) / 255
        corner = np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3)]).mean(0)
        flag = "   <-- not on white" if corner.mean() < 0.9 else ""
        print(f"  {obj:12s} {n:2d} refs at {d:44s} corner {corner.round(2)}{flag}")
    sheet.save(out)
    print("->", out, sheet.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "six_objects.png"))
