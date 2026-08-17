"""Step 1: write an object configuration for each released reconstruction.

FruitNinja releases six and this project originally reported on two of them. The other four have
photographs in the repository already, so the only thing standing between them and a run is a
configuration file. COARSE_DX is not guessed: the two objects that were tuned by hand landed at
129.3 and 120.4 cells across their own longest extent, so a new object is given L/125 and the
tuning that produced the first two becomes a rule.

Two of the four have references in one family only, and that is a property of the object rather
than an oversight. A cinnamon loaf's swirl exists on a transverse cut and becomes stripes
lengthwise; a layer cake's stripes exist on a longitudinal cut and become a single-colour disc
across. Those objects are configured with the family they have, and what the volume produces in
the family they lack is a prediction that can be checked by looking.

The orange and the watermelon are not written here. Their confs were tuned by hand before this
file existed and are the calibration the rule above is derived from; regenerating them would
overwrite the measurements this rule stands on.

    /usr/bin/python3 six/objects.py
"""
import glob, os

import numpy as np
from plyfile import PlyData

ROOT = os.environ.get("FN_ROOT", "/workspace/fn_voxel")
CELLS_ACROSS = 125.0

OBJ = {
    "apple": dict(h="data_finetune_images/apple/horizontal",
                  v="data_finetune_images/apple/vertical",
                  prompt="the {view_cut} cross-sectional view of an apple, pale cream flesh, "
                         "a five-pointed core of dark seeds, thin red skin, macro photo, detailed",
                  clip="a cross-section of an apple", score="fid"),
    "pomegranate": dict(h="data_finetune_images/pomegranate/horizontal",
                        v="data_finetune_images/pomegranate/vertical",
                        prompt="the {view_cut} cross-sectional view of a pomegranate, packed "
                               "translucent red arils in chambers, pale bitter membrane between "
                               "them, macro photo, detailed",
                        clip="a cross-section of a pomegranate", score="fid"),
    "bread": dict(h="data_finetune_images/bread", v="",
                  prompt="the {view_cut} cross-sectional view of a cinnamon swirl loaf, pale "
                         "open crumb, a dark cinnamon spiral, browned crust, macro photo",
                  clip="a slice of cinnamon swirl bread", score="fid"),
    "cake": dict(h="", v="data_finetune_images/cake",
                 prompt="the {view_cut} cross-sectional view of a red velvet layer cake, dark red "
                        "sponge layers separated by white cream, macro photo",
                 clip="a slice of red velvet cake", score="topology"),
}

TEMPLATE = """# {name}. Released by FruitNinja and not used by this project until now; the
# configuration is derived rather than tuned. COARSE_DX is the object's longest extent over {ca:.0f},
# which is where the orange and the watermelon landed when they were set by hand (129.3 and 120.4).
{note}SRC=prefilled/trained_gs/{name}.ply   # the released model, quantised as it is
COARSE_DX={dx:.5f}
SKIN_FRAC=0.95
CFG=config/orange_physics.json
DEMO=config/sphere_demo
REF_H={rh}
REF_V={rv}
ITERS=${{ITERS:-200}}
SCORE={score}
EVAL_REF={eh}
EVAL_REF_V={ev}
PROMPT="{prompt}"
CLIP_PROMPT="{clip}"
"""


def main():
    for name, o in OBJ.items():
        p = os.path.join(ROOT, "prefilled/trained_gs", f"{name}.ply")
        if not os.path.exists(p):
            print(f"{name:12s} model not uploaded yet, skipped")
            continue
        conf = os.path.join(ROOT, "method/objects", f"{name}.conf")
        if os.path.exists(conf) and "refs_white" in open(conf).read():
            print(f"{name:12s} conf already points at refs_white/, left alone "
                  f"(delete it to regenerate, then rerun six/prep.py)")
            continue
        el = PlyData.read(p).elements[0]
        xyz = np.stack([el["x"], el["y"], el["z"]], 1)
        L = float((xyz.max(0) - xyz.min(0)).max())
        dx = L / CELLS_ACROSS
        nh = len([q for q in glob.glob(os.path.join(ROOT, o["h"], "*.png")) if "depth" not in q]) if o["h"] else 0
        nv = len([q for q in glob.glob(os.path.join(ROOT, o["v"], "*.png")) if "depth" not in q]) if o["v"] else 0
        # Writing a conf that points at an empty directory produces a run that trains against
        # nothing and says so only as "refs 0h + 0v" in a line nobody reads.
        if nh + nv == 0:
            raise SystemExit(f"{name}: no photographs at {o['h'] or o['v']!r}. "
                             f"Run six/setup.sh first, and set FN_ROOT.")
        note = ""
        if not o["h"] or not o["v"]:
            fam = "transverse" if o["h"] else "longitudinal"
            note = (f"#\n# References exist for the {fam} family only, which is a property of this\n"
                    f"# object: its interior reads differently along the other axis, and no photograph\n"
                    f"# of that section is in the repository. What a cut of the other family produces\n"
                    f"# is therefore a prediction rather than a fit.\n")
        open(conf, "w").write(TEMPLATE.format(
            name=name, dx=dx, ca=CELLS_ACROSS, note=note, score=o["score"],
            rh=o["h"] or o["v"], rv=o["v"] or o["h"],
            eh=o["h"] or o["v"], ev=o["v"] or o["h"],
            prompt=o["prompt"], clip=o["clip"]))
        print(f"{name:12s} N={len(xyz):9,d}  L={L:.3f}  dx={dx:.5f}   refs {nh}h + {nv}v")


if __name__ == "__main__":
    main()
