"""Are the three arms' renders actually different, and where?

`tbmeasure` gave the control and r1_tb2 the same transverse detail to four decimals, which is
either a real coincidence or a sign that the two arms produced the same pixels -- and if they did,
REF_TRANS_BLEND=2 did not reach the family it is supposed to change and the reading is worthless.
"""
import glob
import hashlib

import numpy as np
from PIL import Image

ARMS = ["r1_pin_full", "r1_tb2", "r1_tb1"]


def one(arm, name):
    f = sorted(glob.glob(f"{arm}/eval_final/{name}_*.png"))
    return f[0] if f else None


for fam in ("rh", "rv"):
    print(f"  === {fam}")
    for k in range(3):
        fs = [one(a, f"{fam}{k}") for a in ARMS]
        if not all(fs):
            continue
        ims = [np.asarray(Image.open(f).convert("RGB"), np.float32) / 255. for f in fs]
        h = [hashlib.md5(open(f, "rb").read()).hexdigest()[:8] for f in fs]
        d01 = float(np.abs(ims[0] - ims[1]).mean())
        d02 = float(np.abs(ims[0] - ims[2]).mean())
        print(f"    {fam}{k}  md5 {' '.join(h)}   "
              f"|control - tb2| {d01:.6f}   |control - tb1| {d02:.6f}")
