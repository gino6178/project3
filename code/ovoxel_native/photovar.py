"""How much do the photographs themselves change along the axis?

The field's colour varies less along the polar axis than across it -- 0.815 of the across-axis
variation on the watermelon, 0.648 on the orange -- and a longitudinal cut shows that as vertical
stripes. The prior is not the cause: it weights the across-axis direction four times harder, so it
has been pushing the other way the whole time.

That leaves the data. A transverse photograph is a disc at one depth; if successive depths look
nearly the same, then "the same pattern at every depth" is what the photographs ask for, the field
obliges, and the stripes are the data's own. This compares two things with no training involved:

  between depths   how much two transverse photographs of DIFFERENT depths differ
  within a depth   how much one photograph differs from itself, split into two halves

If the first is not much larger than the second, the transverse family carries almost no
information about depth at all.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refsel
from PIL import Image

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
OBJS = [o for o in os.environ.get("PV_OBJS", "watermelon_sp,orange_sp").split(",") if o]
RES = int(os.environ.get("RES", "256"))


def load(spec):
    out = []
    for q in sorted(refsel.photos_in(f"{FN}/{spec}")):
        a = np.asarray(Image.open(q).convert("RGB").resize((RES, RES), Image.LANCZOS),
                       np.float32) / 255.
        m = a.min(2) < 0.98
        out.append((a, m, os.path.basename(q)))
    return out


for OBJ in OBJS:
    conf = open(f"{OBJDIR}/{OBJ}.conf").read()
    def spec(k):
        return [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)][0]
    print(f"\n{OBJ}")
    for key, name in (("REF_H=", "transverse"), ("REF_V=", "longitudinal")):
        ph = load(spec(key))
        if len(ph) < 2:
            print(f"  {name}: {len(ph)} photograph, nothing to compare"); continue
        # between: every pair, on the pixels both call foreground
        bet = []
        for i in range(len(ph)):
            for j in range(i + 1, len(ph)):
                m = ph[i][1] & ph[j][1]
                if m.sum() > 64:
                    bet.append(float(np.abs(ph[i][0] - ph[j][0])[m].mean()))
        # within: one photograph against a shifted copy of itself, at the same scale as a cell
        wit = []
        for a, m, _ in ph:
            s = 2
            d = np.abs(a[s:] - a[:-s])
            mm = m[s:] & m[:-s]
            if mm.sum() > 64:
                wit.append(float(d[mm].mean()))
        print(f"  {name}: {len(ph)} photographs")
        print(f"    between two of them          {np.mean(bet):.4f}")
        print(f"    within one, {2} pixels apart   {np.mean(wit):.4f}")
        print(f"    ratio between/within         {np.mean(bet) / max(np.mean(wit), 1e-9):.2f}")
