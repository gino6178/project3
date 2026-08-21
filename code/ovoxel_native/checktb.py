"""Does REF_TRANS_BLEND actually change the transverse target, and where?

Cheap enough to run before an hour of GPU. For a sweep of plane indices the three modes are asked
for their target and compared: mode 0 must be piecewise constant in idx and change only at the
switches, mode 2 must change at the same places and nowhere else, and mode 1 must change at every
index. If mode 1 is ever piecewise constant the blend is not reaching the family.
"""
import os
import sys

import numpy as np

SPEC = sys.argv[1]
N = 24

out = {}
for mode in ("0", "2", "1"):
    os.environ["REF_TRANS_BLEND"] = mode
    for m in [k for k in list(sys.modules) if k == "refsel"]:
        del sys.modules[m]
    import refsel
    refsel._PHOTOS.clear()
    imgs = [np.asarray(refsel.as_array(refsel.solved_photo(SPEC, i, N), 256), np.float32)
            for i in range(N)]
    d = [float(np.abs(imgs[i + 1] - imgs[i]).mean()) for i in range(N - 1)]
    out[mode] = d
    ch = [i for i, x in enumerate(d) if x > 1e-6]
    print(f"  mode {mode}: target changes at plane {ch}")
    print(f"          step sizes {[f'{d[i]:.4f}' for i in ch][:8]}")

nz = lambda m: sum(x > 1e-6 for x in out[m])
print(f"\n  mode 0 changes {nz('0')}/{N-1} times, mode 2 {nz('2')}, mode 1 {nz('1')}")
print(f"  largest single step: mode 0 {max(out['0']):.4f}, mode 2 {max(out['2']):.4f}, "
      f"mode 1 {max(out['1']):.4f}")
