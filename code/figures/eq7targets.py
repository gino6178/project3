"""What the two rules ask adjacent planes for.

    FN_ROOT=... python eq7targets.py OUT.npz REF_DIR N_PLANES

Both rows come from `sds_demo`'s own canonicalisation, so they differ in the assignment rule
and in nothing else. The block rule gives plane j the photograph floor(j M / N); when there are
fewer photographs than planes it gives the same one to several adjacent planes and then changes
all at once. Equation (7) mixes that photograph with the next at the fractional part.
"""
import os
import sys

import numpy as np

FN = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FN)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import sds_demo as sd                                                 # noqa: E402


def main(out, ref_dir, n_planes=16, size=192):
    import cv2
    n_planes = int(n_planes)
    files = sorted(sd._photos_in(os.path.join(FN, ref_dir)))
    M = len(files)
    print(f"  {M} photographs over {n_planes} planes")
    blk, con, which = [], [], []
    for i in range(n_planes):
        t = i * M / n_planes
        k0, k1, w = int(t) % M, (int(t) + 1) % M, float(t - int(t))
        a = np.asarray(sd._blend_canonical(files[k0]), dtype=np.float32)
        b = np.asarray(sd._blend_on_disc(files[k0], files[k1], w), dtype=np.float32) / 255.
        blk.append(cv2.resize(a, (size, size)))
        con.append(cv2.resize(b, (size, size)))
        which.append(k0)
    which = np.array(which)
    print("  block hands out photograph:", " ".join(str(v) for v in which))
    np.savez_compressed(out, block=(np.stack(blk) * 255).astype(np.uint8),
                        cont=(np.stack(con) * 255).astype(np.uint8),
                        which=which, M=M, n=n_planes)
    print("  ->", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 16)
