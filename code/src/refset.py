"""Build a derived reference set: a subset of a directory, or the same photographs rotated.

Two ablations in section 4 need reference sets that are not the ones on disk -- how many
photographs the interior is worth, and what happens when the references are not registered
to each other -- and both were run from directories prepared by hand. A hand-prepared
directory is not a measurement anyone can repeat, so this makes them.

    python refset.py subset  SRC DST 0 3 5      # photographs 0, 3 and 5, in that order
    python refset.py rotate  SRC DST 15 [seed]  # each rotated by an angle drawn from +-15 deg

`subset` keeps the file names, so a set built from indices 0 and 3 still sorts the way the
originals did and the plane-to-photograph assignment is unchanged apart from there being
fewer to spread. `rotate` draws one angle per photograph and writes it into a manifest
beside them, because an injected perturbation that is not recorded cannot be checked.

The rotation fills with white rather than black. A cross-section photograph sits on white,
and a black corner would be read as interior by anything downstream that thresholds on
brightness -- including the fit that places the reference inside the rendered silhouette.
"""
import json
import os
import shutil
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sds_demo import _photos_in                                              # noqa: E402


def main():
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    files = sorted(_photos_in(src))
    if not files:
        raise SystemExit(f"no photographs in {src}")
    os.makedirs(dst, exist_ok=True)

    if mode == "subset":
        idx = [int(a) for a in sys.argv[4:]]
        for i in idx:
            shutil.copy2(files[i], os.path.join(dst, os.path.basename(files[i])))
        print(f"{dst}: {len(idx)} of {len(files)} -> "
              f"{', '.join(os.path.basename(files[i]) for i in idx)}")
        return

    if mode == "rotate":
        import numpy as np
        psi = float(sys.argv[4])
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        rng = np.random.default_rng(seed)
        angles = {}
        for f in files:
            deg = float(rng.uniform(-psi, psi))
            im = Image.open(f).convert("RGB")
            im.rotate(-deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255)).save(
                os.path.join(dst, os.path.basename(f)))
            angles[os.path.basename(f)] = round(deg, 3)
        with open(os.path.join(dst, "rotation.json"), "w") as fh:
            json.dump(dict(source=src, psi=psi, seed=seed, degrees=angles), fh, indent=1)
        print(f"{dst}: {len(files)} rotated within +-{psi} deg, "
              f"actual {min(angles.values()):+.1f} to {max(angles.values()):+.1f}")
        return

    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
