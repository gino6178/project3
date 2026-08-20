"""Does the differentiable alignment start where the moment fit puts it?

    python checkinit.py OUT.png

The drift measured in the joint-optimisation arm only means the optimiser moved the target if
the target started in the right place. If `_moment_init` disagrees with `_fit_disc`, the arm is
measuring the initialisation and not the optimiser. This renders one section, builds the target
both ways, and reports the difference between them.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

FN = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
sys.path += [FN]

import diffalign                                                      # noqa: E402
from section_match import section_target                              # noqa: E402


def main(out, size=512):
    # A synthetic pair is enough and keeps the check free of the trainer: a disc of one radius
    # standing for the render, and a photograph-like disc of another for the reference.
    def disc(n, r, cy, cx, val):
        y, x = np.mgrid[0:n, 0:n]
        m = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
        a = np.ones((n, n, 3), np.float32)
        a[m] = val
        return a

    render = disc(size, 150, 260, 250, (0.95, 0.62, 0.20))
    ref = disc(size, 95, 250, 256, (0.98, 0.70, 0.25))
    rt = torch.from_numpy(render).permute(2, 0, 1).cuda()
    ct = torch.from_numpy(ref).permute(2, 0, 1).cuda()

    par = diffalign._moment_init(ct, rt)
    warped = diffalign._warp(ct, par)

    dr = diffalign._disc(rt).tolist()
    dc = diffalign._disc(ct).tolist()
    dw = diffalign._disc(warped).tolist()
    print(f"  render     centre ({dr[0]:.1f}, {dr[1]:.1f})  radius {dr[2]:.1f}")
    print(f"  reference  centre ({dc[0]:.1f}, {dc[1]:.1f})  radius {dc[2]:.1f}")
    print(f"  warped     centre ({dw[0]:.1f}, {dw[1]:.1f})  radius {dw[2]:.1f}")
    print(f"  the warp should land on the render: "
          f"centre off by ({dw[0]-dr[0]:+.1f}, {dw[1]-dr[1]:+.1f}), "
          f"radius off by {dw[2]-dr[2]:+.1f}")

    grid = np.hstack([render, ref, warped.permute(1, 2, 0).detach().cpu().numpy()])
    Image.fromarray((grid * 255).astype(np.uint8)).save(out)
    print("  ->", out, " render | reference | reference warped by the initialisation")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkinit.png")
