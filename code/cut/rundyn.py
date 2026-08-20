"""Drive `dynamic_cut` at a step the stiff preset can take.

    python rundyn.py PLY OUT FRAMES FLAG [DT] [SUBSTEPS]

The demo's default is dt = 3e-4 at 30 substeps, chosen for the single soft material at
E = 1.2e6. The two-rule field puts the peel at 8.0e6, seven times stiffer, and the CFL limit
goes as 1/sqrt(E): at the same spacing the admissible step falls from 1.03e-3 to about 3.9e-4,
which leaves the default with no margin, and Warp reports it as an illegal memory access rather
than as a divergence. Halving the step and doubling the substeps keeps the frame interval and
restores the margin.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamic_cut as dc                                             # noqa: E402


def main(ply, out, frames, flag, dt=1.0e-4, substeps=90):
    print(f"  dt={dt:g}, {substeps} substeps per frame")
    dc.main(ply, out, frames=int(frames), flag_path=flag,
            substeps=int(substeps), dt=float(dt))


if __name__ == "__main__":
    main(*sys.argv[1:])
