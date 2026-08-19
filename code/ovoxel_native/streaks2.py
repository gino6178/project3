"""Where the oriented structure comes from, given that the cut-face tessellation is exactly flat.

Two questions left:
  - is the ~82 degree peak one orientation or two at right angles, as a projected cubic grid
    would give?
  - does it appear on route 2, whose interior starts at a constant and therefore cannot have
    inherited anything from a released model?

and the direct check: where do the lattice's own axes land in the image, under this camera?
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
W = "/workspace/ovoxel_native"

Z = np.load(f"{W}/out/streaks/spectra.npz")
print("orientation spectra, z-scored; peak and its perpendicular\n")
print(f"  {'variant':<26} {'peak':>6} {'z':>6} {'z at peak+90':>13} {'z at 18k mult':>14}")
for k in Z.files:
    en = Z[k]
    z = (en - en.mean()) / (en.std() + 1e-9)
    p = int(np.argmax(z))
    perp = (p + 90) % 180
    m18 = np.mean([z[(18 * i) % 180] for i in range(10)])
    print(f"  {k:<26} {p:>4}deg {z[p]:>+6.1f} {z[perp]:>+13.1f} {m18:>+14.2f}")

# Where the lattice axes project. The dual grid and the occupancy are both axis aligned in the
# lattice's own frame, so if the pattern is the grid, its orientation is fixed by the camera and
# not by anything trained.
for route in ("1", "2"):
    st = torch.load(f"{W}/state_r{route}.pt", map_location="cpu", weights_only=False)
    C = np.load(f"{W}/cams_mv.npz" if route == "1" else f"{W}/cams_mv_r2.npz")
    mvp = C["eh_mvp"][0].astype(np.float64)
    org = np.asarray(st["org"], np.float64)
    hc = float(st["hc"])
    c0 = st["solid"].float().mean(0).numpy().astype(np.float64) * hc + org

    def proj(p):
        v = np.append(p, 1.0) @ mvp
        return v[:2] / v[3]

    o = proj(c0)
    print(f"\nroute {route}: lattice axes at the evaluation camera")
    for i, nm in enumerate("xyz"):
        e = np.zeros(3); e[i] = 20 * hc
        q = proj(c0 + e) - o
        if np.linalg.norm(q) < 1e-9:
            print(f"  {nm} axis: degenerate (along the view direction)")
            continue
        # image y is flipped relative to clip space
        a = np.degrees(np.arctan2(-q[1], q[0])) % 180
        print(f"  {nm} axis projects to {a:6.1f} deg  (screen length {np.linalg.norm(q):.4f})")
