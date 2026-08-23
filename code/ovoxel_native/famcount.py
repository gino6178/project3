"""How many times does each cell hear from each family?

The two families are not symmetric by construction: a transverse plane covers a whole disc and the
family sweeps the band, while every longitudinal plane passes through the axis, so a cell on the
axis hears from all of them and a cell at the rim hears from a few. If a cell is supervised mostly
by one family, that family decides it, and the field ends up carrying the structure that family
sees -- which is what an interior extruded along the axis looks like.

Counted over one run of the pipeline's own schedule, with the jitter it actually applies, and
reported per cell and against the cell's distance from the axis.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJS = [o for o in os.environ.get("FC_OBJS", "watermelon_sp,orange_sp").split(",") if o]
ITERS = int(os.environ.get("FC_ITERS", "20"))
JIT = float(os.environ.get("FC_JIT", "0.5"))
RES = int(os.environ.get("RES", "512"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)

for OBJ in OBJS:
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    st["interior"] = st["interior"].detach().clone().requires_grad_(True)
    C = np.load(f"{W}/cams_{OBJ}_bal.npz")
    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    NH, NV = H_HI - H_LO, len(C["v_planes"])
    hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
    hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
    hd = C["h_planes"][:, 3]
    vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
    vp = C["v_planes"]
    step_h = float(hd[1] - hd[0])
    N = len(st["interior"])

    def touch(mvp, n, d):
        st["interior"].grad = None
        _, _, k = ON.cut_polygons(st, n, float(d), device=dev)
        if k == 0:
            return torch.zeros(N, device=dev)
        img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(d), RES, exterior=False)
        img.sum().backward()
        g = st["interior"].grad
        return (g.abs().sum(1) > 0).float() if g is not None else torch.zeros(N, device=dev)

    axis = np.asarray(C["h_planes"][0, :3], float); axis /= np.linalg.norm(axis)
    a0 = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(axis, a0); u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    ctr = ((st["solid"].float().mean(0) + 0.5) * st["hc"]
           + torch.as_tensor(st["org"], dtype=torch.float32, device=dev))
    ch = torch.zeros(N, device=dev); cv = torch.zeros(N, device=dev)
    uh = torch.zeros(N, dtype=torch.bool, device=dev)
    uv = torch.zeros(N, dtype=torch.bool, device=dev)
    rng = np.random.default_rng(0)
    for it in range(ITERS):
        for i in range(NH):
            t_ = touch(hmvp, hn, float(hd[H_LO + i]) + step_h * (rng.random() - 0.5) * 2 * JIT)
            ch += t_; uh |= t_ > 0
        for j in range(NV):
            a = np.pi * (j + (rng.random() - 0.5) * 2 * JIT) / NV
            nv = np.cos(a) * u + np.sin(a) * w
            t_ = touch(torch.as_tensor(vmvp[j]),
                       torch.as_tensor(nv, dtype=torch.float32, device=dev),
                       float(-np.dot(nv, np.asarray(ctr.cpu()))))
            cv += t_; uv |= t_ > 0

    cen = (st["solid"].float() + 0.5) * st["hc"] + torch.as_tensor(
        st["org"], dtype=torch.float32, device=dev)
    rel = cen - ctr[None]
    ax_t = torch.as_tensor(axis, dtype=torch.float32, device=dev)
    rad = (rel - (rel @ ax_t)[:, None] * ax_t[None]).norm(dim=1)
    rad = rad / rad.max()
    print(f"\n{OBJ}: {NH} transverse and {NV} longitudinal planes, {ITERS} iterations, "
          f"jitter {JIT:g}")
    print(f"  cells reached at least once: transverse {int(uh.sum()):,} "
          f"({100 * float(uh.float().mean()):.1f}%), longitudinal {int(uv.sum()):,} "
          f"({100 * float(uv.float().mean()):.1f}%)")
    print(f"    both {int((uh & uv).sum()):,} ({100 * float((uh & uv).float().mean()):.1f}%), "
          f"only transverse {int((uh & ~uv).sum()):,}, only longitudinal {int((uv & ~uh).sum()):,}, "
          f"neither {int((~uh & ~uv).sum()):,} ({100 * float((~uh & ~uv).float().mean()):.1f}%)")
    print(f"  total touches: transverse {int(ch.sum()):,}, longitudinal {int(cv.sum()):,}, "
          f"ratio {float(cv.sum() / ch.sum().clamp(min=1)):.2f}")
    print(f"  touches per cell: transverse {float(ch.mean()):.1f}, "
          f"longitudinal {float(cv.mean()):.1f}, ratio {float(cv.mean() / ch.mean()):.2f}")
    print(f"  {'distance from the axis':<24}{'transverse':>12}{'longitudinal':>14}{'ratio':>8}")
    edges = [0, .2, .4, .6, .8, 1.01]
    for a, b in zip(edges, edges[1:]):
        sel = (rad >= a) & (rad < b)
        if int(sel.sum()) < 100:
            continue
        h_, v_ = float(ch[sel].mean()), float(cv[sel].mean())
        print(f"  {f'{a:.1f} to {b:.1f} of the radius':<24}{h_:>12.1f}{v_:>14.1f}"
              f"{v_ / max(h_, 1e-9):>8.2f}")
