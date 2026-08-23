"""Are the blocks where the supervision is thin?

The theory is that a cell's colour is an average of the observations that reached it, so a cell
reached once carries the full noise of one observation and its neighbours carry different noise --
which is what a block is, and why blocks are exactly one cell across. If that is right, the
difference between neighbouring cells should fall as the number of times they were observed rises.
If blockiness is flat in the observation count, the theory is wrong and the blocks come from
somewhere else.

Two quantities, neither of which needs a render of a cut face:

  blockiness   the mean absolute difference between face-adjacent solid cells, which is the field's
               own cell-scale variation
  n_k          how many of the pipeline's own supervised planes touch each cell over one run,
               counted with the jitter the pipeline actually applies
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RUNS = [r for r in os.environ.get("BK_RUNS", "").split(",") if r]
ITERS = int(os.environ.get("BK_ITERS", "40"))
JIT = float(os.environ.get("BK_JIT", "0.5"))
RES = int(os.environ.get("RES", "512"))
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH, NV = H_HI - H_LO, len(C["v_planes"])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
step_h = float(hd[1] - hd[0])
solid = st["solid"]
idx3 = st["idx3"]
lo = torch.as_tensor(st["idx_lo"], dtype=torch.long, device=dev)
N = len(st["interior"])


def counts():
    """How many times each cell is touched over one run of the pipeline's own schedule."""
    st["interior"] = st["interior"].detach().clone().requires_grad_(True)
    c = torch.zeros(N, device=dev)
    rng = np.random.default_rng(0)
    ctr = ((solid.float().mean(0) + 0.5) * st["hc"]
           + torch.as_tensor(st["org"], dtype=torch.float32, device=dev))
    a0 = np.array([0., 0., 1.]) if abs(float(hn[2])) < 0.9 else np.array([1., 0., 0.])
    u2 = np.cross(np.asarray(hn.cpu()), a0); u2 /= np.linalg.norm(u2)
    w2 = np.cross(np.asarray(hn.cpu()), u2)
    for it in range(ITERS):
        for i in range(NH):
            d = float(hd[H_LO + i]) + step_h * (rng.random() - 0.5) * 2 * JIT
            c += _touch(hmvp, hn, d)
        for j in range(NV):
            a = np.pi * (j + (rng.random() - 0.5) * 2 * JIT) / NV
            nv = np.cos(a) * u2 + np.sin(a) * w2
            c += _touch(torch.as_tensor(vmvp[j]),
                        torch.as_tensor(nv, dtype=torch.float32, device=dev),
                        float(-np.dot(nv, np.asarray(ctr.cpu()))))
    return c


def _touch(mvp, n, d):
    st["interior"].grad = None
    _, _, k = ON.cut_polygons(st, n, float(d), device=dev)
    if k == 0:
        return torch.zeros(N, device=dev)
    img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(d), RES, exterior=False)
    img.sum().backward()
    g = st["interior"].grad
    return (g.abs().sum(1) > 0).float() if g is not None else torch.zeros(N, device=dev)


def neighbours():
    """Pairs of face-adjacent solid cells, as two index arrays."""
    key = idx3
    pairs = []
    for ax in range(3):
        a = solid.clone(); a[:, ax] += 1
        q = (a - lo)
        ok = ((q >= 0) & (q < torch.tensor(key.shape, device=dev))).all(1)
        j = torch.full((len(solid),), -1, dtype=torch.long, device=dev)
        qq = q.clamp(min=torch.zeros(3, dtype=torch.long, device=dev),
                     max=torch.tensor([s - 1 for s in key.shape], device=dev))
        j[ok] = key[qq[ok, 0], qq[ok, 1], qq[ok, 2]].long()
        m = j >= 0
        pairs.append(torch.stack([torch.arange(len(solid), device=dev)[m], j[m]], 1))
    return torch.cat(pairs)


nb = neighbours()
print(f"{OBJ}: {N:,} cells, {len(nb):,} adjacent pairs; counting touches over {ITERS} iterations "
      f"of the pipeline's schedule at jitter {JIT:g}")
cnt = counts()
print(f"  touches per cell: median {float(cnt.median()):.0f}, "
      f"10th percentile {float(cnt.quantile(0.1)):.0f}, 90th {float(cnt.quantile(0.9)):.0f}, "
      f"never touched {100 * float((cnt == 0).float().mean()):.1f}%")

for r in RUNS:
    p = torch.load(f"{W}/{r}/params.pt", map_location=dev)
    if "dec_i" in p:
        w = p["dec_i"]["stage1.0.weight"].shape[0]
        n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
        anchor.W_HID, anchor.N_HID = w, n
        di = anchor.ColourDecoder(N, init_rgb=st["interior"].detach()).to(dev)
        di.load_state_dict(p["dec_i"])
        with torch.no_grad():
            col = di()
    else:
        col = p["interior"].to(dev)
    d = (col[nb[:, 0]] - col[nb[:, 1]]).abs().mean(1)
    m = torch.minimum(cnt[nb[:, 0]], cnt[nb[:, 1]])          # the thinner of the pair
    print(f"\n  {r}: blockiness overall {float(d.mean()):.4f}")
    edges = [0, 1, 3, 10, 30, 100, 10 ** 9]
    for a, b in zip(edges, edges[1:]):
        sel = (m >= a) & (m < b)
        if int(sel.sum()) < 50:
            continue
        print(f"    pairs touched {a:>3}-{b if b < 10**9 else '+':>4} times: "
              f"{int(sel.sum()):>8,} pairs, difference {float(d[sel].mean()):.4f}")
