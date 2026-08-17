"""M2: optimise the per-cell feature and the shared decoder against cross-section references.

M1 drew slices with features that were optimised for a different renderer. That showed the
geometry and the lookup are right, and it also showed the cost of not retraining: the cube slice
came out lighter and yellower than the Gaussian one, because those features were fitted under
alpha blending -- each cell's colour was one contribution to a mixture, and reading it back
alone is reading half of a sum. So the features have to be refitted for the renderer that will
actually draw them.

The cube renderer makes this cheap in a way the Gaussian one is not. A slice is a gather: each
pixel takes the feature of the cell containing it, and gather is differentiable in the values it
gathers. There is no rasteriser in the loop, no alpha compositing, no sorting -- the gradient of
a pixel goes to exactly one cell, which is also why a material boundary stays sharp.

Warm start, and it matters for how the result is read: the features begin at what the Gaussian
run learned rather than at noise. This is not a from-scratch comparison of two representations
and is not presented as one; it is the adaptation step the spec's M2 asks for, and starting from
a good field is what makes forty iterations enough.

    python method/common/cube/train.py LATTICE ANCHOR_CKPT FILL_PT REF_DIR OUT [iters]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]
_os.chdir(_FN_ROOT)

from method.common.cube.slice_render import (apply_fill, build_decoder,  # noqa: E402
                                             load_cubes)
import section_match as sm                                              # noqa: E402

DEV = "cuda:0"

# What "the segment walls survived" means, as a number.
#
# M2 was rejected twice on the strength of looking at the slices, which is the right instinct and
# not a criterion. An orange's walls are radial spokes, so on a circle at fixed radius the colour
# oscillates with angle; flattening removes that and nothing else about the image has to change.
# Sampling the section on a polar grid and taking the mean absolute angular derivative over the
# middle radii measures exactly it, and it is scale-free once divided by the mean intensity, so
# the initialisation, the references and any trained result are comparable.
FREEZE_FEAT = _os.environ.get("M2_FREEZE_FEAT", "0") == "1"
STRUCT_W = float(_os.environ.get("M2_STRUCT_W", "0"))


def wall_energy(img, r_lo=0.30, r_hi=0.80, n_r=48, n_a=360):
    """Angular oscillation of a cross-section over its middle radii."""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).to(DEV)
    H, W = img.shape[:2]
    g = img.mean(2) if img.dim() == 3 else img
    fg = (g < 0.97)
    ys, xs = torch.nonzero(fg, as_tuple=True)
    if ys.numel() < 64:
        return float("nan")
    cy, cx = ys.float().mean(), xs.float().mean()
    R = torch.sqrt(((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).float()).max()
    rr = torch.linspace(r_lo, r_hi, n_r, device=DEV)[:, None] * R
    aa = torch.linspace(0, 2 * np.pi, n_a, device=DEV)[None, :]
    py = (cy + rr * torch.sin(aa)).clamp(0, H - 1)
    px = (cx + rr * torch.cos(aa)).clamp(0, W - 1)
    grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)[None]
    pol = F.grid_sample(g[None, None], grid, align_corners=True)[0, 0]
    d = (pol[:, 1:] - pol[:, :-1]).abs().mean()
    return float(d / pol.mean().clamp_min(1e-6))


class CellGrid:
    """The cubes, as one flat array per level plus a lookup from position to index.

    Built once. A slice is then two gathers, and the only per-iteration work is the decode and
    the loss.
    """

    def __init__(self, xyz, feat, h, lvl):
        self.feat = nn.Parameter(feat.clone())
        self.levels = []
        for level in (0, 1):
            sel = lvl == level
            if not bool(sel.any()):
                continue
            c = xyz[sel]
            hh = float(h[sel][0])
            mn = c.min(0).values - hh
            key = ((c - mn) / hh).round().long()
            dims = (key.max(0).values + 2).tolist()
            lut = torch.full((dims[0] * dims[1] * dims[2],), -1, dtype=torch.long, device=DEV)
            lut[key[:, 0] * dims[1] * dims[2] + key[:, 1] * dims[2] + key[:, 2]] = \
                sel.nonzero().squeeze(1)
            self.levels.append((mn, hh, torch.tensor(dims, device=DEV), lut))

    def index_at(self, pts):
        """Index of the smallest occupied cell containing each point, or -1."""
        idx = torch.full((pts.shape[0],), -1, dtype=torch.long, device=DEV)
        for mn, hh, dims, lut in self.levels:            # coarse first, fine overwrites
            q = ((pts - mn) / hh).round().long()
            ok = ((q >= 0) & (q < dims)).all(1)
            got = torch.full_like(idx, -1)
            qq = q[ok]
            got[ok] = lut[qq[:, 0] * int(dims[1]) * int(dims[2])
                          + qq[:, 1] * int(dims[2]) + qq[:, 2]]
            idx = torch.where(got >= 0, got, idx)
        return idx


def plane_basis(n):
    n = n / n.norm()
    a = torch.tensor([0., 0., 1.], device=DEV)
    if abs(float(n @ a)) > 0.9:
        a = torch.tensor([1., 0., 0.], device=DEV)
    u = torch.cross(n, a); u = u / u.norm()
    return n, u, torch.cross(n, u)


def render(grid, decode, centre, n, u, v, radius, size):
    t = torch.linspace(-radius, radius, size, device=DEV)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    pts = (centre[None, None] + gx[..., None] * u + gy[..., None] * v).reshape(-1, 3)
    idx = grid.index_at(pts)
    hit = idx >= 0
    img = torch.ones(pts.shape[0], 3, device=DEV)
    if bool(hit.any()):
        img = img.clone()
        img[hit] = decode(grid.feat[idx[hit]])
    return img.reshape(size, size, 3), hit.reshape(size, size)



def _ssim(a, b):
    mu_a = F.avg_pool2d(a, 11, 1, 5); mu_b = F.avg_pool2d(b, 11, 1, 5)
    sa = F.avg_pool2d(a * a, 11, 1, 5) - mu_a ** 2
    sb = F.avg_pool2d(b * b, 11, 1, 5) - mu_b ** 2
    sab = F.avg_pool2d(a * b, 11, 1, 5) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))).mean()


def patch_loss(img, tgt, mask, n=6, size=128, stat_w=0.3):
    """The loss the main pipeline uses, not a simplification of it.

    A whole-frame loss is dominated by what occupies most of the frame -- the silhouette, the
    mean hue, the radial layout -- and a section can match all of those while being locally
    flat. That is not a hypothesis here: the first version of this file scored the whole slice
    and forty iterations of it took the segment walls out of the orange while the colour got
    closer, which is the same failure the main pipeline had and solved this way.

    Crops are drawn from the foreground only, because a crop of background is two constant
    images and scores perfectly, diluting the gradient in proportion to how much of the frame
    the object does not fill.

    `stat_w` adds the band term: low frequencies compared where they sit, finer octaves only in
    quantity. Statistics have no phase, so a reference whose walls are at different angles from
    ours -- it is a photograph of another orange -- can still say how much structure there
    should be without dictating where.
    """
    fg = (mask.squeeze(-1) > 0.5)
    ys, xs = fg.nonzero(as_tuple=True)
    if ys.numel() < 16:
        return 0.3 * ((img - tgt).abs()).mean() + 0.7 * (1 - _ssim(img.mean(2)[None, None],
                                                                  tgt.mean(2)[None, None]))
    H, W = img.shape[:2]
    size = min(size, H, W)
    pick = torch.randint(0, ys.numel(), (n,), device=ys.device)
    total = 0.0
    for k in range(n):
        y0 = int(max(0, min(int(ys[pick[k]]) - size // 2, H - size)))
        x0 = int(max(0, min(int(xs[pick[k]]) - size // 2, W - size)))
        r = img[y0:y0 + size, x0:x0 + size]
        g = tgt[y0:y0 + size, x0:x0 + size]
        total = total + 0.7 * (1 - _ssim(r.mean(2)[None, None], g.mean(2)[None, None])) \
                      + 0.3 * F.mse_loss(r, g)
        if stat_w > 0:
            pr, pg = r.permute(2, 0, 1)[None], g.permute(2, 0, 1)[None]
            band = 0.0
            for sig in (1.0, 2.0, 4.0):
                kk = int(2 * round(2 * sig) + 1)
                br = F.avg_pool2d(pr, kk, 1, kk // 2)
                bg = F.avg_pool2d(pg, kk, 1, kk // 2)
                band = band + (( (pr - br).abs().mean() - (pg - bg).abs().mean() ) ** 2)
                pr, pg = br, bg
            total = total + stat_w * band
    return total / n


def _wall_t(img):
    """wall_energy, differentiable, on a tensor that requires grad."""
    return torch.tensor(0.0, device=DEV) if img is None else _wall_diff(img)


def _wall_diff(img):
    H, W = img.shape[:2]
    g = img.mean(2)
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV).float(),
                            torch.arange(W, device=DEV).float(), indexing="ij")
    fg = (g < 0.97).float()
    tot = fg.sum().clamp_min(1.0)
    cy, cx = (yy * fg).sum() / tot, (xx * fg).sum() / tot
    R = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2) * fg).max().clamp_min(1.0)
    rr = torch.linspace(0.30, 0.80, 48, device=DEV)[:, None] * R
    aa = torch.linspace(0, 2 * np.pi, 360, device=DEV)[None, :]
    py = (cy + rr * torch.sin(aa)).clamp(0, H - 1)
    px = (cx + rr * torch.cos(aa)).clamp(0, W - 1)
    grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)[None]
    pol = F.grid_sample(g[None, None], grid, align_corners=True)[0, 0]
    return (pol[:, 1:] - pol[:, :-1]).abs().mean() / pol.mean().clamp_min(1e-6)


def main(lattice_dir, anchor_ckpt, fill_pt, ref_dir, out_dir, iters=40, size=512,
         n_planes=16, lo=0.30, hi=0.70):
    _os.makedirs(out_dir, exist_ok=True)
    xyz, feat, h, lvl, lat, ck = load_cubes(lattice_dir, anchor_ckpt)
    if fill_pt and _os.path.exists(fill_pt):
        xyz, feat, h, lvl = apply_fill(xyz, feat, h, lvl, fill_pt)
    print(f"  {xyz.shape[0]:,} cells ({int((lvl == 1).sum()):,} fine)")

    pre, mlp, mlp_s = build_decoder(ck, feat.shape[1])
    c_dim = next(m for m in mlp if hasattr(m, "in_features")).in_features
    shell = (lvl == 1)

    # The head is retrained, the feature is retrained; stage1 is kept frozen as the map from
    # the stored 8-d feature to the head's input, because it also carries the geometry columns
    # the cubes no longer use and refitting it would only relearn an identity on the rest.
    for p in pre.parameters():
        p.requires_grad_(False)

    grid = CellGrid(xyz, feat, h, lvl)
    head = list(mlp.parameters()) + (list(mlp_s.parameters()) if mlp_s else [])
    if FREEZE_FEAT:
        # The hypothesis M2's two failures point at: a cell is visited by many planes at
        # different positions relative to the reference's structure, and the reference is a
        # photograph of a *different* fruit, so the per-cell gradient averages over structures
        # that do not correspond and the cell converges on their mean. Freezing the feature
        # leaves the structure where the initialisation put it and lets the shared decoder fit
        # how that structure maps to colour, which is a far smaller space to flatten from.
        grid.feat.requires_grad_(False)
        opt = torch.optim.Adam([{"params": head, "lr": 1e-3}])
    else:
        opt = torch.optim.Adam([{"params": [grid.feat], "lr": 3e-3},
                                {"params": head, "lr": 1e-4}])

    def decode(f):
        raw = f if pre is None else pre(f)
        cf = raw[:, 11:11 + c_dim] if raw.shape[1] >= 11 + c_dim else raw[:, :c_dim]
        return torch.sigmoid(mlp(cf))

    refs = [p for p in sorted(glob.glob(_os.path.join(ref_dir, "*.png")))
            if not _os.path.splitext(_os.path.basename(p))[0].endswith(("_depth", "_mask"))]
    ref_imgs = [cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255.0 for p in refs]
    print(f"  {len(ref_imgs)} references from {ref_dir}")

    cen = xyz.mean(0)
    axis = torch.tensor([0., 1., 0.], device=DEV)
    n, u, v = plane_basis(axis)
    proj = (xyz - cen) @ n
    d_lo, d_hi = float(proj.min()), float(proj.max())
    radius = float((xyz - cen).norm(dim=1).max()) * 1.06

    # the initialisation's own wall energy, measured once, before anything moves
    with torch.no_grad():
        centre0 = cen + n * (d_lo + (d_hi - d_lo) * 0.5)
        img0, _ = render(grid, decode, centre0, n, u, v, radius, size)
        _wall0 = [wall_energy(img0)]
    refwall = [wall_energy(r) for r in ref_imgs]
    print(f"  wall energy: initialisation {_wall0[0]:.4f}, references "
          f"{np.nanmean(refwall):.4f} (mean of {len(refwall)})")

    for it in range(iters):
        tot = 0.0
        for k in range(n_planes):
            f = lo + (hi - lo) * (k + random.random()) / n_planes
            centre = cen + n * (d_lo + (d_hi - d_lo) * f)
            img, hit = render(grid, decode, centre, n, u, v, radius, size)
            with torch.no_grad():
                ref = ref_imgs[k % len(ref_imgs)]
                tgt = sm.section_target(img.detach().permute(2, 0, 1), ref).permute(1, 2, 0)
            m = hit.unsqueeze(-1).float()
            loss = patch_loss(img, tgt, m)
            if STRUCT_W > 0:
                # Hold the angular oscillation the initialisation had. It is one scalar per
                # slice and it is the thing that was being lost, so penalising its *fall* costs
                # nothing where the fit does not flatten anything.
                we = _wall_t(img)
                loss = loss + STRUCT_W * F.relu(_wall0[0] - we) / max(_wall0[0], 1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        if it % 5 == 0 or it == iters - 1:
            print(f"  iter {it:3d}  loss {tot / n_planes:.5f}", flush=True)
            with torch.no_grad():
                centre = cen + n * (d_lo + (d_hi - d_lo) * 0.5)
                img, _ = render(grid, decode, centre, n, u, v, radius, size)
                cv2.imwrite(_os.path.join(out_dir, f"it_{it:03d}.png"),
                            (img.clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
                w = wall_energy(img)
                print(f"       wall energy {w:.4f}  "
                      f"({100 * w / max(_wall0[0], 1e-9):.0f}% of the initialisation)")

    torch.save({"feat": grid.feat.detach().cpu(), "anchor_xyz": xyz.cpu(),
                "level": lvl.cpu(), "h": h.cpu(),
                "state": {**{f"stage2.{i}.{w}": p.detach().cpu()
                             for i, m in enumerate(mlp) if hasattr(m, "weight")
                             for w, p in (("weight", m.weight), ("bias", m.bias))},
                          **{k: v for k, v in ck["state"].items() if k.startswith("stage1.")}},
                "K": 1}, _os.path.join(out_dir, "cube_model.pt"))
    print(f"  -> {out_dir}/cube_model.pt")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         iters=int(sys.argv[6]) if len(sys.argv) > 6 else 40)
