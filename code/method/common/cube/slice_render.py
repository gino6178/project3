"""M0 and M1 of the cube spec: occupancy from the lattice, and a slice drawn from cubes.

The spec's first pitfall is to do this before touching training: if the slice renderer is
wrong, nothing measured after it means anything. So this reads the lattice and the trained
per-cell features we already have, draws a cross-section by asking which cube each pixel falls
in, and is checked against the same plane drawn by the Gaussian rasteriser. No training, no new
supervision, nothing generative.

What a cube slice is, exactly (spec eq. 11-12): for each pixel of the cross-section image take
its 3-D position on the plane, find the containing cell by integer division, and if that cell is
occupied emit the decoder's RGB for it. Piecewise constant, no interpolation -- the spec is
explicit that interpolation reintroduces smoothing across material boundaries and that it should
be an ablation after the baseline exists, not part of it.

The one thing this cannot inherit is the two-level lattice: a cell's size depends on its level,
so the lookup is done per level, coarse first and fine second, and the finer answer wins where
both exist.

    python method/common/cube/slice_render.py LATTICE ANCHOR_CKPT OUT [n_planes]
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import cv2
import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]
_os.chdir(_FN_ROOT)

DEV = "cuda:0"


def load_cubes(lattice_dir, anchor_ckpt):
    """Occupancy and appearance per cell: spec eq. 6, minus the parts v1 does not need.

    The lattice already answers "which cells are occupied" -- that was M0's work and it was done
    when the lattice was built. What is added here is that a cell is now a *volume* with a size,
    not a point carrying a Gaussian: `h` per cell, from its level.
    """
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)
    ck = torch.load(anchor_ckpt, map_location=DEV)
    xyz = ck["anchor_xyz"].to(DEV).float()
    feat = ck["feat"].to(DEV).float()
    n = xyz.shape[0]
    lvl = lvl[:n].to(DEV)
    h = torch.where(lvl == 1, torch.full_like(lvl, 0, dtype=torch.float32) + float(lat["fine_dx"]),
                    torch.full_like(lvl, 0, dtype=torch.float32) + float(lat["coarse_dx"]))
    return xyz, feat, h, lvl, lat, ck


def build_decoder(ck, feat_dim):
    """The trained appearance head, rebuilt from its state dict.

    Only the visual stage is needed: spec eq. 7 is a feature to RGB map and nothing in a slice
    depends on position offsets, scales or opacities, which is the point of dropping the
    Gaussians in the first place.
    """
    import torch.nn as nn
    st = ck["state"]
    keys = [k for k in st if k.startswith("stage2.") and k.endswith("weight")]
    keys.sort(key=lambda k: int(k.split(".")[1]))
    layers, sd, li = [], {}, 0
    for i, k in enumerate(keys):
        w = st[k]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        sd[f"{li}.weight"] = w
        sd[f"{li}.bias"] = st[k.replace("weight", "bias")]
        li += 1
        if i < len(keys) - 1:
            layers.append(nn.ReLU())
            li += 1
    mlp = nn.Sequential(*layers).to(DEV)
    mlp.load_state_dict(sd)
    mlp.eval()
    # The shell has its own head. `anchor_decoder` splits them because the exterior views can
    # only move shell cells and the cross-sections only interior ones, and with a single head
    # the louder branch wins for both -- so a slice drawn with the wrong head would show the
    # interior in the peel's colours.
    s_keys = [k for k in st if k.startswith("stage2_s.") and k.endswith("weight")]
    mlp_s = None
    if s_keys:
        s_keys.sort(key=lambda k: int(k.split(".")[1]))
        ls, sds, li = [], {}, 0
        for i, k in enumerate(s_keys):
            w = st[k]
            ls.append(nn.Linear(w.shape[1], w.shape[0]))
            sds[f"{li}.weight"] = w; sds[f"{li}.bias"] = st[k.replace("weight", "bias")]
            li += 1
            if i < len(s_keys) - 1:
                ls.append(nn.ReLU()); li += 1
        mlp_s = nn.Sequential(*ls).to(DEV)
        mlp_s.load_state_dict(sds)
        mlp_s.eval()

    # stage1 maps the stored feature to the head's input; if the checkpoint has one, use it.
    k1 = [k for k in st if k.startswith("stage1.") and k.endswith("weight")]
    if k1:
        k1.sort(key=lambda k: int(k.split(".")[1]))
        l1, sd1, li = [], {}, 0
        for i, k in enumerate(k1):
            w = st[k]
            l1.append(nn.Linear(w.shape[1], w.shape[0]))
            sd1[f"{li}.weight"] = w
            sd1[f"{li}.bias"] = st[k.replace("weight", "bias")]
            li += 1
            if i < len(k1) - 1:
                l1.append(nn.ReLU())
                li += 1
        pre = nn.Sequential(*l1).to(DEV)
        pre.load_state_dict(sd1)
        pre.eval()
    else:
        pre = None
    return pre, mlp, mlp_s


@torch.no_grad()
def cube_slice(xyz, rgb, h, lvl, plane_n, plane_d, size=512, margin=1.06):
    """Draw the cross-section at Pi(x) = n.x + d = 0 by cube lookup.

    Two levels, so the lookup runs twice: coarse cells first, fine cells second, and the fine
    answer overwrites where both are occupied. That ordering is the whole of the two-level
    handling -- a finer cell is inside its parent by construction, so "the smallest cell that
    contains this point" is just "the last one to write".
    """
    n = torch.tensor(plane_n, device=DEV, dtype=torch.float32)
    n = n / n.norm()
    # an orthonormal basis of the plane, so the image is axis-aligned in plane space
    a = torch.tensor([0., 0., 1.], device=DEV)
    if abs(float(n @ a)) > 0.9:
        a = torch.tensor([1., 0., 0.], device=DEV)
    u = torch.cross(n, a); u = u / u.norm()
    v = torch.cross(n, u)
    centre = -plane_d * n
    r = float((xyz - centre).norm(dim=1).max()) * margin
    t = torch.linspace(-r, r, size, device=DEV)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    pts = centre[None, None] + gx[..., None] * u + gy[..., None] * v      # (S,S,3)
    flat = pts.reshape(-1, 3)

    img = torch.ones(flat.shape[0], 3, device=DEV)
    hit = torch.zeros(flat.shape[0], dtype=torch.bool, device=DEV)
    for level in (0, 1):
        sel = lvl == level
        if not bool(sel.any()):
            continue
        c, col = xyz[sel], rgb[sel]
        hh = float(h[sel][0])
        mn = c.min(0).values - hh
        key = ((c - mn) / hh).round().long()
        dims = key.max(0).values + 2
        lut = torch.full((int(dims[0]) * int(dims[1]) * int(dims[2]),), -1,
                         dtype=torch.long, device=DEV)
        lut[key[:, 0] * int(dims[1]) * int(dims[2]) + key[:, 1] * int(dims[2]) + key[:, 2]] = \
            torch.arange(c.shape[0], device=DEV)
        q = ((flat - mn) / hh).round().long()
        ok = ((q >= 0) & (q < dims)).all(1)
        idx = torch.full((flat.shape[0],), -1, dtype=torch.long, device=DEV)
        qq = q[ok]
        idx[ok] = lut[qq[:, 0] * int(dims[1]) * int(dims[2]) + qq[:, 1] * int(dims[2]) + qq[:, 2]]
        got = idx >= 0
        img[got] = col[idx[got]]
        hit |= got
    return img.reshape(size, size, 3).cpu().numpy(), hit.reshape(size, size).cpu().numpy()


def apply_fill(xyz, feat, h, lvl, fill_pt):
    """Add the cells `close_and_fill` found, each taking its nearest neighbour's feature.

    A cell added to seal a pinhole has no feature of its own and cannot invent one; the nearest
    occupied cell is what the lattice's own smoothing already does for untrained anchors, and it
    is the honest choice here -- these cells exist to make the volume solid, not to add detail.
    """
    d = torch.load(fill_pt)
    for level, (mn, hh, ni) in d["added"].items():
        if ni.numel() == 0:
            continue
        new_xyz = (mn.to(DEV) + ni.to(DEV).float() * hh)
        src = xyz[lvl == level]
        srcf = feat[lvl == level]
        # Nearest neighbour by grid lookup, not by cdist: 93k new cells against 118k sources is
        # a 23 GB distance matrix even in chunks of eight thousand, and the answer is on a
        # regular grid anyway. Walk outward one ring at a time and take the first hit.
        gmn = src.min(0).values - hh
        gi = ((src - gmn) / hh).round().long()
        gd = (gi.max(0).values + 2).tolist()
        lut = torch.full((gd[0] * gd[1] * gd[2],), -1, dtype=torch.long, device=DEV)
        lut[gi[:, 0] * gd[1] * gd[2] + gi[:, 1] * gd[2] + gi[:, 2]] = \
            torch.arange(src.shape[0], device=DEV)
        q = ((new_xyz - gmn) / hh).round().long()
        idx = torch.full((new_xyz.shape[0],), -1, dtype=torch.long, device=DEV)
        rings = [(dx, dy, dz) for R in range(1, 6)
                 for dx in range(-R, R + 1) for dy in range(-R, R + 1) for dz in range(-R, R + 1)
                 if max(abs(dx), abs(dy), abs(dz)) == R]
        for dx, dy, dz in rings:
            miss = idx < 0
            if not bool(miss.any()):
                break
            p2 = q[miss] + torch.tensor([dx, dy, dz], device=DEV)
            ok = ((p2 >= 0) & (p2 < torch.tensor(gd, device=DEV))).all(1)
            got = torch.full((int(miss.sum()),), -1, dtype=torch.long, device=DEV)
            pp = p2[ok]
            got[ok] = lut[pp[:, 0] * gd[1] * gd[2] + pp[:, 1] * gd[2] + pp[:, 2]]
            cur = idx[miss]; cur[got >= 0] = got[got >= 0]; idx[miss] = cur
        idx = idx.clamp_min(0)
        xyz = torch.cat([xyz, new_xyz])
        feat = torch.cat([feat, srcf[idx]])
        h = torch.cat([h, torch.full((new_xyz.shape[0],), hh, device=DEV)])
        lvl = torch.cat([lvl, torch.full((new_xyz.shape[0],), level, device=DEV, dtype=lvl.dtype)])
    return xyz, feat, h, lvl


def main(lattice_dir, anchor_ckpt, out_dir, n_planes=6, size=512, fill_pt=None):
    _os.makedirs(out_dir, exist_ok=True)
    xyz, feat, h, lvl, lat, ck = load_cubes(lattice_dir, anchor_ckpt)
    if fill_pt and not _os.path.exists(fill_pt):
        # Not a silent skip. A fill that is asked for and not found is a run that quietly
        # measures the unfilled occupancy and reports it as though it were filled, which is how
        # three objects came back with 6.6% to 14.9% holes after the fill had supposedly been
        # applied -- the path had landed in the `size` argument.
        raise SystemExit(f"fill_pt {fill_pt} does not exist")
    if fill_pt:
        n0 = xyz.shape[0]
        xyz, feat, h, lvl = apply_fill(xyz, feat, h, lvl, fill_pt)
        print(f"  occupancy filled: {n0:,} -> {xyz.shape[0]:,} cells")
    pre, mlp, mlp_s = build_decoder(ck, feat.shape[1])
    with torch.no_grad():
        # The head takes a slice of stage1's output, not all of it: the first eleven columns
        # carry the geometry the cubes no longer need -- offset, scale, rotation, opacity --
        # and only the appearance columns reach the colour stage.
        raw = feat if pre is None else pre(feat)
        c_dim = next(m for m in mlp if hasattr(m, "in_features")).in_features
        cf = raw[:, 11:11 + c_dim] if raw.shape[1] >= 11 + c_dim else raw[:, :c_dim]
        out = mlp(cf)
        if mlp_s is not None:
            shell = lvl == 1
            out = torch.where(shell[:, None], mlp_s(cf), out)
        rgb = torch.sigmoid(out).clamp(0, 1)
    print(f"  {xyz.shape[0]:,} cells, {int((lvl == 1).sum()):,} fine   "
          f"decoded colours: {len(torch.unique((rgb * 255).long(), dim=0)):,} distinct")

    c = xyz.mean(0)
    lo, hi = float((xyz - c)[:, 1].min()), float((xyz - c)[:, 1].max())
    for i, frac in enumerate(np.linspace(0.3, 0.7, n_planes)):
        d = lo + (hi - lo) * float(frac)
        img, hit = cube_slice(xyz, rgb, h, lvl, (0., 1., 0.), -(float(c[1]) + d), size=size)
        cv2.imwrite(_os.path.join(out_dir, f"cube_{i:02d}.png"),
                    (img[:, :, ::-1] * 255).astype(np.uint8))
        # A slice that is hole-free is the completion condition for M1; a cube's support is a
        # finite volume, so any gap is a bug in the lookup rather than a coverage parameter.
        ys, xs = np.where(hit)
        if len(xs):
            fill = hit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            from scipy import ndimage
            solid = ndimage.binary_fill_holes(fill)
            gaps = 100.0 * float((solid & ~fill).sum()) / max(float(solid.sum()), 1.0)
        else:
            gaps = float("nan")
        print(f"  plane {i}: covered {100 * hit.mean():5.2f}% of the frame, "
              f"holes inside the section {gaps:.3f}%")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         n_planes=int(sys.argv[4]) if len(sys.argv) > 4 else 6,
         size=int(sys.argv[5]) if len(sys.argv) > 5 else 512,
         fill_pt=sys.argv[6] if len(sys.argv) > 6 else None)
