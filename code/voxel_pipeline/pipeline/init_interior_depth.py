"""Colour the interior by depth from the surface, which every solid has and no hole breaks.

The slice mapping expresses a cell as a fraction of its own slice's bounding box. That is
right while a slice is one simply-connected lump. A ring's slices are annuli: the bounding
box is the whole ring's square, its centre is the hole, and the photograph's centre lands on
the inner rim -- the doughnut came out with the swirl wrapped around the hole instead of
filling the section. Measured, its layer-to-layer agreement initialised at +0.447 against
+0.89 to +0.95 for the three objects whose sections are simply connected.

Depth is indifferent to that. Every cell has a distance to the nearest surface, every
photograph's pixel has a distance to the nearest edge, and both normalise to nought at the
boundary and one at the deepest point. Matching them puts the outside of the photograph on
the outside of the object whatever shape or genus it has -- rind then flesh, crust then
crumb, glaze then dough -- and it needs no axis, no centre and no bounding box.

What it does not carry is where a feature sits within a section: a depth profile is one
dimensional. So this is the mapping for sections that are not simply connected, and the
slice mapping stays for those that are.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os, argparse
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

import torch, numpy as np
from torch import nn
from PIL import Image
from scipy import ndimage
from scene.gaussian_model import GaussianModel

DEV = "cuda:0"
C0 = 0.28209479177387814


def photo_depth_lut(path, nb=64, blur=1.0):
    """Mean colour as a function of normalised distance from the photograph's edge."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.
    bg = np.median(np.concatenate([a[:10].reshape(-1, 3), a[-10:].reshape(-1, 3)]), 0)
    m = ndimage.binary_fill_holes(np.abs(a - bg).max(2) >= 0.10)
    lab, k = ndimage.label(m)
    if k > 1:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    dt = ndimage.distance_transform_edt(m)
    dn = dt / max(dt.max(), 1e-6)
    out = np.zeros((nb, 3), np.float32)
    for i in range(nb):
        sel = m & (dn >= i / nb) & (dn < (i + 1) / nb)
        out[i] = a[sel].mean(0) if sel.sum() > 20 else np.nan
    # fill any empty bins from their neighbours
    for c in range(3):
        v = out[:, c]
        idx = np.arange(nb)
        good = ~np.isnan(v)
        out[:, c] = np.interp(idx, idx[good], v[good])
    if blur > 0:
        out = ndimage.gaussian_filter1d(out, blur, axis=0, mode="nearest")
    return torch.from_numpy(out)


def main(src, photo, out_dir, nb=64, paint_shell=False):
    os.makedirs(out_dir, exist_ok=True)
    g = GaussianModel(0); g.load_ply_zero_sh(os.path.join(src, "gs_fill.ply"))
    lvl = torch.load(os.path.join(src, "cell_level.pt")).to(DEV)
    lat = torch.load(os.path.join(src, "lattice.pt"))
    xyz = g.get_xyz.detach().to(DEV)
    n = min(xyz.shape[0], lvl.shape[0])
    xyz, lvl = xyz[:n], lvl[:n]

    # Occupancy on the coarse lattice. Building it on the fine one leaves every other cell
    # empty -- the interior is spaced at the coarse size and only the skin is fine -- so the
    # solid comes out as a checkerboard and the distance transform finds a boundary one cell
    # away everywhere. Measured that way the ring's greatest depth was 1.41 cells, against a
    # tube twenty-two cells across.
    dxf = float(lat["coarse_dx"])
    mn = xyz.min(0).values
    idx = torch.round((xyz - mn) / dxf).long()
    dims = (idx.max(0).values + 3).tolist()
    idx = idx.clamp(torch.zeros(3, dtype=torch.long, device=DEV),
                    torch.tensor(dims, device=DEV) - 1)
    occ = torch.zeros(dims, dtype=torch.bool, device=DEV)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ_np = np.pad(occ.cpu().numpy(), 1)
    dt = ndimage.distance_transform_edt(occ_np)[1:-1, 1:-1, 1:-1]
    ii = idx.cpu().numpy()
    depth = torch.from_numpy(dt[ii[:, 0], ii[:, 1], ii[:, 2]]).float().to(DEV)
    dmax = float(depth.quantile(0.999))
    dn = (depth / max(dmax, 1e-6)).clamp(0, 0.999)

    lut = photo_depth_lut(photo, nb).to(DEV)
    tgt = lut[(dn * nb).long().clamp(0, nb - 1)]

    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)[:n]
    out = rgb.clone()
    sel = torch.ones_like(lvl, dtype=torch.bool) if paint_shell else (lvl == 0)
    out[sel] = tgt[sel]

    print(f"  最大深度 {dmax:.2f} 格 ({dmax*dxf:.4f} 世界單位)")
    print(f"  重新上色 {int(sel.sum()):,} / {n:,}")
    for lo, hi, nm in [(0.0, 0.15, "貼近表面"), (0.15, 0.5, "中層"), (0.5, 1.0, "最深處")]:
        b = sel & (dn >= lo) & (dn < hi)
        if int(b.sum()) > 100:
            print(f"    {nm:<8} {int(b.sum()):>8,} 格  RGB "
                  f"{[round(float(v),3) for v in out[b].mean(0)]}")

    with torch.no_grad():
        g._features_dc = nn.Parameter(((out - 0.5) / C0).unsqueeze(1).contiguous())
        for k in ["_xyz", "_opacity", "_scaling", "_rotation"]:
            setattr(g, k, nn.Parameter(getattr(g, k).detach()[:n].contiguous()))
        g._features_rest = nn.Parameter(torch.zeros(n, 0, 3, device=DEV))
        g.max_radii2D = torch.zeros(n, device=DEV)
        g.trained = torch.zeros(n, dtype=torch.bool)
        g.is_interior = torch.ones(n, dtype=torch.bool)
    g.save_ply(os.path.join(out_dir, "gs_fill.ply"))
    torch.save(torch.ones(n, dtype=torch.bool), os.path.join(out_dir, "is_interior.pt"))
    torch.save(lvl.cpu(), os.path.join(out_dir, "cell_level.pt"))
    torch.save(lat, os.path.join(out_dir, "lattice.pt"))
    for extra in ["cell_face.pt"]:
        p = os.path.join(src, extra)
        if os.path.exists(p):
            torch.save(torch.load(p), os.path.join(out_dir, extra))
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("photo"); ap.add_argument("out")
    ap.add_argument("--paint-shell", action="store_true")
    a = ap.parse_args()
    main(a.src, a.photo, a.out, paint_shell=a.paint_shell)
