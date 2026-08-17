"""Colour the interior of any shape from its cross-section photographs.

The polar version indexes a cell by radius and angle about an axis, which presumes the
object is a solid of revolution. A loaf is not: its slices are rounded rectangles that
change size and proportion along its length, and there is no radius to normalise by.

What is true of any object with a slicing direction is that a cell sits somewhere inside its
own slice, and that position can be given in the slice's own units. Take the slice's extent
along each in-plane direction at that height, express the cell as a fraction of it, and read
the photograph at the same fraction of its own extent. On a solid of revolution this reduces
to the polar mapping; on a loaf it follows the slice as it narrows toward the ends.

The slicing direction is whatever the lattice recorded -- the longest principal axis for an
elongated object, the renderer's up for a round one -- so nothing here chooses it.

One photograph is not enough. A single transverse photograph carries no information along
the axis, so it can only be extruded, and every longitudinal cut through the result is a set
of vertical stripes -- the transverse pattern smeared. On an orange that is nearly right,
because an orange really does have a pale core running its length. On a watermelon it is
wrong twice over: the white starburst at the centre of a transverse slice became a white
column down the axis, 31.2% of the axis band against the 0.24% its own longitudinal
photographs show, and the training then had to undo at the axis what the initialisation had
put there.

So take both, and compose them so that each cut resembles its own photograph:

    C(u, v, w) = T(u, v) + [ L(rho, w) - mean_w L(rho, .) ]

At fixed w the added term varies only with rho, so a transverse cut keeps T's angular
pattern. At fixed angle it is L's axial variation about its own radial mean, so a
longitudinal cut carries L's structure. With no longitudinal photograph the bracket is zero
and this is exactly the previous behaviour.

Which way round the longitudinal photograph goes is not assumed: the lattice knows the
object's width at every height, the photograph's silhouette has the same profile, and the
orientation is whichever of the four (transpose x flip) matches it best.
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


def photo_lut(path, n=384):
    """The photograph resampled onto its object's own bounding box, in [-1,1]^2."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.
    bg = np.median(np.concatenate([a[:10].reshape(-1, 3), a[-10:].reshape(-1, 3)]), 0)
    m = ndimage.binary_fill_holes(np.abs(a - bg).max(2) >= 0.10)
    lab, k = ndimage.label(m)
    if k > 1:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    gy = np.linspace(y0, y1, n)[:, None]
    gx = np.linspace(x0, x1, n)[None, :]
    gy = np.broadcast_to(gy, (n, n)); gx = np.broadcast_to(gx, (n, n))
    out = np.stack([ndimage.map_coordinates(a[:, :, c], [gy, gx], order=1, mode="nearest")
                    for c in range(3)], -1)
    return torch.from_numpy(out).float()


def photo_stack(path, n=384):
    """Every photograph under `path`, resampled onto its own bounding box. A file gives one.

    One photograph puts its own accidents into the model. A seed sits at some angle in the
    transverse photograph, and extruding that lays a full-height dark band down the object;
    the band is real in the photograph and invented in the model. Across twenty photographs
    the seeds are at unrelated angles and cancel, and what survives is what they agree on --
    the flesh, the pale rind ring, the colour at the centre. That is what an initialisation
    is for: the structure every section shares, with the texture left for training.

    The angular pattern goes with them, which is the price. It is the same effect measured
    for the references (cross-view angular correlation +0.09 when each section is sampled on
    its own), and it is the right way round here -- better a smooth radial profile than one
    photograph's membranes frozen into every slice at the wrong angles.
    """
    if os.path.isdir(path):
        files = sorted(f for f in (os.path.join(path, x) for x in os.listdir(path))
                       if os.path.isfile(f))
    else:
        files = [path]
    return [photo_lut(f, n) for f in files], files


def blend(luts):
    """Per-pixel median, so one odd photograph cannot drag the initialisation with it."""
    return torch.median(torch.stack(luts), dim=0).values if len(luts) > 1 else luts[0]


def _occ(lut):
    """Rough object mask of a resampled photograph, and its width at each row."""
    occ = (lut.mean(2) < 0.93)
    return occ, occ.float().sum(1)


def orient_long(lut, width_profile, quiet=False):
    """Put the longitudinal photograph's axis along its rows, pointing the way the object does.

    A photograph does not say which of its two image axes is the object's slicing axis, and
    guessing wrong lays the axial structure across the object instead of along it. The lattice
    already measured the object's half-width at every height; the photograph's silhouette has
    the same profile, so try the four orientations and keep the one that matches. Degenerate
    for a sphere, where every orientation matches equally -- the scores are printed so that is
    visible rather than silent.
    """
    tgt = width_profile / width_profile.max().clamp_min(1e-6)
    n = lut.shape[0]
    tgt = torch.nn.functional.interpolate(
        tgt[None, None], size=n, mode="linear", align_corners=True)[0, 0]
    best, scores = None, []
    for name, cand in [("as-is", lut), ("transposed", lut.transpose(0, 1))]:
        for flip, c in [("", cand), (" flipped", torch.flip(cand, [0]))]:
            _, w = _occ(c)
            w = w / w.max().clamp_min(1e-6)
            s = float((w - tgt).abs().mean())
            scores.append((s, name + flip, c))
    scores.sort(key=lambda t: t[0])
    best = scores[0]
    if not quiet:
        print("  縱切照片方向: " + "  ".join(f"{nm} {s:.4f}" for s, nm, _ in scores)
              + f"   -> 選用 {best[1]}")
    return best[2]


def main(src, photo, out_dir, nz=48, paint_shell=False, photo_long=None):
    os.makedirs(out_dir, exist_ok=True)
    g = GaussianModel(0); g.load_ply_zero_sh(os.path.join(src, "gs_fill.ply"))
    lvl = torch.load(os.path.join(src, "cell_level.pt")).to(DEV)
    lat = torch.load(os.path.join(src, "lattice.pt"))
    xyz = g.get_xyz.detach().to(DEV)
    n = min(xyz.shape[0], lvl.shape[0])
    xyz, lvl = xyz[:n], lvl[:n]

    up = lat["up"].to(DEV).float(); up = up / up.norm()
    c = xyz.mean(0)
    d = xyz - c
    z = d @ up
    # The in-plane axes come from the object, not from a projected world axis. Picking one
    # arbitrarily rotates the photograph inside the slice -- the loaf rendered as a diamond
    # with its crown pointing sideways. The slice's own principal directions are its width
    # and its height, which is what the photograph's width and height mean too.
    perp = d - z[:, None] * up
    cov = (perp.T @ perp) / perp.shape[0]
    ev, evec = torch.linalg.eigh(cov)
    e1 = evec[:, 2]; e1 = e1 - (e1 @ up) * up; e1 = e1 / e1.norm()   # wider direction
    e2 = torch.cross(up, e1, dim=0)
    a = d @ e1
    b = d @ e2

    # Which way up. A loaf narrows toward its crown and is flat underneath, and so does the
    # photograph; aligning the two by that asymmetry puts the crust where the crust is
    # rather than relying on the mesh and the camera happening to agree about a sign.
    def crown_sign(coord, width):
        hi = width[coord > 0.45 * float(coord.abs().max())].abs()
        lo = width[coord < -0.45 * float(coord.abs().max())].abs()
        hi = float(hi.quantile(0.9)) if hi.numel() > 50 else 1.0
        lo = float(lo.quantile(0.9)) if lo.numel() > 50 else 1.0
        return 1.0 if hi < lo else -1.0
    e2_sign = crown_sign(b, a)
    b = b * e2_sign

    # the slice's own extent at each height, measured from the cells themselves
    zi = ((z - z.min()) / (z.max() - z.min() + 1e-9) * (nz - 1)).long().clamp(0, nz - 1)
    A = torch.zeros(nz, device=DEV); B = torch.zeros(nz, device=DEV)
    for k in range(nz):
        m = zi == k
        if int(m.sum()) > 50:
            A[k] = a[m].abs().quantile(0.98)
            B[k] = b[m].abs().quantile(0.98)
    A = torch.where(A > 0, A, A.max()); B = torch.where(B > 0, B, B.max())
    sm = lambda t: torch.nn.functional.avg_pool1d(t[None, None], 5, 1, 2)[0, 0]
    A, B = sm(A), sm(B)

    ua = (a / A[zi].clamp_min(1e-6)).clamp(-0.999, 0.999)
    ub = (b / B[zi].clamp_min(1e-6)).clamp(-0.999, 0.999)

    tluts, tfiles = photo_stack(photo)
    # Each photograph's own crown direction, found the same way, so they agree with the object
    # and with each other. Orient first, blend second: blending photographs that disagree
    # about which way up they are cancels the very asymmetry this is aligning.
    oriented = []
    for L in tluts:
        L = L.to(DEV)
        N = L.shape[0]
        with torch.no_grad():
            wid = (L.mean(2) < 0.93).float().sum(1)
            if float(wid[:N // 3].mean()) >= float(wid[-N // 3:].mean()):
                L = torch.flip(L, [0])
        oriented.append(L)
    lut = blend(oriented)
    N = lut.shape[0]
    print(f"  橫切照片 {len(tfiles)} 張" + ("（中位數合成）" if len(tfiles) > 1 else ""))
    iy = ((ub * 0.5 + 0.5) * (N - 1)).long().clamp(0, N - 1)
    ix = ((ua * 0.5 + 0.5) * (N - 1)).long().clamp(0, N - 1)
    tgt = lut[iy, ix]

    if photo_long is not None:
        lluts, lfiles = photo_stack(photo_long)
        print(f"  縱切照片 {len(lfiles)} 張" + ("（各自定向後取中位數）" if len(lfiles) > 1 else ""))
        llut = blend([orient_long(L.to(DEV), A, quiet=(len(lfiles) > 1)) for L in lluts])
        M = llut.shape[0]
        locc, _ = _occ(llut)
        # The photograph's own radial mean, down each column and inside its object only, so
        # the residual is the axial variation and nothing else. Columns the mask misses fall
        # back to the column value itself, which makes their residual zero.
        cnt = locc.float().sum(0).clamp_min(1.0)
        colmean = (llut * locc[..., None].float()).sum(0) / cnt[:, None]
        colmean = torch.where((locc.float().sum(0) > 4)[:, None], colmean, llut.mean(0))
        rho = torch.sqrt(ua * ua + ub * ub).clamp(0, 0.999)
        uz = (z - z.min()) / (z.max() - z.min() + 1e-9) * 2.0 - 1.0
        lr = ((0.5 + 0.5 * rho) * (M - 1)).long().clamp(0, M - 1)      # centre -> right edge
        lz = ((uz * 0.5 + 0.5) * (M - 1)).long().clamp(0, M - 1)
        resid = llut[lz, lr] - colmean[lr]
        tgt = (tgt + resid).clamp(0, 1)
        print(f"  縱切殘差 平均 |d| {float(resid.abs().mean()):.4f}  "
              f"最大 {float(resid.abs().max()):.4f}")

    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)[:n]
    out = rgb.clone()
    sel = torch.ones_like(lvl, dtype=torch.bool) if paint_shell else (lvl == 0)
    out[sel] = tgt[sel]
    print(f"  切片軸 {[round(float(v),3) for v in up]}")
    print(f"  平面內主軸 e1 {[round(float(v),3) for v in e1]}  "
          f"e2 {[round(float(v),3) for v in e2]}  冠部方向 {e2_sign:+.0f}")
    print(f"  重新上色 {int(sel.sum()):,} / {n:,}")
    print(f"  平均 RGB  舊 {[round(float(v),3) for v in rgb[sel].mean(0)]}"
          f"  新 {[round(float(v),3) for v in out[sel].mean(0)]}")
    print(f"  切片寬度沿軸 {float(A.min()/A.max()):.2f}–1.00 (窄端/寬端)")

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
    ap.add_argument("--long", dest="photo_long", default=None,
                    help="longitudinal photograph; without it the transverse one is extruded "
                         "and every longitudinal cut is a stripe pattern")
    ap.add_argument("--paint-shell", action="store_true")
    a = ap.parse_args()
    main(a.src, a.photo, a.out, paint_shell=a.paint_shell, photo_long=a.photo_long)
