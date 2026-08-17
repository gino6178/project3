"""Separate the object into material classes without being told what they are.

The rule-based field in `material_field.py` says skin-or-interior from the lattice level and
grades the rest by lightness. That is enough to give the solver two materials and it is not a
decomposition: lightness is not material, it is lightness under whatever the reference
photograph was lit by, and an orange's pith and the pale rim of a segment wall get the same
answer for different reasons.

What the object actually offers is a per-cell appearance that was learned, plus a position on a
lattice. Clustering those groups cells that look and sit alike, and on these objects the groups
that come out are the parts: peel, albedo, flesh, core. Nothing here is told that an orange has
a pith. That is the claim worth making and the one that generalises -- an unsupervised
decomposition into heterogeneous material regions -- and it is separate from the question of
what modulus each region should get, which appearance cannot answer and which is left to an
explicit ordering with an explicit range.

Features, in order of preference:
  * the trained anchor feature, if the run saved one (`anchor_epoch_*.pt`). This is what the
    model learned, and it distinguishes things that decode to a similar colour from different
    causes.
  * otherwise the decoded colour in CIELAB, which is at least perceptually uniform, together
    with the cell's radial position, which is what separates a pale core from a pale rim.

    python report/material_segment.py MODEL.ply LATTICE_DIR OUT.png [K] [ANCHOR.pt]
"""
import os
import sys

import cv2
import numpy as np
import torch

# FN_ROOT, not a hard-coded path. The remote box has a directory of the same
# name, so the chdir succeeded there and every relative path then resolved
# against the wrong tree -- which fails as "file not found" for a file that
# plainly exists. Nine scripts had this and were fixed; these two were missed.
_FN_ROOT = os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)

from scene.gaussian_model import GaussianModel   # noqa: E402

DEV = "cuda:0"
C0 = 0.28209479177387814
# Distinct and light enough to read the structure through: the point of the picture is which
# cells got grouped together, not what the groups are called.
PALETTE = np.array([[214, 96, 60], [ 70, 130, 180], [120, 180, 100], [230, 190, 80],
                    [150, 110, 190], [ 90, 190, 190], [220, 130, 170], [130, 130, 130]],
                   dtype=np.float32) / 255.0


def features(xyz, rgb, level, anchor_feat=None):
    """The space the clustering runs in, standardised so no axis dominates by its units."""
    if anchor_feat is not None:
        f = anchor_feat.to(DEV).float()
        src = "trained anchor feature"
    else:
        lab = cv2.cvtColor((rgb.cpu().numpy()[None] * 255).astype(np.uint8),
                           cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        f = torch.from_numpy(lab).to(DEV)
        src = "decoded colour (CIELAB)"
    c = xyz.mean(0)
    r = (xyz - c).norm(dim=1, keepdim=True)
    r = r / r.max().clamp_min(1e-9)
    # Position is in the space so that a pale core and a pale rim can be told apart, but only
    # weakly: at equal weight with appearance the clustering starts carving homogeneous flesh
    # into shells, which is a partition of the object and not a decomposition of its materials.
    f = torch.cat([f, r * float(f.std()) * 0.6,
                   level.reshape(-1, 1).float() * float(f.std()) * 3.0], 1)
    f = (f - f.mean(0)) / f.std(0).clamp_min(1e-6)
    return f, src


def kmeans(f, k, iters=60, seed=0):
    g = torch.Generator(device=f.device).manual_seed(seed)
    # k-means++ start: a random start on a cloud this elongated puts two centres in the flesh
    # and none in the peel, and the run keeps that.
    idx = torch.randint(0, f.shape[0], (1,), generator=g, device=f.device)
    C = f[idx]
    for _ in range(k - 1):
        d = torch.cdist(f, C).min(1).values ** 2
        C = torch.cat([C, f[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(f, C).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = f[m].mean(0)
    return torch.cdist(f, C).argmin(1), C


def main(ply, lat_dir, out, k=4, anchor_pt=None):
    g = GaussianModel(0)
    g.load_ply_zero_sh(ply)
    xyz = g.get_xyz.detach().to(DEV)
    rgb = (g._features_dc.detach().to(DEV).squeeze(1) * C0 + 0.5).clamp(0, 1)
    lvl = torch.load(os.path.join(lat_dir, "cell_level.pt")).to(DEV).reshape(-1)[:xyz.shape[0]].float()

    af = None
    if anchor_pt and os.path.exists(anchor_pt):
        d = torch.load(anchor_pt, map_location="cpu")
        af = d["feat"] if "feat" in d else None
        if af is not None and af.shape[0] != xyz.shape[0]:
            print(f"  anchor feature has {af.shape[0]} rows for {xyz.shape[0]} cells; ignoring")
            af = None
    f, src = features(xyz, rgb, lvl, af)
    print(f"  clustering {xyz.shape[0]:,} cells on {src}, {f.shape[1]} dims, k={k}")

    lab, _ = kmeans(f, k)

    # Merge classes that differ only in where they are. Two groups whose appearance centroids
    # sit within `MERGE_DE` of each other in CIELAB are the same material seen at two radii,
    # and giving them different moduli would be inventing a boundary the object does not have.
    # This is what decides K: ask for more classes than the object can have and let the ones
    # that carry no appearance difference collapse.
    de = float(os.environ.get("MERGE_DE", "6.0"))
    lab_np = lab.cpu().numpy()
    cent = np.stack([cv2.cvtColor((rgb[lab == j].mean(0).cpu().numpy()[None, None] * 255)
                                  .astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
                     for j in range(k)])
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for a in range(k):
        for b in range(a + 1, k):
            if np.linalg.norm(cent[a] - cent[b]) < de:
                pa, pb = find(a), find(b)
                if pa != pb:
                    parent[max(pa, pb)] = min(pa, pb)
    roots = sorted({find(j) for j in range(k)})
    if len(roots) < k:
        print(f"    merged {k} -> {len(roots)} classes "
              f"(appearance centroids within dE {de:g} are one material)")
        rm = {r: i for i, r in enumerate(roots)}
        lab = torch.tensor([rm[find(int(v))] for v in lab_np], device=DEV)
        k = len(roots)
    order = torch.argsort(torch.tensor([float(rgb[lab == j].mean()) for j in range(k)]))
    remap = torch.zeros(k, dtype=torch.long, device=DEV)
    remap[order.to(DEV)] = torch.arange(k, device=DEV)
    lab = remap[lab]

    for j in range(k):
        m = lab == j
        print(f"    class {j}: {int(m.sum()):>8,} cells ({100*float(m.float().mean()):5.1f}%)  "
              f"mean RGB {[round(float(v),3) for v in rgb[m].mean(0)]}   "
              f"skin fraction {float((lvl[m] > 0.5).float().mean()):.2f}")

    torch.save(lab.cpu(), os.path.splitext(out)[0] + "_labels.pt")

    # Show it where it can be judged: a transverse and a longitudinal slab, actual colour
    # beside class colour.
    c = xyz.mean(0)
    d = xyz - c
    ex = (d.max(0).values - d.min(0).values)
    up = torch.zeros(3, device=DEV); up[int(torch.argmin(ex))] = 1.0
    tiles = []
    for name, axis in (("transverse", up), ("longitudinal", torch.eye(3, device=DEV)[int(torch.argmax(ex))])):
        t = d @ axis
        sel = t.abs() < float(ex.max()) * 0.012
        p = d[sel]
        keep = [i for i in range(3) if abs(float(axis[i])) < 0.5]
        u, v = p[:, keep[0]], p[:, keep[1]]
        S = 420
        U = ((u - u.min()) / (u.max() - u.min()).clamp_min(1e-9) * (S - 1)).long()
        V = ((v - v.min()) / (v.max() - v.min()).clamp_min(1e-9) * (S - 1)).long()
        for what, col in (("colour", rgb[sel].cpu().numpy()),
                          ("classes", PALETTE[lab[sel].cpu().numpy() % len(PALETTE)])):
            img = np.ones((S, S, 3), np.float32)
            img[V.cpu().numpy(), U.cpu().numpy()] = col
            img = (img[:, :, ::-1] * 255).astype(np.uint8)
            cv2.putText(img, f"{name} / {what}", (8, S - 10), 0, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
            tiles.append(img)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print(f"  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         k=int(sys.argv[4]) if len(sys.argv) > 4 else 4,
         anchor_pt=sys.argv[5] if len(sys.argv) > 5 else None)
