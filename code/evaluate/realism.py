"""Two instruments for cut-face realism that survive six reference photographs.

FID and KID need a distribution, and six references do not make one: the Inception covariance has
rank at most five, FID is dominated by its own bias, and KID runs at subset size three. CLIP is
robust at that size and turns out not to discriminate, compressing every candidate and the
references themselves into the top two percent of its range. So this measures the same renders two
other ways, chosen because neither needs a covariance and both answer "is this one real" rather
than "are these two distributions close".

    precision / recall   Kynkaanniemi et al.'s manifold estimate. The real manifold is the union
                         of balls around each reference, each of radius its own k-th nearest
                         neighbour; precision is the fraction of renders inside it, which is
                         fidelity, and recall is the fraction of references inside the renders'
                         manifold, which is coverage. At six references k is small and the estimate
                         is coarse, but it is an estimate of the right quantity rather than a
                         biased estimate of a different one, and precision is exactly the number
                         "how realistic" is asking for.

    DreamSim             a perceptual distance trained on human two-alternative forced choice
                         judgements, which is the judgement this comparison actually wants. Where
                         LPIPS and CLIP are repurposed, this one was built for it. Reported as the
                         mean over renders of the distance to the nearest reference, so a lower
                         number is a render that some photograph agrees with.

    python method/common/eval/realism.py REF_DIR "name=render_dir" ...
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np
import torch

sys.path += [_FN_ROOT]

K = int(_os.environ.get("PR_K", "3"))


def _paths(d, pat="*"):
    out = [p for p in sorted(glob.glob(_os.path.join(d, pat)))
           if _os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg")
           and not _os.path.splitext(_os.path.basename(p))[0].endswith(
               ("_depth", "_mask", "_alpha", "_ref"))]
    return out


def _inception(paths, dev):
    """The same 2048-d pool3 features FID is computed from, so the two are on one footing."""
    from PIL import Image
    from torchmetrics.image.fid import NoTrainInceptionV3
    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(dev).eval()
    out = []
    for i in range(0, len(paths), 32):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((299, 299), Image.LANCZOS)).copy()
               for p in paths[i:i + 32]]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            out.append(net(x).double().cpu())
    return torch.cat(out).numpy()


def _pr(real, fake, k=K):
    """Improved precision and recall: a point is inside a manifold when it lies within some
    sample's own k-th-nearest-neighbour radius."""
    def radii(a):
        d = np.linalg.norm(a[:, None] - a[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        kk = min(k, len(a) - 1)
        return np.sort(d, axis=1)[:, kk - 1]
    dr, df = radii(real), radii(fake)
    d = np.linalg.norm(fake[:, None] - real[None], axis=-1)      # fake x real
    precision = float((d <= dr[None, :]).any(1).mean())
    recall = float((d.T <= df[None, :]).any(1).mean())
    return precision, recall


def _dreamsim(ref_paths, paths, dev):
    from PIL import Image
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device=dev)
    R = torch.cat([preprocess(Image.open(p).convert("RGB")).to(dev) for p in ref_paths])
    best = []
    for i in range(0, len(paths), 16):
        X = torch.cat([preprocess(Image.open(p).convert("RGB")).to(dev) for p in paths[i:i + 16]])
        with torch.no_grad():
            d = torch.stack([model(X, R[j:j + 1].expand_as(X)) for j in range(len(R))], 1)
        best.append(d.min(1).values.cpu())
    return float(torch.cat(best).mean())


def main(ref_dir, *specs):
    # precision and recall need a manifold and a manifold needs samples; DreamSim is a pairwise
    # distance and needs one reference. The doughnut has one reference image per family, which is
    # why it has never had an appearance number, and that is exactly the case worth reporting.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    refs = _paths(ref_dir)
    R = _inception(refs, dev)
    have_ds = True
    try:
        import dreamsim  # noqa: F401
    except ImportError:
        have_ds = False

    print(f"  {len(refs)} references, k = {K}"
          + ("" if have_ds else "   (dreamsim not installed, that column is skipped)"))
    hdr = f"  {'render set':<40} {'precision':>10} {'recall':>8}"
    print(hdr + (f" {'DreamSim':>10}" if have_ds else ""))

    # the references against themselves, split in half, as the floor every column is read against
    h = len(refs) // 2
    p0, r0 = _pr(R[:h], R[h:]) if h > K else (float("nan"), float("nan"))
    line = f"  {'the photographs, split in half':<40} {p0:>10.3f} {r0:>8.3f}"
    if have_ds and h:
        line += f" {_dreamsim(refs[:h], refs[h:], dev):>10.4f}"
    print(line)

    for spec in specs:
        name, d = spec.split("=", 1)
        ps = _paths(d, "rh*_init_0.png")
        if not ps:
            print(f"  {name:<40} {'nothing read':>19}")
            continue
        F = _inception(ps, dev)
        pr, rc = _pr(R, F) if len(R) > K else (float("nan"), float("nan"))
        line = f"  {name:<40} {pr:>10.3f} {rc:>8.3f}"
        if have_ds:
            line += f" {_dreamsim(refs, ps, dev):>10.4f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
