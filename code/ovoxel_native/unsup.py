"""A prior on planes nobody photographed, which is the family of remedy this build has none of.

Every term in the objective acts on the 26 planes that have a photograph. A cell between them is
touched only when a jittered plane happens to cross it, and otherwise it is free -- so the training
can go on sharpening the supervised sections by moving cells that nothing else refers to. That is
visible: the held-out probe reaches its best at outer 20 in most arms and then rises 2 to 3% over
the remaining 305, while the training loss falls 27%. The arms whose planes reach more of the object
degrade least (ovbal 0.4%, az3 1.1%, against ov2 3.1%), which is the same statement from the other
side.

The sparse-view literature answers this by regularising on views that have no ground truth --
RegNeRF renders random unobserved viewpoints and applies a depth-smoothness prior plus a learned
likelihood on the patches; DietNeRF asks for semantic agreement across poses. The requirement is
that the prior says something true without knowing the answer for that view.

What is true of an unphotographed cut of an orange is not where its structure sits -- that is
exactly what is unknown -- but what it is made of: the same kinds of patch as the photographs of
that family. Chamfer between patch sets asks that and nothing else, and this project has already
measured why it is the right one of the three distances here and the wrong one for a supervised
plane:

    between-photograph over within-photograph distance, orange transverse references
      sliced Wasserstein 35.4     Jensen-Shannon 60.1     MSE 10.7     Chamfer 2.44

  Chamfer is the only one below the pixel loss, because it asks about the vocabulary of patches and
  not their amounts or positions. As a target on a supervised plane that is a defect -- it cannot
  place structure, so it let the interior smooth out and cost the detail column. On a plane where
  no position is known, that same insensitivity is the point.

The reference set is the whole family pooled, not one photograph. A random azimuth's section is not
any particular photograph and asking it to be one is the error the depth blend already had to fix.

    SEC_UNSUP        weight; 0 (default) leaves the objective as it was
    SEC_UNSUP_EVERY  apply it every k steps, since it costs one extra render
    SEC_UNSUP_N      patches per side
"""
import os

import numpy as np
import torch

import patchdist

WEIGHT = float(os.environ.get("SEC_UNSUP", "0"))
EVERY = int(os.environ.get("SEC_UNSUP_EVERY", "1"))
NPATCH = int(os.environ.get("SEC_UNSUP_N", "512"))
KIND = os.environ.get("SEC_UNSUP_KIND", "chamfer")

_POOL = {}


def pool(refs, key, n=None, p=None):
    """The family's patches, pooled once and kept.

    `refs` is the list of (3,H,W) reference arrays the trainer already holds. Patches are drawn from
    each and concatenated, so the set describes the family rather than any member of it.
    """
    if key in _POOL:
        return _POOL[key]
    n = NPATCH if n is None else n
    per = max(n // max(len(refs), 1), 16)
    out = []
    for r in refs:
        t = r if torch.is_tensor(r) else torch.as_tensor(r)
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        t = t.float()
        if float(t.max()) > 1.5:
            t = t / 255.0
        m = (t.min(0).values < 0.98).float()
        out.append(patchdist.patches(t, m, per, p))
    _POOL[key] = torch.cat(out)
    print(f"  unsupervised prior: {len(_POOL[key]):,} pooled {KIND} reference patches for {key} "
          f"from {len(refs)} photographs, weight {WEIGHT} every {EVERY} step(s)", flush=True)
    return _POOL[key]


def penalty(img, refs, key, n=None, p=None):
    """Chamfer from this render's patches to the family's, with no target image anywhere."""
    if WEIGHT <= 0:
        return img.new_zeros(())
    Y = pool(refs, key, n, p).to(img.device, img.dtype)
    m = (img.min(0).values < 0.98).float()
    if float(m.sum()) < 400:
        return img.new_zeros(())
    X = patchdist.patches(img, m, NPATCH if n is None else n, p)
    if KIND == "sw":
        return patchdist.sliced_wasserstein(X, Y)
    if KIND == "js":
        return patchdist.sliced_js(X, Y)
    return patchdist.chamfer(X, Y)


def sample_plane(kind, C, H_LO, NH, NV, rng, axis=None, centre=None, az_spacing=None):
    """A plane of that family at a position no photograph covers, and no probe plane either.

    Transverse: a depth drawn uniformly across the supervised band rather than at one of its
    depths. Longitudinal: an azimuth drawn uniformly, turned about the axis, so it is still a
    central section and still the kind of cut the family's photographs are of.
    """
    if kind == "h":
        hd = C["h_planes"][:, 3]
        lo, hi = float(hd[H_LO]), float(hd[H_LO + NH - 1])
        d = lo + (hi - lo) * float(rng.random())
        return C["h_planes"][H_LO, :3], d, C["h_mvp"]
    i = int(rng.integers(0, NV))
    import azjitter
    f = (float(rng.random()) - 0.5) * 2.0
    m2, n2, d2 = azjitter.turn(C["v_mvp"][i], C["v_planes"][i, :3], float(C["v_planes"][i, 3]),
                               axis, centre, f * az_spacing)
    return n2, d2, m2
