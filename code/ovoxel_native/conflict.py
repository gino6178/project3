"""How hard do the two families pull against each other, and does Chamfer let go?

    TAG=ov2 python conflict.py OBJ

The argument for a distributional loss here is not the one it was first tried under. MSE asks the
two families to agree at a PIXEL, and they see different material -- a transverse cut of an orange
shows the radial walls, a longitudinal one the axial fibres -- so a cell both cross is pulled two
ways and settles on a compromise neither photograph contains. Chamfer asks only that a patch belong
to the vocabulary, so a cell should be free to be a legitimate patch of either kind, and the
conflict should relax. That is a claim about gradients, and it has not been measured.

So measure it. For a transverse plane and a longitudinal plane that cross the same cells, take the
gradient each loss puts on the interior colour, and report the cosine between them on the cells
both touch. Aligned is no conflict; opposite is maximal. The same pair, the same cells, once under
the pixel loss and once under Chamfer.

The gradient is taken on the decoded colour rather than on the latents: the latents are shared
through the decoder, which mixes cells that no plane relates, and the question is about what the
two families ask of one cell.
"""
import os
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr

sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/rebuild/project3/code/src")
import ovcut
import ovnative as ON
import patchdist
import refsel
import secloss
import section_match as sm

W = "/workspace/ovoxel_native"
FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
TAG = os.environ.get("TAG", "ov2")
RES = 512
obj = sys.argv[1] if len(sys.argv) > 1 else "orange_sp"


def refdir(which):
    import re
    m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
    return os.path.join(FN, m.group(1))


ON.FDG = ON._load_ovoxel()
st = ovcut.load(obj, TAG)
C = np.load(f"{W}/cams_{obj}.npz")
glctx = dr.RasterizeCudaContext(device="cuda")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH, NV = H_HI - H_LO, len(C["v_mvp"])

base = st["interior"].detach().clone()


def grad_of(kind, idx, lossfn):
    st["interior"] = base.clone().requires_grad_(True)
    if kind == "h":
        mvp, pl = C["h_mvp"], C["h_planes"][H_LO + idx]
        ref = refsel.as_array(refsel.solved_photo(refdir("REF_H"), idx, NH), RES)
    else:
        mvp, pl = C["v_mvp"][idx], C["v_planes"][idx]
        ref = refsel.as_array(refsel.photo(refdir("REF_V"), idx, NV), RES)
    n = torch.as_tensor(pl[:3], dtype=torch.float32, device="cuda")
    img, al, _, _ = ON.render_section(
        st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"),
        n, float(pl[3]), RES)
    with torch.no_grad():
        tgt = sm.section_target(img, ref, alpha=al)
    loss = lossfn(img, tgt)
    g, = torch.autograd.grad(loss, st["interior"], allow_unused=True)
    return torch.zeros_like(base) if g is None else g.detach()


def report(name, lossfn, pairs=6):
    cos, share, mag = [], [], []
    for k in range(pairs):
        gh = grad_of("h", (k * 3) % NH, lossfn)
        gv = grad_of("v", (k * 2) % NV, lossfn)
        th = gh.abs().sum(-1) > 0
        tv = gv.abs().sum(-1) > 0
        both = th & tv
        if int(both.sum()) < 50:
            continue
        a, b = gh[both], gv[both]
        c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        cos.append(float(c.mean()))
        share.append(float(both.sum()) / float((th | tv).sum()))
        mag.append(float(a.norm(dim=-1).mean() + b.norm(dim=-1).mean()) / 2)
    if not cos:
        print(f"  {name:28s} no pair shared enough cells"); return
    print(f"  {name:28s} cosine {np.mean(cos):+.4f}   cells in both families "
          f"{100 * np.mean(share):5.1f}%   mean |grad| {np.mean(mag):.3e}")


print(f"  {obj}, {TAG}: {len(base):,} interior cells, {NH} transverse and {NV} longitudinal planes")
report("pixel loss (SSIM + MSE)", lambda i, t: secloss.patch_loss(i, t))
for kind in ("chamfer", "sw", "js"):
    report(f"{kind} on the patches", lambda i, t, k=kind: patchdist.distance(i, t, k))
print("\n  cosine +1: the two families ask for the same change and there is no conflict"
      "\n  cosine  0: they ask for unrelated changes"
      "\n  cosine -1: whatever one gains the other loses")
