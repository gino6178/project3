"""How much of the photographs did the whole-section denoiser simply copy?

A model trained on three images can pass every texture test by reproducing those three images, and
at 128 pixels a section is large enough to identify the individual fruit -- which is exactly the
size we moved to. So measure it: draw samples, and compare each one to the training photographs.

The reference is the photographs' distance to each other. Two different sections of the same orange
are not identical, and a sample that sits no closer to a photograph than the photographs sit to each
other has not copied anything. A sample that sits much closer has.

Distance is taken over the eight flips and quarter turns, because training augments with those, and
is reported twice: on the image, and on its per-channel mean alone. The second is the material
(the colour of this fruit's flesh), which we WANT the model to reproduce; the first is the layout,
which we do not.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refsel, patchdiff
from PIL import Image

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
N = int(os.environ.get("PD_MEMO_N", "32"))
dev = "cuda" if torch.cuda.is_available() else "cpu"


def dihedral(x):
    """The eight flips and quarter turns, as one batch."""
    out = []
    for k in range(4):
        r = torch.rot90(x, k, (-2, -1))
        out += [r, torch.flip(r, (-1,))]
    return torch.stack(out)


def best_dist(a, bank):
    """Smallest RMS distance from `a` to any image in `bank`, over the eight transforms."""
    d = dihedral(a)                                   # (8,3,P,P)
    e = ((d[:, None] - bank[None, :]) ** 2).mean((2, 3, 4)).sqrt()   # (8,K)
    return float(e.min())


def colour_dist(a, bank):
    ca = a.mean((-2, -1))
    return float(((ca[None] - bank.mean((-2, -1))) ** 2).mean(-1).sqrt().min())


for key, fam in (("REF_H=", "h"), ("REF_V=", "v")):
    ck = f"{W}/pd_{OBJ}_{fam}.pt"
    if not os.path.exists(ck):
        print(f"{OBJ} {fam}: no checkpoint, skipped")
        continue
    st = torch.load(ck, map_location=dev)
    P = st["patch"]
    net = patchdiff.Denoiser().to(dev)
    net.load_state_dict(st["sd"])
    net.eval()

    conf = open(f"{OBJDIR}/{OBJ}.conf").read()
    d = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)][0]
    files = sorted(refsel.photos_in(f"{FN}/{d}"))
    bank, shells = [], []
    for p in files:
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
        t = torch.as_tensor(a).permute(2, 0, 1).to(dev)
        z = patchdiff.fit_shell(t, (t.min(0).values < 0.98), P)
        bank.append(z[:3]); shells.append((z[3:] > 0.5).float())
    bank, shells = torch.stack(bank), torch.stack(shells)

    ab = patchdiff.schedule(device=dev)
    # the model is given the photographs' own shells, so any difference left is the flesh
    ms = shells[torch.randint(0, len(shells), (N,), device=shells.device)]
    s = patchdiff.sample(net, ab, n=N, p=P, device=dev, masks=ms).clamp(0, 1)

    ds = [best_dist(s[i], bank) for i in range(len(s))]
    cs = [colour_dist(s[i], bank) for i in range(len(s))]
    # leave-one-out: how far apart are the photographs themselves?
    dr = [best_dist(bank[i], torch.cat([bank[:i], bank[i + 1:]])) for i in range(len(bank))]
    cr = [colour_dist(bank[i], torch.cat([bank[:i], bank[i + 1:]])) for i in range(len(bank))]

    print(f"\n{OBJ} {fam}: {len(files)} photographs at {P}px, {N} samples")
    print(f"  sample -> nearest photograph   image {np.mean(ds):.4f} +- {np.std(ds):.4f}"
          f"   colour {np.mean(cs):.4f}")
    print(f"  photograph -> other photograph image {np.mean(dr):.4f} +- {np.std(dr):.4f}"
          f"   colour {np.mean(cr):.4f}")
    print(f"  ratio (image) {np.mean(ds) / max(np.mean(dr), 1e-9):.2f}x"
          f"   -- below 1 means the samples sit closer to the data than the data sits to itself")
