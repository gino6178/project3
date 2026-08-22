"""Train the denoiser on one family's photographs and look at what it samples from noise.

Nothing is wired to the field yet. If a model trained on these patches cannot produce a patch that
looks like this fruit's flesh, there is no point going further -- and unlike the pretrained prior,
this one can be checked against the exact data it was trained on.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/workspace/ovoxel_native")
import refsel, patchdiff
from PIL import Image

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = "/workspace/ovoxel_native"
OBJ = os.environ.get("OBJ", "orange_sp")
CHUNK = int(os.environ.get("PD_CHUNK", "500"))
TAG = os.environ.get("PD_TAG", "")          # keeps arms of a comparison out of each other's files
RESUME = os.environ.get("PD_RESUME", "1") == "1"
dev = "cuda"
conf = open(f"{OBJDIR}/{OBJ}.conf").read()

FAMS = [f for f in os.environ.get("PD_FAM", "h,v").split(",") if f]
for key, fam in (("REF_H=", "h"), ("REF_V=", "v")):
    if fam not in FAMS:
        continue
    d = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)][0]
    files = sorted(refsel.photos_in(f"{FN}/{d}"))
    imgs, masks = [], []
    for p in files:
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
        t = torch.as_tensor(a).permute(2, 0, 1).to(dev)
        imgs.append(t)
        masks.append((t.min(0).values < 0.98))
    print(f"\n{OBJ} {fam}: {len(files)} photographs, patch {patchdiff.PATCH}, "
          f"chunk of {CHUNK} steps")
    ck = f"{W}/pd_{OBJ}_{fam}{TAG}.pt"
    init = torch.load(ck, map_location=dev, weights_only=False) \
        if (RESUME and os.path.exists(ck)) else None
    if init is not None and init.get("dim") != patchdiff.DIM:
        raise SystemExit(f"checkpoint was trained at dim {init.get('dim')}, this run is "
                         f"{patchdiff.DIM}; delete {ck} or set PD_DIM to match")
    done = init.get("done", 0) if init is not None else 0

    ev = patchdiff.eval_batch(imgs, masks, device=dev)
    t0 = time.time()
    net, ab, opt = patchdiff.train(imgs, masks, device=dev, steps=CHUNK, init=init)
    before = init.get("eval") if init is not None else None
    after = patchdiff.evaluate(net, ab, ev)
    done += CHUNK
    print(f"  {CHUNK} steps in {time.time() - t0:.0f}s   {done} total")
    if before is None:
        print(f"  held-out loss {after:.4f}  (first chunk)")
    else:
        print(f"  held-out loss {before:.4f} -> {after:.4f}   "
              f"{'still improving' if after < before - 1e-4 else 'FLAT -- more steps buy nothing'}")
    torch.save({"sd": net.state_dict(), "opt": opt.state_dict(), "patch": patchdiff.PATCH,
                "T": patchdiff.T, "dim": patchdiff.DIM, "done": done, "eval": after,
                "shift": patchdiff.SHIFT, "fit": patchdiff.FIT, "mask": patchdiff.MASK}, ck)

    real, rmask = patchdiff.crops_from(imgs, masks, 16)
    real, rmask = real.to(dev), rmask.to(dev)
    # the same shells top and bottom, so the two halves of the sheet are directly comparable and
    # the only thing left to judge is what the model put inside them
    s = patchdiff.sample(net, ab, n=16, device=dev, masks=rmask)

    def grid(x, k=8):
        a = (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        rows = [np.concatenate(list(a[i * k:(i + 1) * k]), 1) for i in range(len(a) // k)]
        return np.concatenate(rows, 0)

    def grad(u):
        u = u.astype(np.float32) / 255
        return float(np.mean([np.abs(np.diff(u, axis=1)).mean(),
                              np.abs(np.diff(u, axis=0)).mean()]))

    gs, gr = grid(s), grid(real)
    print(f"  sampled patches gradient {grad(gs):.5f}   real patches {grad(gr):.5f}")
    big = np.concatenate([gr, np.full((6, gr.shape[1], 3), 255, np.uint8), gs], 0)
    Image.fromarray(big).resize((big.shape[1] * 4, big.shape[0] * 4), Image.NEAREST) \
        .save(f"{W}/pd_{OBJ}_{fam}{TAG}.jpg", quality=92)
    print(f"  SHEET pd_{OBJ}_{fam}{TAG}.jpg  (top two rows real, bottom two sampled)")
