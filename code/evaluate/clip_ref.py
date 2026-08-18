"""CLIP similarity to the photographs, which is the reference-based score FID cannot be here.

FID and KID compare distributions, and a distribution needs samples: with six reference
photographs the Inception covariance has rank at most five and FID is dominated by its own bias,
which this paper says at length and then reports FID anyway because it is what the prior work
reports. CLIP asks a different question that survives the sample size. An embedding distance needs
no covariance, so six references give six honest numbers rather than one unstable one, and the
score is the mean over renders of the best similarity to any reference -- the same selection rule
the perceptual scores use, so the arms stay comparable.

Two scores, because they fail differently:

    CLIP-I   cosine similarity between a render's image embedding and the photographs'. Reference
             based, and sensitive to whether the render looks like this fruit.
    CLIP-T   cosine similarity to a text prompt, which needs no reference at all. Known to sit at
             its ceiling on this task -- every candidate and the photographs themselves land within
             a few tenths -- so it is reported to show that it cannot separate rather than to claim
             that it does.

    python method/common/eval/clip_ref.py PROMPT REF_DIR "name=render_dir" ...
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import glob
import sys

import numpy as np

sys.path += [_FN_ROOT]

MODEL = _os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14")


def _embed(paths, model, proc, dev, text=None):
    import torch
    from PIL import Image
    if text is not None:
        t = proc(text=[text], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            f = model.get_text_features(**t)
        return torch.nn.functional.normalize(f, dim=-1)
    out = []
    for i in range(0, len(paths), 32):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
        t = proc(images=ims, return_tensors="pt").to(dev)
        with torch.no_grad():
            f = model.get_image_features(**t)
        out.append(torch.nn.functional.normalize(f, dim=-1))
    import torch as _t
    return _t.cat(out)


def main(prompt, ref_dir, *specs):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    refs = sorted(p for p in glob.glob(_os.path.join(ref_dir, "*.png"))
                  + glob.glob(_os.path.join(ref_dir, "*.jpg"))
                  if not any(k in p for k in ("_mask", "_alpha", "_depth")))
    R = _embed(refs, model, proc, dev)
    T = _embed(None, model, proc, dev, text=prompt)
    # the references against each other, leave-one-out: the ceiling any render could reach
    sim = (R @ R.T).cpu().numpy()
    np.fill_diagonal(sim, -1)
    floor_i = float(sim.max(1).mean())
    floor_t = float((R @ T.T).mean())
    print(f"  {len(refs)} references from {ref_dir}")
    print(f"  {'render set':<40} {'CLIP-I':>8} {'CLIP-T':>8}   n")
    print(f"  {'the photographs against each other':<40} {100*floor_i:>7.2f} {100*floor_t:>7.2f}"
          f"   {len(refs)}")
    for spec in specs:
        name, d = spec.split("=", 1)
        ps = sorted(glob.glob(_os.path.join(d, "rh*_init_0.png")))
        if not ps:
            print(f"  {name:<40} {'nothing read':>17}")
            continue
        F = _embed(ps, model, proc, dev)
        ci = float((F @ R.T).max(1).values.mean())
        ct = float((F @ T.T).mean())
        print(f"  {name:<40} {100*ci:>7.2f} {100*ct:>7.2f}   {len(ps)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])
