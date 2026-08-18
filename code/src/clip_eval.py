"""The paper's evaluation: CLIP score against a prompt, and consistency between views.

Everything measured on the interior so far has compared a rendered plane to one of the
photographs, and the photographs are of a different orange -- its segments are not ours, its
membranes are at other angles -- so a per-pixel score there rewards copying a structure the
object does not have, and cannot say whether the structure it does have is right.

The paper does not use a per-pixel score. It reports (Table 1) a CLIP score against a
category-specific prompt, which needs no reference at all, and FID/KID against the collected
photographs, which compare distributions rather than pixels and so do not demand the same
segments. Table 2 adds the cosine similarity between the CLIP features of different views,
which is consistency measured without any reference either.

    python clip_eval.py "the cross-section of an orange" run/snap/iter_0029/h*_init_0.png
"""
import argparse
import sys

import numpy as np
import torch
from PIL import Image

MODEL = "openai/clip-vit-base-patch32"


def embed(paths, model, proc, device):
    ims = [Image.open(p).convert("RGB") for p in paths]
    with torch.no_grad():
        x = proc(images=ims, return_tensors="pt").to(device)
        f = model.get_image_features(**x)
    return f / f.norm(dim=-1, keepdim=True)


def main(prompt, paths, model_name=MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)

    img = embed(paths, model, proc, device)
    with torch.no_grad():
        t = proc(text=[prompt], return_tensors="pt", padding=True).to(device)
        txt = model.get_text_features(**t)
    txt = txt / txt.norm(dim=-1, keepdim=True)

    # CLIPScore as Hessel et al. define it: 100 * max(0, cosine). The 2.5 rescaling they use
    # for captioning is dropped here, as the paper's numbers (24-33) are on this scale.
    per = (100 * (img @ txt.T).clamp_min(0)).reshape(-1)
    # Consistency: how alike the views are to each other in CLIP's feature space. A model whose
    # slices disagree scores low here however plausible each slice is on its own.
    g = img @ img.T
    iu = torch.triu_indices(len(paths), len(paths), offset=1)
    cos = g[iu[0], iu[1]]
    print(f"  {len(paths)} views   CLIP score {per.mean():.1f} "
          f"(min {per.min():.1f}, max {per.max():.1f})   "
          f"cross-view cosine {cos.mean():.3f}")
    return float(per.mean()), float(cos.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    main(a.prompt, a.images)
