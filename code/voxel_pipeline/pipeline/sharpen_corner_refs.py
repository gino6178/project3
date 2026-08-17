"""Restore the peel texture on the blended corner references, without letting them drift.

Fourteen directions is what the skin initialiser wants; the question has only ever been where
the eight corners come from. Sampling them independently at strength 0.95 is what the code
turned off, and the measurement says why: each direction returns its own fruit and the average
of eight fruits has no texture left, gradient 0.034 against the six faces' 0.141 -- a flat
orange disc. Blending them from the three faces they sit between keeps the fruit but inherits
the same flatness for the same reason, an average of three images being smoother than any of
them, and it measured 0.132.

So blend first and then sample lightly. The blend fixes the identity -- every pixel came from
a reference already accepted -- and a low strength keeps it: at 0.3 the sampler restores the
dimpling it recognises rather than composing a new fruit, because most of the latent it starts
from is the blend's own. The strength is the whole control here, and it is the parameter that
was wrong before, not the idea.

    python voxel_pipeline/pipeline/sharpen_corner_refs.py cube_or14b_prep cube_or14s_prep \\
        config/cube_prompts/orange.json 0.3
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

NEG = ("cross section, cut in half, sliced, halved, wedge, segments, pulp, interior, "
       "flesh, seeds, watermark, text, drop shadow, cast shadow, shadow, vignette, "
       "dark rim, specular highlight, glare, reflection, gradient background")


def main(src_dir, out_dir, prompts_json, strength=0.3, seed=1234,
         model="sd2-community/stable-diffusion-2-depth"):
    os.makedirs(out_dir, exist_ok=True)
    prompts = json.load(open(prompts_json))
    prompt = prompts.get("front") or next(iter(prompts.values()))
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(model).to("cuda:0")

    for f in sorted(os.listdir(src_dir)):
        if not f.endswith("_ref.png"):
            continue
        name = f[:-len("_ref.png")]
        img = Image.open(os.path.join(src_dir, f)).convert("RGB")
        if not name.startswith("c"):
            img.save(os.path.join(out_dir, f))          # the six faces pass through untouched
            continue
        orig = img.size
        sq = img.resize((512, 512)) if orig != (512, 512) else img
        t = torch.from_numpy(np.asarray(sq, np.float32) / 255.).permute(2, 0, 1)[None]
        d = pipe.depth_estimator(F.interpolate(t, size=(384, 384), mode="bilinear",
                                               align_corners=False).to("cuda:0")).predicted_depth
        d = F.interpolate(d[None], size=(512, 512), mode="bilinear",
                          align_corners=False).squeeze(0)
        g = torch.Generator(pipe.device).manual_seed(int(seed))
        # One seed for all eight. Reseeding per direction is what made them different fruits.
        r = pipe(prompt=prompt, image=sq, depth_map=d, negative_prompt=NEG,
                 strength=float(strength), guidance_scale=12, num_inference_steps=50,
                 generator=g, return_dict=False)
        out = (r[0][0] if isinstance(r, tuple) else r.images[0])
        (out.resize(orig) if out.size != orig else out).save(os.path.join(out_dir, f))
        print(f"  {name} sharpened at strength {strength}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.3)
