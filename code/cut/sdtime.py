"""What a generative interior costs per target, on this machine, with the prior work's own call.

    python sdtime.py [N]

FruitNinja supervises a section by sampling a depth-conditioned latent diffusion model and using
what comes back as the target. The call is the one in `sds_demo`: 512 x 512, fifty inference
steps, guidance 10, conditioned on the section's own depth map. This times that call and nothing
else, so the number is the marginal cost of one supervised section under a generative interior,
independent of which weights are loaded -- the schedule and the architecture set the cost, and
a fine-tuned checkpoint is the same U-Net.

What this does not measure, and what therefore has to be added to it, is the fine-tuning of that
checkpoint on the object's own photographs. Neither the data nor the schedule for that stage is
released, so it is a cost we can name and not one we can time.
"""
import os
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL = os.environ.get("SD_MODEL_H", "sd2-community/stable-diffusion-2-depth")


def main(n=5):
    from diffusers import StableDiffusionDepth2ImgPipeline
    t0 = time.perf_counter()
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(MODEL).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    print(f"  {MODEL} loaded in {time.perf_counter()-t0:.1f} s")

    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((512, 512, 3)) * 60 + 180).astype("uint8"))
    depth = torch.from_numpy(rng.random((1, 512, 512)).astype("float32")).to("cuda:0")
    prompt = ("the horizontal cross-sectional view of a watermelon, red flesh with scattered "
              "black seeds, white pith and green rind, macro photo, detailed")

    def one(strength):
        g = torch.Generator(device="cuda:0").manual_seed(0)
        return pipe(prompt=prompt, image=img, depth_map=depth, strength=strength,
                    guidance_scale=10, num_inference_steps=50, generator=g, return_dict=False)

    for s in (0.6, 0.25):
        one(s); torch.cuda.synchronize()                      # warm up, per strength
        ts = []
        for _ in range(int(n)):
            t = time.perf_counter(); one(s); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        m = statistics.median(ts)
        print(f"  strength {s:4.2f}: {m*1000:7.0f} ms per target   "
              f"(min {min(ts)*1000:.0f}, max {max(ts)*1000:.0f})")
        print(f"    one pass over 16 planes x 2 families = 32 targets: {32*m:6.1f} s")
        print(f"    the paper regenerates every plane every iteration; at 200 iterations "
              f"that is {200*32*m/3600:6.1f} h")
    print(f"  pipeline resident: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB allocated")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 5)
