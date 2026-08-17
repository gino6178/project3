"""Blended latent diffusion for the shell painting: keep, refine, generate.

Adjusting the img2img strength cannot express what the painting needs. A view is looking at
three kinds of surface at once -- some it has never seen, some it saw badly and can improve,
some it saw well and must not touch -- and one number applies the same treatment to all three.
Low, and the unpainted regions inherit whatever the initialisation left there; the navel came
back as a knot of white beads because that is what a blurred navel sharpens into. High, and
nothing is anchored: at 0.95 the sampler returned decorative borders, metal studs and a
watermark reading "YA MAGSS", because depth2img has no mask and a free hand.

TEXTure does it with a mask instead, following Blended Diffusion: at every denoising step the
latent is overwritten, outside the region being painted, with the *current* render noised to
that same step,

    z_i  <-  z_i * m + noise(z_render, t_i) * (1 - m)

so the frozen regions are dragged back to what is already painted at every step rather than
being left to drift, while the painted regions are free. Their mask is

    m = 0            keep
      = checkerboard refine, first 25 steps
      = 1            refine, after 25
      = 1            generate

and the checkerboard is the part worth stealing: over the first steps it interleaves new noise
with the existing content at latent resolution, so what is generated is forced to line up with
what is there before it is allowed to depart from it.

`diffusers` calls `callback_on_step_end` after each scheduler step and takes back whatever
latents it returns, so none of the loop has to be reimplemented.
"""
import torch


class BlendedPaint:
    """Per-step latent blending against a fixed reference latent.

    `mask` is at latent resolution, 1 where the model may paint. `refine` marks where the
    checkerboard applies for the first `cb_steps` steps.
    """

    def __init__(self, pipe, init_latent, gen, refine, cb_steps=25, generator=None):
        self.pipe = pipe
        self.z0 = init_latent
        self.gen = gen
        self.refine = refine
        mask = (gen + refine).clamp(0, 1)
        self.cb_steps = int(cb_steps)
        self.noise = torch.randn(init_latent.shape, generator=generator,
                                 device=init_latent.device, dtype=init_latent.dtype)
        h, w = mask.shape[-2:]
        yy = torch.arange(h, device=mask.device).reshape(-1, 1)
        xx = torch.arange(w, device=mask.device).reshape(1, -1)
        self.checker = ((yy + xx) % 2 == 0).to(mask.dtype).reshape(1, 1, h, w)

    def __call__(self, pipe, i, t, kw):
        z = kw["latents"]
        # The latents now sit at the *next* timestep, so the reference has to be noised to it.
        # `i` counts the img2img loop, which starts partway down `scheduler.timesteps`, so it
        # cannot index that list -- find where this `t` actually is.
        ts = pipe.scheduler.timesteps
        k = int((ts == t.to(ts.device)).nonzero().reshape(-1)[0])
        nxt = ts[k + 1] if k + 1 < len(ts) else None
        if nxt is None:
            zr = self.z0
        else:
            zr = pipe.scheduler.add_noise(self.z0, self.noise.to(z.dtype),
                                          nxt.reshape(1).to(z.device))
        cb = self.checker if i < self.cb_steps else 1.0
        m = (self.gen + self.refine * cb).clamp(0, 1).to(z.dtype)
        kw["latents"] = z * m + zr.to(z.dtype) * (1.0 - m)
        return kw


def encode(pipe, img):
    """Image (PIL, 512x512) -> its latent, scaled as the scheduler expects."""
    import numpy as np
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype="float32") / 255.)
    x = (x.permute(2, 0, 1)[None] * 2.0 - 1.0).to(pipe.device, pipe.vae.dtype)
    with torch.no_grad():
        lat = pipe.vae.encode(x).latent_dist.mode()
    return lat * pipe.vae.config.scaling_factor


def to_latent_mask(m_img, size=64):
    """An image-space weight to latent resolution, by averaging -- the values are kept.

    A hard boundary between the free and the frozen region is what blended diffusion pays for:
    the model sees a discontinuity it did not draw and explains it, so the first run came back
    with grey registration marks ringing the `up` view and yellow starbursts ringing `down`,
    both landing on the circle where the 60 degree cone ends. A ramp gives it nothing to
    explain. Thresholding here would put the hard edge back one stage later."""
    m = m_img.reshape(1, 1, *m_img.shape[-2:]).float()
    return torch.nn.functional.adaptive_avg_pool2d(m, (size, size))
