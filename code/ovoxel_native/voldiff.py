"""Diffuse the lattice itself, supervised by a 2D model along three axes.

The interior is generated, not fitted: a plane nobody photographed has nothing producing its
content, and the blocks are what fills the silence. SDS was the obvious way to bring a diffusion
prior in and it does not work here -- it is an optimisation towards the mode, and at every guidance
setting tried (7.5, 15, 30) it drove a held-out cut to a render gradient of 0.098 to 0.118 against
the photographs' 0.0174, seven times too much texture and visibly unreal. Ordinary sampling from the
same model, at guidance 7.5, produces sections whose gradient is 0.024 -- the right order, and they
look like oranges. The difference is sampling against optimising, and it is the whole reason this
file exists rather than an SDS term.

There is no 3D-native model to sample from, so the 2D one supervises the volume along each axis in
turn and the results are averaged:

    for each round, with the strength decreasing
        for each axis
            take slices, run the 2D model on each, write them back
        average the three axes' volumes
        restore every voxel a supervised plane crosses

The average is what makes it three-dimensional. A voxel appears in one slice per axis, so agreeing
with all three at once is a constraint no single slice imposes, and it is imposed by construction
rather than asked for by a loss. The restore is the other half of the division: the photographs keep
what they can see -- 93.4% of the orange's cells -- and the model supplies the rest.

    VD_ROUNDS     passes over the three axes
    VD_STRENGTH   img2img strength on the first round, halving each round
    VD_SLICES     slices per axis per round; all of them is far too slow, so they are drawn
    VD_STEPS      denoising steps per slice
    VD_CFG        guidance; 7.5 is where the samples were measured to be believable
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL = os.environ.get("VD_MODEL", "sd2-community/stable-diffusion-2-depth")
ROUNDS = int(os.environ.get("VD_ROUNDS", "3"))
STRENGTH = float(os.environ.get("VD_STRENGTH", "0.35"))
NSLICE = int(os.environ.get("VD_SLICES", "24"))
STEPS = int(os.environ.get("VD_STEPS", "30"))
CFG = float(os.environ.get("VD_CFG", "7.5"))
SIDE = int(os.environ.get("VD_SIDE", "512"))
# How much of the photograph-determined field is put back after each round.
#
# 1 restores every cell a supervised plane ever crossed, and with the transverse family jittered
# that is 93.4% of the orange -- which leaves the diffusion 6.6% to write, all of it in the caps,
# where the held-out planes do not pass. The first run changed the held-out transverse cuts by
# nothing at all and the longitudinal ones by 1.3%.
#
# 0 restores nothing and lets the low strength do the anchoring instead, which is what stops img2img
# drifting: every slice already contains the real content, because the field it is a slice of was
# fitted to the photographs, so the model is continuing a picture rather than inventing one.
KEEP = float(os.environ.get("VD_KEEP", "1"))
# How the three axes are combined: "seq" applies each to the result of the last, "avg" averages them.
#
# Averaging is the obvious way to make a voxel agree with all three directions and it destroys the
# thing it was meant to preserve. Three plausible textures averaged is a smooth one -- the same
# arithmetic that makes the blend of two photographs carry 86% of their gradient, and that drawing
# one instead of averaging two was the largest single improvement measured in this work. Run with
# the average, the volume's held-out transverse gradient fell from 0.01553 to 0.01233 while the
# photographs sit at 0.01744: the stripes went and so did the structure.
#
# Sequential keeps every voxel a sample rather than a mean. Each axis sees what the previous one
# wrote, so consistency accumulates instead of being averaged in.
MODE = os.environ.get("VD_MODE", "seq")
# Which axes to diffuse along. Slices perpendicular to the polar axis ARE transverse cuts, and the
# transverse family is the one the photographs supervise well -- 92% of the cells, each plane
# jittered. Running the model over those slices overwrites the best-determined part of the field,
# and it shows: with all three axes the held-out transverse gradient fell from 0.01554 to 0.01304
# while the longitudinal one rose past its baseline. Leaving that axis out puts the generator where
# the supervision is thin and keeps it away from where the supervision is good.
AXES = [int(a) for a in os.environ.get("VD_AXES", "0,1,2").split(",")]
# Take only a band of what the model returns, and keep our own layout.
#
# Both ways of using img2img whole failed, and they failed in opposite directions. At low strength
# the stripes survive the round trip untouched -- the model is a preserver there, not a generator.
# At high strength the section collapses to a smooth blob, because a longitudinal cut of an orange
# is not something a general model has seen; it draws transverse discs beautifully and has no mode
# to fall into for this.
#
# What the model is good at, in both regimes, is local texture. So take the band it is good at and
# leave the rest: subtract a small blur to drop detail finer than a cell, which the lattice cannot
# store anyway, and subtract a large blur to drop the layout, which the photographs already decided.
# What is left is added to our own slice.
#
# The band matters because of what the resolution allows. A photograph reduced to the lattice's 120
# cells keeps 49% of its gradient, and our renders already carry 1.8 times THAT -- so the problem was
# never too little high frequency. It is that the high frequency present is stripes rather than
# segments, and this replaces one with the other instead of adding to either.
BAND = os.environ.get("VD_BAND", "0") == "1"
BAND_LO = float(os.environ.get("VD_BAND_LO", "3"))    # finer than a cell at 512: unstorable
BAND_HI = float(os.environ.get("VD_BAND_HI", "20"))   # coarser than this: the photographs' job
BAND_W = float(os.environ.get("VD_BAND_W", "1.0"))
# The prompt has to follow the slice's own direction.
#
# A slice perpendicular to the polar axis is a transverse cut and shows radial segment membranes; a
# slice containing the axis is a longitudinal cut and shows the segments running lengthwise. Asking
# for one on the other is asking the model to draw the wrong fruit's geometry, and it obliges: the
# first band-transfer figure put radial membranes onto a longitudinal slice and it looked like
# structure appearing where there had been stripes. It was the wrong structure.
#
# The object's own conf carries the phrase with a {view_cut} placeholder for exactly this reason --
# "the {view_cut} cross-sectional view of a navel orange, ..." -- so the two prompts are that string
# substituted, and a fine-tune bound to either phrase would be reached by the right one.
POLAR_AXIS = int(os.environ.get("VD_POLAR_AXIS", "1"))


def prompt_for(ax, base):
    """`base` carries {view_cut}; the axis decides which cut it is."""
    cut = "horizontal" if ax == POLAR_AXIS else "vertical"
    return base.replace("{view_cut}", cut) if "{view_cut}" in base else base
NEG = "watermark, text, letters, logo, signature, border, frame"


def _pipe(device="cuda"):
    """The depth pipeline, because this checkpoint's UNet takes five channels.

    `StableDiffusionImg2ImgPipeline.from_pretrained` accepts this model without complaint and then
    fails at the first convolution -- the weights want 4 latent channels plus 1 of depth. Depth for a
    slice through a volume is not a meaningful quantity, so the pipeline's own estimator supplies it
    and what it produces is treated as conditioning noise rather than as geometry.
    """
    from diffusers import StableDiffusionDepth2ImgPipeline
    p = StableDiffusionDepth2ImgPipeline.from_pretrained(MODEL, dtype=torch.float16).to(device)
    p.set_progress_bar_config(disable=True)
    return p


def _slice(vol, ax, i):
    return vol.index_select(ax, torch.tensor([i], device=vol.device)).squeeze(ax)


def _put(vol, ax, i, sl):
    idx = torch.tensor([i], device=vol.device)
    vol.index_copy_(ax, idx, sl.unsqueeze(ax))


def _blur(x, s):
    import torch.nn.functional as _F
    k = int(2 * round(3 * s) + 1)
    g = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(g ** 2) / (2 * s * s)); g = g / g.sum()
    c = x.shape[1]
    x = _F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), padding=(0, k // 2), groups=c)
    return _F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), padding=(k // 2, 0), groups=c)


def denoise_slice(pipe, sl, prompt, strength, gen):
    """One slice through the model and back at its own size.

    The slice is resized to the model's own resolution and back. Anything else asks a 512-trained
    network to work at 120 pixels, where it has never seen a fruit.
    """
    a = (sl.clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(a).resize((SIDE, SIDE), Image.BICUBIC)
    out = pipe(prompt=prompt, image=im, depth_map=None, negative_prompt=NEG, strength=strength,
               guidance_scale=CFG, num_inference_steps=STEPS, generator=gen).images[0]
    if not BAND:
        out = out.resize((a.shape[1], a.shape[0]), Image.BICUBIC)
        return torch.as_tensor(np.asarray(out, np.float32) / 255.,
                               device=sl.device, dtype=sl.dtype)
    # band-pass at the model's own resolution, then bring the band down to the slice's
    o = torch.as_tensor(np.asarray(out, np.float32) / 255.,
                        device=sl.device, dtype=sl.dtype).permute(2, 0, 1)[None]
    band = _blur(o, BAND_LO) - _blur(o, BAND_HI)
    import torch.nn.functional as _F
    band = _F.interpolate(band, (a.shape[0], a.shape[1]), mode="area")[0].permute(1, 2, 0)
    return (sl + BAND_W * band).clamp(0, 1)


def run(vol, solid, known_vol, prompt, device="cuda", seed=0, log=print):
    """The loop. `vol` is (X,Y,Z,3); `known_vol` is a boolean of the same shape."""
    pipe = _pipe(device)
    gen = torch.Generator(device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = vol.clone()
    s = STRENGTH
    for r in range(ROUNDS):
        acc = torch.zeros_like(vol)
        cnt = torch.zeros(vol.shape[:3] + (1,), device=device, dtype=vol.dtype)
        nw = 0
        for ax in AXES:
            # slices that contain something, drawn rather than taken in order so successive rounds
            # do not keep visiting the same planes
            has = (solid.sum(dim=[d for d in range(3) if d != ax]) > 0).nonzero()[:, 0]
            pick = has[rng.choice(len(has), min(NSLICE, len(has)), replace=False)]
            pr = prompt_for(ax, prompt)
            for i in pick.tolist():
                sl = _slice(vol, ax, i)
                out = denoise_slice(pipe, sl, pr, s, gen)
                if MODE == "seq":
                    # written straight back, so the next axis reads it
                    _put(vol, ax, i, torch.where(_slice(solid, ax, i)[..., None],
                                                 out, _slice(vol, ax, i)))
                    nw += int(_slice(solid, ax, i).sum())
                else:
                    sel = [slice(None)] * 3
                    sel[ax] = slice(i, i + 1)
                    acc[tuple(sel)] += out.unsqueeze(ax)
                    cnt[tuple(sel)] += 1
        if MODE != "seq":
            touched = cnt[..., 0] > 0
            vol = torch.where(touched[..., None], acc / cnt.clamp(min=1), vol)
            nw = int(touched.sum())
        if KEEP >= 1:
            vol = torch.where(known_vol[..., None], keep, vol)
        elif KEEP > 0:
            vol = torch.where(known_vol[..., None], KEEP * keep + (1 - KEEP) * vol, vol)
        vol = torch.where(solid[..., None], vol, torch.ones_like(vol))
        log(f"  round {r + 1}/{ROUNDS}  {MODE}  strength {s:.3f}  "
            f"{nw:,} voxel writes  "
            f"mean change {float((vol - keep).abs().mean()):.5f}")
        s *= 0.5
    return vol
