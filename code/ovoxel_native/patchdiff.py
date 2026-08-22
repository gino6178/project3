"""A denoiser trained on this object's own photographs, in the diffusion formulation.

Every attempt to bring a pretrained diffusion model in has failed, and always for the same reason:
its distribution is not ours. Generic SD-2 scores a 16-pixel blocked render three times better than
a real photograph; asked to rebuild a longitudinal section from high noise it collapses to a smooth
blob, because a cut along the axis of an orange is not something it has seen. The prior was the
problem, not the machinery.

So train the denoiser instead. The data is what we already have -- the family's photographs -- and
the object is the ordinary one:

    x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) eps,     minimise || eps_theta(x_t, t) - eps ||^2

Two things make this tractable on two to twelve photographs. It works on patches, so a single 512
photograph is tens of thousands of training examples rather than one; and the patch size is the one
the separability measurement picked out -- below about 32 pixels a patch carries no information
about WHICH fruit it came from (a linear probe scores 50 to 53% against a chance of 50), so a model
of that size learns the family's texture and cannot memorise an individual. Above it the orange's
transverse photographs start to be told apart, 62% at 128.

That is the same reasoning, and the same measurement, that set the patch critic's receptive field --
and the critic is the one term of six that improved the field's structure instead of smoothing it.
This is its generative counterpart.

    PD_PATCH    patch side; 32 or below by the measurement above
    PD_STEPS    training steps
    PD_DIM      width of the network
    PD_T        diffusion steps
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = int(os.environ.get("PD_PATCH", "32"))
STEPS = int(os.environ.get("PD_STEPS", "4000"))
DIM = int(os.environ.get("PD_DIM", "64"))
T = int(os.environ.get("PD_T", "200"))
LR = float(os.environ.get("PD_LR", "2e-4"))
BATCH = int(os.environ.get("PD_BATCH", "64"))
# Whole sections rather than crops of them. At 32 pixels a patch carries the family's texture and
# nothing about which fruit it came from, which is why that size was chosen -- but it also carries
# no structure, and the samples showed it: right material, no membranes, no direction. Structure
# lives at the scale of the section, so the model has to see the section.
#
# Three photographs at that size is not a dataset, and the model will memorise them unless it is
# stopped. What stops it is augmentation: flips, quarter turns, and a scale jitter, so a photograph
# is never presented twice the same way. That is not a substitute for having more photographs and
# the memorisation it leaves behind is measured rather than assumed -- see `pdmemo.py`.
FULL = os.environ.get("PD_FULL", "0") == "1"
AUG = os.environ.get("PD_AUG", "1") == "1"
# The shell is already known. Every cut we will ever ask this model about comes with its polygon, so
# which pixels are inside the object is given, not something to be inferred -- and the first
# whole-section run spent most of its capacity inferring it anyway: the samples got the silhouette
# right and the flesh empty. Under MASK the shell is an input: noise is added only inside it, the
# loss is taken only inside it, and the section is cropped to its own bounding box so the whole
# frame is interior. Same pixel budget, all of it on the part we cannot look up.
MASK = os.environ.get("PD_MASK", "1") == "1"
# How far inside the shell a pixel sits, normalised so that 0 is the cut's edge and 1 its deepest
# point. The shell alone already bought the rind: once the boundary was given, the model learnt
# that white pith sits next to it. Depth states that relation directly instead of making a stack of
# dilated convolutions measure it, and it is the one interior coordinate that carries no angle --
# so unlike a radius-and-angle frame it is not destroyed by the rotations we augment with, and the
# field can compute it from the cut polygon for any cut we ever take.
COORD = os.environ.get("PD_COORD", "0") == "1"
# How a section is placed in the frame. "bbox" crops each section to its own shell and blows it up,
# which spends every pixel on flesh but destroys the one thing the photographs already agree on:
# they were taken through a fixed camera, so within a family they share a scale, and a small polar
# section IS small. The volume sampler slices that same shared frame, so a prior trained on
# per-section blow-ups would be reading the volume at the wrong scale. "frame" keeps the shared one.
FIT = os.environ.get("PD_FIT", "frame")
BG = float(os.environ.get("PD_BG", "0"))   # what sits outside the shell, in [-1, 1]


# Noise is added per pixel and independently; an image is not. The same schedule therefore destroys
# much less of a 128-pixel section than of a 32-pixel one, and we went from 32 to 128 without
# touching it -- measured in `pdsnr.py`, the layout at membrane scale still correlated 0.90 with the
# clean section at 95% of the way through the forward process, against 0.54 at the same relative
# scale at 32 pixels. Everything the model had to invent was crammed into the last few timesteps,
# which uniform sampling of t visits a few percent of the time.
#
# SHIFT scales the signal-to-noise ratio of the whole schedule by (SHIFT)^2, which is the standard
# correction for resolution: at 128 pixels with a 32-pixel reference, SHIFT = 32/128.
SHIFT = float(os.environ.get("PD_SHIFT", "1"))


def schedule(t_max=T, device="cuda", shift=None):
    """Cosine, which behaves better than linear at these very short schedules."""
    s = 0.008
    t = torch.arange(t_max + 1, device=device, dtype=torch.float32) / t_max
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    k = (SHIFT if shift is None else shift) ** 2
    if k != 1.0:
        ab = (k * ab / (1 - ab + k * ab)).clamp(1e-5, 1.0)
    return ab


class Denoiser(nn.Module):
    """Small enough to be trained in minutes on one object's photographs.

    No downsampling: at 32 pixels there is nothing to downsample to, and a stack of dilated
    convolutions covers the patch while keeping every layer at full resolution, which is what
    matters when the output is a per-pixel noise estimate.
    """

    def __init__(self, dim=DIM):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.inp = nn.Conv2d(3 + (1 if MASK else 0) + (1 if COORD else 0), dim, 3, padding=1)
        self.blocks = nn.ModuleList()
        for d in [int(x) for x in os.environ.get("PD_DIL", "1,2,4,8,4,2,1").split(",")]:
            self.blocks.append(nn.ModuleDict(dict(
                c1=nn.Conv2d(dim, dim, 3, padding=d, dilation=d),
                c2=nn.Conv2d(dim, dim, 3, padding=d, dilation=d),
                g1=nn.GroupNorm(8, dim), g2=nn.GroupNorm(8, dim),
                emb=nn.Linear(dim, dim))))
        self.out = nn.Conv2d(dim, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, m=None):
        if m is not None:
            x = torch.cat([x, m], 1)
        e = self.temb(t.view(-1, 1).float())
        h = self.inp(x)
        for b in self.blocks:
            r = h
            h = F.silu(b["g1"](b["c1"](h)))
            h = h + b["emb"](e)[..., None, None]
            h = F.silu(b["g2"](b["c2"](h)))
            h = h + r
        return self.out(h)


def _augment(x):
    """A flip, a quarter turn and a scale jitter, so a photograph is never seen twice the same."""
    if np.random.rand() < 0.5:
        x = torch.flip(x, [-1])
    if np.random.rand() < 0.5:
        x = torch.flip(x, [-2])
    k = int(np.random.randint(4))
    if k:
        x = torch.rot90(x, k, (-2, -1))
    if np.random.rand() < 0.7:
        p = x.shape[-1]
        f = float(np.random.uniform(0.8, 1.25))
        h = max(8, int(x.shape[-2] * f))
        x = F.interpolate(x[None], (h, h), mode="bilinear", align_corners=False)[0]
        x = F.interpolate(x[None], (p, p), mode="area")[0]
    return x


def depth_map(m):
    """Distance to the outside of the shell, normalised to [0, 1]. Computed once per photograph."""
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(m.detach().cpu().numpy().astype(np.uint8))
    return torch.as_tensor(d / max(d.max(), 1e-9), dtype=torch.float32, device=m.device)[None]


def fit_shell(img, m, p, extra=None):
    """The section cropped to its own shell and brought to the lattice's own resolution.

    Every cell a cut face can show, and nothing finer, because nothing finer can be stored. The
    fourth channel is the shell.
    """
    parts = [img[None], m[None, None].float()] + ([extra[None]] if extra is not None else [])
    z = torch.cat(parts, 1)
    if FIT == "frame":
        return F.interpolate(z, (p, p), mode="area")[0]
    ys, xs = m.nonzero(as_tuple=True)
    if ys.numel() < 16:
        return None
    cy, cx = (int(ys.min()) + int(ys.max())) / 2, (int(xs.min()) + int(xs.max())) / 2
    h = max(int(ys.max()) - int(ys.min()), int(xs.max()) - int(xs.min())) / 2 * 1.02
    y0, y1, x0, x1 = round(cy - h), round(cy + h), round(cx - h), round(cx + h)
    pad = (max(0, -x0), max(0, x1 - img.shape[-1]), max(0, -y0), max(0, y1 - img.shape[-2]))
    z = F.pad(z, pad)
    y0, y1, x0, x1 = y0 + pad[2], y1 + pad[2], x0 + pad[0], x1 + pad[0]
    return F.interpolate(z[..., y0:y1, x0:x1], (p, p), mode="area")[0]


def crops_from(images, masks, n, p=None, generator=None, depths=None):
    """`n` training examples as (image, shell[, depth]).

    Whole sections under FULL, crops of them otherwise. `depths` are the maps `depth_map` returns,
    one per photograph, computed once by the caller because there are only ever a handful of them.
    """
    p = PATCH if p is None else p
    out = []
    for _ in range(n):
        k = np.random.randint(len(images))
        img, m = images[k], masks[k]
        if FULL:
            z = fit_shell(img, m, p, depths[k] if (COORD and depths is not None) else None)
            if z is None:
                continue
            out.append(_augment(z) if AUG else z)
            continue
        H, W = img.shape[-2:]
        ys, xs = m[p // 2:H - p // 2, p // 2:W - p // 2].nonzero(as_tuple=True)
        if ys.numel() < 4:
            continue
        j = int(torch.randint(0, ys.numel(), (1,), generator=generator))
        y0, x0 = int(ys[j]), int(xs[j])
        c = [img[:, y0:y0 + p, x0:x0 + p], m[None, y0:y0 + p, x0:x0 + p].float()]
        if COORD and depths is not None:
            c.append(depths[k][:, y0:y0 + p, x0:x0 + p])
        out.append(_augment(torch.cat(c, 0)) if AUG else torch.cat(c, 0))
    if not out:
        return None, None
    z = torch.stack(out)
    return z[:, :3], torch.cat([(z[:, 3:4] > 0.5).float(), z[:, 4:]], 1)


def eval_batch(images, masks, device="cuda", n=32, seed=1234, depths=None):
    """A fixed set of examples, times and noise, held apart from training.

    The training loss is drawn from a fresh random time at every step, and its variance across
    steps is larger than the improvement we are looking for between two short chunks. This batch
    is the same every time it is built, so two numbers taken from it are comparable.
    """
    st_n, st_t = np.random.get_state(), torch.get_rng_state()
    np.random.seed(seed); torch.manual_seed(seed)
    x0, m = crops_from(images, masks, n, depths=depths)
    x0, m = x0.to(device) * 2 - 1, m.to(device)
    if MASK:
        x0 = m[:, :1] * x0 + (1 - m[:, :1]) * BG
    g = torch.Generator(device).manual_seed(seed)
    t = torch.randint(0, T, (n,), device=device, generator=g)
    eps = torch.randn(x0.shape, device=device, generator=g)
    np.random.set_state(st_n); torch.set_rng_state(st_t)
    return x0, m, t, eps


@torch.no_grad()
def evaluate(net, ab, batch):
    """The held-out loss on `eval_batch`, in the same units as the training loss."""
    x0, m, t, eps = batch
    a = ab[t].view(-1, 1, 1, 1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    if MASK:
        xt = m[:, :1] * xt + (1 - m[:, :1]) * x0
        w = m[:, :1].expand_as(eps)
        return float(((net(xt, t.float() / T, m) - eps) ** 2 * w).sum() / w.sum().clamp(min=1))
    return float(F.mse_loss(net(xt, t.float() / T), eps))


def train(images, masks, device="cuda", steps=STEPS, log=print, init=None, depths=None):
    """Fit the denoiser, or carry on fitting one. Returns the model, the schedule and the optimiser.

    `init` is a checkpoint from a previous chunk. Training in chunks and looking at the held-out
    loss after each one is how we find out whether more steps are still buying anything, rather
    than paying for a long run and finding out at the end.
    """
    net = Denoiser().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    if init is not None:
        net.load_state_dict(init["sd"])
        if init.get("opt") is not None:
            opt.load_state_dict(init["opt"])
    ab = schedule(T, device)
    hist = []
    for i in range(steps):
        x0, m = crops_from(images, masks, BATCH, depths=depths)
        if x0 is None:
            continue
        x0, m = x0.to(device) * 2 - 1, m.to(device)
        if MASK:
            x0 = m[:, :1] * x0 + (1 - m[:, :1]) * BG
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        eps = torch.randn_like(x0)
        a = ab[t].view(-1, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        if MASK:
            xt = m[:, :1] * xt + (1 - m[:, :1]) * x0   # noise only where there is something to make
            pred = net(xt, t.float() / T, m)
            w = m[:, :1].expand_as(eps)
            loss = ((pred - eps) ** 2 * w).sum() / w.sum().clamp(min=1)
        else:
            loss = F.mse_loss(net(xt, t.float() / T), eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if (i + 1) % max(steps // 5, 1) == 0:
            log(f"    step {i + 1:5d}/{steps}  loss {np.mean(hist[-200:]):.4f}")
    return net, ab, opt


@torch.no_grad()
def sample(net, ab, n=8, p=None, device="cuda", steps=None, seed=0, masks=None):
    """Ancestral sampling from noise, to see what the model has actually learnt.

    Under MASK the shell comes in as `masks`: the model is asked to fill a given outline, which is
    the only question it will ever be asked in the field.
    """
    p = PATCH if p is None else p
    steps = T if steps is None else steps
    g = torch.Generator(device).manual_seed(seed)
    m = torch.ones(n, 1, p, p, device=device) if masks is None else masks.to(device)
    x = torch.randn(n, 3, p, p, device=device, generator=g)
    if MASK:
        x = m[:, :1] * x + (1 - m[:, :1]) * BG
    for t in reversed(range(steps)):
        tt = torch.full((n,), t, device=device)
        e = net(x, tt.float() / T, m if MASK else None)
        a, ap = ab[t], ab[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)
        if t > 0:
            beta = 1 - a / ap
            noise = torch.randn(x.shape, device=device, generator=g)
            x = ap.sqrt() * x0 + (1 - ap - beta).clamp(min=0).sqrt() * e + beta.sqrt() * noise
        else:
            x = x0
        if MASK:
            x = m[:, :1] * x + (1 - m[:, :1]) * BG   # outside the shell nothing is ever generated
    return (x + 1) / 2
