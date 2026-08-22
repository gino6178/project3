"""A patch critic small enough to see texture and not identity.

The pixel loss asks this fruit to be that fruit, which is impossible: the photographs are of
different oranges and their sections disagree at 10.7 times the rate two halves of one photograph
do. What the family does agree on is local texture. Measured, on patches with their mean removed, by
asking a linear probe which of two photographs of the SAME family each patch came from:

                        p=8   p=16   p=32   p=64  p=128
    orange transverse    50%    50%    51%    58%    62%
    orange longitudinal  55%    57%    58%    55%    52%
    melon transverse     52%    52%    53%    52%    50%
    melon longitudinal   52%    53%    54%    53%    55%

Chance is 50%. Below about 32 pixels a patch carries no information about which fruit it came from,
and above it the orange's transverse photographs start to be told apart -- 62% at 128. So a critic
with a receptive field under 32 can only judge whether a patch looks like this kind of interior,
which is the question worth asking; a larger one drifts back towards "be this photograph", which is
what the pixel term already does and does badly for the reason above.

The architecture is three stride-2 convolutions and a 1x1 head, whose receptive field is 22 pixels
by construction and is checked by `selftest` rather than asserted. Spectral norm and a hinge loss
because this runs inside a per-scene fit that has no tolerance for a discriminator that wins.

    SEC_CRITIC      weight on the field's adversarial term; 0 (default) leaves the objective alone
    SEC_CRITIC_LR   the critic's own learning rate
    SEC_CRITIC_N    crops per step, from the render and from the photographs alike
    SEC_CRITIC_P    crop side; larger than the receptive field so each crop gives several positions
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

WEIGHT = float(os.environ.get("SEC_CRITIC", "0"))
LR = float(os.environ.get("SEC_CRITIC_LR", "1e-4"))
NCROP = int(os.environ.get("SEC_CRITIC_N", "8"))
P = int(os.environ.get("SEC_CRITIC_P", "64"))
# R1 defaults OFF. With spectral norm already bounding the Lipschitz constant, adding a gradient
# penalty on top pinned the critic: over 4,480 steps its hinge loss never left 1.999, which is what
# a critic that outputs exactly zero scores, and the adaptive weight then amplified that meaningless
# gradient by a hundred. Two Lipschitz controls at once is one too many.
R1 = float(os.environ.get("SEC_CRITIC_R1", "0"))
# and a ceiling on the adaptive weight, so a critic that fails again cannot be amplified into noise
LAM_MAX = float(os.environ.get("SEC_CRITIC_LAM_MAX", "10"))
# Whether the critic also speaks on planes with no photograph, as a multiple of its weight there.
#
# This is the point of having it. A photographed plane already has a pixel term that says what it
# should look like; a plane nobody photographed has nothing, and the blocks are what fills the
# silence. The critic is the only term measured to work here that can speak without a target: it
# was trained on this family's photographs, its receptive field is 22 pixels which is below the 32
# at which a patch starts to identify which fruit it came from, and it is the one statistic of six
# that increased the field's structure rather than smoothing it away.
#
# A diffusion prior was the obvious alternative and was measured instead of assumed. Generic SD-2
# scores a 16-pixel blocked render three times better than a real photograph, and worse the less
# blocked it is -- its noise-prediction error is a measure of how predictable an image is, not of
# how much it looks like a cut orange, so following its score would drive the field towards exactly
# the defect. That is written up rather than worked around.
UNSUP = float(os.environ.get("SEC_CRITIC_UNSUP", "0"))
# Judge the high-frequency subbands rather than the crop. Measured, the field a drawn-plane run
# produces carries 2.2 times the photographs' gradient, so what is wrong with its texture is not
# that there is too little of it -- it is grain where there should be membranes. A critic given the
# crop can win on tone or on smoothness; a critic given only LH, HL and HH has nothing to look at
# except how the high frequencies are arranged, which is the question actually being asked.
#
# One Haar level halves the resolution, so the critic's 22-pixel receptive field covers 44 pixels of
# the render at WAVE=1 and 88 at WAVE=2.
WAVE = int(os.environ.get("SEC_CRITIC_WAVE", "0"))


def dwt(x):
    """One orthonormal Haar level. Returns LL and the three high-frequency subbands stacked."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, torch.cat([lh, hl, hh], -3)


def bands(x, levels=None):
    """The high-frequency subbands of the last level, as 3*C channels."""
    levels = WAVE if levels is None else levels
    for _ in range(max(levels, 1)):
        ll, hi = dwt(x)
        x = ll
    return hi


class PatchCritic(nn.Module):
    """Receptive field 22: conv4/s2 -> 4 -> conv4/s2 -> 10 -> conv4/s2 -> 22 -> 1x1."""

    def __init__(self, ch=(32, 64, 128), in_ch=None):
        super().__init__()
        c0 = in_ch if in_ch is not None else (9 if WAVE > 0 else 3)
        L = []
        for c in ch:
            L += [SN(nn.Conv2d(c0, c, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            c0 = c
        L += [SN(nn.Conv2d(c0, 1, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def crops(img, mask, n=None, p=None, generator=None):
    """`n` crops of side `p` centred on the section, as (n,3,p,p)."""
    n = NCROP if n is None else n
    p = P if p is None else p
    H, W = img.shape[-2:]
    p = min(p, H, W)
    ys, xs = mask.nonzero(as_tuple=True)
    if ys.numel() < 8:
        return None
    k = torch.randint(0, ys.numel(), (n,), device=ys.device, generator=generator)
    out = []
    for i in range(n):
        y0 = int(ys[k[i]]) - p // 2
        x0 = int(xs[k[i]]) - p // 2
        y0 = max(0, min(y0, H - p))
        x0 = max(0, min(x0, W - p))
        out.append(img[..., y0:y0 + p, x0:x0 + p])
    return torch.stack(out)


def prep(x):
    """What the critic is actually shown: the subbands under WAVE, the crop itself otherwise."""
    return _norm(bands(x) if WAVE > 0 else x)


def _norm(x):
    """Each crop centred AND scaled to unit deviation.

    Centring is what stops the critic winning on tone, and tone is exactly the part that says which
    fruit it is -- the orange's three transverse photographs differ in section mean from 0.37 to
    0.145 in blue.

    Scaling is what makes spectral norm and a hinge loss compatible. Interior texture varies by about
    a tenth, so mean-centred crops arrive at magnitude 0.1; four spectrally normed layers have
    Lipschitz constant at most one, so the critic's output cannot exceed about 0.1 either, and the
    hinge asks it to reach 1. It is not that the critic will not separate them -- it cannot, and its
    loss sits at 2.0 for the whole run looking like a critic that has nothing to say. Unit deviation
    puts the input in the range the two mechanisms were designed for together.
    """
    x = x - x.mean((-1, -2), keepdim=True)
    return x / x.std((-1, -2), keepdim=True).clamp_min(1e-3)


class Trainer:
    """The critic and its optimiser, and the two terms that come out of it."""

    def __init__(self, refs_by_kind, device, dtype=torch.float32):
        self.D = {}
        self.opt = {}
        self.real = {}
        self.acc = {}
        self.pending = []
        for kind, refs in refs_by_kind.items():
            d = PatchCritic().to(device=device, dtype=dtype)
            self.D[kind] = d
            self.opt[kind] = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.0, 0.9))
            pool = []
            for r in refs:
                t = r if torch.is_tensor(r) else torch.as_tensor(r)
                if t.dim() == 3 and t.shape[-1] == 3:
                    t = t.permute(2, 0, 1)
                t = t.to(device=device, dtype=dtype)
                if float(t.max()) > 1.5:
                    t = t / 255.0
                m = t.min(0).values < 0.98
                if int(m.sum()) > 64:
                    pool.append((t, m))
            self.real[kind] = pool
            print(f"  critic {kind}: receptive field 22, {len(pool)} photographs to draw real "
                  f"crops from, weight {WEIGHT}, lr {LR}"
                  + (f", judging {WAVE}-level Haar subbands (field {22 * 2 ** WAVE} px of the "
                     f"render)" if WAVE > 0 else ""), flush=True)

    def _real_batch(self, kind, n):
        pool = self.real[kind]
        out = []
        for _ in range(n):
            t, m = pool[np.random.randint(len(pool))]
            c = crops(t, m, 1)
            if c is not None:
                out.append(c[0])
        return torch.stack(out) if out else None

    def adaptive(self, pixel_term, adv_term, img):
        """VQGAN's balance: scale the adversarial term by the ratio of the two gradient norms.

        A fixed weight cannot work here. Measured at initialisation the critic's gradient is a tenth
        of the pixel term's, because an untrained critic outputs nearly nothing; a few hundred steps
        later it is far larger. Any constant chosen from either end is wrong at the other. Taking the
        ratio each step against the same tensor -- the render, which both terms pass through -- keeps
        the adversarial term a correction of a stated size whatever the critic is currently worth.
        """
        try:
            gp = torch.autograd.grad(pixel_term, img, retain_graph=True, create_graph=False)[0]
            ga = torch.autograd.grad(adv_term, img, retain_graph=True, create_graph=False)[0]
        except RuntimeError:
            return 1.0
        lam = gp.norm() / (ga.norm() + 1e-8)
        return float(lam.clamp(0.0, LAM_MAX).detach())

    def step(self, kind, img):
        """The field's term against the critic as it currently stands, and the update deferred.

        The two cannot be interleaved. A training step draws several planes, and updating the critic
        after the first of them changes the parameters the first plane's term was built on -- the
        backward pass then finds a tensor at the wrong version and stops. Every plane in a step is
        therefore scored by the same critic, and the updates it earns are applied once the field has
        taken its own step.

        Returns (field term, d loss) as before; the d loss is from the pending batch and is the same
        number that will be optimised in `flush`.
        """
        if WEIGHT <= 0 or kind not in self.D:
            return img.new_zeros(()), float("nan")
        m = img.min(0).values < 0.98
        fake = crops(img, m)
        if fake is None:
            return img.new_zeros(()), float("nan")
        real = self._real_batch(kind, fake.shape[0])
        if real is None:
            return img.new_zeros(()), float("nan")
        D = self.D[kind]
        with torch.no_grad():
            dl = float(F.relu(1.0 - D(prep(real))).mean()
                       + F.relu(1.0 + D(prep(fake.detach()))).mean())
            self.acc[kind] = float((D(prep(real)).mean() > D(prep(fake)).mean()).float())
        self.pending.append((kind, fake.detach(), real))
        return -D(prep(fake)).mean(), dl

    def flush(self):
        """Apply every update the step earned, after the field has moved."""
        if not self.pending:
            return
        by = {}
        for kind, fake, real in self.pending:
            by.setdefault(kind, []).append((fake, real))
        self.pending = []
        for kind, items in by.items():
            D, opt = self.D[kind], self.opt[kind]
            fake = torch.cat([a for a, _ in items])
            real = torch.cat([b for _, b in items])
            real = prep(real).requires_grad_(R1 > 0)
            opt.zero_grad(set_to_none=True)
            dr, df = D(real), D(prep(fake))
            dl = F.relu(1.0 - dr).mean() + F.relu(1.0 + df).mean()
            if R1 > 0:
                g = torch.autograd.grad(dr.sum(), real, create_graph=True)[0]
                dl = dl + 0.5 * R1 * (g ** 2).sum((1, 2, 3)).mean()
            dl.backward()
            opt.step()


def wavetest():
    """Haar must be orthonormal and invertible, or the subbands are not the image's frequencies."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    ll, hi = dwt(x)
    e_in = float((x ** 2).sum())
    e_out = float((ll ** 2).sum() + (hi ** 2).sum())
    lh, hl, hh = hi.split(3, -3)
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    y = torch.zeros_like(x)
    y[..., 0::2, 0::2], y[..., 0::2, 1::2] = a, b
    y[..., 1::2, 0::2], y[..., 1::2, 1::2] = c, d
    print(f"  Haar: energy in {e_in:.4f} out {e_out:.4f} (ratio {e_out / e_in:.6f}), "
          f"reconstruction error {float((x - y).abs().max()):.2e}")
    flat = torch.ones(1, 3, 32, 32)
    _, hf = dwt(flat)
    print(f"  a flat image has high-frequency energy {float(hf.abs().max()):.2e}")


def selftest():
    """The receptive field is the whole premise, so measure it rather than trust the arithmetic."""
    torch.manual_seed(0)
    d = PatchCritic(in_ch=3)          # measured on the plain critic: WAVE only changes the input
    x = torch.zeros(1, 3, 96, 96, requires_grad=True)
    out = d(x)
    cy, cx = out.shape[-2] // 2, out.shape[-1] // 2
    out[0, 0, cy, cx].backward()
    g = x.grad.abs().sum((0, 1))
    ys, xs = (g > 0).nonzero(as_tuple=True)
    h = int(ys.max() - ys.min()) + 1
    w = int(xs.max() - xs.min()) + 1
    print(f"critic selftest: receptive field {h}x{w} pixels "
          f"({'under 32, as required' if max(h, w) <= 32 else 'TOO LARGE'})")
    assert max(h, w) <= 32, (h, w)
    return True


if __name__ == "__main__":
    selftest()
