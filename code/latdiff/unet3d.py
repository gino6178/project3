"""A small 3-D UNet over the lattice latent.

Deliberately shallow. The object is about 117 cells across and this has three levels of stride-2
downsampling over 3x3x3 convolutions, so its receptive field is a few tens of cells -- a patch of
the object, not the object. That is the property SinDiffusion identifies as the one that lets a
diffusion model be trained on a single example without memorising it, and there is exactly one
example here: the lattice stage 1 fitted.
"""
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    f = th.exp(-math.log(10000) * th.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * f[None]
    return th.cat([th.cos(a), th.sin(a)], dim=-1)


class Res(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.e(F.silu(e))[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    def __init__(self, cin, cout, base=64, mults=(1, 2, 4)):
        super().__init__()
        emb = base * 4
        self.emb = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.base = base
        chs = [base * m for m in mults]
        self.inp = nn.Conv3d(cin, chs[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for ch in chs:
            self.down.append(Res(c, ch, emb)); c = ch
            self.pool.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
        self.mid1, self.mid2 = Res(c, c, emb), Res(c, c, emb)
        self.up = nn.ModuleList()
        for ch in reversed(chs):
            self.up.append(Res(c + ch, ch, emb)); c = ch
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv3d(c, cout, 3, padding=1))

    def forward(self, x, t, cond=None):
        e = self.emb(timestep_embedding(t, self.base))
        h = self.inp(x if cond is None else th.cat([x, cond], 1))
        skips = []
        for d, p in zip(self.down, self.pool):
            h = d(h, e); skips.append(h); h = p(h)
        h = self.mid2(self.mid1(h, e), e)
        for u, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[2:], mode="nearest")
            h = u(th.cat([h, s], 1), e)
        return self.out(h)
