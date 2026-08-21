"""A continuous interior field: three orthogonal feature planes instead of one feature per cell.

The per-cell field has no spatial prior of any kind. A cell carries its own 8-d latent, and if no
supervised plane ever crosses it, nothing in the objective refers to it: it keeps whatever the
prefit wrote, and against neighbours that moved it reads as coloured speckle. Measured on the
orange at 9/17, 19.6% of the cells are in that state. `fieldreg` states the missing coupling as a
penalty; this states it as an architecture instead, so the question of which is the better answer
can be asked with the rest of the run held identical.

A cell's feature is read from three axis-aligned planes at the cell's own position, bilinearly, and
summed:

    feat(x, y, z) = P_xy[x, y] + P_xz[x, z] + P_yz[y, z]        (3, C, R, R) -> (N, C)

then through the same two-stage MLP the per-cell decoder uses, so the only thing that differs
between the two arms is where the feature comes from. Two consequences follow from the
interpolation and neither is a free lunch:

  * a cell no plane ever crosses still has a defined feature, because its neighbours' supervision
    moves the same plane texels. That is the point.
  * the field can no longer represent two adjacent cells as unrelated, so whatever structure is
    genuinely at the scale of one cell cannot be expressed. `patchdist` already showed what
    blurring the interior costs -- detail 0.1772 -> 0.1574 -- so this must be read on the detail
    column and not only on the held-out score.

Interface-compatible with `anchor.ColourDecoder` on purpose: `feat`, `forward`, `pin_colour`,
`param_groups`, and a state dict. `feat` is a property returning a non-leaf tensor, so anything
that reaches for `dec.feat.grad` gets None rather than a wrong answer -- the trainer's per-cell
coverage probe is one of those, and it is simply unavailable here, which is stated rather than
silently zero.
"""
import os

import torch
import torch.nn as nn

# TRIPLANE=2 keeps the per-cell table and adds the planes underneath it, rather than replacing it.
#
# Replacing it lost in both columns, and the reason is in the same sentence as the motivation: an
# interpolating field cannot represent two adjacent cells as unrelated, so it fills the cells no
# plane reaches AND removes the ability to state anything at the scale of one cell. The per-cell
# table has the opposite pair of properties. Summing them gives each its own job:
#
#     feat = triplane(x, y, z) + residual[cell]
#
# The residual starts at zero, so a cell no plane ever crosses keeps whatever the planes interpolate
# there and contributes nothing of its own -- which is the fill -- while a cell that is supervised
# can move away from the interpolated value as far as it needs, which is the detail. Neither is
# possible in the other arm.
HYBRID = os.environ.get("TRIPLANE", "0") == "2"
RES = int(os.environ.get("TRIPLANE_RES", "192"))
C_FEAT = int(os.environ.get("TRIPLANE_DIM", "16"))
INIT = float(os.environ.get("TRIPLANE_INIT", "0.01"))


def _uv(centres):
    """Each cell's position on the three planes, in the [-1, 1] grid_sample expects.

    `centres` is (N, 3) in lattice coordinates. The extent is taken from the cells themselves, so
    the planes cover the object and not the bounding box of an empty region around it.
    """
    lo = centres.min(0).values
    hi = centres.max(0).values
    n = (centres - lo) / (hi - lo).clamp(min=1e-9) * 2.0 - 1.0          # (N, 3) in [-1, 1]
    # grid_sample takes (x, y) with x indexing the LAST axis, so each pair is reversed relative to
    # the axis pair it names. Getting this wrong transposes a plane, which a round object hides.
    return torch.stack([torch.stack([n[:, 1], n[:, 0]], -1),            # P_xy sampled at (y, x)
                        torch.stack([n[:, 2], n[:, 0]], -1),            # P_xz at (z, x)
                        torch.stack([n[:, 2], n[:, 1]], -1)], 0)        # P_yz at (z, y)


class TriplaneDecoder(nn.Module):
    """`anchor.ColourDecoder` with the per-cell feature table replaced by three planes."""

    def __init__(self, centres, init_rgb=None, c_feat=C_FEAT, res=RES, c_dim=16):
        super().__init__()
        self.planes = nn.Parameter(torch.randn(3, c_feat, res, res) * INIT)
        # exactly zero, not small noise: the residual must start by saying nothing, so that an
        # unsupervised cell is the interpolated value and not the interpolated value plus a
        # arbitrary offset that nothing will ever correct
        self.resid = nn.Parameter(torch.zeros(len(centres), c_feat)) if HYBRID else None
        self.register_buffer("uv", _uv(centres), persistent=False)
        self.register_buffer("nograd", torch.zeros(len(centres), dtype=torch.bool),
                             persistent=False)
        self.stage1 = nn.Sequential(
            nn.Linear(c_feat, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, c_dim))
        self.stage2 = nn.Sequential(nn.Linear(c_dim, 64), nn.ReLU(inplace=True),
                                    nn.Linear(64, 3))
        self.pin = None
        self.res, self.c_feat = res, c_feat
        with torch.no_grad():
            self.stage1[-1].weight.mul_(0.01)
            self.stage1[-1].bias.zero_()
            self.stage2[-1].weight.mul_(0.01)
            if init_rgb is not None:
                p = init_rgb.mean(0).clamp(1e-3, 1 - 1e-3)
                self.stage2[-1].bias.copy_(torch.log(p / (1 - p)))

    @property
    def feat(self):
        """(N, c_feat), the three planes sampled at every cell and summed."""
        g = self.uv[:, None]                                    # (3, 1, N, 2)
        s = nn.functional.grid_sample(self.planes, g, mode="bilinear",
                                      padding_mode="border", align_corners=True)
        f = s[:, :, 0].permute(0, 2, 1).sum(0)                  # (3, C, 1, N) -> (N, C)
        if self.resid is not None:
            f = f + self.resid
        if bool(self.nograd.any()):
            # the trainer's `feat.grad[is_outer] = 0`, which has nowhere to land when the feature
            # is not a leaf: detaching the masked rows stops the gradient at the same place while
            # leaving the value alone
            f = torch.where(self.nograd[:, None], f.detach(), f)
        return f

    def set_nograd(self, mask):
        self.nograd.copy_(mask.to(self.nograd.device, torch.bool))

    def forward(self):
        rgb = torch.sigmoid(self.stage2(self.stage1(self.feat)))
        if self.pin is not None:
            m, t = self.pin
            rgb = torch.where(m[:, None], t, rgb)
        return rgb

    def pin_colour(self, mask, target):
        self.pin = (mask, target)

    def param_groups(self, lr_feat=0.005, lr_mlp=0.002):
        feat = ["planes"] + (["resid"] if self.resid is not None else [])
        mlp = [p for n, p in self.named_parameters() if n not in feat]
        ps = [self.planes] + ([self.resid] if self.resid is not None else [])
        return [dict(params=ps, lr=lr_feat), dict(params=mlp, lr=lr_mlp)]


def selftest():
    """The two things that would be wrong and invisible: the axis pairing, and the interpolation."""
    torch.manual_seed(0)
    c = torch.rand(2000, 3) * 10
    d = TriplaneDecoder(c, init_rgb=torch.rand(2000, 3))
    f = d.feat
    assert f.shape == (2000, C_FEAT), f.shape
    # a cell and a cell one step away in x must read different texels, and the same must hold for
    # y and z -- if a pair is reversed, one of these collapses
    for ax in range(3):
        a = torch.zeros(1, 3); b = torch.zeros(1, 3)
        b[0, ax] = 10.0
        both = torch.cat([a, b])
        dd = TriplaneDecoder(torch.cat([both, c]), init_rgb=torch.rand(2002, 3))
        ff = dd.feat
        assert float((ff[0] - ff[1]).abs().max()) > 1e-6, f"axis {ax} does not move the feature"
    # interpolation: a cell between two others is not independent of them
    g = d.feat
    assert torch.allclose(f, g), "feat is not deterministic"
    n = sum(p.numel() for p in d.parameters())
    print(f"triplane selftest OK: {3 * C_FEAT * RES * RES:,} plane floats + MLP = {n:,} total, "
          f"against {2000 * 8:,} for the same cells per-cell")
    return True


if __name__ == "__main__":
    selftest()
