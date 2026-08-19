"""The anchor decoder's colour path, on an O-Voxel state.

`ANCHOR=1 ANCHOR_K=1 ANCHOR_DIM=8 ANCHOR_SPLIT=1 ANCHOR_PREFIT=1`, taken from
`src/anchor_decoder.py` rather than from a description of it.  What is carried across is the whole
of what produces a colour there:

    feat (N, f_dim)  --stage1-->  cf (N, c_dim)  --stage2-->  sigmoid  -->  rgb

    stage1   Linear(f_dim,128) ReLU Linear(128,128) ReLU Linear(128, c_dim)
    stage2   Linear(c_dim,64)  ReLU Linear(64,3)
    init     feat ~ N(0, 0.01); last layer of each stage scaled by 0.01; stage2's final bias set
             to logit(mean initial colour), so the decoder starts at one flat colour and the
             prefit has to produce the rest

f_dim = ANCHOR_DIM = 8 and c_dim = 16, which is `AnchorDecoder.__init__`'s own default and is not
reachable from the environment there either.

Two things in the original are dropped, and both are Gaussian parameters this representation does
not have: stage1's output width is `K*(3+3+4+1+c_dim)` there because each anchor emits K children
carrying an offset, a scale, a quaternion and an opacity, and `forward` slices the colour latent
out of that at `raw[:, 11:11+c_dim]`.  With ANCHOR_K=1 and no primitive to parameterise, the
other 11 outputs have nowhere to go and no gradient; stage1 emits c_dim directly.  Nothing about
the coupling changes -- that comes from the shared weights, which is the point.

ANCHOR_SPLIT=1 maps onto this representation exactly.  There it separates level-0 cells from
level-1 shell cells because "the exterior branch can only move shell cells and the cross-sections
only interior ones"; here those two populations are already two tensors -- `interior`, one value
per solid coarse cell, and `surf_rgb`, one per dual vertex -- so the split is two decoders, one
per tensor, and cannot leak by construction.

SHELL_PIN is `col_pin`: the pinned rows are overwritten with the target after the head rather than
held by a residual, which is what stops the gradient there as well as the drift.
"""
import os

import torch
import torch.nn as nn

F_DIM = int(os.environ.get("ANCHOR_DIM", "8"))
C_DIM = int(os.environ.get("ANCHOR_C_DIM", "16"))
CHUNK = int(os.environ.get("ANCHOR_CHUNK", "262144"))


class ColourDecoder(nn.Module):
    """One head. Two of these are an ANCHOR_SPLIT=1 decoder."""

    def __init__(self, n, init_rgb=None, f_dim=F_DIM, c_dim=C_DIM):
        super().__init__()
        self.feat = nn.Parameter(torch.randn(n, f_dim) * 0.01)
        self.stage1 = nn.Sequential(
            nn.Linear(f_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, c_dim))
        self.stage2 = nn.Sequential(nn.Linear(c_dim, 64), nn.ReLU(inplace=True),
                                    nn.Linear(64, 3))
        self.pin = None                      # (mask, target), col_pin's role
        with torch.no_grad():
            self.stage1[-1].weight.mul_(0.01)
            self.stage1[-1].bias.zero_()
            self.stage2[-1].weight.mul_(0.01)
            if init_rgb is not None:
                p = init_rgb.mean(0).clamp(1e-3, 1 - 1e-3)
                self.stage2[-1].bias.copy_(torch.log(p / (1 - p)))

    def _s1(self, x):
        """Chunked and recomputed in the backward pass, for ANCHOR_CHUNK's reason: the hidden
        layers are 128 wide, so holding every row's activations at these lattice sizes is the
        whole card."""
        if not (CHUNK and x.requires_grad):
            return self.stage1(x)
        from torch.utils.checkpoint import checkpoint
        return torch.cat([checkpoint(self.stage1, x[i:i + CHUNK], use_reentrant=False)
                          for i in range(0, x.shape[0], CHUNK)], 0)

    def forward(self):
        rgb = torch.sigmoid(self.stage2(self._s1(self.feat)))
        if self.pin is not None:
            m, t = self.pin
            rgb = torch.where(m[:, None], t, rgb)
        return rgb

    def pin_colour(self, mask, target):
        self.pin = (mask, target)

    def param_groups(self, lr_feat=0.005, lr_mlp=0.002):
        """anchor_decoder.param_groups' own split: the features move faster than the weights they
        are read through, because every row has its own feature and one MLP serves all of them."""
        mlp = [p for n, p in self.named_parameters() if n != "feat"]
        return [dict(params=[self.feat], lr=lr_feat), dict(params=mlp, lr=lr_mlp)]


def prefit(dec, target_rgb, steps=800, lr=0.01, tol=5e-5, verbose=True, tag=""):
    """Fit the decoder to the appearance the state already carries, before any gradient.

    ANCHOR_PREFIT. The features start as noise and the head starts at the mean colour, so the
    decoder begins with exactly one distinct colour where the state has hundreds of thousands.
    Without this every arm would start from flat grey and the sections would have to invent the
    exterior through a rim.

    `steps` is a ceiling and `tol` is the exit: the original stops as soon as it fits, because
    left running it "diverged and collapsed back to a single colour, undoing the whole point".
    """
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    last = float("nan")
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        rgb = dec()
        loss = nn.functional.mse_loss(rgb, target_rgb)
        loss.backward()
        opt.step()
        last = float(loss)
        if last < tol:
            if verbose:
                print(f"  prefit {tag} converged at step {i}, mse {last:.6f}", flush=True)
            break
        if verbose and (i % 200 == 0 or i == steps - 1):
            with torch.no_grad():
                q = (rgb * 255).round().to(torch.uint8)
                print(f"  prefit {tag} {i:>5}  mse {last:.6f}  "
                      f"distinct colours {torch.unique(q, dim=0).shape[0]:,}", flush=True)
    with torch.no_grad():
        err = float((dec() - target_rgb).abs().mean())
    # Drop the prefit's last gradient. Under SHELL_PIN this decoder is not in the training
    # optimiser, so nothing else ever clears it: `opt.zero_grad()` only touches parameters the
    # optimiser owns. Left behind, a stale non-zero `feat.grad` makes the coverage probe -- which
    # reads grads before the ownership mask is applied -- report a pinned exterior as 100%
    # supervised, when col_pin means it takes no gradient at all.
    dec.zero_grad(set_to_none=True)
    print(f"  prefit {tag} done: mse {last:.6f}, mean |decode - seed| {err:.5f}", flush=True)
    return dec


def voxel_smooth_anchors(dec, xyz, trained, grid=16):
    """Voxel Smoothing (paper 3.3), on the anchor FEATURES, as anchor_decoder.py performs it.

        C = sum_i w_i C_i / sum_i w_i

    "untrained Gaussians are assigned colors using a distance-weighted average of nearby trained
    Gaussians ... w_i is the inverse distance weight based on the Euclidean distance between the
    untrained Gaussian and each trained Gaussian within the same voxel."

    The average is over features and not over colours for the reason the original gives: colour is
    decoder output and does not persist between decodes, while `feat` does. The substitution is
    exact for what the smoothing is for -- stage2 is a fixed, smooth map within an iteration, so an
    untrained anchor given the weighted mean of its trained neighbours' features decodes to
    approximately the weighted mean of their colours, and unlike averaging colours it still holds
    after the next decode.

    `trained` is the pipeline's `gaussians.trained`: the accumulated mask of what the sections
    actually supervised, so a cell no plane ever crossed is a target and never a source.

    One translation, and it is the difference between this working and eating the exterior. In the
    pipeline SHELL_PIN and SEC_SKIP_OUTER draw the *same* mask -- both call
    `occupancy.surface_cells` at SHELL_PIN_LAYERS -- so the surface cells are simultaneously
    untrained and colour-pinned, and `col_pin` overwrites whatever the smoothing wrote into their
    features. Here the pin lives on the other decoder (the dual grid is the skin), so those cells
    have no col_pin to protect them and must be marked trained instead. Without that they are the
    bulk of what has no gradient and the smoothing hands the rind the mean of the flesh behind it.
    """
    with torch.no_grad():
        f = dec.feat
        t = trained
        if int(t.sum()) == 0 or int((~t).sum()) == 0:
            return 0
        mn, mx = xyz.min(0)[0], xyz.max(0)[0]
        cell = torch.where((mx - mn) > 0, (mx - mn) / grid, torch.ones_like(mx))
        idx = ((xyz - mn) / cell).floor().long().clamp(0, grid - 1)
        key = idx[:, 0] * grid * grid + idx[:, 1] * grid + idx[:, 2]
        filled = 0
        for k in key[~t].unique():
            in_cell = key == k
            src, dst = in_cell & t, in_cell & (~t)
            if int(src.sum()) == 0 or int(dst.sum()) == 0:
                continue
            w = 1.0 / (torch.cdist(xyz[dst], xyz[src]) + 1e-8)
            f[dst] = (w / w.sum(1, keepdim=True)) @ f[src]
            filled += int(dst.sum())
    return filled
