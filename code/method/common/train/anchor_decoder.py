"""Anchor points on the lattice, each decoding into K Gaussians through a shared MLP.

Storing one colour per cell gives N parameters that never share a gradient, and a cell is
only visible in the few cross-sections whose plane passes through it. Measured: raising the
per-cell learning rate 30x moved the interior a long way and left the angular structure
correlation at +0.10, exactly where it started -- the interior went from one flat colour to
another. Per-cell parameters cannot be fixed by giving them a bigger step.

A shared decoder changes what is being learned. The MLP takes gradient from every view in
every iteration, so it accumulates orders of magnitude more updates than any one cell. On
an independent cut -- a plane the training never supervised -- this is the only variant
that scores positive (+0.274, against -0.225 for 3DGS and -0.172 for per-cell colours).

Two stages, following Scaffold-GS:

  stage 1   anchor feature -> per-child offset, scale, rotation, opacity, colour feature
  stage 2   colour feature (+ view direction) -> RGB

Offsets are bounded to one cell by a tanh, which is what keeps connectivity exact: the
anchors sit on the lattice and every child inherits its anchor's piece, the same
parent/child relation the physics uses to carry millions of render primitives on tens of
thousands of simulated particles.
"""
import os
import torch
import torch.nn as nn

# Where the decoded children start, as a fraction of the cell: sigmoid(SCALE_BIAS). prefit
# only fits colour, so whatever this bias puts the scale head at is where it stays.
#
# The released -1.0 lands the children at 0.19 of a cell against the 0.50 the voxel model it
# replaces uses, and the lattice stops covering itself: a transverse section of the orange
# leaks background over 3.26 percent of its own area, worse than the 1.80 of the undecoded
# model, and the leak renders as a white column through the middle of the cut. That column
# was mistaken for structure twice -- once when a sweep of this bias was judged on angular
# contrast, which a gap raises rather than lowers and which therefore picked -1.0 as best,
# and once when the trained orange's sections were read as having grown a pale core. The
# model had no pale core to grow: 0.15 percent of its interior primitives were pale against
# 5.35 percent at initialisation. The white was the background.
#
# Coverage is the observable, and it settles it: leakage falls 3.26 -> 1.49 -> 0.29 -> 0.02
# -> 0.01 percent as the bias goes -1.0 -> -0.5 -> 0.0 -> 0.5 -> 1.0, while the rendered
# radial membranes are unchanged throughout. 1.0 puts the children at 0.51 of a cell, which
# is the size the lattice was built with.
SCALE_BIAS = float(os.environ.get("ANCHOR_SCALE_BIAS", "1.0"))
# Separate colour and geometry heads for the shell and the interior. They are supervised by
# branches that want different things, and with one head the louder branch wins for both.
SPLIT_HEADS = os.environ.get("ANCHOR_SPLIT", "0") == "1"

OPACITY_FREEZE = float(os.environ.get("OPACITY_FREEZE", "0"))
PHYS_HEAD = os.environ.get("ANCHOR_PHYS", "0") == "1"
POS_FREEZE = os.environ.get("POS_FREEZE", "0") == "1"
# Pin the shell as well as the interior. Default on: the shell showed the same collapse for
# the same reason, and its silhouette comes from the fill rather than from fitting.
FREEZE_SHELL = os.environ.get("ANCHOR_FREEZE_SHELL", "1") == "1"
# A shell-only scale bias. None means "use SCALE_BIAS for everything", which is what every
# run before this did. sigmoid(1.0) is 0.73 of a cell and sigmoid(3.0) is 0.95.
_ssb = os.environ.get("ANCHOR_SCALE_BIAS_SHELL", "")
SHELL_SCALE_BIAS = float(_ssb) if _ssb else None
# Hold the scale at whatever the bias puts it at, and let it take no gradient.
SCALE_FREEZE = os.environ.get("SCALE_FREEZE", "0") == "1"
# Anchors per chunk through the decoder MLPs, recomputed in the backward pass. 0 disables.
ANCHOR_CHUNK = int(os.environ.get("ANCHOR_CHUNK", "0"))
C0 = 0.28209479177387814


class AnchorDecoder(nn.Module):
    def __init__(self, anchor_xyz, cell_size, K=4, f_dim=32, c_dim=16,
                 init_rgb=None, view_dependent=False, level=None):
        super().__init__()
        self.K = K
        self.c_dim = c_dim
        self.view_dependent = view_dependent
        self.register_buffer("anchor_xyz", anchor_xyz)
        N = anchor_xyz.shape[0]

        # Per-anchor cell size, so one decoder covers every octree level. Splitting inner
        # and outer into separate decoders bakes in the assumption that an object has a
        # distinct shell; an object without one leaves the second decoder empty.
        c = torch.as_tensor(cell_size, dtype=torch.float32)
        self.register_buffer("cell", (c if c.dim() else c.expand(N)).reshape(-1, 1))
        # Scale is capped by the distance to the actual nearest anchor, not by the cell
        # size. A coarse cell sitting behind the skin had a cap twice the skin's own cell,
        # so its children punched through and showed flesh colour on the peel; and the
        # hard minimum() had exactly zero gradient at the cap, where p95 and the maximum
        # both sat -- those children were stuck there.
        self.register_buffer("cap", (c if c.dim() else c.expand(N)).reshape(-1, 1).clone())

        self.feat = nn.Parameter(torch.randn(N, f_dim) * 0.01)
        # Set lazily by set_colour(); None until something writes a colour. persistent=False
        # for the same reason is_interior is: it is state of this run, and putting it in the
        # state_dict makes every prefit cache written before it unloadable.
        self.col_res = None
        self.geom_frozen = None
        # Per-anchor offsets, not only MLP output. Nearby anchors have nearly identical
        # features, so a shared decoder gives them nearly the same four-point arrangement:
        # measured cosine similarity between anchors' offset patterns was +0.414, and that
        # repeated pattern reads as the streaks visible across the peel.
        self.offset = nn.Parameter(torch.randn(N, K, 3) * 0.01)

        per_child = 3 + 3 + 4 + 1 + c_dim          # offset, scale, rot, opacity, colour
        self._per_child = per_child
        in2 = c_dim + (3 if view_dependent else 0)

        def _mk():
            s1 = nn.Sequential(
                nn.Linear(f_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, K * per_child))
            s2 = nn.Sequential(nn.Linear(in2, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))
            return s1, s2

        self.stage1, self.stage2 = _mk()

        # Physics head. One number per anchor, in [0, 1], saying how stiff this cell is
        # relative to the rest of this object -- not a modulus. The scale that turns it into
        # one is a property of what the object is, not of what it looks like, and is applied
        # outside (see report/material_field.CATEGORY_RANGES); keeping the two apart is what
        # stops the network being asked for a number it has no way to know.
        #
        # It reads the same feature the visual head does. That is the point: material and
        # appearance are not independent -- peel is the colour of peel -- so predicting one
        # from the feature that produces the other is a constraint on the feature, and the
        # head costs 1.2k parameters.
        self.phys = nn.Sequential(nn.Linear(f_dim, 32), nn.ReLU(inplace=True),
                                  nn.Linear(32, 1)) if PHYS_HEAD else None
        # A second head for the shell, when the caller supplies the levels. One head has to
        # serve both, and the two are supervised by branches that disagree: on the doughnut
        # the exterior's pink glaze reached the crumb through the shared colour head and the
        # depth contrast fell from 0.214 to 0.133 in twenty iterations. Splitting them means
        # the exterior branch can only move shell cells and the cross-sections only interior
        # ones. It costs 35k parameters, against 8.8M for the per-anchor features.
        self.split = level is not None
        if level is not None:
            # Registered whether or not the heads are split: OPACITY_FREEZE needs to know
            # which anchors are interior, and that is the same information.
            # persistent=False: it is derived from the lattice, not learned, and putting it
            # in the state_dict makes every prefit cache written before it unloadable.
            self.register_buffer("is_interior",
                                 (torch.as_tensor(level).reshape(-1)[:N] == 0),
                                 persistent=False)
        if self.split:
            lv = torch.as_tensor(level).reshape(-1)[:N]
            self.register_buffer("is_shell", (lv == 1))
            self.stage1_s, self.stage2_s = _mk()

        with torch.no_grad():
            heads = [(self.stage1, self.stage2)]
            if self.split:
                heads.append((self.stage1_s, self.stage2_s))
            for s1, s2 in heads:
                s1[-1].weight.mul_(0.01)           # offsets start near the anchor
                s1[-1].bias.zero_()
                s2[-1].weight.mul_(0.01)
                if init_rgb is not None:
                    p = init_rgb.mean(0).clamp(1e-3, 1 - 1e-3)
                    s2[-1].bias.copy_(torch.log(p / (1 - p)))

    def stiffness(self):
        """Relative stiffness per anchor, in [0, 1]. None when the head is not built."""
        if self.phys is None:
            return None
        return torch.sigmoid(self.phys(self.feat)).squeeze(1)

    def forward(self, cam_center=None):
        """Returns xyz, rgb, opacity, scale, rotation for K * N children."""
        N = self.anchor_xyz.shape[0]

        def _s1(net, x):
            """Stage 1 in chunks, recomputed in the backward pass.

            The hidden layers are 128 wide, so running every anchor at once stores 128 floats
            per anchor per layer and autograd keeps all of them. At the lattice sizes this file
            was written for that is a gigabyte and fine; at three million anchors it is the
            whole card, and the run dies inside `F.linear` with 22.5 GB allocated before the
            first iteration. Checkpointing trades that for one extra forward pass per chunk in
            the backward, which is a few percent of an iteration that also rasterises fifty
            views.
            """
            if not (ANCHOR_CHUNK and x.requires_grad):
                return net(x)
            from torch.utils.checkpoint import checkpoint
            return torch.cat([checkpoint(net, x[i:i + ANCHOR_CHUNK], use_reentrant=False)
                              for i in range(0, x.shape[0], ANCHOR_CHUNK)], 0)

        if self.split:
            raw = torch.empty(N, self.K * self._per_child, device=self.feat.device,
                              dtype=self.feat.dtype)
            m = self.is_shell
            raw[~m] = _s1(self.stage1, self.feat[~m])
            raw[m] = _s1(self.stage1_s, self.feat[m])
            raw = raw.view(N * self.K, -1)
        else:
            raw = _s1(self.stage1, self.feat).view(N * self.K, -1)
        anc = self.anchor_xyz.repeat_interleave(self.K, dim=0)
        cell = self.cell.repeat_interleave(self.K, dim=0)

        cap = self.cap.repeat_interleave(self.K, dim=0)
        # Half a cell, not a whole one. tanh saturates at +-1, so multiplying by the cell
        # size let a child sit a full cell from its anchor -- inside the neighbouring cell,
        # with eight cells' worth of reach. That contradicts the invariant this file states
        # and relies on ("every child inherits its anchor's piece"), and it shows: on the
        # orange the shell/flesh colour boundary is one lattice step wide, and after
        # decoding 8.6% of the primitives at r/R 0.90-0.94 carry peel colour where the
        # input model had 0.0%, which renders as the speckle across the albedo ring.
        off = torch.tanh(raw[:, 0:3] + self.offset.view(-1, 3)) * (cell * 0.5)
        if POS_FREEZE:
            # Hold the interior on the lattice, which is what the trainer says it does and
            # then does not: `# lattice anchoring: interior positions never move` sits inside
            # `if not ANCHOR:`, so under ANCHOR=1 nothing holds them. Measured against the
            # fill they wander a mean of 0.41 to 0.56 of a cell spacing -- rarely far enough
            # to leave their own cell (0.01% to 0.96% do), so the cell indexing survives, but
            # far enough to smear the field within it, which is part of the speckle a section
            # renders with.
            #
            # An anchor is the fill's own position, so zeroing the offset puts the child back
            # exactly where the fill put it rather than somewhere merely nearby. The shell
            # keeps its offsets: the silhouette is fitted, not given.
            _m = self.is_interior.reshape(-1, 1).repeat_interleave(self.K, 0)
            off = torch.where(_m, torch.zeros_like(off), off)
        xyz = anc + off
        # softplus, not a saturating activation: scale must stay positive and span a wide
        # range. The cap keeps a child inside its own cell, which the physics binding and
        # the connectivity argument both rely on.
        # bounded by cap and smooth everywhere: no hard clamp, so nothing can park on a
        # boundary with no gradient
        # A separate bias for the shell, because the shell is the only place the lattice
        # fails to cover itself. Recovered from two renders of the same model on white and
        # on black, alpha = 1 - (c_white - c_black): the interior reaches 0.999 and holds it,
        # while the outer 8% of a section sits at 0.892 straight out of the fill, before a
        # single iteration. One cell of skin, cut tangentially by the section plane, has too
        # few primitives along those rays -- and at 0.5 of the spacing each Gaussian's sigma
        # is half the distance to its neighbour, so they meet in a trough rather than
        # overlapping. Raising only the shell's bias widens them without touching an
        # interior that is already opaque.
        _b = SCALE_BIAS
        if SHELL_SCALE_BIAS is not None and hasattr(self, "is_interior"):
            _m = self.is_interior.reshape(-1, 1).repeat_interleave(self.K, 0)
            _b = torch.where(_m, torch.full_like(raw[:, 3:4], SCALE_BIAS),
                             torch.full_like(raw[:, 3:4], SHELL_SCALE_BIAS))
        scale = cap * torch.sigmoid(raw[:, 3:6] + _b)
        if SCALE_FREEZE:
            # The shortcut that survived pinning the opacity. Nothing bounds the scale, so a
            # target brighter than the render can still be met by shrinking until the
            # background shows between the primitives -- and it is: over stage 1 the shell
            # went from 0.500 of its cell to 0.076, a 6.6x shrink, which took the rim's alpha
            # from 0.892 down to 0.792 while the interior, many primitives deep along any ray,
            # barely moved (0.500 -> 0.408). Pinning holds the size the lattice was built with.
            scale = (cap * torch.sigmoid(torch.zeros_like(raw[:, 3:6]) + _b)).detach()
        rot = torch.nn.functional.normalize(
            raw[:, 6:10] + torch.tensor([1., 0., 0., 0.], device=raw.device), dim=1)
        opac = torch.sigmoid(raw[:, 10:11] + 4.0)
        if OPACITY_FREEZE > 0:
            # Hold the interior opaque and let it take no gradient.
            #
            # Nothing in the loss says a cell must be there. A section target is brighter
            # than the render wherever it asks for pith or albedo, the background behind
            # the cut is white, and the cheapest way for a cell to get brighter is to stop
            # existing -- so the model satisfies the target by deleting the interior
            # rather than by colouring it. Measured on the orange at K=4 d=32: 0.6% of
            # interior cells below opacity 0.5 at initialisation, 73.0% by iteration 30,
            # and that is on the photograph target, before any diffusion is involved. The
            # four finished objects all carry it to some degree, and the ranking matches
            # how good their interiors look -- watermelon 14.9%, doughnut 32.8%, loaf
            # 30.6%, orange 56.8%.
            #
            # The released method never meets this because it does not let opacity move at
            # all: it overwrites every entry with 10000 and sigmoid saturates, so the
            # gradient is zero and colour is the only way to satisfy anything.
            #
            # This applied to the interior only, on the argument that the shell still needs
            # a silhouette and freezing it would stop the object from having an edge. That
            # argument does not hold here and the measurement says so. The skin comes from
            # the fill, which comes from the scan, so the silhouette is given rather than
            # fitted -- and left free the shell takes the identical shortcut for the
            # identical reason. Sixty iterations against a real transverse photograph, with
            # the interior pinned: shell opacity 0.952 -> 0.392 while its colour did not
            # move at all, (0.501, 0.500, 0.500) to (0.523, 0.482, 0.479) against an
            # exterior reference of (0.859, 0.411, 0.170). The exterior render's brown cast
            # is exactly that -- a grey half-transparent skin over an orange interior and a
            # white background. Pinning both leaves colour as the only route for either
            # branch, which is the whole point.
            #
            # ANCHOR_FREEZE_SHELL=0 restores the interior-only behaviour.
            _keep = torch.full_like(opac, float(OPACITY_FREEZE)).detach()
            if hasattr(self, "is_interior") and not FREEZE_SHELL:
                _m = self.is_interior.reshape(-1, 1).repeat_interleave(self.K, 0)
                opac = torch.where(_m, _keep, opac)
            else:
                opac = _keep
        cf = raw[:, 11:11 + self.c_dim]

        if self.view_dependent:
            if cam_center is None:
                cam_center = self.anchor_xyz.mean(0) + torch.tensor(
                    [0., 0., 3.], device=xyz.device)
            d = torch.nn.functional.normalize(
                cam_center.reshape(1, 3).to(xyz.device) - xyz, dim=1)
            cf = torch.cat([cf, d], dim=1)
        if self.split:
            ms = self.is_shell.repeat_interleave(self.K)
            rgb = torch.empty(cf.shape[0], 3, device=cf.device, dtype=cf.dtype)
            rgb[~ms] = self.stage2(cf[~ms])
            rgb[ms] = self.stage2_s(cf[ms])
            rgb = torch.sigmoid(rgb)
        else:
            rgb = torch.sigmoid(self.stage2(cf))
        # A place for edits made to the decoded colour to survive the next decode.
        #
        # `write_into` assigns `gaussians._features_dc` a fresh tensor every time, so anything
        # written into the previous one is gone -- which is why the paper's `voxel_smoothing`
        # is inert here even where it is reachable. The decoder has no memory of a colour, only
        # of a feature, so an operation defined on colours has nowhere to be stored. This is
        # that place: an additive per-primitive residual, applied after the head and carried
        # across decodes. It takes no gradient, so it moves only when something writes to it.
        if self.col_res is not None:
            rgb = (rgb + self.col_res).clamp(0.0, 1.0)
        if self.geom_frozen is not None:
            # Hold the shell's geometry as well as its colour.
            #
            # Pinning the colour alone leaves `xyz`, `scale` and `rot` to be recomputed from a
            # feature the cross-section branch is still moving, so the cells the projection
            # painted drift and resize under it. Nothing renders wrong at first because the
            # colour is right; what appears is banding, the lattice's own rows becoming visible
            # on the peel as the cells stop tiling. Measured on the orange: exterior
            # high-frequency energy 3.39 at the projection and 17.82 after two hundred
            # iterations, with the shell's colour and opacity bit-identical throughout. On the
            # watermelon the same drift opened the shell and the trained red interior showed
            # through it.
            m, gx, gs, gr = self.geom_frozen
            xyz = torch.where(m[:, None], gx, xyz)
            scale = torch.where(m[:, None], gs, scale)
            rot = torch.where(m[:, None], gr, rot)
        return xyz, rgb, opac, scale, rot

    def freeze_geometry(self, mask):
        """Record the shell's current geometry and reproduce it on every later decode."""
        with torch.no_grad():
            xyz, _, _, scale, rot = self()
            m = mask.reshape(-1).to(xyz.device)
            if m.shape[0] != xyz.shape[0]:
                m = m[:xyz.shape[0]] if m.shape[0] > xyz.shape[0] else \
                    torch.cat([m, torch.zeros(xyz.shape[0] - m.shape[0], dtype=torch.bool,
                                              device=xyz.device)])
            self.geom_frozen = (m, xyz.detach().clone(), scale.detach().clone(),
                                rot.detach().clone())
        return int(m.sum())

    def set_colour(self, target, mask=None):
        """Make the next decode produce `target`, by storing the difference from what it does.

        `target` is (N*K, 3) in RGB. With a mask, only those primitives are pinned and the rest
        keep whatever the heads give them. Used by voxel smoothing, which is defined on colours
        rather than on features and would otherwise be discarded on the next decode.
        """
        with torch.no_grad():
            if self.col_res is None:
                # plain attribute assignment: `col_res` is declared as None in __init__, and
                # register_buffer refuses a name that already exists. `.to()` on the module
                # will not follow it, so it is created on the parameters' own device.
                self.col_res = torch.zeros(self.anchor_xyz.shape[0] * self.K, 3,
                                           device=self.feat.device, dtype=self.feat.dtype)
            self.col_res.zero_()
            base = self()[1]
            d = target.to(base.device, base.dtype) - base
            self.col_res.copy_(d if mask is None else torch.where(
                mask.reshape(-1, 1).to(base.device), d, torch.zeros_like(d)))
        return int(self.col_res.abs().sum(1).gt(1e-6).sum())

    def to_sh(self, rgb):
        return ((rgb - 0.5) / C0).unsqueeze(1)

    def param_groups(self, lr_feat=0.005, lr_mlp=0.002):
        mlp = list(self.stage1.parameters()) + list(self.stage2.parameters())
        if self.split:
            mlp += list(self.stage1_s.parameters()) + list(self.stage2_s.parameters())
        if self.phys is not None:
            mlp += list(self.phys.parameters())
        return [{"params": [self.feat], "lr": lr_feat, "name": "anchor_feat"},
                {"params": [self.offset], "lr": lr_feat, "name": "anchor_offset"},
                {"params": mlp, "lr": lr_mlp, "name": "anchor_mlp"}]


# No K_shell. It was a parameter here and an ANCHOR_KSHELL environment variable in the
# trainer, and neither ever reached the decoder -- install() accepted it and built
# AnchorDecoder with K alone, so setting it changed nothing and said nothing. Giving the
# shell a different child count is not a one-line change either: forward() lays the K
# children of an anchor out consecutively and every consumer recovers them with
# repeat_interleave(K) or view(-1, 3), and a ragged count breaks all of it. Better no knob
# than one that looks like it works.
def install(gaussians, level_path, lattice_path, K=4, f_dim=32, view_dependent=False):
    """Build the decoder from a trained all-voxel model and rewire the optimizer."""
    dev = gaussians.get_xyz.device
    lvl = torch.load(level_path).to(dev)
    lat = torch.load(lattice_path)
    xyz = gaussians.get_xyz.detach()
    n = min(xyz.shape[0], lvl.shape[0])
    rgb = (gaussians._features_dc.detach().squeeze(1) * C0 + 0.5).clamp(0, 1)[:n]
    anchors = xyz[:n].contiguous()
    cells = torch.where(lvl[:n] == 0,
                        torch.full((n,), float(lat["coarse_dx"]), device=dev),
                        torch.full((n,), float(lat["fine_dx"]), device=dev))

    dec = AnchorDecoder(anchors, cells, K=K, f_dim=f_dim,
                        init_rgb=rgb, view_dependent=view_dependent,
                        level=(lvl[:n] if SPLIT_HEADS else None)).to(dev)
    # The cap is the distance to the nearest other anchor, not the nominal cell size. A
    # coarse cell behind the skin is nominally twice as wide as the skin's own cells, so
    # capping by cell let its children reach through the peel; the true local spacing is
    # what actually says how far a primitive may spread before it overlaps its neighbour.
    with torch.no_grad():
        nnd = torch.empty(anchors.shape[0], device=dev)
        for s0 in range(0, anchors.shape[0], 4000):
            q = anchors[s0:s0 + 4000]
            best = torch.full((q.shape[0],), 1e9, device=dev)
            for j in range(0, anchors.shape[0], 200000):
                dm = torch.cdist(q, anchors[j:j + 200000])
                dm[dm < 1e-9] = 1e9
                best = torch.minimum(best, dm.min(1).values)
                del dm
            nnd[s0:s0 + 4000] = best
        dec.cap.copy_(nnd.reshape(-1, 1))
    print(f"  scale cap from local spacing: median {float(nnd.median()):.6f}  "
          f"p05 {float(nnd.quantile(0.05)):.6f}  p95 {float(nnd.quantile(0.95)):.6f}")
    groups = dec.param_groups()
    # The directional terms train too.
    #
    # The decoder produces a mean colour and nothing else, so a shell that carries spherical
    # harmonics keeps them frozen at whatever the initialisation fitted -- including the fit's
    # one real defect, a dark ring at the limb where no reference ever looked and the
    # polynomial had to extrapolate. Training sees that ring in every view and can remove it,
    # but only if the coefficients are in the optimiser. They are a plain parameter on the
    # model rather than an output of the MLP, which is exactly why they need adding by hand.
    _rest = getattr(gaussians, "_features_rest", None)
    if _rest is not None and _rest.shape[1] > 0:
        gaussians._features_rest = torch.nn.Parameter(_rest.detach().clone())
        groups.append({"params": [gaussians._features_rest],
                       "lr": float(os.environ.get("SH_LR", "0.002")), "name": "sh_rest"})
        print(f"  spherical-harmonic terms are trainable: "
              f"{gaussians._features_rest.numel():,} coefficients")
    opt = torch.optim.Adam(groups, eps=1e-15)
    n0 = int((lvl[:n] == 0).sum())
    print(f"anchor decoder: {anchors.shape[0]:,} anchors (coarse {n0:,} / fine "
          f"{anchors.shape[0]-n0:,}) x K={K} -> {anchors.shape[0]*K:,} gaussians, "
          f"{sum(p.numel() for p in dec.parameters()):,} params")
    return dec, opt


def prefit(dec, target_rgb, steps=800, lr=0.01, verbose=True, tol=5e-5):
    """Fit the decoder to the appearance the input model already carries.

    The anchor features start as noise and the colour head starts at the model's mean
    colour, so the decoder begins with exactly one distinct colour where the input has
    184,339. Every bit of the photo-reconstructed shell appearance is discarded at
    initialisation, and cross-section supervision cannot put it back because the shell
    appears in those views only as a rim. Fitting the decoder to the existing colours
    first costs no renders, and afterwards the cross-section loss only has to change what
    is actually wrong, which is the interior.
    """
    tgt = target_rgb.repeat_interleave(dec.K, dim=0)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        _, rgb, _, _, _ = dec(None)
        loss = torch.nn.functional.mse_loss(rgb, tgt)
        loss.backward()
        opt.step()
        # Stop once it fits. Left running to 2500 steps this diverged and collapsed back
        # to a single colour, undoing the whole point of the step.
        if float(loss) < tol:
            if verbose:
                print(f"  prefit converged at step {i}, mse {float(loss):.6f}")
            break
        if verbose and (i % 400 == 0 or i == steps - 1):
            with torch.no_grad():
                q = (rgb * 255).round().to(torch.uint8)
                print(f"  prefit {i:>5}  mse {float(loss):.5f}  "
                      f"distinct colours {torch.unique(q, dim=0).shape[0]:,}")
    return dec


def write_into(gaussians, dec, cam_center=None):
    """Replace the model's per-primitive tensors with the decoder's output.

    The tensors stay in the autograd graph, so the transform chain, the rasteriser and the
    losses all flow gradient back to the anchors and the MLP with no change to the loop.
    """
    xyz, rgb, opac, scale, rot = dec(cam_center)
    gaussians._xyz = xyz
    gaussians._features_dc = dec.to_sh(rgb)
    # Keep whatever directional terms the model came in with. The decoder produces the mean
    # colour and nothing else, so overwriting this with an empty tensor threw away the
    # per-cell directionality on the first decode -- before a single iteration had run.
    _rest = getattr(gaussians, "_features_rest", None)
    if _rest is None or _rest.shape[1] == 0:
        gaussians._features_rest = torch.zeros(xyz.shape[0], 0, 3, device=xyz.device)
    gaussians._opacity = torch.logit(opac.clamp(1e-6, 1 - 1e-6))
    gaussians._scaling = torch.log(scale.clamp_min(1e-9))
    gaussians._rotation = rot
    gaussians.max_radii2D = torch.zeros(xyz.shape[0], device=xyz.device)
    return xyz.shape[0]


def build_axial(anchor_xyz, cell, up, fine_dx, steps=(1, 2, 4, 8)):
    """Pair each anchor with the one a cell away along the object's axis.

    Measured on both fruits, training takes the interior apart rather than building it: the
    angular pattern of a horizontal layer agreed with the other layers at +0.889 after
    initialisation and +0.468 after fifty iterations on the orange, +0.946 to +0.340 on the
    watermelon. Nothing in the loop asks the forty-eight supervised planes to describe one
    object -- each is fitted to its own photograph, of its own fruit, and the shared decoder
    settles between contradictions by grinding the structure down.

    What the sections have in common is what the object is made of. A citrus fruit's segment
    walls, a watermelon's fibres and a loaf's crumb all run along the object's axis, so cells
    a step apart along it hold the same material. That is a statement about the material, not
    about symmetry, so it holds for an irregular shape as readily as for a ball.

    On a lattice the pair is exact and needs no threshold: the neighbour is at a known offset,
    and either a cell is there or it is not. Only the axial direction is paired, so detail
    within a cross-section -- the membranes, the seeds -- is left entirely free.
    """
    dev = anchor_xyz.device
    u = torch.as_tensor(up, dtype=torch.float32, device=dev)
    u = u / u.norm()
    base = anchor_xyz.min(0).values
    q = torch.round((anchor_xyz - base) / fine_dx).to(torch.int64)
    dims = q.max(0).values + 2
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    order = torch.argsort(key)
    skey, sidx = key[order], order

    # Several separations, not one. A single-cell pair leaves the colour free to drift a
    # little at every step, and the layers the interior is actually judged over are about
    # six cells apart -- far enough that a chain of small permitted differences adds up to
    # no coupling at all. Measured: single-cell pairs cut the neighbour difference from
    # 0.045 to 0.030 while layer-to-layer agreement moved only 0.469 to 0.477. Pairing at
    # one, two, four and eight cells ties the scale that is being asked about directly.
    srcs, dsts = [], []
    for k in steps:
        step = torch.round((anchor_xyz + u * (cell.reshape(-1, 1) * k) - base)
                           / fine_dx).to(torch.int64)
        step = step.clamp(torch.zeros(3, dtype=torch.int64, device=dev), dims - 1)
        tkey = (step[:, 0] * dims[1] + step[:, 1]) * dims[2] + step[:, 2]
        pos = torch.searchsorted(skey, tkey).clamp(max=skey.shape[0] - 1)
        hit = skey[pos] == tkey
        src = hit.nonzero().squeeze(1)
        dst = sidx[pos[src]]
        keep = src != dst
        srcs.append(src[keep]); dsts.append(dst[keep])
    return torch.cat(srcs), torch.cat(dsts)


def build_neighbours(anchor_xyz, cell, fine_dx, level=None):
    """Pair each anchor with its six face neighbours, one cell away.

    The released trainer regularises space by averaging `_features_dc` over the cells of a
    512-cubed grid every 101 iterations. That cannot be ported literally: it averages stored
    colours, and here the colour is decoder output, so there is nothing in the model to
    average. It also would not do anything if it could -- with one primitive per lattice cell,
    a 512-cubed grid puts every primitive alone in its own cell and the mean is the value.

    What it is doing, though, is asking neighbouring space to agree, and that does port. The
    quantity to smooth is the anchor feature, since it is what the colour is decoded from, and
    the neighbourhood is the lattice's own, which needs no grid and no threshold: the face
    neighbour is at a known offset and either occupied or not.

    Pairs never cross a level. A coarse interior anchor and the skin anchor outside it are one
    step apart in space but describe different materials, and averaging across that boundary
    walks flesh colour into the peel -- the same failure the scale cap had.
    """
    dev = anchor_xyz.device
    base = anchor_xyz.min(0).values
    q = torch.round((anchor_xyz - base) / fine_dx).to(torch.int64)
    dims = q.max(0).values + 2
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    order = torch.argsort(key)
    skey, sidx = key[order], order
    c = cell.reshape(-1, 1)
    srcs, dsts = [], []
    for ax in range(3):
        for sgn in (1, -1):
            u = torch.zeros(3, device=dev); u[ax] = sgn
            step = torch.round((anchor_xyz + u * c - base) / fine_dx).to(torch.int64)
            step = step.clamp(torch.zeros(3, dtype=torch.int64, device=dev), dims - 1)
            tkey = (step[:, 0] * dims[1] + step[:, 1]) * dims[2] + step[:, 2]
            pos = torch.searchsorted(skey, tkey).clamp(max=skey.shape[0] - 1)
            hit = skey[pos] == tkey
            src = hit.nonzero().squeeze(1)
            dst = sidx[pos[src]]
            keep = src != dst
            if level is not None:
                keep = keep & (level[src] == level[dst])
            srcs.append(src[keep]); dsts.append(dst[keep])
    return torch.cat(srcs), torch.cat(dsts)


def voxel_smooth_anchors(dec, trained, grid=16):
    """Voxel Smoothing (paper 3.3), on the anchors rather than on stored colours.

        C = sum_i w_i * C_i / sum_i w_i

    "untrained Gaussians are assigned colors using a distance-weighted average of nearby
    trained Gaussians ... w_i is the inverse distance weight based on the Euclidean distance
    between the untrained Gaussian and each trained Gaussian within the same voxel."

    The version in `train_voxel.voxel_smoothing` is faithful and unreachable: the anchor branch
    of `density_and_prune` returns before it, and it would not survive if it ran, because
    `write_into` assigns `_features_dc` a fresh tensor on every decode and a `.data.copy_` into
    the previous one is discarded. Colour is decoder output here; the only thing that persists
    between iterations is `dec.feat`. So the average is taken over features.

    That substitution is exact for the property the smoothing is for. `stage2` is a fixed map
    from feature to colour within an iteration, and it is smooth, so an untrained anchor given
    the weighted mean of its trained neighbours' features decodes to approximately the weighted
    mean of their colours -- and, unlike averaging colours, it still holds after the next
    decode. What it is not is a way to give an untrained anchor a colour no trained anchor has,
    which is what the paper wants and what this preserves.

    A cell that never appeared in any supervised plane has no gradient and keeps whatever the
    prefit gave it, which is the input's flat colour. Grid resolution follows the paper's rule
    of thumb, that a voxel hold under 1% of the particles; the released 512^3 puts almost every
    primitive alone in its own voxel, where the average is a no-op.
    """
    with torch.no_grad():
        f = dec.feat
        xyz = dec.anchor_xyz
        # The flag is per primitive and the anchors are per cell, and the children of one
        # anchor are contiguous, so fold K children down to their parent. An anchor counts as
        # trained if any of its children was: they share the feature this writes to, so one
        # supervised child makes the whole anchor a legitimate source and not a target.
        t = trained.reshape(-1).to(xyz.device)
        if t.shape[0] == xyz.shape[0] * dec.K:
            t = t.reshape(xyz.shape[0], dec.K).any(1)
        else:
            t = t[:xyz.shape[0]]
        if t.sum() == 0 or (~t).sum() == 0:
            return 0
        mn, mx = xyz.min(0)[0], xyz.max(0)[0]
        cell = torch.where((mx - mn) > 0, (mx - mn) / grid, torch.ones_like(mx))
        idx = ((xyz - mn) / cell).floor().long().clamp(0, grid - 1)
        key = idx[:, 0] * grid * grid + idx[:, 1] * grid + idx[:, 2]
        filled = 0
        for k in key[~t].unique():
            in_cell = key == k
            src, dst = in_cell & t, in_cell & (~t)
            if src.sum() == 0 or dst.sum() == 0:
                continue
            w = 1.0 / (torch.cdist(xyz[dst], xyz[src]) + 1e-8)
            f[dst] = (w / w.sum(1, keepdim=True)) @ f[src]
            filled += int(dst.sum())
    return filled


def smooth_features(dec, src, dst, alpha):
    """Blend each anchor's feature toward the mean of its occupied face neighbours."""
    with torch.no_grad():
        f = dec.feat
        acc = torch.zeros_like(f)
        cnt = torch.zeros(f.shape[0], 1, device=f.device, dtype=f.dtype)
        acc.index_add_(0, src, f[dst])
        cnt.index_add_(0, src, torch.ones_like(cnt[src]))
        has = cnt.squeeze(1) > 0
        mean = acc[has] / cnt[has]
        f[has] = (1.0 - alpha) * f[has] + alpha * mean
    return int(has.sum())
