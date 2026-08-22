"""Sample the interior as a 3D volume, using the two per-family 2D priors. TPDM, on our lattice.

The published recipe (Lee et al., ICCV 2023) writes the 3D distribution as a product of the
distributions of its slices taken in two perpendicular directions,

    p(x)  proportional to  q_transverse(x)^alpha  q_longitudinal(x)^beta,

trains one ordinary 2D diffusion model per direction, and samples the volume by ALTERNATING: each
reverse step picks a direction with probability alpha/(alpha+beta), denoises every slice in that
direction, and writes them back. Alternating is the point -- averaging two directions' estimates at
every step is what blurs the result, and averaging is what this file replaced.

Two things differ from the paper, both forced by our geometry rather than chosen:

  * our two families are not two axes of the grid. Transverse planes are parallel but oblique to
    the lattice, and longitudinal planes turn about the polar axis. So a slice is gathered from the
    cells it passes through and scattered back to them, rather than being a row of the array.

  * a slice of a Cartesian grid of independent noise is again independent noise, which is what lets
    the paper feed 3D noise to a 2D model. Interpolating would not preserve that, so GATHERING is
    nearest-cell: each slice pixel is one cell, and the noise a slice carries is the noise those
    cells carry. Scattering is not under that constraint -- what goes back is an estimate of the
    noise, not noise -- and nearest-cell scattering leaves holes, because an oblique plane's pixels
    do not land one per cell. So it writes trilinearly to the eight cells around the exact point,
    which took the transverse family from 79.5% of the interior to nearly all of it.

A cell no plane touched in a step is left alone for that step rather than updated with a zero
estimate, which would drive it towards grey.

What comes out is a prior sample: an interior that looks, along both families, like the
photographs of this fruit. It knows nothing yet about WHICH photograph belongs on which plane --
that is the measurement term, and it is applied separately by the fitting the pipeline already does.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchdiff

P = int(os.environ.get("TP_P", "128"))           # slice resolution the priors were trained at
# The reference implementation's convention exactly (`is_primary_tern`): a float K updates the
# primary family with probability 1 - 1/K, an integer K updates it K-1 times for every one update
# of the auxiliary. Transverse is primary here, because it is the family with more planes.
K = os.environ.get("TP_K", "4")
AZ = int(os.environ.get("TP_AZ", "192"))         # longitudinal planes per step
SEED = int(os.environ.get("TP_SEED", "0"))
WINDOW = 1.05                                    # set by Slicer.window from the photographs
# Where a slice's clean estimate comes from. "prior" asks a trained denoiser, which is the paper's
# setup and needs a dataset to have been trained on. "exemplar" draws one of the object's own
# photographs and uses it directly: with three photographs per family there is no distribution over
# layouts to learn, but there are three real layouts to insist on, and the alternating chain then
# has to find one volume that satisfies a draw on every plane of both families at once.
#
# The draw is fixed per plane for the whole chain. Redrawing every step would ask each plane to be
# all three photographs, and the only volume that satisfies that is their average -- the flat
# result every attempt in this line has produced.
MODE = os.environ.get("TP_MODE", "prior")
ROT = os.environ.get("TP_ROT", "1") == "1"
# The drawn photograph is placed slightly larger than the slice's shell and then cut back to it.
# Fitting the two shells exactly leaves notches wherever they disagree in outline, and a notch is
# a cell with no target at all; overfilling costs only the rim of the photograph, which the mask
# discards anyway. What is supervised is still exactly the mask.
OVER = float(os.environ.get("TP_OVER", "1.10"))
# Two things the first exemplar run got wrong, both visible in its residual: it never moved.
#
# PICK: the paper's denoiser estimates the clean slice FROM the noisy one, so each step carries the
# other family's answer forward. A photograph drawn once and pasted ignores the noisy slice
# entirely, so every step threw away what the other family had just done and the two families
# simply overwrote each other for ever. "nearest" restores the dependence: of this family's
# photographs, the one closest to what the volume currently says this slice is.
#
# LAM: a step that sets the estimate equal to a photograph is a hard projection onto that family's
# constraints, and the two families' constraints do not intersect. Relaxing the step is what makes
# alternating projections converge to a compromise instead of oscillating between the two.
# Both families in the same step, rather than one family per step. Alternating is the paper's
# scheme and it is right when one model carries a measurement constraint the other does not; here
# the two families are equals, and alternating made each step undo the last -- measured, the
# residual to one family fell only while the other's rose. Fusing sums both families' estimates
# into one clean estimate before the step, which is the product-of-experts the factorisation says
# it is, and it is what Score-Fusion (2025) does to the same pair of perpendicular models.
FUSE = os.environ.get("TP_FUSE", "1") == "1"
W_H = float(os.environ.get("TP_W_H", "1"))
W_V = float(os.environ.get("TP_W_V", "1"))
PICK = os.environ.get("TP_PICK", "nearest")
LAM = float(os.environ.get("TP_LAM", "0.5"))
# Relaxation is what makes the two families meet instead of overwriting each other, but holding it
# at a half all the way down means the last steps are still averaging the answer with the previous
# one, and detail cannot settle. It is loosened towards the end: LAM at the start of the chain,
# LAM_END by t=0.
LAM_END = float(os.environ.get("TP_LAM_END", "1.0"))
# Every cell is crossed by a hundred planes and the write-back averaged all of them, which is where
# the high frequencies went: the planes disagree in detail and agreeing on nothing is what an
# average returns. Raising the trilinear weights to a power concentrates each cell on the planes
# that pass closest to it -- 1 is the old average, and large values approach taking only the
# nearest plane.
SHARP = float(os.environ.get("TP_SHARP", "8"))


def dense(st):
    """The interior field as a dense (3, X, Y, Z) volume plus the mask of cells that exist."""
    idx = st["idx3"].long()
    m = idx >= 0
    v = torch.zeros((3,) + tuple(idx.shape), device=idx.device)
    c = st["interior"][idx[m]]
    v[:, m] = c.T
    return v, m


def cell_centres(st, device):
    G = st["idx3"].shape
    g = torch.meshgrid(*[torch.arange(n, device=device) for n in G], indexing="ij")
    ijk = torch.stack(g, -1).float() + torch.as_tensor(st["idx_lo"], device=device).float() + 0.5
    return ijk * st["hc"] + torch.as_tensor(st["org"], dtype=torch.float32, device=device)


def plane_basis(n, prefer=None):
    """Two unit vectors spanning the plane with normal n.

    `prefer` pins one of them: for a longitudinal plane it is the polar axis, so that the vertical
    of the placed photograph is the object's axis. Without it the basis is arbitrary, and a
    photograph of a cut down the axis was being laid into the volume sideways.
    """
    n = n / np.linalg.norm(n)
    if prefer is not None:
        w = np.asarray(prefer, float) - n * float(np.dot(prefer, n))
        w /= np.linalg.norm(w)
        return np.cross(w, n), w
    a = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    return u, np.cross(n, u)


class Slicer:
    """Gathers slices from the volume by nearest cell, and scatters values back the same way."""

    def __init__(self, st, device):
        self.G = tuple(st["idx3"].shape)
        self.hc, self.device = st["hc"], device
        self.org = torch.as_tensor(st["org"], dtype=torch.float32, device=device)
        self.lo = torch.as_tensor(st["idx_lo"], dtype=torch.float32, device=device)
        self.span = WINDOW
        c = cell_centres(st, device)
        self.centre = c.reshape(-1, 3).mean(0)
        self.extent = float((c.reshape(-1, 3).max(0).values - c.reshape(-1, 3).min(0).values).max())

    def window(self, frac_target, st, cams, mask, device):
        """Scale the sampling window so slices fill the frame the way the photographs do.

        The priors see whole photographs, and how much of a photograph the fruit covers is a
        property of the camera that took it. The sampler must present slices at the same coverage
        or the model is reading the volume at the wrong scale, so the window is set from a measured
        fraction rather than a guess.
        """
        self.frac = frac_target
        n = np.asarray(cams["h_planes"][0, :3], float)
        c = cell_centres(st, device).reshape(-1, 3)
        d = float((c @ torch.as_tensor(n / np.linalg.norm(n), dtype=torch.float32,
                                       device=device)).median())
        lo, hi = 0.3, 4.0
        for _ in range(24):                       # bisect on the window, it is monotone
            self.span = 0.5 * (lo + hi)
            flat, ok, _, _ = self.index(n, d)
            got = float((mask.reshape(-1)[flat.reshape(-1)].float() * ok.reshape(-1)).mean())
            if got > frac_target:
                lo = self.span
            else:
                hi = self.span
        return self.span, got

    def index(self, n, d, prefer=None):
        """Flat cell index for each pixel of a P x P grid on the plane {x : n.x = d}, and a mask."""
        u, w = plane_basis(np.asarray(n, float), prefer)
        s = torch.linspace(-0.5, 0.5, P, device=self.device) * self.extent * self.span
        gu, gw = torch.meshgrid(s, s, indexing="ij")
        u = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        nn = torch.as_tensor(np.asarray(n, float) / np.linalg.norm(n),
                             dtype=torch.float32, device=self.device)
        base = self.centre - nn * (self.centre @ nn - float(d))
        pts = base + gu[..., None] * u + gw[..., None] * w
        ijk = torch.round((pts - self.org) / self.hc - 0.5 - self.lo).long()
        ok = ((ijk >= 0) & (ijk < torch.tensor(self.G, device=self.device))).all(-1)
        ijk = ijk.clamp(min=torch.zeros(3, dtype=torch.long, device=self.device),
                        max=torch.tensor([g - 1 for g in self.G], device=self.device))
        flat = (ijk[..., 0] * self.G[1] + ijk[..., 1]) * self.G[2] + ijk[..., 2]
        # and the eight cells around the exact point, for scattering back
        u = (pts - self.org) / self.hc - 0.5 - self.lo
        f0 = torch.floor(u)
        fr = u - f0
        sf, sw = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    q = (f0 + torch.tensor([dx, dy, dz], device=self.device, dtype=torch.float32))
                    good = ((q >= 0) & (q < torch.tensor(self.G, device=self.device))).all(-1)
                    q = q.long().clamp(min=torch.zeros(3, dtype=torch.long, device=self.device),
                                       max=torch.tensor([g - 1 for g in self.G],
                                                        device=self.device))
                    sf.append((q[..., 0] * self.G[1] + q[..., 1]) * self.G[2] + q[..., 2])
                    w = ((fr[..., 0] if dx else 1 - fr[..., 0]) *
                         (fr[..., 1] if dy else 1 - fr[..., 1]) *
                         (fr[..., 2] if dz else 1 - fr[..., 2]))
                    sw.append(w * good * ok)
        return flat, ok, torch.stack(sf), torch.stack(sw)

    def gather(self, vol, flat, ok):
        c = vol.shape[0]
        return vol.reshape(c, -1)[:, flat.reshape(-1)].reshape((c,) + flat.shape) * ok

    def scatter(self, num, den, sf, sw, val):
        """Trilinear write-back: each slice pixel contributes to the eight cells around it."""
        c = val.shape[0]
        for f, w in zip(sf, sw):
            fl = f.reshape(-1)
            num.reshape(c, -1).index_add_(1, fl, (val * w).reshape(c, -1))
            den.reshape(-1).index_add_(0, fl, w.reshape(-1))


def planes_of(st, cams, fam, device):
    """Every plane of one family, spaced so that together they touch every cell they can.

    Each is (normal, offset, in-plane direction to pin) -- the last is the polar axis for the
    longitudinal family and nothing for the transverse, whose sections are the same under any
    turn about that axis.
    """
    if fam == "h":
        n = np.asarray(cams["h_planes"][0, :3], float)
        c = cell_centres(st, device).reshape(-1, 3).cpu().numpy()
        proj = c @ (n / np.linalg.norm(n))
        lo, hi = proj.min(), proj.max()
        step = st["hc"]
        return [(n, float(d), None) for d in np.arange(lo, hi + step, step)]
    ax = np.asarray(cams["h_planes"][0, :3], float)     # the polar axis: transverse normal
    u, w = plane_basis(ax)
    out = []
    for k in range(AZ):
        a = np.pi * k / AZ
        nn = np.cos(a) * u + np.sin(a) * w              # a plane containing the axis
        c = cell_centres(st, device).reshape(-1, 3).mean(0).cpu().numpy()
        out.append((nn, float(c @ nn), ax))
    return out


def place(photo, pmask, mask, device, gen, turns=True):
    """Put a photograph on a slice: its shell scaled onto the slice's shell, the rest left alone."""
    import torch.nn.functional as F
    ys, xs = pmask.nonzero(as_tuple=True)
    if ys.numel() < 16:
        return None
    a = photo[:, int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
    if ROT:
        # quarter turns only where they are legitimate. A transverse section is the same under any
        # turn about the polar axis; a longitudinal one is not -- turning it lays the fruit's axis
        # sideways, which no volume can satisfy alongside the transverse family. Mirroring across
        # the axis is fine for both.
        if turns:
            k = int(torch.randint(0, 4, (1,), generator=gen, device=gen.device))
            a = torch.rot90(a, k, (-2, -1))
        if bool(torch.randint(0, 2, (1,), generator=gen, device=gen.device)):
            a = torch.flip(a, (-1,))
    ys, xs = mask.nonzero(as_tuple=True)
    if ys.numel() < 16:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hy, hx = (y1 - y0) / 2 * OVER, (x1 - x0) / 2 * OVER
    b = F.interpolate(a[None], (max(int(2 * hy), 2), max(int(2 * hx), 2)), mode="bilinear",
                      align_corners=False)[0]
    out = torch.zeros_like(photo)
    ty0, tx0 = int(round(cy - hy)), int(round(cx - hx))
    sy0, sx0 = max(0, -ty0), max(0, -tx0)
    ty0, tx0 = max(0, ty0), max(0, tx0)
    h = min(b.shape[-2] - sy0, out.shape[-2] - ty0)
    w = min(b.shape[-1] - sx0, out.shape[-1] - tx0)
    out[:, ty0:ty0 + h, tx0:tx0 + w] = b[:, sy0:sy0 + h, sx0:sx0 + w]
    return out * mask


def photo_targets(idx_f, sl, mask, photos, pmasks, device, seed, turns=True):
    """The family's photographs, each placed on each plane's own shell. Fixed for the whole run.

    Under PICK="fixed" one of them is drawn per plane and that is the plane's target for ever.
    Under "nearest" all of them are kept and the step chooses.
    """
    gen = torch.Generator(device).manual_seed(seed)
    mf = mask[None].float()
    out = []
    for flat, ok, _, _ in idx_f:
        m = sl.gather(mf, flat, ok)[0] > 0.5
        ks = range(len(photos)) if PICK == "nearest" else \
            [int(torch.randint(0, len(photos), (1,), generator=gen, device=device))]
        cand = [place(photos[k], pmasks[k], m, device, gen, turns) for k in ks]
        cand = [c for c in cand if c is not None]
        out.append(None if not cand else torch.stack(cand) * 2 - 1)
    return out


def load_prior(path, device):
    """The model and the schedule it was trained under.

    The schedule is read from the checkpoint. It used to be parsed out of the file name, which
    worked until a checkpoint was copied under another name: the sampler then ran a shifted model
    against an unshifted schedule and produced pure colour noise, which is exactly what a model
    asked for the noise at the wrong level will do.
    """
    st = torch.load(path, map_location=device, weights_only=False)
    net = patchdiff.Denoiser(dim=st.get("dim", patchdiff.DIM)).to(device)
    net.load_state_dict(st["sd"]); net.eval()
    if "shift" not in st:
        raise SystemExit(f"{os.path.basename(path)} predates the schedule being recorded; "
                         f"retrain it, or the sampler cannot know what noise it was trained on")
    return net, st.get("T", patchdiff.T), float(st["shift"])


@torch.no_grad()
def sample(st, cams, nets, device="cuda", steps=None, log=print, photos=None, state=None,
           chunk=None):
    """The alternating reverse chain, over the whole interior volume."""
    (net_h, T, sh), (net_v, Tv, shv) = nets["h"], nets["v"]
    ab = patchdiff.schedule(T, device, shift=sh)
    steps = T if steps is None else steps
    g = torch.Generator(device).manual_seed(SEED)

    vol0, mask = dense(st)
    sl = Slicer(st, device)
    sl.span = WINDOW
    fams = {f: planes_of(st, cams, f, device) for f in ("h", "v")}
    idx = {f: [sl.index(n, d, pr) for n, d, pr in fams[f]] for f in ("h", "v")}
    cov = {f: torch.zeros(mask.numel(), device=device) for f in ("h", "v")}
    for f in ("h", "v"):
        for flat, ok, sf, sw in idx[f]:
            for q, w in zip(sf, sw):
                cov[f].index_add_(0, q.reshape(-1), (w ** SHARP).reshape(-1))
    for f in ("h", "v"):
        hit = (cov[f].reshape(mask.shape) > 0) & mask
        log(f"  {f}: {len(fams[f])} planes, they touch {100. * hit.sum() / mask.sum():.1f}% "
            f"of the {int(mask.sum()):,} interior cells")

    tgt = {}
    if MODE == "exemplar":
        for f in ("h", "v"):
            tgt[f] = photo_targets(idx[f], sl, mask, *photos[f], device, SEED + (f == "v"),
                                   turns=(f == "h"))
            n_ok = sum(t is not None for t in tgt[f])
            # what a residual of this size means: two different photographs of this fruit, laid on
            # the same plane, differ by this much. A volume that sits at that distance from its
            # targets is as close to them as they are to each other, and no arrangement of cells
            # can do better while every plane insists on a different photograph.
            tot, n = 0., 0
            for c in tgt[f]:
                if c is None or len(c) < 2:
                    continue
                d = (c[:, None] - c[None]).abs().mean((2, 3, 4))
                iu = torch.triu_indices(len(c), len(c), 1)
                tot += float(d[iu[0], iu[1]].mean()); n += 1
            log(f"  {f}: a photograph drawn for {n_ok} of {len(tgt[f])} planes; two different "
                f"photographs on the same plane differ by {tot / max(n, 1):.4f}")

    def residual(cur):
        """How far the volume's clean estimate is from what each family's planes were given."""
        out = {}
        for ff in ("h", "v"):
            if not tgt.get(ff):
                continue
            tot, n = 0., 0
            for (fl, o, _, _), tg in zip(idx[ff], tgt[ff]):
                if tg is None:
                    continue
                mm = sl.gather(mask[None].float(), fl, o)[0] > 0.5
                d = (sl.gather(cur, fl, o)[None] - tg).abs()[:, :, mm].mean((1, 2))
                tot += float(d.min()); n += 1
            out[ff] = tot / max(n, 1)
        return out

    if state is None:
        state = dict(x=torch.randn(vol0.shape, device=device, generator=g) * mask,
                     prev=torch.zeros_like(vol0), t=steps - 1, frames=[], hist=[])
    x = state["x"].to(device)
    prev = state["prev"].to(device)
    t_end = max(-1, state["t"] - (chunk if chunk else steps))
    t0 = time.time()
    for t in range(state["t"], t_end, -1):
        fams = ("h", "v") if FUSE else \
            (("h",) if (float(K) == int(float(K))
                        and (steps - 1 - t) % int(float(K)) != int(float(K)) - 1)
             or (float(K) != int(float(K))
                 and float(torch.rand((), device=device, generator=g)) > 1. / float(K))
             else ("v",))
        # each family accumulates on its own and is normalised before the two are combined.
        # Sharing one accumulator looks like equal weighting and is not: sharpened weights hand
        # each cell to whichever plane passes closest, and transverse planes are parallel and a
        # cell apart while longitudinal ones fan out, so the transverse family won almost every
        # cell -- measured, its residual halved while the longitudinal family's rose.
        acc = {}
        mf = mask[None].float()
        a_t = ab[t]
        for f in fams:
            num = torch.zeros_like(x); den = torch.zeros(mask.numel(), device=device)
            net = nets[f][0]
            wf = (W_H if f == "h" else W_V) if FUSE else 1.0
            for i in range(0, len(idx[f]), 32):
                batch = idx[f][i:i + 32]
                im = torch.stack([sl.gather(x, fl, o) for fl, o, _, _ in batch])
                mk = torch.stack([sl.gather(mf, fl, o) for fl, o, _, _ in batch])
                if MODE == "exemplar":
                    keep = [j for j in range(len(batch)) if tgt[f][i + j] is not None]
                    if not keep:
                        continue
                    est = (im[keep] / a_t.sqrt()).clamp(-1, 1)
                    sel = []
                    for jj, j in enumerate(keep):
                        c = tgt[f][i + j]
                        if len(c) == 1:
                            sel.append(c[0]); continue
                        mm = mk[keep][jj] > 0.5
                        dd = ((c - est[jj][None]).abs() * mm[None]).sum((1, 2, 3))
                        sel.append(c[int(dd.argmin())])
                    x0s = torch.stack(sel)
                    bat = [batch[j] for j in keep]
                else:
                    tt = torch.full((len(batch),), t / T, device=device)
                    e = net(im, tt, mk)
                    x0s = ((im - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-1, 1)
                    bat = batch
                for j, (_, _, sf, sw) in enumerate(bat):
                    sl.scatter(num, den, sf, [w ** SHARP for w in sw], x0s[j])
            acc[f] = (num, den.reshape(mask.shape), wf)

        tot = torch.zeros_like(x); wsum = torch.zeros_like(x[:1])
        for f, (num, den3, wf) in acc.items():
            hit = (den3 > 1e-4)[None].float()
            tot = tot + wf * hit * (num / den3.clamp(min=1e-4))
            wsum = wsum + wf * hit
        touched = wsum[0] > 1e-6
        x0 = (tot / wsum.clamp(min=1e-6)).clamp(-1, 1)
        lam = LAM + (LAM_END - LAM) * (1 - t / max(steps - 1, 1))
        if MODE == "exemplar" and lam < 1.0:
            x0 = torch.where(touched[None], (1 - lam) * prev + lam * x0, prev)
        prev = x0

        a, ap = ab[t], ab[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        eps = (x - a.sqrt() * x0) / (1 - a).clamp(min=1e-6).sqrt()
        if t > 0:
            beta = 1 - a / ap
            z = torch.randn(x.shape, device=device, generator=g)
            xn = ap.sqrt() * x0 + (1 - ap - beta).clamp(min=0).sqrt() * eps + beta.sqrt() * z
        else:
            xn = x0
        x = torch.where(touched[None] & mask[None], xn, x) * mask

        if t % max(steps // 100, 1) == 0 and len(state["frames"]) < 120:
            fr = []
            for ff in ("h", "v"):
                fl, o, _, _ = idx[ff][len(idx[ff]) // 2]
                fr.append(sl.gather((x + 1) / 2, fl, o).clamp(0, 1))
            a4 = torch.cat(fr, -1).permute(1, 2, 0).clamp(0, 1)
            state["frames"].append((t, (a4 * 255).to(torch.uint8).cpu().numpy()))
        if t % max((chunk or steps) // 4, 1) == 0:
            r = residual(x0)
            state["hist"].append((t, r.get("h"), r.get("v")))
            log(f"    t {t:5d}  |x| {float(x[:, mask].abs().mean()):.3f}  lam {lam:.2f}  "
                f"residual  h {r.get('h', 0):.4f}  v {r.get('v', 0):.4f}   "
                f"{time.time() - t0:.0f}s")

    state.update(x=x, prev=prev, t=t_end)
    return (x + 1) / 2, mask, state
