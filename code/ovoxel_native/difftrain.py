"""Solve the interior as a diffusion chain, with the photographs as the targets and the planes
drawn afresh at every step.

Two things this changes from the pipeline's fitting.

The planes are not a fixed schedule. Measured, the fixed 26 cut faces touch about half the orange's
cells and that is a ceiling -- the rest never receive a gradient, whatever they were initialised to
is what they keep, and every "unsupervised plane" problem in this line comes from there. Planes
drawn uniformly instead accumulate: 100 draws reach 85% of the interior, 400 reach 98%, and a run
makes hundreds of steps anyway, so the coverage is free.

The interior starts as noise and is carried down a schedule rather than fitted straight. Each step
takes a few gradient steps towards the photographs of whichever planes were drawn -- that is the
denoise -- and then puts back the noise the next noise level calls for. Early steps therefore
cannot lock in the first few draws, which is exactly the failure a straight fit has when its planes
keep moving: whatever it fits first is what the untouched cells stay at.

There is no learnt prior anywhere in this. The only thing that says what an interior looks like is
the photographs.
"""
import os, sys, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import batchrender
import critic
import refsel
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
STATE = os.environ.get("STATE", f"{W}/state_{OBJ}.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_{OBJ}_bal.npz")
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
RES = int(os.environ.get("RES", "512"))
T = int(os.environ.get("DT_T", "400"))
# Planes per step. This was four because four was what the one-at-a-time renderer could afford,
# not because four was a good number: the gradient of every step was decided by four cut faces.
# `batchrender` rasterises them in one pass and drops the exterior, which the interior receives no
# gradient from at all and which was more than half the cost of a section.
K = int(os.environ.get("DT_K", "32"))
M = int(os.environ.get("DT_M", "2"))          # gradient steps per noise level
# How long a drawn set of planes is kept before the next is drawn. Redrawing at every gradient step
# gives each plane one step and then abandons it, which is enough to spread the coverage but not
# enough to resolve any of the faces it touches; holding the draw lets each one settle before the
# next is taken.
HOLD = int(os.environ.get("DT_HOLD", "0")) or M
LR = float(os.environ.get("DT_LR", "0.05"))
# Cosine decay to a floor, over the noise levels and the noiseless steps together. At a fixed rate
# the last steps move as far as the first, which on a chain that is meant to be settling is the
# difference between resolving a face and jittering around it.
LR_END = float(os.environ.get("DT_LR_END", "0")) 
FIXED = os.environ.get("DT_FIXED", "0") == "1"    # the old schedule, for comparison
# How the planes are chosen. "random" draws each one independently. "cycle" walks a formula
# instead: the transverse depth advances by a radical inverse and the longitudinal azimuth by the
# golden angle, each family on its own counter, so any prefix of the run is spread rather than
# spread on average -- and the schedule is reproducible from the step number alone, with no random
# state in it. Every PERIOD draws the sequence is shifted, because a formula that guarantees even
# coverage in one pass also guarantees it repeats itself in the next: measured on coverage alone,
# the unshifted cycle saturated at 95% while independent draws went on to 98%.
SEQ = os.environ.get("DT_SEQ", "random")
PERIOD = int(os.environ.get("DT_PERIOD", "64"))
PHI = (1 + 5 ** 0.5) / 2


def radical(k, base=2):
    f, r = 1.0, 0.0
    while k:
        f /= base
        r += f * (k % base)
        k //= base
    return r
# The last denoise step, with no noise put back after it. A DDPM's final step is exactly this, and
# without it the run ends at whatever the second-to-last noise level left, which measured was an
# error of 0.10 against the 0.03 the same planes reach when they are allowed to settle.
POLISH = int(os.environ.get("DT_POLISH", "150"))
# How much of the schedule's noise is actually put back. 1 is the chain as written; 0 leaves the
# noise only in the initialisation, which turns the run into plain fitting with drawn planes and is
# the ablation that says whether the chain earns its place at all.
NOISE = float(os.environ.get("DT_NOISE", "1"))
# How many of the step's planes are also shown to the critic. Every plane in this run is drawn, so
# none of them has a photograph of its own; the pixel term is answering with the nearest one, and
# the critic is the term that can say something about a face no photograph was taken at without
# pretending one was.
CRIT_N = int(os.environ.get("DT_CRIT_N", "4"))
DIAG = os.environ.get("DT_DIAG", "0") == "1"
dev = "cuda"

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH = H_HI - H_LO
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
NV = len(vp)

conf = open(f"{OBJDIR}/{OBJ}.conf").read()


def photos(key):
    spec = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)][0]
    return f"{FN}/{spec}"


PH, PV = photos("REF_H="), photos("REF_V=")
ref_h = [torch.as_tensor(refsel.as_array(refsel.solved_photo(PH, i, NH), RES),
                         device=dev).permute(2, 0, 1) for i in range(NH)]
ref_v = [torch.as_tensor(refsel.as_array(refsel.solved_photo(PV, i, NV), RES),
                         device=dev).permute(2, 0, 1) for i in range(NV)]
print(f"{OBJ}: {len(st['interior']):,} cells, {NH} transverse and {NV} longitudinal photographs, "
      f"{'fixed planes' if FIXED else 'planes ' + SEQ}, T={T}, {K} planes held for {HOLD} steps")

step_h = float(hd[H_LO + 1] - hd[H_LO]) if NH > 1 else 1.0
lo, hi = float(hd[H_LO]) - step_h / 2, float(hd[H_HI - 1]) + step_h / 2
rng = np.random.default_rng(0)
_kh = _kv = 0


def draw():
    """A plane, the photograph that speaks for it, and which family it came from."""
    global _kh, _kv
    if SEQ == "cycle":
        if (_kh + _kv) * NH % (NH + NV) < NH:
            _kh += 1
            u = (radical(_kh) + (PHI * (_kh // PERIOD)) % 1.0) % 1.0
            d = lo + u * (hi - lo)
            i = int(np.clip(round((d - float(hd[H_LO])) / step_h), 0, NH - 1))
            return hmvp, hn, d, ref_h[i], "h"
        _kv += 1
        f = ((_kv * PHI) + (PHI * PHI * (_kv // PERIOD))) % 1.0
        j = int(f * NV) % NV
        a = f * NV - j
        nv = (1 - a) * vp[j, :3] + a * vp[(j + 1) % NV, :3]
        nv = nv / np.linalg.norm(nv)
        d = float(np.dot(nv, vp[j, :3] * vp[j, 3]))
        return vmvp[j], torch.as_tensor(nv, dtype=torch.float32, device=dev), d, \
            ref_v[j if a < 0.5 else (j + 1) % NV], "v"
    if rng.random() < NH / (NH + NV):
        if FIXED:
            i = int(rng.integers(NH)); return hmvp, hn, float(hd[H_LO + i]), ref_h[i], "h"
        d = rng.uniform(lo, hi)
        i = int(np.clip(round((d - float(hd[H_LO])) / step_h), 0, NH - 1))
        return hmvp, hn, d, ref_h[i], "h"                # the nearest depth's photograph
    j = int(rng.integers(NV))
    if FIXED:
        return vmvp[j], torch.as_tensor(vp[j, :3], dtype=torch.float32, device=dev), \
            float(vp[j, 3]), ref_v[j], "v"
    a = rng.random()                                     # an azimuth between two cameras
    nv = (1 - a) * vp[j, :3] + a * vp[(j + 1) % NV, :3]
    nv = nv / np.linalg.norm(nv)
    d = float(np.dot(nv, vp[j, :3] * vp[j, 3]))
    return vmvp[j], torch.as_tensor(nv, dtype=torch.float32, device=dev), d, \
        ref_v[j if a < 0.5 else (j + 1) % NV], "v"


def schedule(t_max, device):
    s = 0.008
    t = torch.arange(t_max + 1, device=device, dtype=torch.float32) / t_max
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0)


def set_lr(frac):
    if LR_END <= 0:
        return LR
    r = LR_END + (LR - LR_END) * 0.5 * (1 + math.cos(math.pi * min(max(frac, 0.), 1.)))
    for gp in opt.param_groups:
        gp["lr"] = r
    return r


ab = schedule(T, dev)
crit = None
if critic.WEIGHT > 0:
    crit = critic.Trainer({"h": ref_h, "v": ref_v}, dev)
g = torch.Generator(dev).manual_seed(0)
z = (torch.rand(st["interior"].shape, device=dev, generator=g) * 2 - 1).requires_grad_(True)
opt = torch.optim.Adam([z], lr=LR)
t0 = time.time()
TOTAL = T * M + POLISH
for t in reversed(range(T)):
    set_lr(((T - 1 - t) * M) / TOTAL)
    held = None
    for _m in range(M):
        if _m % HOLD == 0:
            held = [draw() for _ in range(K)]
        st["interior"] = (z.clamp(-1, 1) + 1) / 2
        img, al = batchrender.render_batch(st, glctx, [h[0] for h in held], [h[1] for h in held],
                                           [h[2] for h in held], RES)
        tg = torch.stack([h[3] for h in held])
        m = (al > 0.5).float()
        pix = (((img - tg).abs() * m).sum(dim=(1, 2, 3)) /
               m.sum(dim=(1, 2, 3)).clamp(min=1) / 3).mean()
        loss = pix
        if crit is not None:
            adv, nadv = 0., 0
            for j in range(min(CRIT_N, len(held))):
                term, _ = crit.step(held[j][4], img[j])
                if float(term) != 0.:
                    adv = adv + term; nadv += 1
            if nadv:
                adv = adv / nadv
                lam = crit.adaptive(pix, adv, img)
                loss = pix + lam * critic.WEIGHT * adv
                if DIAG and (t % max(T // 8, 1) == 0) and _m == 0:
                    gp = torch.autograd.grad(pix, img, retain_graph=True)[0].norm()
                    ga = torch.autograd.grad(adv, img, retain_graph=True)[0].norm()
                    acc = np.mean([v for v in crit.acc.values()]) if crit.acc else float("nan")
                    print(f"      critic: adv {float(adv):+.4f}  lam {lam:.3f}  "
                          f"|grad pixel| {float(gp):.3e}  |grad adv| {float(ga):.3e}  "
                          f"tells real from fake {acc:.2f} of the time", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if crit is not None:
            crit.flush()
    if t > 0:                                   # put back the noise this level calls for
        with torch.no_grad():
            a = ab[t - 1]
            z.data = (a.sqrt() * z.data.clamp(-1, 1) +
                      NOISE * (1 - a).sqrt() *
                      torch.randn(z.shape, device=dev, generator=g)).clamp(-1, 1)
    if t % max(T // 8, 1) == 0:
        print(f"    t {t:4d}  a_bar {float(ab[t]):.4f}  lr {opt.param_groups[0]['lr']:.4f}  "
              f"loss {float(loss):.4f} (pixel {float(pix):.4f})"
              f"   {time.time() - t0:.0f}s", flush=True)

for _p in range(POLISH):
    set_lr((T * M + _p) / TOTAL)
    if _p % HOLD == 0:
        held = [draw() for _ in range(K)]
    st["interior"] = (z.clamp(-1, 1) + 1) / 2
    img, al = batchrender.render_batch(st, glctx, [h[0] for h in held], [h[1] for h in held],
                                       [h[2] for h in held], RES)
    tg = torch.stack([h[3] for h in held])
    m = (al > 0.5).float()
    pix = (((img - tg).abs() * m).sum(dim=(1, 2, 3)) /
           m.sum(dim=(1, 2, 3)).clamp(min=1) / 3).mean()
    loss = pix
    if crit is not None:
        adv, nadv = 0., 0
        for j in range(min(CRIT_N, len(held))):
            term, _ = crit.step(held[j][4], img[j])
            if float(term) != 0.:
                adv = adv + term; nadv += 1
        if nadv:
            adv = adv / nadv
            loss = pix + crit.adaptive(pix, adv, img) * critic.WEIGHT * adv
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if crit is not None:
        crit.flush()
print(f"  after {POLISH} noiseless steps: loss {float(loss):.4f}   {time.time() - t0:.0f}s")

with torch.no_grad():
    st["interior"] = (z.clamp(-1, 1) + 1) / 2
    e_h = [float((lambda im, al: (((im - ref_h[i]).abs() * (al > 0.5)).sum() /
                                  (al > 0.5).float().sum().clamp(min=1) / 3))(
        *ON.render_section(st, glctx, hmvp, hn, float(hd[H_LO + i]), RES)[:2]))
        for i in range(NH)]
    print(f"  photographed transverse planes: mean error {np.mean(e_h):.4f}")
    cols = []
    for i in (0, NH // 2, NH - 1):
        img, _, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[H_LO + i]), RES)
        cols.append(torch.cat([ref_h[i], img.clamp(0, 1)], -2))
    for f in (0.3, 0.5, 0.7):                    # depths no photograph was taken at
        d = lo + f * (hi - lo)
        img, _, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
        cols.append(torch.cat([torch.ones_like(img), img.clamp(0, 1)], -2))
    sheet = torch.cat(cols, -1).permute(1, 2, 0).clamp(0, 1)
Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(
    f"{W}/difftrain_{OBJ}{'_fixed' if FIXED else ''}.jpg", quality=92)
torch.save({"interior": st["interior"].detach().cpu()},
           f"{W}/difftrain_{OBJ}{'_fixed' if FIXED else ''}.pt")
print(f"SHEET difftrain_{OBJ}{'_fixed' if FIXED else ''}.jpg  (three photographed planes with "
      f"their photographs above, then three depths nobody photographed)")
