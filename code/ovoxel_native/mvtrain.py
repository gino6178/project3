"""The O-Voxel-native representation trained the way stage_train trains everything else.

One program. What varies is variables, not code paths:

    ROUTE=1     build_orange/lattice     the released ply, quantised as it is
    ROUTE=2     build_orange_r2/skin     shape from make_shape.py's ellipsoid SDF, exterior from
                                         skin_project.py's six-view projection.  Nothing in this
                                         arm comes from a pre-existing Gaussian model.
    ANCHOR=1    interior and surf_rgb are decoded from an 8-d per-cell feature through a shared
                MLP (see anchor.py).  ANCHOR=0 keeps them as free per-cell RGB, which is what
                produced the speckle.
    SHELL_PIN   the exterior is not trained.
    EXT_VIEWS   how many of the six exterior directions supervise.

Everything else is stage_train's, carried across: 200 outer iterations each sweeping all 16
transverse depths (centers[H_LO:H_HI] of 24, jittered +-0.5 of a step) and all 10 longitudinal
azimuths ((180/N_VPLANES)*i); both reference families with sds_demo's own assignment -- equation
(27)'s solved permutation and phases transverse, equation (14)'s continuous depth blend
longitudinal; SECTION_MATCH=1 unchanged, SEC_RIND_MATCH and SEC_PATH_MATCH off as they default;
SEC_SKIP_OUTER's mask from occupancy.surface_cells at two layers.

There is no diffusion path. REF_WARMUP is 10^7 in stage_train, so past_warmup() is never true and
the photograph is the target for the whole run.

The trained state is: two feature tensors, two small MLPs, and the dual grid's own geometry. No
opacity, no covariance, no spherical harmonics, no Gaussian anywhere in it or in the renderer.
"""
import json, os, random, sys, time
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON
import nvdiffrast.torch as dr
import section_match as sm
import refsel
import anchor
import azjitter
import fieldreg
import overlap
import patchdist
import refalign
import critic
import secloss
import styleloss
import triplane
import unsup

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
# The two reference families, relative to FN_ROOT. The orange's were the only ones this build
# ever used, which was fine while it was a single-object experiment and is not once the same
# program has to run on the rest -- every object carries its own pair in objects/<obj>.conf.
REF_H = os.path.join(FN, os.environ.get("REF_H", "secref_orraw_hsep"))
REF_V = os.path.join(FN, os.environ.get("REF_V", "secref_orraw_vsep"))
EXT = os.path.join(FN, os.environ.get("EXT_DIRS", "cube_or6_prep"))

ROUTE = os.environ.get("ROUTE", "1")
STATE = os.environ.get("STATE", f"{W}/state_r{ROUTE}.pt")
CAMS = os.environ.get("CAMS", f"{W}/cams_mv.npz" if ROUTE == "1" else f"{W}/cams_mv_r2.npz")
OUT = os.environ.get("OUT", f"{W}/r{ROUTE}")
ANCHOR = os.environ.get("ANCHOR", "1") == "1"
PREFIT = os.environ.get("ANCHOR_PREFIT", "1") == "1"
ITERS = int(os.environ.get("ITERS", "200"))
RES = int(os.environ.get("ABL_RES", "512"))
JITTER = float(os.environ.get("JITTER", "0.5"))
EXT_VIEWS = int(os.environ.get("EXT_VIEWS", "6"))
SHELL_PIN = os.environ.get("SHELL_PIN", "1") == "1"
SEC_SKIP_OUTER = float(os.environ.get("SEC_SKIP_OUTER", "0.10"))
SKIP_LAYERS = int(os.environ.get("SHELL_PIN_LAYERS", "2"))
LR_FEAT = float(os.environ.get("LR_FEAT", "0.005"))
LR_MLP = float(os.environ.get("LR_MLP", "0.002"))
LR_RGB = float(os.environ.get("LR_RGB", "0.02"))
LR_GEO = float(os.environ.get("LR_GEO", "1e-3"))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "20"))
VOXEL_SMOOTH = os.environ.get("VOXEL_SMOOTH", "1") == "1"
# The pipeline's cross-family reconciliation, which is off there and reported only in its
# averaging mode. See xcons.py for why "copy" is the mode worth measuring and why the
# transverse family is the one that should win.
SEC_XCONS = float(os.environ.get("SEC_XCONS", "0"))
SEC_XCONS_AT = int(os.environ.get("SEC_XCONS_AT", "0"))
SEC_JOINT = os.environ.get("SEC_JOINT", "1") == "1"
# The interior field as three interpolating planes rather than one latent per cell. See
# triplane.py: it is an architectural answer to the same gap `SEC_TV` answers as a penalty, so the
# two are meant to be compared with everything else held identical. Two things it turns off, both
# because they are per-cell operations with no counterpart: the coverage probe on feat_i, which
# would otherwise report a non-leaf's absent gradient as zero coverage, and VOXEL_SMOOTH, whose
# whole job -- filling untrained cells from trained neighbours -- is what the interpolation does.
TRIPLANE = os.environ.get("TRIPLANE", "0") in ("1", "2")
# How much each family's loss counts, as a multiplier on the plane-count weighting it already has.
#
# The plane count decides two separate things and they had no way of being set apart: how much of
# the object a family reaches, and how much of the gradient it gets. A family contributes one plane
# per step at weight |family|/G over G steps, so its total weight per outer iteration is exactly its
# plane count -- which is why changing the split changes both at once. Measured on the orange, the
# two halves of that pull in opposite directions: moving 14/12 to 9/17 improved the longitudinal
# column by 9.7% and left 19.6% of the cells with no gradient at all, because the transverse family
# is the one that sweeps and cutting it from 14 to 9 took its reach from 91.1% to 71.8%.
#
# SEC_FAM_W sets the weights directly, so the split can keep the reach of one arm and the balance of
# another: `SEC_FAM_W=9,17` on 14/12 planes asks for 9:17 of gradient over 93% of the object.
# Normalised so the total is what it would have been, and the default of 1,1,1 is the old
# behaviour exactly.
_famw = [float(x) for x in os.environ.get("SEC_FAM_W", "1,1,1").split(",")]
while len(_famw) < 3:
    _famw.append(1.0)
FAM_W = _famw[:3]
FAM_W_EFF = [0.0, 0.0, 0.0]
# How far a longitudinal plane may be moved along its own normal, as a fraction of the object's
# radius. 0, and it should stay 0: the asymmetry it was written to remove is in the data, not in
# this file.
#
# The transverse family is one camera and 24 depths, of which 16 are supervised and each is
# jittered by half a step, because there are photographs at many depths. The longitudinal family is
# one camera per azimuth and `centers[len//2]` -- the middle depth, one plane per azimuth, never
# moved -- because every longitudinal photograph is a CENTRAL section. Moving the plane off the
# axis therefore asks the model to reproduce a central cut at an off-centre depth: dumping the
# target at 0%, 5% and 12% of the radius gives the same full central section all three times,
# because that is the only photograph there is.
#
# Measured on the orange, against the same arm with this off (0.0859 rh, 0.2300 rv): 0.05 gives
# 0.1186 / 0.2224, 0.12 gives 0.1235 / 0.2326, 0.25 gives 0.1122 / 0.2527. The longitudinal gain
# is small, the transverse cost is 36%, and it grows with the jitter. What the held-out
# longitudinal planes are asking for -- they sit 4.6% to 12.3% off the axis while every supervised
# one sits at 3.1% to 3.4% -- is a photograph that does not exist.
JITTER_V = float(os.environ.get("JITTER_V", "0"))
# half-width in pixels of the band around the other family's crossings, inside which the
# distributional term is taken; 0 applies it to the whole face, which is what was measured and
# found harmful
SEC_DIST_BAND = int(os.environ.get("SEC_DIST_BAND", "0"))
# How far a longitudinal plane may be turned about the axis, as a fraction of the spacing between
# the supervised azimuths. This is the family's own degree of freedom, unlike JITTER_V.
#
# Why it exists, measured by geometry alone on the orange's 770,182 solid cells: the transverse
# family reaches 92.4% of them, because it has 16 depths and each is jittered half a step, so it
# sweeps. The longitudinal family reaches 15.9%, because it is ten fixed sheets through the axis
# that never move, and two sheets do not cover what is between them. Cells both families reach:
# 14.1%. So 84% of the object has no longitudinal supervision at all and a held-out longitudinal
# cut passes almost entirely through cells only the other family constrained -- which is a more
# basic cause of the vertical striping than the references all being central sections.
#
# Turning at 0.5 of the 18-degree spacing takes the longitudinal family to 97.7% and both families
# to 90.4%, and it keeps the plane through the axis, so it is still a central section and the
# photographs are still the right kind.
JITTER_AZ = float(os.environ.get("JITTER_AZ", "0"))
# Whether each plane's reference is chosen at the position the plane was jittered to rather than at
# its unjittered index, for both families by the same rule.
#
# It sounds as though it should matter -- with JITTER_AZ a longitudinal plane turns most of the way
# towards its neighbour and is otherwise still supervised by its own photograph -- and measured, it
# does not. On the orange at 5,200 steps against the same arm without it: 0.0867 transverse and
# 0.2255 longitudinal against 0.0859 and 0.2257, which is inside the noise, and on top of the
# turning 0.1193/0.2026 against 0.1220/0.2015, better on one column and worse on the other. It also
# costs half again as much per step, since the blend it asks for misses refsel's cache.
#
# Kept because the negative result is worth more than the code is, and off.
REF_FOLLOW = os.environ.get("REF_FOLLOW", "0") == "1"
# The jitter is quantised before the reference is asked for. refsel caches a blend by its mixing
# weight rounded to three places, so a continuous jitter misses the cache every step and pays for
# two disc detections and two resizes -- measured, 330 ms a step became 758. Sixteen positions
# across the spacing is finer than the plane spacing itself and the cache holds.
REF_STEPS = int(os.environ.get("REF_STEPS", "16"))


def _q(f):
    return round(f * REF_STEPS) / REF_STEPS if REF_FOLLOW else f
SEC_XCONS_HOLD = os.environ.get("SEC_XCONS_HOLD", "0") == "1"
SEC_XCONS_MODE = os.environ.get("SEC_XCONS_MODE", "copy")
ABL_INTERVAL = int(os.environ.get("ABL_INTERVAL", "30"))
ABL_GRID = int(os.environ.get("ABL_GRID", "16"))
dev = "cuda"

os.makedirs(OUT, exist_ok=True)
for s in ("eval_init", "eval_final"):
    os.makedirs(f"{OUT}/{s}", exist_ok=True)
random.seed(0); torch.manual_seed(0); np.random.seed(0)

st = torch.load(STATE, map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(CAMS)

H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH = H_HI - H_LO
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
h_step = float(hd[1] - hd[0])
NV = len(C["v_planes"])
enames = [str(x) for x in C["e_names"]][:EXT_VIEWS]

print(f"route {ROUTE}: {STATE}")
print(f"  {len(st['interior']):,} solid coarse cells, {len(st['dual_v']):,} dual vertices")
print(f"{NH} transverse depths (d {hd[H_LO]:+.4f} .. {hd[H_HI-1]:+.4f}, step {h_step:+.4f}), "
      f"{NV} longitudinal azimuths, {len(enames)} exterior views {enames}")
_dist = os.environ.get("SEC_DIST", "0")
_distdesc = ("" if _dist not in ("sw", "chamfer", "js") else
             f", then {_dist} between the two patch distributions at "
             f"{os.environ.get('SEC_DIST_W', '1.0')} from "
             f"{os.environ.get('SEC_DIST_START', '0.5')} of the run, keeping "
             f"{os.environ.get('SEC_DIST_MIX', '0.3')} of the pixel term")
_lossdesc = (f"0.7(1-SSIM)+0.3MSE on {secloss.SEC_PATCH_N} crops of {secloss.SEC_PATCH}px, "
             f"band term at {secloss.SEC_PATCH_STAT}{_distdesc}") if secloss.SEC_PATCH > 0 \
    else "whole-frame L1 (SEC_PATCH=0)"
print(f"  section loss: {_lossdesc}")
print(f"ANCHOR={int(ANCHOR)}  SHELL_PIN={int(SHELL_PIN)}  SEC_SKIP_OUTER={SEC_SKIP_OUTER}  "
      f"JITTER={JITTER}  ITERS={ITERS} outer  RES={RES}  "
      f"FLAT_INIT={os.environ.get('FLAT_INIT', 'off')}")

# ---------------------------------------------------------------- what the sections may not paint
from occupancy import surface_cells                                   # noqa: E402
hc, org = st["hc"], st["org"]
centres = (st["solid"].float() + 0.5) * hc + torch.as_tensor(org, dtype=torch.float32, device=dev)
if SEC_SKIP_OUTER > 0:
    is_outer = surface_cells(centres, hc, layers=SKIP_LAYERS)
else:
    is_outer = torch.zeros(len(centres), dtype=torch.bool, device=dev)
print(f"  sections skip the exterior, {SKIP_LAYERS} cells deep by the occupancy: "
      f"{int(is_outer.sum()):,} of {is_outer.numel():,} coarse cells "
      f"({100*float(is_outer.float().mean()):.1f}%)")

# ---------------------------------------------------------------- the model
seed_i = st["interior"].detach().clone()
seed_s = st["surf_rgb"].detach().clone()

# Image-initialised only: the interior does not start from the released model's.
#
# On route 1 the lattice is the released ply quantised, so `ovnative.build` seeds `interior` from
# that model's own f_dc -- and ANCHOR_PREFIT in train_voxel.py does the same thing, fitting the
# decoder to `gaussians._features_dc` over every cell. Both are faithful to the pipeline; the page
# says as much. This removes that one input and nothing else: the exterior still comes from the
# quantised ply, because appearance from a captured model is what route 1 *is*, and only the
# interior starts neutral.
#
# 0.5 is not an invented constant. It is `ovnative.build`'s own unseeded value and it is what
# route 2 actually starts from: that state's interior is exactly [0.5, 0.5, 0.5] at the 25th,
# 50th and 75th percentiles, the spread coming only from cells next to the painted skin.
FLAT_INIT = float(os.environ.get("FLAT_INIT", "-1"))
if FLAT_INIT >= 0:
    seed_i = torch.full_like(seed_i, FLAT_INIT)
    st["interior"] = seed_i.clone()
    print(f"  interior initialised flat at {FLAT_INIT}: the released model's interior colours are "
          f"not an input to this arm")
st["dual_v"] = st["dual_v"].detach().clone().requires_grad_(not SHELL_PIN)
st["split_w"] = st["split_w"].detach().clone().requires_grad_(not SHELL_PIN)

groups = []
if ANCHOR:
    if TRIPLANE:
        # the interior only. The exterior is a dual-vertex tensor, not a cell field, and under
        # SHELL_PIN it takes no gradient at all -- there is nothing for an interpolating field to
        # do there and swapping it too would confound the two.
        _ctr = (st["solid"].float() + 0.5) * float(st["hc"]) \
            + torch.as_tensor(st["org"], dtype=torch.float32, device=st["solid"].device)
        dec_i = triplane.TriplaneDecoder(_ctr.cpu(), init_rgb=seed_i.cpu()).to(dev)
        dec_i.set_nograd(is_outer)
        print(f"  interior field: triplane {triplane.RES}x{triplane.RES}x{triplane.C_FEAT} "
              f"x3 = {3*triplane.C_FEAT*triplane.RES**2:,} floats, against "
              f"{len(seed_i)*anchor.F_DIM:,} per-cell; "
              f"{int(is_outer.sum()):,} outer cells detached from the gradient", flush=True)
    else:
        dec_i = anchor.ColourDecoder(len(seed_i), init_rgb=seed_i).to(dev)
    dec_s = anchor.ColourDecoder(len(seed_s), init_rgb=seed_s).to(dev)
    if PREFIT:
        t0 = time.time()
        anchor.prefit(dec_i, seed_i, tag="interior")
        anchor.prefit(dec_s, seed_s, tag="surf_rgb")
        print(f"  prefit took {time.time()-t0:.0f}s", flush=True)
    if SHELL_PIN:
        # col_pin: overwrite after the head. Stops the gradient as well as the drift.
        dec_s.pin_colour(torch.ones(len(seed_s), dtype=torch.bool, device=dev), seed_s)
    groups += dec_i.param_groups(LR_FEAT, LR_MLP)
    if not SHELL_PIN:
        groups += dec_s.param_groups(LR_FEAT, LR_MLP)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"  decoder: feat {anchor.F_DIM}-d per cell, stage1 {anchor.F_DIM}->128->128->"
          f"{anchor.C_DIM}, stage2 {anchor.C_DIM}->64->3, two heads (ANCHOR_SPLIT)")
else:
    dec_i = dec_s = None
    st["interior"] = st["interior"].detach().clone().requires_grad_(True)
    st["surf_rgb"] = st["surf_rgb"].detach().clone().requires_grad_(not SHELL_PIN)
    groups += [dict(params=[st["interior"]], lr=LR_RGB)]
    if not SHELL_PIN:
        groups += [dict(params=[st["surf_rgb"]], lr=LR_RGB)]
    n_train = sum(p.numel() for g in groups for p in g["params"])
if not SHELL_PIN:
    groups += [dict(params=[st["dual_v"], st["split_w"]], lr=LR_GEO)]
    n_train += st["dual_v"].numel() + st["split_w"].numel()
# Weight decay, and it is AdamW rather than Adam's own `weight_decay` because the two are not the
# same thing: Adam scales the decay by the same running second moment it scales the gradient by, so
# a rarely-touched cell -- which is most of them here -- decays at a different rate from a busy one.
# Decoupled decay treats every parameter alike, which is what a prior on the field should do.
WD = float(os.environ.get("WEIGHT_DECAY", "0"))
# per-group weight_decay wins where a group sets it (the hybrid's residual does), and WD is the
# default for the groups that do not
_anywd = WD > 0 or any(g.get("weight_decay", 0) for g in groups)
opt = (torch.optim.AdamW(groups, weight_decay=WD) if _anywd else torch.optim.Adam(groups))
# Cosine decay to LR_FLOOR of the initial rate. There was no schedule at all: a constant rate for
# 4,550 steps on a loss that stops falling after a few hundred leaves the field wandering near the
# minimum rather than settling into it, and the held-out probe reaches its best at outer 20 in most
# arms and then rises for the remaining 305. Every comparable per-scene fit -- NeRF, Plenoxels,
# Instant-NGP -- decays the rate; this one did not.
LR_DECAY = os.environ.get("LR_DECAY", "0") == "1"
LR_FLOOR = float(os.environ.get("LR_FLOOR", "0.02"))
_lr0 = [g["lr"] for g in opt.param_groups]


# Where the cosine finishes. At 1.0 the rate only reaches the floor on the last step, which is a
# schedule for a run that is still learning at the end; the probe says this one stops learning after
# about a twentieth of the budget. LR_DECAY_END=0.6 completes the cosine at 60% and holds the floor
# for the rest, so the last 40% of the run can only make small corrections.
LR_DECAY_END = float(os.environ.get("LR_DECAY_END", "1.0"))


def set_lr(frac):
    if not LR_DECAY:
        return
    import math
    f = min(max(frac / max(LR_DECAY_END, 1e-6), 0.0), 1.0)
    m = LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * f))
    for g, l0 in zip(opt.param_groups, _lr0):
        g["lr"] = l0 * m
print(f"  trainable: {n_train:,} floats "
      f"({'interior only, the exterior is pinned' if SHELL_PIN else 'interior and exterior'})")

feat_params = ([] if TRIPLANE else
               ([dec_i.feat] + ([dec_s.feat] if not SHELL_PIN else [])) if ANCHOR else [])


def decode():
    """write_into's role: put the decoded colours where the renderer reads them, in the graph."""
    if ANCHOR:
        st["interior"] = dec_i()
        st["surf_rgb"] = dec_s()


# ---------------------------------------------------------------- held-out cuts
ehm = torch.as_tensor(C["eh_mvp"], dtype=torch.float32, device=dev)
ehp = C["eh_planes"]
evm = torch.as_tensor(C["ev_mvp"], dtype=torch.float32, device=dev)
evp = C["ev_planes"]
# The probe and the reported score were the same six planes per family, so every decision taken by
# looking at the probe -- when to stop, which arm to keep -- was taken on the set the arm is then
# scored against. VAL_N of each family becomes validation, watched during training; the rest is
# test and is rendered but never looked at until the run is over.
VAL_N = int(os.environ.get("VAL_N", "3"))
VAL_H = list(range(min(VAL_N, len(ehp))))
VAL_V = list(range(min(VAL_N, len(evp))))
TEST_H = [i for i in range(len(ehp)) if i not in VAL_H]
TEST_V = [i for i in range(len(evp)) if i not in VAL_V]
print(f"  held-out split: validation {len(VAL_H)}h+{len(VAL_V)}v (the probe), "
      f"test {len(TEST_H)}h+{len(TEST_V)}v (scored, never watched)", flush=True)


def dump(folder):
    with torch.no_grad():
        decode()
        for i in range(len(ehp)):
            n = torch.as_tensor(ehp[i, :3], dtype=torch.float32, device=dev)
            img, _, _, _ = ON.render_section(st, glctx, ehm[i], n, float(ehp[i, 3]), RES)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            tag = "val" if i in VAL_H else "test"
            cv2.imwrite(f"{folder}/rh{i}_init_0.png", (a[:, :, ::-1] * 255).astype(np.uint8))
            cv2.imwrite(f"{folder}/{tag}_rh{i}.png", (a[:, :, ::-1] * 255).astype(np.uint8))
        for i in range(len(evp)):
            n = torch.as_tensor(evp[i, :3], dtype=torch.float32, device=dev)
            img, _, _, _ = ON.render_section(st, glctx, evm[i], n, float(evp[i, 3]), RES)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            tag = "val" if i in VAL_V else "test"
            cv2.imwrite(f"{folder}/rv{i}_init_0.png", (a[:, :, ::-1] * 255).astype(np.uint8))
            cv2.imwrite(f"{folder}/{tag}_rv{i}.png", (a[:, :, ::-1] * 255).astype(np.uint8))
    print(f"  -> {folder}", flush=True)


if refsel.SAMPLE:
    print("  references: drawn per step, one photograph rather than the average of two "
          "(REF_SAMPLE=1)", flush=True)
refs_h = [refsel.as_array(refsel.solved_photo(REF_H, j, NH), RES) for j in range(NH)]
refs_v = [refsel.as_array(refsel.photo(REF_V, i, NV), RES) for i in range(NV)]

_FLIP_H = os.environ.get("REF_H_FLIP", "")
_FLIP_V = os.environ.get("REF_V_FLIP", "")
if refalign.ENABLED or _FLIP_H or _FLIP_V:
    # Registered against the shell, which is pinned and therefore already the object's own shape:
    # the silhouette of each cut and the rind inside it come from the released model, so nothing
    # the interior is supposed to learn is used to decide which way up its own supervision goes.
    with torch.no_grad():
        _shh = []
        for j in range(NH):
            _s, _, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[H_LO + j]), RES)
            _shh.append(_s.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        _shv = []
        for i in range(NV):
            _n3 = torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32, device=dev)
            _s, _, _, _ = ON.render_section(
                st, glctx, torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev),
                _n3, float(C["v_planes"][i, 3]), RES)
            _shv.append(_s.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        refs_h, _nh = refalign.orient_family(refs_h, _shh, "transverse", _FLIP_H)
        refs_v, _nv = refalign.orient_family(refs_v, _shv, "longitudinal", _FLIP_V)
    print("  reference orientation against the pinned shell:", flush=True)
    for _n in (_nh, _nv):
        if _n:
            print(_n, flush=True)
refs_e = {nm: cv2.imread(os.path.join(EXT, f"{nm}_ref.png"))[:, :, ::-1].astype(np.float32) / 255.
          for nm in enames}
print(f"  references: {len(refs_h)} transverse, {len(refs_v)} longitudinal, "
      f"{len(refs_e)} exterior", flush=True)

dump(OUT + "/eval_init")


def probe():
    """The same twelve held-out cuts, measured the same way every time."""
    with torch.no_grad():
        decode()
        tot, n = 0.0, 0
        for i in VAL_H:
            nn_ = torch.as_tensor(ehp[i, :3], dtype=torch.float32, device=dev)
            im, al, _, _ = ON.render_section(st, glctx, ehm[i], nn_, float(ehp[i, 3]), RES)
            tot += float((im - sm.section_target(im, refs_h[i % NH], alpha=al)).abs().mean()); n += 1
        for i in VAL_V:
            nn_ = torch.as_tensor(evp[i, :3], dtype=torch.float32, device=dev)
            im, al, _, _ = ON.render_section(st, glctx, evm[i], nn_, float(evp[i, 3]), RES)
            tot += float((im - sm.section_target(im, refs_v[i % NV], alpha=al)).abs().mean()); n += 1
    return tot / n


def apply_masks(section):
    """Ownership, enforced on the gradient rather than on the render, so the training image and
    the evaluation image are the same picture.

    The per-cell part is exact. The shared MLP is not, and cannot be: it serves every cell, so a
    section moving it moves the exterior's decoded colour too. That is equally true of the
    decoder this is ported from, where the interior head is shared across is_outer cells that the
    section render excludes -- the coupling that removes the speckle is the same coupling that
    makes a purely per-cell ownership impossible. SHELL_PIN's col_pin is what closes it, by
    overwriting after the head.
    """
    if section:
        if ANCHOR:
            # the triplane has no per-cell leaf to zero, so the same mask is applied inside its
            # feature read instead -- set once, below, rather than every step
            if not TRIPLANE and dec_i.feat.grad is not None:
                dec_i.feat.grad[is_outer] = 0
            for p in dec_s.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        else:
            if st["interior"].grad is not None:
                st["interior"].grad[is_outer] = 0
            if st["surf_rgb"].grad is not None:
                st["surf_rgb"].grad.zero_()
        for k in ("dual_v", "split_w"):
            if st[k].grad is not None:
                st[k].grad.zero_()
    else:
        if ANCHOR:
            for p in dec_i.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        elif st["interior"].grad is not None:
            st["interior"].grad.zero_()


TK = ["feat_i", "feat_s", "dual_v", "split_w"] if ANCHOR else \
     ["interior", "surf_rgb", "dual_v", "split_w"]


def rows(k):
    # the hybrid keeps a per-cell residual, which is a leaf and does carry a gradient, so the
    # coverage probe works there and reports exactly what it should: which cells the planes moved
    # away from the interpolated field. The pure triplane has no such tensor and no coverage.
    if k == "feat_i" and TRIPLANE:
        r = getattr(dec_i, "resid", None)
        # the pure triplane has no residual; fall back to the computed feature so the touch buffers
        # still have a length. It is not a leaf, so its grad stays None and coverage stays
        # unmeasurable there, which is the honest answer rather than a zero
        return r if r is not None else dec_i.feat
    return {"feat_i": dec_i.feat if ANCHOR else None,
            "feat_s": dec_s.feat if ANCHOR else None,
            "interior": st.get("interior"), "surf_rgb": st.get("surf_rgb"),
            "dual_v": st["dual_v"], "split_w": st["split_w"]}[k]


# One entry per longitudinal plane: its last target, and the geometry needed to project onto
# it. Under SEC_XCONS_HOLD the reconciled target is kept and reused rather than re-derived,
# because section_target rebuilds each target from the current render every pass and would
# rebuild the contradiction with it.
_vcache, _vheld = {}, {}
if SEC_XCONS > 0:
    import xcons

touch = {k: torch.zeros(len(rows(k)), dtype=torch.bool, device=dev) for k in TK}
_tvpolar = np.asarray(C["h_planes"][0, :3], float)
_tvpairs = fieldreg.face_pairs(st, dev, _tvpolar) if fieldreg.WEIGHT > 0 else None
upd_i = torch.zeros(len(seed_i), dtype=torch.bool, device=dev)
hist, l1hist, probes = [], [], [(0, probe())]
dhist, lamhist = [], []
print(f"  probe at 0: {probes[0][1]:.5f}", flush=True)
t0 = time.time()
steps = 0
def family_shares(nh, nv, ne):
    """Each family's total loss weight per outer iteration.

    Without SEC_FAM_W a family's share is its plane count, which is what the code has always done
    and what couples reach to weight. With it, the requested ratio is scaled so the three shares
    still sum to what they summed to -- otherwise asking for 9:17 on 14/12 planes multiplies the
    whole loss by 12.7 and changes the effective learning rate rather than the balance.
    """
    counts = [nh, nv, ne]
    if FAM_W == [1.0, 1.0, 1.0]:
        return counts
    want = [FAM_W[i] if counts[i] else 0.0 for i in range(3)]
    tot_w, tot_c = sum(want), sum(counts)
    if tot_w <= 0:
        return counts
    return [w * tot_c / tot_w for w in want]


def groups_for_iteration():
    """The planes of one outer iteration, grouped into what each gradient step sees.

    SEC_JOINT=1 (default) puts one transverse and one longitudinal plane in every step, so the two
    families are optimised together and a cell they share meets both constraints at once rather
    than alternately. Under the old rule the step order was a shuffle of all the planes and a step
    saw exactly one of them, so a cell in both families was pulled to one photograph and then to
    the other, and what it settled on depended on which came last.

    The families are different sizes, so the shorter one is cycled through fresh permutations and
    each family's loss is scaled by (its plane count / the number of groups). That keeps each
    family's total weight over an outer iteration exactly what it was; without it, cycling the
    shorter family would silently give it more say.

    SEC_JOINT=0 restores one plane per step, which is what every arm before this was trained
    under.
    """
    hs = [("h", i) for i in range(NH)]
    vs = [("v", i) for i in range(NV)]
    es = [("e", i) for i in range(len(enames))]
    for L in (hs, vs, es):
        random.shuffle(L)
    if not SEC_JOINT:
        allp = hs + vs + es
        random.shuffle(allp)
        return [[(k, i, 1.0)] for k, i in allp]
    G = max(len(hs), len(vs), len(es), 1)
    SH = family_shares(len(hs), len(vs), len(es))
    FAM_W_EFF[:] = SH
    out = []
    for g in range(G):
        grp = []
        for L, share in ((hs, SH[0]), (vs, SH[1]), (es, SH[2])):
            if not L:
                continue
            if g and g % len(L) == 0:
                random.shuffle(L)
            grp.append((*L[g % len(L)], share / G))
        out.append(grp)
    return out


_g0 = groups_for_iteration()
print(f"  a gradient step sees {'+'.join(sorted({k for g in _g0 for k, _, _ in g}))} "
      f"({len(_g0)} steps per outer iteration, {len(_g0[0])} planes each)"
      if SEC_JOINT else
      f"  a gradient step sees one plane ({len(_g0)} steps per outer iteration)", flush=True)
_vrad = float((st["solid"].max(0).values - st["solid"].min(0).values).max()) \
    * float(st["hc"]) / 2.0
# the object's axis is the transverse family's normal, and the centre is where the longitudinal
# planes all pass; both are fixed for the run
_axis = np.asarray(C["h_planes"][0, :3], float)
_cen = ((st["solid"].float().mean(0) + 0.5) * float(st["hc"])).cpu().numpy() \
    + np.asarray(st["org"])
_az_spacing = np.radians(180.0 / max(NV, 1))
if JITTER_AZ > 0:
    assert azjitter.check(C["v_mvp"][0], C["v_planes"][0, :3], float(C["v_planes"][0, 3]),
                          _axis, _cen), "the axis rotation does not return the identity"
    print(f"  longitudinal planes turned by up to {JITTER_AZ:.2f} of the "
          f"{np.degrees(_az_spacing):.1f}-degree spacing, about {np.round(_axis, 3)}", flush=True)
_crit = (critic.Trainer({"h": refs_h, "v": refs_v}, dev) if critic.WEIGHT > 0 else None)
_nstep = len(_g0)
TV_ANNEAL = float(os.environ.get("SEC_TV_ANNEAL", "1"))
TOTAL_STEPS = _nstep * ITERS
_urng = np.random.default_rng(0)
# Where a plane's jitter comes from. The schedule already moves each plane within its own slot --
# half a spacing either way -- but it moves it by an independent draw every time, and independent
# draws clump: measured on coverage alone, a low-discrepancy sequence reached 85.6% of the interior
# in 100 planes where independent draws reached 77.9%, and on a noise-start solve of the orange it
# was worth 2.0% of error and 27 points of the excess texture on planes nobody photographed.
#
# Each plane keeps its own counter, because two families interleaved on one counter each see a
# strided subsequence, and a strided subsequence of a low-discrepancy sequence is not one.
SEC_SCHED = os.environ.get("SEC_SCHED", "random")
_PHI = (1 + 5 ** 0.5) / 2
_sched_n = {}


def _seq(key, base=2):
    """The next value of this plane's own sequence, in [0, 1)."""
    k = _sched_n.get(key, 0) + 1
    _sched_n[key] = k
    if base == 0:                                  # golden angle, for an azimuth
        return (k * _PHI) % 1.0
    f, r = 1.0, 0.0
    while k:
        f /= base
        r += f * (k % base)
        k //= base
    return r


def _jit(key, base=2):
    """A jitter fraction in [-1, 1], drawn or walked depending on SEC_SCHED."""
    if SEC_SCHED != "cycle":
        return (random.random() - 0.5) * 2.0
    return (_seq(key, base) - 0.5) * 2.0


print(f"  plane jitter: {SEC_SCHED}", flush=True)

for j in range(ITERS):
    for grp in groups_for_iteration():
        decode()
        loss, _l1s, _kinds, _dls, _lams = None, [], [], [], []
        for kind, i, wfam in grp:
            if kind == "h":
                _f = _jit(("h", i)) * JITTER
                d = float(hd[H_LO + i]) + h_step * _f
                img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
                # the target follows the plane. Both families jitter, and picking the reference by
                # the integer index leaves a plane that has moved most of the way towards its
                # neighbour still supervised by its own photograph -- which is exactly the two
                # families ceasing to be treated alike, and it is worse for the longitudinal one
                # because its jitter spans the whole spacing.
                # REF_SAMPLE re-draws every step, so the reference cannot come from the list
                # built once at startup -- that would draw the whole run's targets in one go and
                # fix them, which is the opposite of the point
                ref = (refs_h[i] if not (REF_FOLLOW or refsel.SAMPLE) else refsel.as_array(
                    refsel.solved_photo(REF_H, i + (_q(_f) if REF_FOLLOW else 0), NH), RES))
            elif kind == "v":
                n = torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32, device=dev)
                # The same jitter the transverse family has always had, on the parameter that
                # plays the same role. A transverse plane is moved along its normal by up to half
                # a depth step, so over a run it sweeps the cells between the supervised depths;
                # a longitudinal plane was pinned, so the cells between the supervised azimuths
                # were reached by the transverse family alone.
                #
                # That is what the held-out longitudinal renders were showing. On the orange the
                # supervised planes come out as sections of an orange and a held-out plane five
                # degrees away comes out as vertical columns, from the same model and the same
                # renderer -- five degrees at sixty cells of radius is five cells, which is a
                # different set of cells. The supervised planes sit 3.1 to 3.4% of the radius off
                # the axis and the held-out ones 4.6 to 12.3%, so the band this sweeps is the band
                # they are drawn from.
                dv = float(C["v_planes"][i, 3])
                _vm = torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev)
                if JITTER_V > 0:
                    dv += _vrad * JITTER_V * (random.random() - 0.5) * 2.0
                _fv = 0.0
                if JITTER_AZ > 0:
                    # about the axis, which is the transverse family's normal: the plane stays
                    # through the axis, so it stays a central section and the photographs stay the
                    # right kind, while it sweeps the cells between the ten fixed azimuths
                    _fv = _jit(("v", i), base=0) * JITTER_AZ
                    _a = _fv * _az_spacing
                    _m2, _n2, dv = azjitter.turn(C["v_mvp"][i], C["v_planes"][i, :3], dv,
                                                 _axis, _cen, _a)
                    _vm = torch.as_tensor(_m2, dtype=torch.float32, device=dev)
                    n = torch.as_tensor(_n2, dtype=torch.float32, device=dev)
                img, al, _, _ = ON.render_section(st, glctx, _vm, n, dv, RES)
                ref = (refs_v[i] if not (REF_FOLLOW or refsel.SAMPLE) else refsel.as_array(
                    refsel.photo(REF_V, i + (_q(_fv) if REF_FOLLOW else 0), NV), RES))
            else:
                img, al, _, _ = ON.render_exterior(
                    st, glctx, torch.as_tensor(C["e_mvp"][i], dtype=torch.float32, device=dev), RES)
                ref = refs_e[enames[i]]
            with torch.no_grad():
                if kind == "v" and SEC_XCONS_HOLD and j > SEC_XCONS_AT and i in _vheld:
                    tgt = _vheld[i]
                else:
                    tgt = sm.section_target(img, ref, alpha=al)
                if SEC_XCONS > 0 and j >= SEC_XCONS_AT:
                    if kind == "v":
                        _vcache[i] = (tgt, torch.as_tensor(C["v_mvp"][i], dtype=torch.float32,
                                                           device=dev),
                                      torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32,
                                                      device=dev),
                                      float(C["v_planes"][i, 3]))
                    elif kind == "h" and _vcache:
                        # The transverse target wins along every line it shares with a cached
                        # longitudinal one; under "copy" reconcile writes into the longitudinal.
                        _tot, _dis = 0, 0.0
                        for _vi, (_vt, _vm, _vn, _vd) in _vcache.items():
                            _k, _e = xcons.reconcile(tgt, hmvp, hn, d, _vt, _vm, _vn, _vd,
                                                     centres, RES, band=float(h_step),
                                                     weight=SEC_XCONS, mode=SEC_XCONS_MODE)
                            _tot += _k; _dis += _e * _k
                            if SEC_XCONS_HOLD and _k:
                                _vheld[_vi] = _vt
                        if _tot and j % 10 == 0:
                            print(f"  cross-section consistency at {j}: {_tot:,} px reconciled, "
                                  f"disagreement {_dis / max(_tot, 1):.4f}", flush=True)
            _pl = (secloss.patch_loss(img, tgt) if secloss.SEC_PATCH > 0
                   else (img - tgt).abs().mean())
            # Asked of the render and the family, not of the render and this plane's own target: the
            # pixel term already says where the structure goes, and this says what it should be made
            # of. On the exterior views there is no family of sections to compare against.
            # The critic sees the render whether or not this plane has a photograph. On a
            # supervised plane it sits beside the pixel term; the point is that it is the only term
            # that can also speak on a plane nobody photographed, which is where the blocks are.
            if critic.WEIGHT > 0 and kind != "e" and _crit is not None:
                _adv, _dl = _crit.step(kind, img)
                _lam = _crit.adaptive(_pl, _adv, img) if float(_adv) != 0.0 else 0.0
                _pl = _pl + critic.WEIGHT * _lam * _adv
                _dls.append(_dl)
                _lams.append(_lam)
            if styleloss.WEIGHT > 0 and kind != "e":
                _pl = _pl + styleloss.WEIGHT * styleloss.penalty(
                    img, refs_h if kind == "h" else refs_v, kind)
            # The distributional term, in its second stage. `schedule` returns (0, 1) until
            # SEC_DIST_START of the run has passed and while SEC_DIST is off, so this is the
            # existing objective exactly unless it is asked for.
            _wd, _wp = patchdist.schedule(j, ITERS)
            if _wd > 0:
                # Where the term is applied, not only whether. The two families disagree only on
                # the cells they both cross, and two planes meet in a line: measured on the orange,
                # 1.0% of the cells either family touches are touched by both, and the gradient
                # cosine there is -0.4285 under the pixel loss and -0.1160 under Chamfer. Applying
                # it everywhere relaxes a conflict that exists on a hundredth of the object and
                # blinds the rest to how much structure it should have.
                _dm = None
                if SEC_DIST_BAND > 0:
                    _others = ([(C["v_planes"][q, :3], C["v_planes"][q, 3]) for q in range(NV)]
                               if kind == "h" else
                               [(C["h_planes"][H_LO + q, :3], C["h_planes"][H_LO + q, 3])
                                for q in range(NH)])
                    _pl_now = (hn.cpu().numpy(), d) if kind == "h" else \
                        (C["v_planes"][i, :3], float(C["v_planes"][i, 3]))
                    _mvp_now = hmvp if kind == "h" else \
                        torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev)
                    _b = overlap.band(_mvp_now.cpu().numpy(), _pl_now[0], _pl_now[1], _others,
                                      RES, half_px=SEC_DIST_BAND, span=_vrad)
                    _fg = ((tgt.min(0).values < 0.98) | (img.min(0).values < 0.98))
                    _dm = torch.as_tensor(_b, device=dev) & _fg
                    if int(_dm.sum()) < 400:
                        _dm = None
                if _dm is not None or SEC_DIST_BAND <= 0:
                    _pl = _wp * _pl + _wd * patchdist.distance(
                        img, tgt, mask=(None if _dm is None else _dm.float()))
            loss = _pl * wfam if loss is None else loss + _pl * wfam
            with torch.no_grad():
                _l1s.append(float((img - tgt).abs().mean()))
            _kinds.append(kind)

        # The critic on a plane nobody photographed: one extra render, scored against the family
        # with no target image anywhere. The critic is trained on it as a fake too -- otherwise it
        # only ever learns to separate photographs from SUPERVISED renders, and the unsupervised
        # ones, which are the blocky ones, stay outside anything it has an opinion about.
        if critic.WEIGHT > 0 and critic.UNSUP > 0 and _crit is not None:
            _uk2 = "h" if (steps % 2 == 0) else "v"
            try:
                _n2, _d2, _m2 = unsup.sample_plane(_uk2, C, H_LO, NH, NV, _urng,
                                                   axis=_axis, centre=_cen,
                                                   az_spacing=_az_spacing)
                _i2, _, _, _ = ON.render_section(
                    st, glctx, torch.as_tensor(_m2, dtype=torch.float32, device=dev),
                    torch.as_tensor(_n2, dtype=torch.float32, device=dev), float(_d2), RES)
                _a2, _dl2 = _crit.step(_uk2, _i2)
                # the same balance the supervised planes were given this step, since there is no
                # pixel term here to take a ratio against
                _lam2 = float(np.mean(_lams)) if _lams else 1.0
                loss = loss + critic.WEIGHT * critic.UNSUP * _lam2 * _a2
                _dls.append(_dl2)
            except RuntimeError:
                pass

        # The prior on a plane nobody photographed. One extra render per step, at a position
        # drawn fresh each time, scored against the family's pooled patches and against no target
        # image at all -- see unsup.py. It is here rather than inside the plane loop because it is
        # not one of the supervised planes and must not be weighted as if it were.
        if unsup.WEIGHT > 0 and (steps % unsup.EVERY) == 0:
            _uk = "h" if (steps // max(unsup.EVERY, 1)) % 2 == 0 else "v"
            _un, _ud, _um = unsup.sample_plane(_uk, C, H_LO, NH, NV, _urng,
                                               axis=_axis, centre=_cen, az_spacing=_az_spacing)
            _ut = torch.as_tensor(_un, dtype=torch.float32, device=dev)
            _umt = torch.as_tensor(_um, dtype=torch.float32, device=dev)
            try:
                _ui, _, _, _ = ON.render_section(st, glctx, _umt, _ut, float(_ud), RES)
                loss = loss + unsup.WEIGHT * unsup.penalty(
                    _ui, refs_h if _uk == "h" else refs_v, _uk)
            except RuntimeError:
                pass

        # The spatial prior, on the whole field rather than on what these planes happened to
        # cross: a cell is coupled to its neighbours whether or not either was supervised this
        # step, which is the point -- the cells the schedule reaches rarely are the ones with
        # nothing else holding them. Once per step, not once per plane, because it does not
        # depend on which plane was drawn.
        if fieldreg.WEIGHT > 0:
            # Coarse to fine, expressed on the prior rather than on the lattice. The lattice cannot
            # be refined -- it is the object's own occupancy -- so the frequency the field is
            # allowed to carry is controlled by how hard the neighbours are tied together: strong
            # early, so the run settles the low frequencies first, and released later so detail can
            # appear. TV_ANNEAL is the multiplier on the weight at the start, reaching 1 at the end.
            _tvw = fieldreg.WEIGHT
            if TV_ANNEAL > 1.0:
                _f = steps / max(TOTAL_STEPS, 1)
                _tvw = fieldreg.WEIGHT * (TV_ANNEAL ** (1.0 - min(_f, 1.0)))
            loss = loss + _tvw * fieldreg.penalty(st["interior"], _tvpairs, polar=_tvpolar)
        l1_now = float(np.mean(_l1s))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for k in TK:
            g = rows(k).grad
            if g is not None:
                touch[k] |= (g.abs().sum(-1) > 0)
        # the masks are the sections' unless every plane in this step was an exterior view
        apply_masks(any(k != "e" for k in _kinds))
        if ANCHOR and not TRIPLANE and dec_i.feat.grad is not None:
            # gaussians.trained: what the sections actually supervised, accumulated
            upd_i.__ior__(dec_i.feat.grad.abs().sum(-1) > 0)
        set_lr(steps / max(TOTAL_STEPS, 1))
        opt.step()
        if _crit is not None:
            _crit.flush()
        if not ANCHOR:
            with torch.no_grad():
                st["interior"].clamp_(0, 1)
                if not SHELL_PIN:
                    st["surf_rgb"].clamp_(0, 1)
        if not SHELL_PIN:
            with torch.no_grad():
                st["split_w"].clamp_(1e-3, 1 - 1e-3)
        hist.append(float(loss)); l1hist.append(l1_now); steps += 1
        if _dls:
            dhist.append(float(np.nanmean(_dls)))
            lamhist.append(float(np.mean(_lams)) if _lams else 0.0)
    if ANCHOR and VOXEL_SMOOTH and not TRIPLANE and j > 0 and j % ABL_INTERVAL == 0:
        nfill = anchor.voxel_smooth_anchors(dec_i, centres, upd_i | is_outer, ABL_GRID)
        print(f"  voxel smoothing at outer {j}: {int(upd_i.sum()):,}/{len(centres):,} trained, "
              f"{nfill:,} untrained cells' features filled from trained neighbours", flush=True)
    if (j + 1) % PROBE_EVERY == 0 or j == 0:
        probes.append((j + 1, probe()))
        _dtxt = ""
        if _crit is not None and dhist:
            _dtxt = (f"  D {np.mean(dhist[-_nstep:]):.4f}"
                     f" lam {np.mean(lamhist[-_nstep:]):.3g}")
        print(f"  outer {j+1:4d}/{ITERS}  steps {steps:,}  loss {np.mean(hist[-_nstep:]):.5f}  "
              f"L1 {np.mean(l1hist[-_nstep:]):.5f}  "
              f"probe {probes[-1][1]:.5f}{_dtxt}  {time.time()-t0:.0f}s", flush=True)

el = time.time() - t0
print(f"\ntrained {ITERS} outer iterations = {steps:,} gradient steps in {el:.1f}s "
      f"({1000*el/max(steps,1):.0f} ms/step)")
print(f"  loss first {_nstep} mean {np.mean(hist[:_nstep]):.5f} -> "
      f"last {_nstep} mean {np.mean(hist[-_nstep:]):.5f}\n"
      f"  L1   first {_nstep} mean {np.mean(l1hist[:_nstep]):.5f} -> "
      f"last {_nstep} mean {np.mean(l1hist[-_nstep:]):.5f}")
print("coverage over the schedule actually run (with jitter):")
for k in TK:
    print(f"  {k:<10} {int(touch[k].sum()):>9,} / {len(touch[k]):>9,}  "
          f"({100*float(touch[k].float().mean()):5.1f}%)")

with torch.no_grad():
    decode()
    for nm, seed, cur in (("interior", seed_i, st["interior"]), ("surf_rgb", seed_s, st["surf_rgb"])):
        d = (cur - seed).norm(dim=-1)
        # the speckle, measured rather than looked at: how much a cell differs from the cells
        # next to it in the same coarse row
        print(f"  {nm:<10} moved mean {float(d.mean()):.4f} max {float(d.max()):.4f}; "
              f"decoded spread {cur.std(0).cpu().numpy().round(4)}")

dump(OUT + "/eval_final")
save = {"dual_v": st["dual_v"].detach().cpu(), "split_w": st["split_w"].detach().cpu()}
if ANCHOR:
    save["dec_i"] = {k: v.detach().cpu() for k, v in dec_i.state_dict().items()}
    save["dec_s"] = {k: v.detach().cpu() for k, v in dec_s.state_dict().items()}
else:
    save["interior"] = st["interior"].detach().cpu()
    save["surf_rgb"] = st["surf_rgb"].detach().cpu()
torch.save(save, OUT + "/params.pt")
json.dump(dict(route=ROUTE, anchor=int(ANCHOR), shell_pin=int(SHELL_PIN),
               ext_views=len(enames), iters=ITERS, steps=steps, loss=hist, probe=probes,
               coverage={k: float(touch[k].float().mean()) for k in TK}),
          open(OUT + "/hist.json", "w"))
np.savez(OUT + "/touch.npz", **{k: touch[k].cpu().numpy() for k in TK},
         is_outer=is_outer.cpu().numpy())
print("probe curve: " + "  ".join(f"{i}:{v:.5f}" for i, v in probes))
print("TRAIN_OK")
