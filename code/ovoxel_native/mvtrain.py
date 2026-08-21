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
import fieldreg
import patchdist
import secloss

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
opt = torch.optim.Adam(groups)
print(f"  trainable: {n_train:,} floats "
      f"({'interior only, the exterior is pinned' if SHELL_PIN else 'interior and exterior'})")

feat_params = ([dec_i.feat] + ([dec_s.feat] if not SHELL_PIN else [])) if ANCHOR else []


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


def dump(folder):
    with torch.no_grad():
        decode()
        for i in range(len(ehp)):
            n = torch.as_tensor(ehp[i, :3], dtype=torch.float32, device=dev)
            img, _, _, _ = ON.render_section(st, glctx, ehm[i], n, float(ehp[i, 3]), RES)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            cv2.imwrite(f"{folder}/rh{i}_init_0.png", (a[:, :, ::-1] * 255).astype(np.uint8))
        for i in range(len(evp)):
            n = torch.as_tensor(evp[i, :3], dtype=torch.float32, device=dev)
            img, _, _, _ = ON.render_section(st, glctx, evm[i], n, float(evp[i, 3]), RES)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            cv2.imwrite(f"{folder}/rv{i}_init_0.png", (a[:, :, ::-1] * 255).astype(np.uint8))
    print(f"  -> {folder}", flush=True)


refs_h = [refsel.as_array(refsel.solved_photo(REF_H, j, NH), RES) for j in range(NH)]
refs_v = [refsel.as_array(refsel.photo(REF_V, i, NV), RES) for i in range(NV)]
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
        for i in range(len(ehp)):
            nn_ = torch.as_tensor(ehp[i, :3], dtype=torch.float32, device=dev)
            im, al, _, _ = ON.render_section(st, glctx, ehm[i], nn_, float(ehp[i, 3]), RES)
            tot += float((im - sm.section_target(im, refs_h[i % NH], alpha=al)).abs().mean()); n += 1
        for i in range(len(evp)):
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
            if dec_i.feat.grad is not None:
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
_tvpairs = fieldreg.face_pairs(st, dev) if fieldreg.WEIGHT > 0 else None
upd_i = torch.zeros(len(seed_i), dtype=torch.bool, device=dev)
hist, l1hist, probes = [], [], [(0, probe())]
print(f"  probe at 0: {probes[0][1]:.5f}", flush=True)
t0 = time.time()
steps = 0
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
    out = []
    for g in range(G):
        grp = []
        for L in (hs, vs, es):
            if not L:
                continue
            if g and g % len(L) == 0:
                random.shuffle(L)
            grp.append((*L[g % len(L)], len(L) / G))
        out.append(grp)
    return out


_g0 = groups_for_iteration()
print(f"  a gradient step sees {'+'.join(sorted({k for g in _g0 for k, _, _ in g}))} "
      f"({len(_g0)} steps per outer iteration, {len(_g0[0])} planes each)"
      if SEC_JOINT else
      f"  a gradient step sees one plane ({len(_g0)} steps per outer iteration)", flush=True)
_nstep = len(_g0)
for j in range(ITERS):
    for grp in groups_for_iteration():
        decode()
        loss, _l1s, _kinds = None, [], []
        for kind, i, wfam in grp:
            if kind == "h":
                d = float(hd[H_LO + i]) + h_step * ((random.random() - 0.5) * 2.0 * JITTER)
                img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
                ref = refs_h[i]
            elif kind == "v":
                n = torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32, device=dev)
                img, al, _, _ = ON.render_section(
                    st, glctx, torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev),
                    n, float(C["v_planes"][i, 3]), RES)
                ref = refs_v[i]
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
            # The distributional term, in its second stage. `schedule` returns (0, 1) until
            # SEC_DIST_START of the run has passed and while SEC_DIST is off, so this is the
            # existing objective exactly unless it is asked for.
            _wd, _wp = patchdist.schedule(j, ITERS)
            if _wd > 0:
                _pl = _wp * _pl + _wd * patchdist.distance(img, tgt)
            loss = _pl * wfam if loss is None else loss + _pl * wfam
            with torch.no_grad():
                _l1s.append(float((img - tgt).abs().mean()))
            _kinds.append(kind)

        # The spatial prior, on the whole field rather than on what these planes happened to
        # cross: a cell is coupled to its neighbours whether or not either was supervised this
        # step, which is the point -- the cells the schedule reaches rarely are the ones with
        # nothing else holding them. Once per step, not once per plane, because it does not
        # depend on which plane was drawn.
        if fieldreg.WEIGHT > 0:
            loss = loss + fieldreg.WEIGHT * fieldreg.penalty(st["interior"], _tvpairs)
        l1_now = float(np.mean(_l1s))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for k in TK:
            g = rows(k).grad
            if g is not None:
                touch[k] |= (g.abs().sum(-1) > 0)
        # the masks are the sections' unless every plane in this step was an exterior view
        apply_masks(any(k != "e" for k in _kinds))
        if ANCHOR and dec_i.feat.grad is not None:
            # gaussians.trained: what the sections actually supervised, accumulated
            upd_i.__ior__(dec_i.feat.grad.abs().sum(-1) > 0)
        opt.step()
        if not ANCHOR:
            with torch.no_grad():
                st["interior"].clamp_(0, 1)
                if not SHELL_PIN:
                    st["surf_rgb"].clamp_(0, 1)
        if not SHELL_PIN:
            with torch.no_grad():
                st["split_w"].clamp_(1e-3, 1 - 1e-3)
        hist.append(float(loss)); l1hist.append(l1_now); steps += 1
    if ANCHOR and VOXEL_SMOOTH and j > 0 and j % ABL_INTERVAL == 0:
        nfill = anchor.voxel_smooth_anchors(dec_i, centres, upd_i | is_outer, ABL_GRID)
        print(f"  voxel smoothing at outer {j}: {int(upd_i.sum()):,}/{len(centres):,} trained, "
              f"{nfill:,} untrained cells' features filled from trained neighbours", flush=True)
    if (j + 1) % PROBE_EVERY == 0 or j == 0:
        probes.append((j + 1, probe()))
        print(f"  outer {j+1:4d}/{ITERS}  steps {steps:,}  loss {np.mean(hist[-_nstep:]):.5f}  "
              f"L1 {np.mean(l1hist[-_nstep:]):.5f}  "
              f"probe {probes[-1][1]:.5f}  {time.time()-t0:.0f}s", flush=True)

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
