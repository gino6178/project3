import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import random
import torchvision.transforms as T
# Gaussian splatting dependencies
from scene.gaussian_model import GaussianModel
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from mpm_solver_warp.engine_utils import *
from pytorch_msssim import ssim

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
import torchvision.transforms as transforms
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from cross_section import *
from sds_demo import *
# `import *` skips underscore names, and the exterior branch needs the section branch's own
# disc fit -- the reference has to be put on the render's silhouette before they are compared.
from sds_demo import _fit_disc
from section_match import section_target
import section_consistency

import os as _os
import contextlib as _cl
def _nullctx():
    return _cl.nullcontext()
ABL_GRID      = int(_os.environ.get("ABL_GRID", "16"))        # item2: 512 = released
ABL_TRAINED   = _os.environ.get("ABL_TRAINED", "1") == "1"    # item3+4
ABL_IDW       = _os.environ.get("ABL_IDW", "1") == "1"        # item5
ABL_INTERVAL  = int(_os.environ.get("ABL_INTERVAL", "30"))    # item6: 101 = released
ABL_ITERS     = int(_os.environ.get("ABL_ITERS", "3000"))
ABL_RES       = int(_os.environ.get("ABL_RES", "512"))
LATTICE_PURE  = _os.environ.get("LATTICE_PURE", "0") == "1"
HF_W          = float(_os.environ.get("HF_W", "0"))
JITTER        = float(_os.environ.get("JITTER", "0"))
ALLVOXEL      = _os.environ.get("ALLVOXEL", "0") == "1"
# Prompt for the whole object; empty keeps the released behaviour of preserving the
# input model's exterior instead of supervising it.
EXT_PROMPT    = _os.environ.get("EXT_PROMPT", "")
# The cross-section branch takes forty-eight planes times thirty steps per outer iteration
# against the exterior's ten, so through the shared decoder the colour head is pulled toward
# flesh by two orders of magnitude more gradient. Invisible on an orange, where peel and
# flesh agree; on a watermelon the shell trained away from its own green reference toward
# brick red for the whole run. These two put the exterior on comparable footing.
# Two GPUs on one run rather than two runs on one GPU each. The loop supervises 10
# vertical planes, 16 horizontal ones and EXT_VIEWS exterior views per outer iteration, and
# every one of them is an independent render and backward against a fixed reference -- so a
# rank can take half of each and the gradients can be averaged, which is ordinary data
# parallelism with the planes as the batch. All three counts are even, so both ranks call
# backward the same number of times and the collective never deadlocks.
# Fix the seed. The decoder's features start as torch.randn and nothing was seeding it, so
# two runs of the same code on the same input diverged from the first step: the doughnut's
# longitudinal leakage came out 0.78% one run and 0.43% the next, the orange's 1.11% and
# 2.04%, with nothing changed between them. Differences that size were being read as the
# effect of changes that were in fact within this spread. Both ranks seed identically, which
# they must -- they hold their own copy of the decoder and average gradients by hand.
SEED          = int(_os.environ.get("SEED", "1234"))
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)

DDP_WORLD     = int(_os.environ.get("WORLD_SIZE", "1"))
DDP_RANK      = int(_os.environ.get("RANK", "0"))
# How many vertical planes the loop supervises. This was the literal 10 in three places --
# the loop bound, its progress line, and the set_plane call that tells _photo which of the
# family's photographs this plane gets. _photo maps plane to file as idx * len(files) // n,
# so the moment those three stopped agreeing every vertical plane would quietly draw a
# different photograph than the one it was supposed to, with nothing to show for it. The
# horizontal branch already derives its count from the planes themselves; this is the same
# thing named once. Keep it even, or the two ranks call backward a different number of
# times and the collective deadlocks.
N_VPLANES     = int(_os.environ.get("N_VPLANES", "10"))
# How the horizontal planes are sampled. The released trainer walks 70 steps along the camera
# direction and supervises centers[10:60], fifty planes; ours walked 24 and took centers[4:20],
# sixteen. The depth range covered is nearly the same -- 0.143 to 0.857 of it against 0.167 to
# 0.833 -- so the difference is density, not coverage, and density is the part that decides how
# much of the interior is ever looked at. Named rather than written into the loop so the two
# can be compared. Keep H_HI - H_LO even for the same DDP reason as N_VPLANES.
H_STEPS       = int(_os.environ.get("H_STEPS", "24"))
H_LO          = int(_os.environ.get("H_LO", "4"))
H_HI          = int(_os.environ.get("H_HI", "20"))
# How often the cross-section reference is refitted to the current render. It was a
# hardcoded 30, which for a 30-iteration run means once, at iteration 0 -- while JITTER moves
# each slab by up to half a step every iteration and the slab's disc changes size with it. On
# a sphere the outermost supervised planes sit at depth 0.652R, where dr/r = 0.099 per unit
# of jitter, so the target is fitted to a disc about 5% off the one being rendered for the
# rest of the run. Under REF_PHOTO refitting is a resize and a paste with no diffusion, so
# doing it every iteration removes the mismatch for no real cost.
REF_INTERVAL  = int(_os.environ.get("REF_INTERVAL", "30"))
# Once refinement takes over, the target is a diffusion sample rather than a paste, so the
# released method's interval applies again: regenerate every 30 and read the saved file in
# between. Refitting a photograph every iteration is free; sampling 26 views every iteration
# is not, and the released code is evidence that 30 is enough for a target that only moves as
# fast as the model does.
REF_REFINE_INTERVAL = int(_os.environ.get("REF_REFINE_INTERVAL", "30"))


# How many SDS epochs a target gets. The released schedule is 30 - j//100, written for a
# 3000-iteration run: it goes 30 to 0 across that, and across 150 it goes 30 to 28, so the
# grinding never lets up and the anneal that gives the released loop its terminus is inert.
# SDS_ANNEAL=1 spends the same 30-to-0 range over the run actually being done, measured from
# the warmup handover, so the target fades to the render the way it was meant to.
SDS_ANNEAL = _os.environ.get("SDS_ANNEAL", "0") == "1"


def sds_epochs(j):
    """How many SDS steps a target gets.

    The paper, section 4.1: "For each reference view, we initially apply 20 SDS optimization
    steps for generation, followed by 3-4 refinement steps per iteration", over a run of
    "between 120 and 200 iterations".

    The released code instead computes 30 - j//100, which over its own 3000 iterations runs 30
    down to 0 and over 150 runs 30 down to 28. So at the paper's own run length the released
    schedule applies thirty steps every time -- seven to ten times the refinement the paper
    describes -- and that is enough to grind the structure out: measured here, the visible
    angular structure fell from 0.0161 at the handover to 0.00045 by iteration 149.

    SDS_SCHEDULE=paper follows the paper: the first regeneration after the handover generates,
    every one after it refines.
    """
    mode = _os.environ.get("SDS_SCHEDULE", "released")
    if mode == "paper":
        warm = int(_os.environ.get("REF_WARMUP", "0"))
        first = int(_os.environ.get("SDS_GEN_STEPS", "20"))
        rest = int(_os.environ.get("SDS_REFINE_STEPS", "4"))
        return first if j <= warm else rest
    if not SDS_ANNEAL:
        return 30 - j // 100
    warm = int(_os.environ.get("REF_WARMUP", "0"))
    t = min(max((j - warm) / max(ABL_ITERS - warm, 1), 0.0), 1.0)
    return max(int(round(30 * 0.5 * (1.0 + math.cos(math.pi * t)))), 1)


def ref_interval(j):
    return REF_REFINE_INTERVAL if past_warmup() else REF_INTERVAL


def rebuild_ref(j):
    """Build a new target this iteration, or reuse the one already on disk?

    The two regimes are not the same question. Before the first regeneration the target is the
    photograph refitted to the current render -- a resize and a paste, no diffusion -- so it is
    rebuilt often and tracks the render as it grows. After a regeneration the target is a
    generated image that took a sampler to make, and it has to stand until the criterion says
    the fit has stopped paying, or it is not a target at all.

    Conflating them cost a run: with the rebuild interval left at 1, the iteration after a
    regeneration overwrote the generated target with the photograph again, so the regenerated
    reference survived exactly one iteration out of two hundred. The archive is what showed it
    -- 2.41% dark on every iteration except two, where it was 4.20% -- which is the reason the
    archive is on.
    """
    import sds_demo as _sd
    if float(_os.environ.get("REF_CONV_TAU", "0")) > 0:
        if _sd._CONV.get("fired_any"):
            return past_warmup()
        return j % REF_INTERVAL == 0
    return j % ref_interval(j) == 0


# What to hand the sampler as the depth condition for a cross-section.
#
# The released code runs a monocular estimator over the rendered slab. That estimator has never
# seen a cut plane; handed a picture of an orange slice it answers with its prior, which is a
# rounded orange. Measured on one of our sections: over the face the estimate spans 305 to 761
# with the centre 240 nearer than the rim, a relative variation of 15.6% -- a dome, for a
# surface that is flat by construction. So the sampler is being told "draw a sphere whose skin
# looks like this" when the truth is "draw a plane whose pattern is this".
#
# The lattice knows the real answer and the rasteriser already returns it: a thin slab seen
# head-on has constant depth over its face and nothing behind it, which is exactly the alpha
# channel. SD2-depth wants near-high, so the face is 1 and the background 0.
#
# This is the same correction stage 2 of the pipeline already applies to the exterior
# references -- conditioning them on the rasteriser's depth buffer rather than an estimate --
# which was never carried across to the section branch because that branch came over from the
# released trainer unchanged.
#
# Worth being clear about what this does not fix: it is not why the closed loop diverges.
# Iterating the generator on its own output twelve times blows the speckle up by a factor of
# nine under the estimate and by 9.02 under the true plane, so the dome is a wrong condition
# and a separate problem from the drift.
# Default is "estimate", which is what the three finished objects were trained with. The
# plane condition is the correct one -- the estimator hands the sampler a dome for a surface
# that is flat by construction -- but correct is not the same as what those runs did, and
# changing a default silently rewrites what "reproduce the baseline" means. Opt in.
SECTION_DEPTH = _os.environ.get("SECTION_DEPTH", "estimate")


def section_depth(rendering, alpha, pipe, size=(512, 512)):
    if SECTION_DEPTH == "estimate":
        d = F.interpolate(rendering.unsqueeze(0), size=(384, 384), mode="bilinear",
                          align_corners=False)
        d = pipe.depth_estimator(d.to("cuda:0")).predicted_depth
        return F.interpolate(d.unsqueeze(0), size=size, mode="bilinear",
                             align_corners=False).squeeze(0)
    a = alpha.reshape(1, 1, *alpha.shape[-2:]).float()
    return F.interpolate(a, size=size, mode="bilinear", align_corners=False).squeeze(0)


# Keep every target that gets generated, next to the render it was generated from. Without
# this both are overwritten in place -- one file per view, rewritten each time -- so by the
# end of a run there is no record of what the model was ever asked to become, only what it
# ended up as. That is exactly the wrong half to keep when the question is whether a target
# was reasonable: the released target is a blurred blob at the start of its anneal and the
# render itself is fine, and telling those apart afterwards is impossible from the render
# alone. Pairs, because a target only means something against its own input.
REF_LOG = int(_os.environ.get("REF_LOG", "30"))


def log_ref(out, tag, j, src, ref, what=None):
    """Archive one target and its source render. Call only where the target was regenerated."""
    if REF_LOG <= 0 or j % REF_LOG:
        return
    d = _os.path.join(out, "reflog", f"iter_{j:05d}")
    _os.makedirs(d, exist_ok=True)
    src.save(_os.path.join(d, f"{tag}_src.png"))
    ref.save(_os.path.join(d, f"{tag}_ref.png"))
    # And what it is, in numbers, on the log's own line. Every reference failure in this
    # project was diagnosed after the fact by opening the images and measuring them by hand --
    # a reference regenerated as an orange, a reference blended until its detail was below the
    # photograph's, a reference drifting darker every round. All three are one line each here,
    # so the run says it went wrong while it is going wrong.
    try:
        import numpy as _np, cv2 as _cv2
        _a = _np.asarray(ref.convert("RGB")).astype(_np.float32) / 255.0
        _fg = _np.abs(_a - 1).max(2) > 0.06
        if _fg.sum() > 500:
            _lum = _a.mean(2)
            _det = float(_np.abs(_cv2.Laplacian(_lum, _cv2.CV_32F))[_fg].mean()) * 1000
            _dark = 100.0 * float((_fg & (_lum < 0.22)).sum()) / float(_fg.sum())
            _rgb = _a[_fg].mean(0)
            _line = (f"  reflog j={j} {tag}: RGB "
                     f"({_rgb[0]:.3f},{_rgb[1]:.3f},{_rgb[2]:.3f})  detail {_det:.2f}e-3  "
                     f"dark {_dark:.2f}%  {what or ''}")
            print(_line, flush=True)
            with open(_os.path.join(out, "reflog", "stats.csv"), "a") as _f:
                _f.write(f"{j},{tag},{_rgb[0]:.4f},{_rgb[1]:.4f},{_rgb[2]:.4f},"
                         f"{_det:.3f},{_dark:.3f}\n")
    except Exception as _e:                       # logging must never take a run down
        print(f"  reflog j={j} {tag}: stats failed ({_e})", flush=True)
    # What produced it, so the archive says what was done rather than needing the schedule
    # reconstructed from the launch line afterwards. Name the phase, not just the number:
    # sampling_strength() clamps its cosine below the warmup and so reports the handover
    # strength for iterations where no img2img ran at all.
    # One file per tag, not one per iteration. A single strength.txt was overwritten by
    # whichever branch logged last, so an exterior target regenerated by img2img at 0.35 was
    # filed under "photograph" because a section had written after it.
    with open(_os.path.join(d, f"{tag}_how.txt"), "w") as f:
        if what is not None:
            _what = what
        elif not past_warmup():
            _what = "photograph"
        elif _os.environ.get("REF_SAMPLING", "1") == "1":
            _what = f"img2img {sampling_strength():.4f}"
        else:
            _what = "sds"
        f.write(_what + "\n")


# Keep a checkpoint and the section renders every N iterations, so progress can be measured
# rather than assumed -- a single end-state render cannot show whether it was still improving.
SNAP_INTERVAL = int(_os.environ.get("SNAP_INTERVAL", "0"))
# The six cube directions, in the order make_cube_refs writes them, so view ttt is compared
# against the reference actually taken from that direction. Without this the exterior branch
# sweeps its own azimuths and elevations, and each view is matched to a picture of a
# different side of the object.
EXT_CUBE      = ([(0, 90), (0, -90), (0, 0), (90, 0), (180, 0), (270, 0)]
                 if _os.environ.get("EXT_CUBE", "0") == "1" else None)
# Weight on axial material coherence. Nothing else in the loop asks the supervised planes to
# describe the same object, and without it the interior's layer-to-layer agreement falls from
# +0.889 to +0.468 over a run.
# Only the horizontal sections: rotating one about the view axis gives another valid picture
# of the same fruit, which is exactly the free parameter to drop. A vertical section's
# orientation is tied to the axis and is not free in the same way.
# Match the reference to the section actually rendered, component by component, instead of
# rescaling it to the bounding disc of every foreground pixel at once. See section_match.py.
SECTION_MATCH = _os.environ.get("SECTION_MATCH", "0") == "1"
PHASE_ALIGN   = _os.environ.get("PHASE_ALIGN", "0") == "1"
AXIAL_W       = float(_os.environ.get("AXIAL_W", "0"))
# The released trainer's spatial regulariser, ported to anchor features. Off by default.
SMOOTH_W      = float(_os.environ.get("SMOOTH_W", "0"))
SMOOTH_INTERVAL = int(_os.environ.get("SMOOTH_INTERVAL", "101"))
# The paper's voxel smoothing, on the anchors. Default on: it is one of the three things
# section 3.3 asks for, and under ANCHOR nothing else performs it.
VOXEL_SMOOTH  = _os.environ.get("VOXEL_SMOOTH", "1") == "1"
SEC_ROLL      = _os.environ.get("SEC_ROLL", "0") == "1"
# Patch supervision for the sections. SEC_PATCH is the crop side in pixels (0 is
# off, whole-frame as before), SEC_PATCH_N how many crops per plane per iteration,
# SEC_PATCH_STAT the weight of the per-crop band term.
SEC_PATCH     = int(_os.environ.get("SEC_PATCH", "0"))
SEC_PATCH_N   = int(_os.environ.get("SEC_PATCH_N", "4"))
SEC_PATCH_STAT= float(_os.environ.get("SEC_PATCH_STAT", "0"))
# Auxiliary material task. PHYS_TARGET names a per-anchor relative-stiffness field in [0, 1]
# -- the unsupervised decomposition, saved by report/material_segment.py -- and PHYS_W is what
# the head's agreement with it is worth against the section loss. PHYS_TV smooths the
# prediction over lattice neighbours except where the appearance itself has an edge, so a
# boundary is allowed exactly where the object has one.
PHYS_TARGET   = _os.environ.get("PHYS_TARGET", "")
PHYS_W        = float(_os.environ.get("PHYS_W", "0.05"))
PHYS_TV       = float(_os.environ.get("PHYS_TV", "0.01"))
# How far to pull the two families of section targets toward each other where
# their planes meet. 0 is off; 1 replaces both with their mean along the line.
SEC_XCONS     = float(_os.environ.get("SEC_XCONS", "0"))
# When to start reconciling, and whether to keep the result. Applied from iteration zero the
# reconciliation achieves nothing cumulative: `section_target` re-derives each target from the
# current render every iteration, so the contradiction it removed is rebuilt before the next
# pass and the measured disagreement oscillates instead of falling. Waiting until the render
# has settled, and then reusing the reconciled targets rather than re-deriving them, lets the
# corrections compound -- the model is then trained against a target set that agrees with
# itself, which is the state the whole idea is aiming at.
SEC_XCONS_AT  = int(_os.environ.get("SEC_XCONS_AT", "0"))
SEC_XCONS_HOLD = _os.environ.get("SEC_XCONS_HOLD", "0") == "1"
# How the two families are made to agree. "mean" splits the difference and blurs whatever they
# disagree about, which is precisely the discrete features -- a seed averaged with no seed is a
# smudge. "copy" lets the transverse family win and the longitudinal adopt it, so the structure
# survives with a definite position. See section_consistency.reconcile.
SEC_XCONS_MODE = _os.environ.get("SEC_XCONS_MODE", "mean")
EXT_HF        = _os.environ.get("EXT_HF", "0") == "1"
EXT_COL_W     = float(_os.environ.get("EXT_COL_W", "1.0"))
EXT_VIEWS     = int(_os.environ.get("EXT_VIEWS", "10"))
# Regenerate the exterior references from the model's own render every this many iterations.
# 0 keeps the single generation the branch used to do. Iterating is what makes the exterior
# behave like the sections: the target follows the model instead of being a fixed picture at
# a framing the render never matches.
EXT_REF_INTERVAL = int(_os.environ.get("EXT_REF_INTERVAL", "0"))
# Rescale a pre-made exterior reference onto the render's silhouette. Default on: without it
# the loss compares two differently framed objects and the chroma term averages the render
# over the reference's background.
EXT_FIT_DISC  = _os.environ.get("EXT_FIT_DISC", "1") == "1"
# img2img strength for a regenerated exterior reference. Low, because the render already is
# the object: at 0.95 from a flat grey shell the sampler returned a shaded ball carrying a
# small picture of an orange, gradient 0.034 against the six faces' 0.141.
EXT_REFINE_S = float(_os.environ.get("EXT_REFINE_S", "0.35"))
# Exponent on the cosine between the surface normal and the view, as a per-pixel weight on
# the exterior loss. 0 is off, which is every run before this one; higher makes a view
# speak only for what it faces squarely.
EXT_FACING = float(_os.environ.get("EXT_FACING", "0"))
# How much a named face's view weighs against a filler direction's, in the exterior loss.
# 1.0 treats them alike, which is what lets six plain-peel views outvote the one that
# shows the calyx.
EXT_NAMED_LOSS_W = float(_os.environ.get("EXT_NAMED_LOSS_W", "1.0"))
# Partition the surface between the directions and let each supervise only its own share.
EXT_VORONOI = _os.environ.get("EXT_VORONOI", "0") == "1"
EXT_VORONOI_DEBUG = _os.environ.get("EXT_VORONOI_DEBUG", "0") == "1"
# Softmax temperature on the direction alignment. 0 keeps the hard argmax. Smaller is
# sharper; at 0.05 two adjacent directions cross over within a few degrees.
EXT_VORONOI_TAU = float(_os.environ.get("EXT_VORONOI_TAU", "0"))
# Weight of an unmasked term alongside the masked one, to couple the cells to each other.
EXT_VORONOI_MIX = float(_os.environ.get("EXT_VORONOI_MIX", "0"))
# Supervise the exterior by frequency: low frequencies per pixel, high ones by their energy
# alone. EXT_ERODE is how far inside both silhouettes the comparison stays, in pixels.
EXT_BAND     = _os.environ.get("EXT_BAND", "0") == "1"
EXT_BAND_W   = float(_os.environ.get("EXT_BAND_W", "30.0"))
EXT_ERODE    = int(_os.environ.get("EXT_ERODE", "10"))
# The finest octave the shell can actually produce, in pixels. Below it the texture term only
# forbids excess; at and above it the reference's energy is matched from both sides.
EXT_BAND_REACH = float(_os.environ.get("EXT_BAND_REACH", "0.5"))
# The fraction of the shell, by depth, that the section planes may not touch.
SEC_SKIP_OUTER = float(_os.environ.get("SEC_SKIP_OUTER", "0"))
# Weight on matching the *distribution* of colour inside a crop rather than its colour pixel by
# pixel. See `get_quant_loss`.
SEC_QUANT = float(_os.environ.get("SEC_QUANT", "0"))
# One iteration's per-plane residuals, drained at the end of the iteration into the
# convergence test that decides when a reference has stopped paying. See
# `sds_demo.converged_enough`.
_RESID = []
# How often to write a resumable checkpoint. 0 disables it.
CKPT_INTERVAL = int(_os.environ.get("CKPT_INTERVAL", "0"))
# Whether a snapshot carries the weights as well as the pictures.
SNAP_PLY = _os.environ.get("SNAP_PLY", "1") == "1"
# Match each octave's whole distribution by sorting, rather than only its mean magnitude.
EXT_BAND_SW  = _os.environ.get("EXT_BAND_SW", "0") == "1"
# Weight on SSIM against the reference photograph, the quantity the exterior is judged by.
EXT_SSIM_W   = float(_os.environ.get("EXT_SSIM_W", "0"))
# Penalty on the achromatic part of the directional terms: what makes a patch brighten and
# darken as the camera moves, without touching what makes it a feature.
SH_BAL_W     = float(_os.environ.get("SH_BAL_W", "0"))
# Half-angle, in degrees, of the cap at each pole whose colour training may not change.
POLE_PIN     = float(_os.environ.get("POLE_PIN", "0"))
_pole_tgt, _pole_msk = [None], [None]
# Keep the shell exactly as the projection left it; only the interior learns.
SHELL_PIN    = _os.environ.get("SHELL_PIN", "0") == "1"
# Pin the shell's geometry too, not only its colour. Colour alone leaves position and
# scale to be recomputed from a feature the interior branch keeps moving, and the cells
# stop tiling: the lattice's rows appear as banding on the peel, and on a thinner shell
# they part and the interior shows through.
SHELL_PIN_GEOM = _os.environ.get("SHELL_PIN_GEOM", "1") == "1"
_shell_tgt, _shell_msk = [None], [None]
_AXES_CACHE = [None]
# Supervise the exterior from the same directions the skin was coloured from, instead of six.
#
# The initialisation covers the sphere with thirty-two scattered directions and comes out
# uniform -- the low-frequency saturation dip is 0.042 at worst and 0.6% of the silhouette is
# more than 0.05 below the median. Training with six fixed views puts it back: 0.102 and
# 18.6%, and the dips land between the six, at the poles and around the equator, which is
# exactly where no view looks straight on. Six directions learned well and the gaps drift.
#
# Point this at a reference directory carrying `dirs.json` and each iteration takes the next
# EXT_VIEWS of them, so over a run every direction is used and none is privileged. The
# reference for a direction is the one generated for it, so the stem scar stays where it
# belongs -- which is what a random camera with a fixed reference could not offer.
EXT_DIRS = _os.environ.get("EXT_DIRS", "")
_EXT_DIRS = []
if EXT_DIRS:
    import json as _json
    _dj = _os.path.join(EXT_DIRS, "dirs.json")
    if _os.path.exists(_dj):
        _EXT_DIRS = [(n, v[0], v[1]) for n, v in _json.load(open(_dj)).items()
                     if _os.path.exists(_os.path.join(EXT_DIRS, f"{n}_ref.png"))]
        print(f"exterior supervised from {len(_EXT_DIRS)} directions in {EXT_DIRS}")

    else:
        raise SystemExit(f"EXT_DIRS={EXT_DIRS} has no dirs.json")
EXT_REPEAT    = int(_os.environ.get("EXT_REPEAT", "1"))
EXT_ELEV      = [-60, -40, -20, 0, 20, 40, 60, -50, -10, 10, 30, 50,
                 -30, 0, 45, -45, 15, -15, 55, -55, 35, -35, 25, -25]

# The released README's step five: "specify your new model ID ... replace these values with
# your own fine-tuned model IDs". The released code never does, so every run of it uses the
# base model, whose mode for an orange section is a smooth disc -- which is what SDS then
# converges to. Settable now, and still the base model when unset.
SD_MODEL_VERTICAL = _os.environ.get("SD_MODEL_V", "sd2-community/stable-diffusion-2-depth")
SD_MODEL_HORIZONTAL = _os.environ.get("SD_MODEL_H", "sd2-community/stable-diffusion-2-depth")

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def save_img(rendering, path, frame, f_prefix=""):
    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    assert args.output_path is not None
    cv2_img *= 255 
    path = os.path.join(path, f"{f_prefix}{frame}.png".rjust(8, "0"))
    cv2.imwrite(
        path,
        cv2_img,
    )
    return path

def load_checkpoint(model_path, iteration=-1, gs_path=None):
    if gs_path:
        checkpt_path = gs_path
        print("using ", gs_path)
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    # Take the degree from the file rather than assuming zero.
    #
    # `load_ply_zero_sh` discards the higher-order terms on purpose, so a shell whose cells
    # carry a direction-dependent colour loaded here as its own mean and trained as though the
    # directionality had never been fitted -- the very thing it exists to keep. The names are
    # already being read two lines up; use them.
    n_extra = len(extra_f_names)
    sh_degree = 0
    while 3 * ((sh_degree + 1) ** 2 - 1) < n_extra:
        sh_degree += 1
    if 3 * ((sh_degree + 1) ** 2 - 1) != n_extra:
        raise SystemExit(f"{checkpt_path} has {n_extra} f_rest fields, which is no SH degree")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply_zero_sh(checkpt_path)
    if sh_degree > 0:
        import numpy as _np
        from torch import nn as _nn
        e = _np.stack([_np.asarray(plydata.elements[0][a]) for a in extra_f_names],
                      1).astype(_np.float32)
        e = e.reshape(e.shape[0], 3, (sh_degree + 1) ** 2 - 1).transpose(0, 2, 1)
        gaussians._features_rest = _nn.Parameter(torch.from_numpy(e).cuda())
        gaussians.active_sh_degree = sh_degree
        print(f"directional appearance: SH degree {sh_degree}, "
              f"mean |coefficient| {float(gaussians._features_rest.abs().mean()):.4f}")
    return gaussians

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--guidance_config", type=str, default="./config/guidance/ms_guidance.yaml")
    parser.add_argument("--white_bg", type=bool, default=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--gs_path", type=str, default=None)
    parser.add_argument("--gs_ori_path", type=str, default=None)
    parser.add_argument("--flip", action="store_true")
    args = parser.parse_args()

    if DDP_WORLD > 1:
        import torch.distributed as dist
        # Each rank is launched with one visible device, so "cuda:0" -- which this script
        # hardcodes throughout -- already means that rank's own GPU and nothing has to be
        # rewritten to carry a device around.
        dist.init_process_group("nccl", rank=DDP_RANK, world_size=DDP_WORLD)
        print(f"[rank {DDP_RANK}/{DDP_WORLD}] {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")
    if not os.path.exists(args.guidance_config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    train = args.train

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.physics_config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path, gs_path=args.gs_path)
    # Voxel variant: interior points sit at exact cell centres (see filling.py) and
    # their positions are held there for the whole run -- the cell *is* the primitive.
    # Everything else (colour, scale, rotation) still trains normally, and the
    # exterior surface is untouched.
    _flag = os.path.join(os.path.dirname(args.gs_path), "is_interior.pt")
    if os.path.exists(_flag):
        gaussians.is_interior = torch.load(_flag).cuda()
        gaussians.lattice_pure = LATTICE_PURE
        print(f"voxel mode: {int(gaussians.is_interior.sum())} interior points pinned"
              f"{'  (lattice_pure)' if LATTICE_PURE else ''}")
    else:
        raise SystemExit(f"missing {_flag} -- rerun internal_filling.py")

    # The outermost layer answers to the exterior alone.
    #
    # A cut plane selects every primitive within `surf_dis` of it, the shell included, and the
    # section reference shows the peel there as a thin pale rim -- which is what a cut peel
    # looks like. So the same cells are told "bright orange peel" by the exterior branch and
    # "pale rim" by whichever planes pass through them, and one colour cannot be both. The
    # planes sit at fixed heights and fixed azimuths, so their footprint on the surface is a
    # grid of parallels and meridians, and that is exactly the wireframe that appeared over the
    # trained peel -- it survived turning the view partition off, because it never came from
    # there.
    #
    # The shell is several cells thick and the conflict is only about its outside, so split the
    # ownership by depth rather than arbitrating the colour: the outermost layer is the
    # exterior's, everything under it stays the sections'. The rim a cut exposes is the layer
    # beneath, which is what it is in a real peel.
    if SEC_SKIP_OUTER > 0:
        with torch.no_grad():
            # By radius over every primitive. `is_interior` cannot be used for this -- in voxel
            # mode it is true for all 985,492 of them, so its complement is empty -- and the
            # outermost layer is a geometric fact anyway, not a flag.
            _p = gaussians.get_xyz.detach()
            _rr = (_p - _p.mean(0)).norm(dim=1).float()
            _thr = torch.quantile(_rr, 1.0 - SEC_SKIP_OUTER)
            gaussians.is_outer = _rr > _thr
        print(f"  sections will skip the outer {100*SEC_SKIP_OUTER:.0f}% by radius: "
              f"{int(gaussians.is_outer.sum()):,} primitives")
    else:
        gaussians.is_outer = torch.zeros_like(gaussians.is_interior.reshape(-1), dtype=torch.bool)
    gaussians_ori = load_checkpoint(model_path, gs_path=args.gs_ori_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background_b = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    )
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    )
    # The background the *sections* are rendered against, which is a different question from
    # the one the exterior views ask.
    #
    # A cut face is composited over this, so it is what shows through wherever the interior is
    # thin. On white, transparency is the cheapest brightness the model can buy: a section
    # target is brighter than the render wherever it asks for pith or albedo, and a cell can
    # meet that either by learning a paler colour or by getting out of the way. It gets out of
    # the way -- the interior goes from 0% of cells below opacity 0.5 to 61-64% in fifty
    # iterations, on the photograph target, with no diffusion involved.
    #
    # Darken the background and the shortcut inverts: less opacity now means darker, which is
    # away from the target, so the only way left to satisfy it is colour. This constrains
    # nothing about what the model may represent -- a genuine void still renders as a void --
    # which is what freezing the interior could not say for itself.
    #
    # Default 1.0 leaves every existing run bit-identical.
    _sbg = float(_os.environ.get("SECTION_BG", "1.0"))
    background_sec = torch.full((3,), _sbg, dtype=torch.float32, device="cuda")
    params = load_params_from_gs(gaussians, pipeline)
    init_opacity = params["opacity"]
    max_value = init_opacity.max()
    max_value = 10000
    # Detach the tensor from the computation graph, modify it, and then reattach it
    with torch.no_grad():
        gaussians._opacity.copy_(init_opacity.clone().detach().fill_(max_value))

    # init the scene
    print("Initializing scene and pre-processing...")

    def create_3d_grid(gaussians, grid_size):
        xyz = gaussians.get_xyz # Shape (N, 3)
        device = xyz.device

        # Get the min and max coordinates to define the grid boundaries
        min_coords = xyz.min(dim=0)[0]
        max_coords = xyz.max(dim=0)[0]
        
        # Calculate the dimensions of each grid cell
        cell_dimensions = (max_coords - min_coords) / torch.tensor(grid_size, device=device)

        # Create a dictionary to hold the grid
        grid = {}

        # Iterate over each gaussian and assign it to a grid cell
        for idx in tqdm(range(xyz.size(0)), desc="Creating 3D Grid"):
            # Determine the grid cell for the current point
            cell_coords = ((xyz[idx] - min_coords) / cell_dimensions).floor().long()

            # Convert cell coordinates to a tuple for the dictionary key
            cell_key = tuple(cell_coords.tolist())
            
            if cell_key not in grid:
                grid[cell_key] = []
            
            grid[cell_key].append(idx)

        return grid

    def voxel_smoothing(gaussians, grid_size=None):
        """Voxel Smoothing, paper section 3.3.

            C = sum_i w_i * C_i / sum_i w_i

        "untrained Gaussians are assigned colors using a distance-weighted average
        of nearby trained Gaussians... C_i represents the color of each nearby
        *trained* Gaussian, and w_i is the inverse distance weight"

        Three things matter and were all lost in the released implementation:
          * only *untrained* Gaussians are written to -- supervised colour must not
            be overwritten;
          * only *trained* Gaussians are read from -- otherwise the black,
            never-supervised interior is what gets propagated;
          * weights are inverse distance, not a flat mean.
        Grid resolution follows the paper's rule of thumb ("each voxel contains
        fewer than 1% of the total particles"); the released 512^3 leaves 98.8% of
        voxels holding a single point, which makes the average a no-op.
        """
        grid_size = ABL_GRID if grid_size is None else grid_size
        xyz = gaussians.get_xyz.detach()
        fdc = gaussians._features_dc.detach()          # (N,1,3)
        trained = gaussians.get_trained()
        if not ABL_TRAINED:                      # released behaviour: no distinction
            trained = torch.ones_like(trained)   # -> every point is both source and target
        if trained.sum() == 0:
            return 0

        mn, mx = xyz.min(0)[0], xyz.max(0)[0]
        cell = (mx - mn) / grid_size
        cell = torch.where(cell > 0, cell, torch.ones_like(cell))
        idx = ((xyz - mn) / cell).floor().long().clamp(0, grid_size - 1)
        key = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]

        new_fdc = fdc.clone()
        filled = 0
        tgt_all = (~trained) if ABL_TRAINED else torch.ones_like(trained)
        for k in key[tgt_all].unique():
            in_cell = key == k
            src = in_cell & trained
            tgt = in_cell & ((~trained) if ABL_TRAINED else torch.ones_like(trained))
            if src.sum() == 0 or tgt.sum() == 0:
                continue
            d = torch.cdist(xyz[tgt], xyz[src])                    # (T,S)
            w = (1.0 / (d + 1e-8)) if ABL_IDW else torch.ones_like(d)
            w = w / w.sum(1, keepdim=True)
            new_fdc[tgt] = (w @ fdc[src].squeeze(1)).unsqueeze(1)
            filled += int(tgt.sum())

        gaussians._features_dc.data.copy_(new_fdc)
        return filled


    def preprocess_particles(gaussians, pipeline, preprocessing_params, args):
        # Load parameters from gaussians and pipeline
        params = load_params_from_gs(gaussians, pipeline)

        init_pos = params["pos"]
        init_cov = params["cov3D_precomp"]
        init_screen_points = params["screen_points"]
        init_opacity = params["opacity"]
        init_shs = params["shs"]

        # Rotate and translate object
        if args.debug:
            log_dir = "./log"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            particle_position_tensor_to_ply(init_pos, os.path.join(log_dir, "init_particles.ply"))

        transformed_pos, scale_origin, original_mean_pos = transform2origin(init_pos)
        transformed_pos = shift2center111(transformed_pos)

        # Modify covariance matrix accordingly
        init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        init_cov = scale_origin * scale_origin * init_cov

        if args.debug:
            particle_position_tensor_to_ply(transformed_pos, os.path.join(log_dir, "transformed_particles.ply"))

        device = "cuda:0"
        mpm_init_pos = transformed_pos.to(device=device)
        mpm_init_cov = init_cov
        # Return processed outputs
        return init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points

    filling_params = preprocessing_params["particle_filling"]

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    class TrainingArgs:
        def __init__(self):
            self.position_lr_init = 0.001
            self.position_lr_final = 0.0002
            self.position_lr_delay_mult = 0.02
            self.position_lr_max_steps = 600
            # Adam moves a parameter by about lr per step regardless of the loss scale,
            # so 151 iterations buy 151 * 0.001 = 0.15 in SH units, which is 0.043 in RGB.
            # Turning saturated flesh into a white membrane needs dRGB ~ 0.4-0.65, i.e.
            # dSH ~ 1.4-2.3 -- a factor of 10-15 more than the budget. That is why raising
            # the high-frequency weight 80x moved luminance only 0.024: reweighting cannot
            # buy steps.
            self.feature_lr = float(_os.environ.get("FEATURE_LR", "0.001"))
            self.opacity_lr = 0.01
            self.scaling_lr = 0.001
            self.rotation_lr = 0.01
            self.percent_dense = 0.01
            self.density_start_iter = 0
            self.density_end_iter = 3000
            self.densification_interval = 50
            self.opacity_reset_interval = 700
            self.densify_grad_threshold = 0.01

    training_args = TrainingArgs()
    gaussians.spatial_lr_scale = 0.1
    gaussians.training_setup(training_args)

    ANCHOR = _os.environ.get("ANCHOR", "0") == "1"
    _dec = None
    _nb_src, _nb_dst = [None], [None]
    _phys_tgt = [None]      # the auxiliary material target, loaded on first use
    _held = {}              # reconciled section targets, kept across iterations
    _axial = [None, None]
    _anchor_rgb = None
    _skin = None
    SHELL_W = float(_os.environ.get("SHELL_W", "3.0"))
    DARK_W = float(_os.environ.get("DARK_W", "10.0"))
    DARK_FLOOR = float(_os.environ.get("DARK_FLOOR", "0.25"))
    if ANCHOR:
        import anchor_decoder
        _d = os.path.dirname(args.gs_path)
        _dec, _opt = anchor_decoder.install(
            gaussians, os.path.join(_d, "cell_level.pt"),
            os.path.join(_d, "lattice.pt"),
            K=int(_os.environ.get("ANCHOR_K", "4")),
            f_dim=int(_os.environ.get("ANCHOR_DIM", "32")),
            view_dependent=_os.environ.get("ANCHOR_VIEWDEP", "0") == "1")
        # Key the cache by the shape it was fitted for. It was "anchor_prefit.pt" and
        # nothing else, so a cache written at K=1 d=8 was handed to a K=4 d=32 decoder and
        # load_state_dict raised on eight separate size mismatches. Same class of trap as
        # the reference cache keyed on a prefix: a name that does not mention what varies.
        # Key the cache by everything that changes the decoder's shape, not by some of it. K and
        # the feature width were in the name; view-dependence was not, and it widens stage2's
        # input by the three components of the view direction -- so a cache fitted without it
        # was handed to a decoder expecting 19 inputs and load_state_dict raised on a 16-vs-19
        # mismatch. Same trap the comment below describes, one field short.
        _vd = "_vd" if _dec.view_dependent else ""
        _pf = os.path.join(_d, f"anchor_prefit_K{_dec.K}_d{_dec.feat.shape[1]}{_vd}.pt")
        _pf_old = os.path.join(_d, "anchor_prefit.pt")
        if not os.path.exists(_pf) and os.path.exists(_pf_old) and _dec.K == 1 \
                and _dec.feat.shape[1] == 8:
            _pf = _pf_old        # the existing caches were all written at K=1 d=8
        if _os.environ.get("ANCHOR_PREFIT", "1") == "1" and not os.path.exists(_pf):
            # Fit it here when there is no cache. The guard below used to require the file
            # and say nothing when it was missing, so for every input prepared since --
            # none of which had one -- prefit and the shell anchor were both skipped in
            # silence, and the decoder trained from random features and a single mean
            # colour. Measured on the watermelon: the input carries a brightness spread of
            # 0.195, with a green shell at (0.24, 0.44, 0.23) against red flesh at
            # (0.81, 0.27, 0.26), and iteration 0 logged a spread of 0.0019. Everything the
            # reference photographs and the exterior projection had established was
            # discarded before the first step.
            _tgt = (gaussians._features_dc.detach().squeeze(1).cuda()
                    * anchor_decoder.C0 + 0.5).clamp(0, 1)[:_dec.anchor_xyz.shape[0]]
            if DDP_RANK == 0:
                anchor_decoder.prefit(_dec, _tgt)
                # Write it whole, then move it into place. Both ranks look for this file at
                # the same moment, and rank 1 read one that rank 0 was still writing --
                # torch.load on a half-written archive raises inside tarfile, so the second
                # GPU died at startup and the run silently continued on one.
                _tmp = _pf + f".tmp{os.getpid()}"
                torch.save(_dec.state_dict(), _tmp)
                os.replace(_tmp, _pf)
                print(f"anchor prefit computed and cached to {_pf}")
            else:
                # Wait for rank 0 rather than fitting it again: the fit is deterministic
                # given the input, so a second copy would only cost another minute.
                import time as _t
                for _ in range(3600):
                    if os.path.exists(_pf):
                        break
                    _t.sleep(1)
                print(f"anchor prefit awaited from rank 0: {_pf}")
        if _os.environ.get("ANCHOR_PREFIT", "1") == "1" and os.path.exists(_pf):
            # Hand the decoder the appearance the input model already carries. Without
            # this the decoder starts at a single colour where the input has 184,339, and
            # the ten exterior views cannot climb back: their target differs from the
            # current render only in detail, so the gradient is weak, while the
            # cross-section target differs wildly and dominates. The shell came out
            # uniformly yellow for exactly that reason.
            # The prefit predates any head added since it was cached, and the physics head is
            # one: it has no prefit to restore and starts from its own initialisation. Loading
            # non-strictly restores what the cache has and leaves the rest alone; anything the
            # cache is genuinely missing would show up as an untrained visual head, which the
            # printed mse would make obvious.
            _missing, _unexpected = _dec.load_state_dict(torch.load(_pf), strict=False)
            _missing = [k for k in _missing if not k.startswith("phys.")]
            if _missing or _unexpected:
                print(f"  prefit cache: missing {_missing}, unexpected {_unexpected}")
            print(f"anchor prefit loaded from {_pf}")
        # Resume, if asked. A checkpoint written by CKPT_INTERVAL carries the decoder's state
        # and its per-cell features, which together are the whole learned model in the anchor
        # path -- the ply is decoded output. Loading them after the prefit means the prefit is
        # simply overwritten by something later and better, which is what a resume is.
        #
        # The references are deliberately *not* restored from the checkpoint here: they live in
        # the output directory and are rebuilt or reused by the ordinary path. Restoring the
        # weights without them would be the dangerous case -- carrying on against targets that
        # are not the ones the weights were fitted to -- so the checkpoint keeps a copy of them
        # for inspection and RESUME_REFS copies them into place when a resume needs them.
        _rs = _os.environ.get("RESUME", "")
        if _rs and _dec is not None:
            _rp = _rs if _rs.endswith(".pt") else os.path.join(_rs, "anchor.pt")
            _ck = torch.load(_rp, map_location="cuda:0")
            _dec.load_state_dict(_ck["state"], strict=False)
            with torch.no_grad():
                _dec.feat.copy_(_ck["feat"].to(_dec.feat.device))
            print(f"resumed decoder and {tuple(_ck['feat'].shape)} features from {_rp} "
                  f"(written at iteration {_ck.get('iter', '?')})", flush=True)
            if _os.environ.get("RESUME_REFS", "1") == "1":
                import glob as _g2, shutil as _sh2
                _n = 0
                for _f2 in _g2.glob(os.path.join(os.path.dirname(_rp), "[hv]*_ref.png")):
                    _sh2.copy2(_f2, args.output_path); _n += 1
                print(f"  and {_n} references it was fitted against", flush=True)
            # Keep the shell where prefit put it. The cross-section branch supervises 16
            # horizontal planes plus the vertical ones against a photo of a cut orange,
            # against only 10 exterior views, and that photo's overall colour drags the
            # whole model with it: the peel starts orange straight out of prefit and has
            # drifted yellow by iteration 151. Anchoring only the level-1 (skin) anchors
            # to their prefit colour leaves the interior free, which is the part the
            # cross-sections are actually meant to teach.
            with torch.no_grad():
                _lv = torch.load(os.path.join(_d, "cell_level.pt")).cuda()
                _skin = (_lv == 1).repeat_interleave(_dec.K)
                _, _rgb0, _, _, _ = _dec(None)
                _anchor_rgb = _rgb0.detach().clone()
            SHELL_W = float(_os.environ.get("SHELL_W", "3.0"))
        if AXIAL_W > 0:
            _lat = torch.load(os.path.join(_d, "lattice.pt"))
            _cell = _dec.cell.reshape(-1)
            _axial[0], _axial[1] = anchor_decoder.build_axial(
                _dec.anchor_xyz, _cell, _lat["up"].cuda(), float(_lat["fine_dx"]))
            print(f"axial pairs: {_axial[0].shape[0]:,} of {_cell.shape[0]:,} anchors "
                  f"({_axial[0].shape[0]/_cell.shape[0]*100:.1f}%)")
        if SMOOTH_W > 0:
            _lat2 = torch.load(os.path.join(_d, "lattice.pt"))
            _lv2 = torch.load(os.path.join(_d, "cell_level.pt")).cuda()
            _nb_src[0], _nb_dst[0] = anchor_decoder.build_neighbours(
                _dec.anchor_xyz, _dec.cell.reshape(-1), float(_lat2["fine_dx"]), _lv2)
            print(f"neighbour pairs: {_nb_src[0].shape[0]:,} over "
                  f"{_dec.anchor_xyz.shape[0]:,} anchors "
                  f"({_nb_src[0].shape[0]/_dec.anchor_xyz.shape[0]:.2f} per anchor)")
        gaussians.optimizer = _opt
        gaussians.is_interior = torch.ones(
            gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        gaussians.lattice_pure = False        # nothing to densify or prune any more

    _gprof = [None]
    _gacc = [torch.zeros(72, device="cuda"), 0]
    _CAMC = [None]

    def _cap(cam):
        _CAMC[0] = getattr(cam, "camera_center", None)
        return cam

    def _bw(loss):
        """Extra priors, then backward -- called wherever the loop backpropagates.

        No part of an orange's flesh is black. The cross-section branch renders a thin
        slab through plane_filter, so of the four children an anchor decodes only the
        front one is ever seen; the rest are unconstrained and drift to extremes. Measured
        along the chain: the input reconstruction carries 19.4% near-black interior
        primitives, all-voxel quantisation cleans that to 3.0%, prefit preserves 3.0% --
        and 151 iterations of training push it back to 19.6%, with mean luminance rising
        at the same time. The distribution polarises because the loss only sees a
        projection. A floor on luminance forbids black without prescribing any pattern.
        """
        if ANCHOR and _dec is not None and AXIAL_W > 0 and _axial[0] is not None:
            # Cells a step apart along the axis hold the same material, so their colours
            # should agree. L1 rather than L2: a real boundary -- a seed, a membrane wall
            # ending -- should cost its difference once and not be squared into submission.
            _, _c, _, _, _ = _dec(None)
            _ck = _c.view(-1, _dec.K, 3).mean(1)
            loss = loss + AXIAL_W * (_ck[_axial[0]] - _ck[_axial[1]]).abs().mean()
        if ANCHOR and _dec is not None and (DARK_W > 0 or SHELL_W > 0):
            _, _c, _, _, _ = _dec(None)
            if SHELL_W > 0 and _anchor_rgb is not None:
                loss = loss + SHELL_W * torch.nn.functional.mse_loss(
                    _c[_skin], _anchor_rgb[_skin])
            if DARK_W > 0:
                _lum = _c.mean(1)
                loss = loss + DARK_W * torch.relu(DARK_FLOOR - _lum).pow(2).mean()
        loss.backward()
        if DDP_WORLD > 1 and _dec is not None:
            import torch.distributed as dist
            for _p in _dec.parameters():
                if _p.grad is not None:
                    dist.all_reduce(_p.grad, op=dist.ReduceOp.SUM)
                    _p.grad /= DDP_WORLD

    def _align_phase(render, gt, nb=72, r_lo=0.25, r_hi=0.80):
        """Rotate the reference to the angle that best matches the render, before comparing.

        A cross-section photograph has no canonical angular origin: rotating it about the
        disc centre gives an equally valid picture of the same fruit, so its absolute phase
        carries no information about the object being built. Comparing against it pointwise
        nonetheless demands that phase be reproduced -- and since the photographs are of
        different fruits, the demands contradict each other. Averaging patterns of
        independent phase cancels the angular structure exactly, which is what the interior
        showed: with one photograph the layer-to-layer agreement rose over training,
        0.946 to 0.972, and with twenty it fell to 0.361.

        Removing the phase is not a prior about oranges. It is dropping a nuisance parameter
        that the data does not constrain, the same way point clouds are aligned before their
        shapes are compared. Once dropped, the photographs agree on where the pattern sits
        and their average keeps it, so they can contribute detail and colour while the
        pattern's placement stays with whatever the model already has.
        """
        H, W = render.shape[-2:]
        yy = torch.arange(H, device=render.device).view(-1, 1).float() - (H - 1) / 2
        xx = torch.arange(W, device=render.device).view(1, -1).float() - (W - 1) / 2
        rr = torch.sqrt(yy ** 2 + xx ** 2)
        R = float(rr.max()) * 0.5
        band = (rr > r_lo * R * 2) & (rr < r_hi * R * 2)
        ang = torch.atan2(yy.expand(H, W), xx.expand(H, W))
        bin_ = ((ang + math.pi) / (2 * math.pi) * nb).long().clamp(0, nb - 1)[band]

        def prof(img):
            v = img.mean(0)[band]
            acc = torch.zeros(nb, device=img.device).index_add_(0, bin_, v)
            cnt = torch.zeros(nb, device=img.device).index_add_(
                0, bin_, torch.ones_like(v))
            q = acc / cnt.clamp_min(1)
            return (q - q.mean()) / (q.std() + 1e-6)

        with torch.no_grad():
            a, b = prof(render), prof(gt)
            # Accumulate this iteration's renders, and align against the mean of the last
            # one. Aligning each photograph to its own plane's render leaves every phase
            # assignment stationary -- with no term coupling the layers, each plane's
            # objective is minimised independently, so an inconsistent assignment is as
            # good as a consistent one and nothing pulls it back. Measured, that is what
            # happened: 0.468 without alignment, 0.450 with it. One shared target instead
            # makes every plane's loss pull toward the same pattern, which is the
            # alternating minimisation of the phases and the field together.
            _gacc[0] = _gacc[0] + a
            _gacc[1] = _gacc[1] + 1
            tgt = _gprof[0] if _gprof[0] is not None else a
            corr = torch.stack([(torch.roll(b, k) * tgt).mean() for k in range(nb)])
            k = int(corr.argmax())
            if k == 0:
                return gt
            th = torch.tensor(k * 2 * math.pi / nb, device=gt.device)
            c, sn = torch.cos(th), torch.sin(th)
            mat = torch.tensor([[c, -sn, 0.], [sn, c, 0.]], device=gt.device).unsqueeze(0)
            grid = F.affine_grid(mat, (1, 3, H, W), align_corners=False)
            out = F.grid_sample(gt.unsqueeze(0), grid, align_corners=False,
                                padding_mode="border")[0]
        return out

    # One regeneration per view per iteration: the exterior loop can run a view more than
    # once (EXT_REPEAT), and without this each repeat would resample and the later repeats
    # would train against a target the earlier ones had already moved.
    _ext_last = {}

    def _ext_gt(rendering, rendering_ori, ttt, j):
        """What the exterior branch should match.

        By default it matches a render of the model training started from, which preserves
        that model's appearance -- correct when the input was a scan and the peel was the
        one part already right. A generated shell has no appearance to preserve: ours
        begins flat grey and is coloured by the cross-section photograph's outer ring,
        which is the cut rim rather than the intact peel, and this branch then holds that
        wrong colour for the whole run because no other view ever sees the outside. Given a
        prompt for the whole object, sample the exterior the way the sections are sampled,
        so the outside is supervised rather than merely preserved.
        """
        if not EXT_PROMPT:
            return rendering_ori.detach()
        if _EXT_DIRS:
            _nm = _EXT_DIRS[(j * EXT_VIEWS + ttt) % len(_EXT_DIRS)][0]
            ref_path = os.path.join(EXT_DIRS, f"{_nm}_ref.png")
        else:
            ref_path = os.path.join(args.output_path, f"o{ttt}_ref.png")
        # Generated once when missing, and again every EXT_REF_INTERVAL iterations if that is
        # set. It used to be once only, on the argument that the shell would by then have been
        # dragged toward flesh colour by the cross-sections through a shared decoder, so
        # resampling would ask the generator to turn cut fruit back into whole fruit. Under
        # ANCHOR_SPLIT the shell has its own colour head and the sections cannot reach it, so
        # that argument no longer holds -- and holding the reference fixed has a cost that was
        # not being paid attention to. A pre-made reference is framed however it was made:
        # `cube_or3_prep` has the orange at 461 of 512 pixels across, our render has it at 316,
        # a factor of 1.46 in width and 2.05 in area. The two are then compared pixel by pixel
        # by SSIM, and the chroma term averages the render over the *reference's* mask, most of
        # which is our background -- which is why the exterior drives itself to a dark
        # oversaturated red speckle instead of orange. Regenerating from the current render
        # inherits the render's own framing, so the mismatch cannot arise.
        _stale = (EXT_REF_INTERVAL > 0 and j > 0 and j % EXT_REF_INTERVAL == 0
                  and _ext_last.get(ttt) != j)
        if _stale:
            _ext_last[ttt] = j
        if _stale or not os.path.exists(ref_path):
            d = F.interpolate(rendering.unsqueeze(0), size=(384, 384),
                              mode="bilinear", align_corners=False)
            d = pipe.depth_estimator(d.to("cuda:0")).predicted_depth
            d = F.interpolate(d.unsqueeze(0), size=(512, 512),
                              mode="bilinear", align_corners=False).squeeze(0)
            save_img(rendering, args.output_path, 0, f"o{ttt}_init_")
            cur = Image.open(os.path.join(args.output_path, f"o{ttt}_init_0.png"))
            # A regeneration refines this view's own render, so it resamples rather than
            # reusing the one-shot, and at EXT_REFINE_S rather than the 0.95 a flat grey
            # shell needed. The initialisation says which orange this is; the sampler only
            # has to put the peel back.
            _new = exterior_ref(cur, d, pipe, EXT_PROMPT, fresh=_stale,
                                strength=(EXT_REFINE_S if _stale else None))
            _new.save(ref_path)
            # Keep the pair. A regenerated exterior target is overwritten in place, so without
            # this there is no record of whether the loop is developing the peel or drifting.
            log_ref(args.output_path, f"o{ttt}", j, cur, _new,
                    what=(f"exterior img2img {EXT_REFINE_S:.2f}" if _stale else
                          "exterior one-shot "
                          + _os.environ.get("EXT_STRENGTH", "0.95")))
        # Put the reference on the render's silhouette before comparing them.
        #
        # A pre-made reference is framed however it was made, and the good ones are not framed
        # like our render: `cube_or3_prep` has the orange 461 pixels across in a 512 frame and
        # the render has it at 316, a factor of 1.46 in width and 2.05 in area. SSIM then
        # compares peel against background over the whole annulus between the two silhouettes,
        # which no arrangement of the model can fix, and the chroma term averages the render
        # over the reference's mask -- mostly our white background, so the only way to pull
        # that average down to the target is to drive the object far darker and more saturated
        # than the target. That is exactly what came out: a dark disc speckled red.
        #
        # `_fit_disc` is the section branch's own answer to the same problem and takes both
        # discs from their border colour, so it needs nothing said about either image.
        _ref_img = Image.open(ref_path).convert("RGB")
        if EXT_FIT_DISC:
            save_img(rendering, args.output_path, 0, f"o{ttt}_init_")
            _ref_img = _fit_disc(_ref_img, Image.open(
                os.path.join(args.output_path, f"o{ttt}_init_0.png")))
        return transform(_ref_img).to(device)

    def _dec_now(g):
        """Rebuild the primitives from the anchors, once per view.

        One decode per iteration is not enough: several views are rendered and each calls
        backward(), which frees the graph, so the second view hits "backward through the
        graph a second time". Decoding per view also gives the view-dependent colour stage
        the right camera.
        """
        if ANCHOR:
            anchor_decoder.write_into(g, _dec, _CAMC[0])
        return g


    transform = transforms.ToTensor()
    device = "cuda:0"
    view_count = 180
    epochs = 400
    prev_loss = 0
    track_loss = 0
    steps_per_c = 3

    pos = gaussians.get_xyz
    def training_step(gaussians, loss, grad_update_mask=None, viewspace_point_tensor=None, visibility_filter=None):
        # In anchor mode the optimised parameters are the anchor features and the shared
        # MLP, not one row per primitive, so a per-primitive visibility mask has nothing
        # to index. The mask still does its job: it zeroes the rendered gradient upstream,
        # which is what reaches the decoder.
        if not ANCHOR:
          for group in gaussians.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if grad_update_mask != None:
                        param.grad[~grad_update_mask] = 0
                    if group['name'] == 'opacity':
                        param.grad[:] = 0
                    if group['name'] == 'xyz':
                        # lattice anchoring: interior positions never move
                        param.grad[gaussians.get_is_interior()] = 0
        # Record which primitives this cross-section actually supervised, so
        # voxel_smoothing can tell trained from untrained (paper section 3.3).
        if grad_update_mask is not None:
            gaussians.trained = gaussians.get_trained() | grad_update_mask.to(
                gaussians.get_xyz.device)
        if not ANCHOR:
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        gaussians.optimizer.step()
        # The decoder rebuilds every per-primitive tensor at the top of each iteration, so
        # projecting them in place here would edit a non-leaf and do nothing useful.
        with torch.no_grad():
          if not ANCHOR:
            gaussians._opacity.copy_(init_opacity.clone().detach().fill_(max_value))
            gaussians._scaling.clamp_(max=-16)
            # Keep the DC colour term inside the range that maps to valid RGB.
            # render_utils.convert_SH ends in clamp_min(sh2rgb + 0.5, 0), so a Gaussian
            # pushed below -0.5/C0 renders pure black *and* sits in the flat part of the
            # clamp, where its colour gradient is exactly zero -- it can never come back.
            # Projecting onto the feasible set after each step keeps every primitive in
            # the region where gradients still flow. The loss is unchanged.
            _SH_LIM = 0.5 / 0.28209479177387814
            gaussians._features_dc.clamp_(-_SH_LIM, _SH_LIM)
          pass
        gaussians.optimizer.zero_grad()

    def purge_free_from_lattice():
        """Delete free Gaussians from the volume the lattice is supposed to own.

        Blocking densification of interior points is necessary but not sufficient:
        exterior points split inward, their children carry is_interior=False, and those
        children keep splitting. Measured after 150 iterations: 58419 free Gaussians in
        the interior region against 137051 lattice points (70.1% lattice), and the
        interior's nearest-neighbour spacing CV sat at 0.618 -- barely better than plain
        3DGS at 0.642, and far from the 0.394 the lattice reaches on its own.

        Membership is decided by enclosure, not by cell occupancy. fill_particles only
        fills cells that were *empty*, so lattice cells and free-Gaussian cells are
        disjoint by construction and an "is this cell occupied by a voxel" test matches
        exactly nothing (measured: 0 of 347412). A cell is interior iff filled cells
        exist on both sides of it along every axis -- the same ray-cast criterion the
        filling itself uses, evaluated here on the lattice.
        """
        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            interior = gaussians.get_is_interior()
            if interior.sum() < 10:
                return
            lat = xyz[interior]
            # cell size = the modal nearest-neighbour spacing of the lattice itself
            s = lat[torch.randperm(lat.shape[0], device=lat.device)[:4000]]
            dd = []
            for i in range(0, s.shape[0], 2000):
                dm = torch.cdist(s[i:i + 2000], lat)
                dm[dm < 1e-9] = 1e9
                dd.append(dm.min(1).values)
            dx = float(torch.cat(dd).median())
            mn = lat.min(0).values
            dims = (((lat.max(0).values - mn) / dx).round().long() + 1)
            occ = torch.zeros(dims.tolist(), dtype=torch.bool, device=xyz.device)
            il = ((lat - mn) / dx).round().long().clamp(
                torch.zeros(3, dtype=torch.long, device=xyz.device), dims - 1)
            occ[il[:, 0], il[:, 1], il[:, 2]] = True
            enc = torch.ones_like(occ)
            for ax in range(3):
                fwd = torch.cummax(occ, dim=ax).values
                bwd = torch.flip(torch.cummax(torch.flip(occ, [ax]), dim=ax).values, [ax])
                enc &= fwd & bwd

            ix = ((xyz - mn) / dx).round().long()
            cand = ((ix >= 0) & (ix < dims)).all(1) & ~interior
            drop = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
            if cand.any():
                j = ix[cand]
                drop[cand] = enc[j[:, 0], j[:, 1], j[:, 2]]
            if drop.any():
                gaussians.prune_points(drop)
                print(f"  lattice_pure: pruned {int(drop.sum())} free gaussians from the "
                      f"lattice volume (dx={dx:.6f}, {int(enc.sum())} enclosed cells)")

    def density_and_prune(e=0):
        # Nothing to densify when the primitives are decoder output: cloning a row of a
        # generated tensor does not create a new parameter. Capacity comes from K and from
        # the MLP instead.
        #
        # Smoothing does port, though, and used not to. The released trainer averages stored
        # colours over the cells of a 512-cubed grid every 101 iterations; here the colour is
        # decoder output so there is nothing stored to average, and one primitive per lattice
        # cell would leave every primitive alone in its own grid cell anyway. The equivalent
        # is to blend each anchor's feature toward its occupied face neighbours' -- see
        # anchor_decoder.build_neighbours. Off by default so the change is opt-in.
        if ANCHOR and SHELL_PIN:
            # Hold the whole shell at the colour the projection gave it.
            #
            # With the exterior branch off the shell should not move at all, and it does: 0.789
            # for the ply itself, 0.781 once the decoder has fitted it, 0.778 after thirty
            # iterations. Almost all of that is the first step -- the decoder reproduces a
            # million cell colours through an MLP and the prefit's residual is the 0.008 -- and
            # only 0.002 is anything training did. Neither is wanted here: the exterior is taken
            # directly from the released model's renders precisely so that nothing approximates
            # it.
            #
            # `set_colour` stores the difference between what the decoder produces and what is
            # wanted, and the decode adds it back, so the shell renders exactly the projection
            # while the interior trains normally. It is re-applied each epoch because the
            # residual is computed against a decoder that keeps moving under it.
            if _shell_tgt[0] is None:
                _lv = torch.load(os.path.join(os.path.dirname(args.gs_path), "cell_level.pt"))
                _lv = _lv.reshape(-1).to(gaussians.get_xyz.device)
                _K = max(1, gaussians.get_xyz.shape[0] // _lv.shape[0])
                _shell_msk[0] = (_lv != 0).repeat_interleave(_K)[:gaussians.get_xyz.shape[0]]
                _g0 = GaussianModel(0)
                _g0.load_ply_zero_sh(args.gs_path)
                _shell_tgt[0] = (_g0._features_dc.detach().squeeze(1) * 0.28209479177387814
                                 + 0.5).clamp(0, 1)
                del _g0
                if DDP_RANK == 0:
                    print(f"  shell pinned to the projection: "
                          f"{int(_shell_msk[0].sum()):,} primitives")
            _dec.set_colour(_shell_tgt[0], _shell_msk[0])
            if SHELL_PIN_GEOM and _dec.geom_frozen is None:
                # Once, after the colour is pinned for the first time: the geometry to hold is
                # the one the projection was measured on, not whatever a later epoch produces.
                _n = _dec.freeze_geometry(_shell_msk[0])
                if DDP_RANK == 0:
                    print(f"  shell geometry frozen: {_n:,} primitives hold their position, "
                          f"scale and rotation")
        if ANCHOR and POLE_PIN > 0:
            # Hold the two poles at what the painting put there.
            #
            # The calyx is the one feature a single reference carries, and the exterior branch
            # averages over directions, so it loses every time: 0.166 at initialisation and
            # 0.093 after training, 0.167 after sequential painting and 0.044 after training
            # that. No weighting fixes it -- the vote is taken between frames and a constant on
            # a loss term divides out of Adam.
            #
            # A cap is not a weighting. Everything outside it trains as before; inside it the
            # colour is what the painting decided, restored after each epoch through the same
            # residual the voxel smoothing uses, because the decoder would otherwise overwrite
            # it on the next decode.
            if _pole_tgt[0] is None:
                _p = _dec.anchor_xyz.repeat_interleave(_dec.K, 0)
                _n = _p - _p.mean(0)
                _n = _n / _n.norm(dim=1, keepdim=True).clamp_min(1e-9)
                _up = torch.tensor([0., 1., 0.], device=_n.device)
                _pole_msk[0] = (_n @ _up).abs() > math.cos(math.radians(POLE_PIN))
                _pole_tgt[0] = _dec()[1].detach().clone()
                if DDP_RANK == 0:
                    print(f"  poles pinned: {int(_pole_msk[0].sum()):,} primitives within "
                          f"{POLE_PIN:.0f} degrees of the axis hold their painted colour")
            _dec.set_colour(_pole_tgt[0], _pole_msk[0])
        if ANCHOR:
            # The auxiliary material task. The head predicts, per anchor, how stiff that cell
            # is relative to the rest of this object; the target is the unsupervised
            # decomposition, so nothing here is being told what the parts are. It runs on its
            # own step because it needs no render: the gradient reaches the shared feature,
            # which is the whole reason to do it inside training rather than after it.
            if _dec is not None and _dec.phys is not None and PHYS_TARGET:
                if _phys_tgt[0] is None:
                    _t = torch.load(PHYS_TARGET, map_location="cpu").float().reshape(-1)
                    _n = _dec.anchor_xyz.shape[0]
                    if _t.shape[0] < _n:
                        print(f"  PHYS_TARGET has {_t.shape[0]} entries for {_n} anchors; "
                              f"the auxiliary task is off")
                        _phys_tgt[0] = False
                    else:
                        _phys_tgt[0] = _t[:_n].to(_dec.feat.device)
                        print(f"  auxiliary material task on: {_n:,} anchors, target in "
                              f"[{float(_phys_tgt[0].min()):.2f}, {float(_phys_tgt[0].max()):.2f}]")
                if _phys_tgt[0] is not False:
                    r = _dec.stiffness()
                    l_phys = PHYS_W * torch.nn.functional.mse_loss(r, _phys_tgt[0])
                    if PHYS_TV > 0 and _nb_src[0] is not None:
                        # Smooth, except where the appearance already has an edge. A material
                        # boundary is allowed exactly where the object shows one, so the weight
                        # falls off with the colour difference across the pair rather than being
                        # uniform -- otherwise the peel's own boundary is the first thing this
                        # term erases.
                        with torch.no_grad():
                            _c = _dec()[1].detach()
                            _c = _c.view(_dec.anchor_xyz.shape[0], _dec.K, 3).mean(1)
                            _w = torch.exp(-(_c[_nb_src[0]] - _c[_nb_dst[0]]).abs().mean(1) / 0.05)
                        l_phys = l_phys + PHYS_TV * (
                            _w * (r[_nb_src[0]] - r[_nb_dst[0]]) ** 2).mean()
                    l_phys.backward()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if DDP_RANK == 0 and e % 20 == 0:
                        with torch.no_grad():
                            _e = float((r - _phys_tgt[0]).abs().mean())
                        print(f"  material head: mean |r - target| {_e:.4f}")
            # The paper's own smoothing, on the anchors. `voxel_smoothing` below is faithful
            # and unreachable from here, and would not survive a decode if it were reached.
            if VOXEL_SMOOTH and e > 0 and e % ABL_INTERVAL == 0:
                tr = gaussians.get_trained()
                n = anchor_decoder.voxel_smooth_anchors(_dec, tr, ABL_GRID)
                if DDP_RANK == 0:
                    t = int(tr.sum()); N = tr.numel()
                    print(f"  voxel smoothing: trained {t:,}/{N:,} ({100*t/N:.1f}%), "
                          f"untrained anchors filled from trained neighbours: {n:,}")
            if SMOOTH_W > 0 and SMOOTH_INTERVAL > 0 and e > 0 and e % SMOOTH_INTERVAL == 0:
                n = anchor_decoder.smooth_features(_dec, _nb_src[0], _nb_dst[0], SMOOTH_W)
                if DDP_RANK == 0:
                    print(f"  smoothed {n:,} anchor features toward their neighbours "
                          f"(alpha={SMOOTH_W})")
            return
        if e == 0 and LATTICE_PURE:
            # clear the primitives that were already buried inside the shell before
            # training starts -- orange_raw.ply ships ~8300 dark leftovers there
            purge_free_from_lattice()
        if e > 1 and e % 10 == 0:
            print("before densifying")
            print(gaussians.get_xyz.shape)
            grads = gaussians.xyz_gradient_accum / gaussians.denom
            grads[grads.isnan()] = 0.0
            mean_grads = torch.mean(grads)
            max_grads = torch.max(grads)
            min_grads = torch.min(grads)
            print(f"Gradient Statistics:")
            print(f"Mean: {mean_grads.item()}")
            print(f"Max: {max_grads.item()}")
            print(f"Min: {min_grads.item()}")
            gaussians.densify_and_prune(0.0002, min_opacity=0.0000001, extent=4, max_screen_size=None)
            print("after densifying")
            print(gaussians.get_xyz.shape)
            if LATTICE_PURE:
                purge_free_from_lattice()
        
        if e > 0 and e % ABL_INTERVAL == 0:
            with torch.no_grad():
                tr = int(gaussians.get_trained().sum())
                n = gaussians.get_xyz.shape[0]
                filled = voxel_smoothing(gaussians)
                print(f"voxel smoothing: trained {tr}/{n} ({tr/n*100:.1f}%), "
                      f"untrained filled from trained neighbours: {filled}")
    
    _BLUR_K = {}

    def _blur(t, sigma):
        """Gaussian blur of a (3,H,W) tensor, separable, differentiable."""
        if sigma not in _BLUR_K:
            r = max(int(3 * sigma), 1)
            x = torch.arange(-r, r + 1, device=t.device, dtype=torch.float32)
            k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
            _BLUR_K[sigma] = (k / k.sum(), r)
        k, r = _BLUR_K[sigma]
        u = t.unsqueeze(0)
        u = F.conv2d(u, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), padding=(0, r), groups=3)
        u = F.conv2d(u, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), padding=(r, 0), groups=3)
        return u.squeeze(0)

    def get_band_loss(rendering, ground_truth, mask, w_stat=1.0,
                      sig=(0.5, 1.0, 2.0, 4.0, 8.0)):
        """Supervise the low frequencies where they sit, and the high ones only in quantity.

        A per-pixel L2 over the whole frame is what makes the trained peel look like beads, and
        the reason is that most of what it is comparing cannot be matched. The reference is a
        photograph of *an* orange, not of this one: its dimples fall where the generator put
        them, ours fall on the lattice, and no assignment of cell colours brings the two into
        register. The optimiser still has a million free colours, so it does the only thing that
        lowers the error -- it fits the reference's texture as seen from that one direction. From
        every other direction that fit is noise, and with thirty-two directions each imposing its
        own, the cells end up with the variance we see as grain. Measured: energy at 4-8px went
        to 1.8 times the photographs' own while the peel visibly worsened.

        The same term also spans the silhouette, where the two disagree by construction -- the
        render's outline is the lattice's and the reference's is the generator's -- so the
        largest residuals in the frame sit on a boundary nothing can align. That is what drives
        the edge primitives, and it shows up as a bright rim (limb over centre 1.13 against the
        photographs' 1.05) and a ragged outline (0.44% radial roughness against 0.29%).

        So: blur both to `sig[-1]` and compare per pixel, which is alignable and carries the
        hue, the shading and features like the calyx; and for everything finer, compare only the
        mean energy per octave, a handful of scalars per frame. Statistics have no phase, so no
        direction can impose its own pattern on a cell, and the term is still satisfied exactly
        when the peel has the right amount of texture at each scale. The mask is eroded well
        inside both silhouettes so the boundary is never compared at all.
        """
        lo_r, lo_g = _blur(rendering, sig[-1]), _blur(ground_truth, sig[-1])
        m = mask
        den = m.sum().clamp_min(1.0)
        loss = ((lo_r - lo_g) ** 2 * m).sum() / den / 3.0
        # One-sided: penalise having *more* texture than the photograph at a scale, never less.
        #
        # A two-sided match asks the render to reach the reference's energy in every band, and
        # the reference's is concentrated at one to two pixels, which a per-cell colour cannot
        # produce -- a shell cell's Gaussian already covers several pixels. Asked for something
        # unreachable, the optimiser supplies it where it can, an octave up, and the run came
        # back with more grain than the one it was meant to fix (mid-band error 0.633 against
        # 0.408). Excess is the defect and deficiency is a property of the representation, so
        # only excess is penalised.
        # Per pixel, against a per-frame scalar. Comparing two scalars per band leaves one
        # number to carry the whole gradient, and it was far too weak to hold the null space
        # down: the low-frequency term constrains only the blurred image, so every cell pattern
        # that averages to the same blur is free, and stochastic gradients fill that freedom
        # with noise. Energy at 4-8px still rose from 1.26 to 1.97 times the photographs'.
        # Penalising each pixel's own excess over the frame's target gives a dense gradient and
        # is still phase-free -- the target is a single number, so no direction can put its
        # pattern anywhere in particular.
        # Two-sided where the representation can answer, one-sided where it cannot.
        #
        # Purely one-sided says "never more texture than the photograph" and nothing says
        # "have any", so the peel decayed to a smooth ball; so the bands the shell can actually
        # reach are matched from both directions and only the rest is merely capped.
        #
        # Where that line falls is a fact about the shell and was worth measuring rather than
        # assuming. Colour every primitive with independent noise and render: the result is the
        # finest thing this representation can put on the screen, and it peaks at one to two
        # pixels with 0.0147 of energy there against the photographs' 0.0088. The shell is not
        # the limit. `reach` was set to 4 on the belief that it was, which left everything below
        # four pixels merely capped -- never asked for -- and the trained peel came out at half
        # the photographs' energy at 0.5-2px and 1.3 times theirs at 2-8px. Energy in the wrong
        # octaves is what "coarse marbling instead of dimples" is.
        stat = rendering.new_zeros(())
        pr, pg = rendering, ground_truth
        sel = m.reshape(-1) > 0.5 if EXT_BAND_SW else None
        for s in sig:                                   # ascending, so each band is an octave
            br, bg = _blur(rendering, s), _blur(ground_truth, s)
            if EXT_BAND_SW:
                # Match the band's whole distribution, not its mean.
                #
                # A mean magnitude per octave is one number, and a great many textures share
                # it: with the octaves finally placed correctly the peel came out as a
                # connected labyrinth of veins where the photograph has separate round pits,
                # and both have the same energy at every scale. Heitz et al. make exactly this
                # point about the Gram matrix in neural texture synthesis and replace it with a
                # sliced Wasserstein distance, which is zero only when the distributions agree.
                #
                # Here the "slices" are already given -- the octaves and the three channels --
                # so each one is a 1-D problem, and 1-D Wasserstein is sorting: pair the k-th
                # largest response in the render with the k-th largest in the photograph. Pits
                # are a few strong extremes over a quiet field and a labyrinth is many middling
                # ones, so their sorted profiles differ even where their means do not. Sorting
                # discards position, so this stays as phase-free as the mean was, which is the
                # property that keeps a direction from imposing its own pattern.
                for c in range(3):
                    a = (pr - br)[c].reshape(-1)[sel]
                    b = (pg - bg)[c].reshape(-1)[sel]
                    stat = stat + F.mse_loss(torch.sort(a).values,
                                             torch.sort(b).values.detach()) / 3.0
            else:
                tgt = ((pg - bg).abs() * m).sum() / den / 3.0
                e = (pr - br).abs()
                d = e - tgt if s >= EXT_BAND_REACH else F.relu(e - tgt)
                stat = stat + (d ** 2 * m).sum() / den / 3.0
            pr, pg = br, bg
        return loss + w_stat * stat

    def _inner_mask(rendering, ground_truth, erode=10):
        """Where both silhouettes agree, minus a margin. No gradient flows through it."""
        with torch.no_grad():
            fg = ((rendering.mean(0) < 0.95) & (ground_truth.mean(0) < 0.95)).float()
            if erode > 0:
                k = 2 * erode + 1
                fg = -F.max_pool2d(-fg.reshape(1, 1, *fg.shape), k, 1, erode).reshape(fg.shape)
            return fg.unsqueeze(0)

    def get_hf_loss(rendering, ground_truth):
        """Match spatial gradients, which the mean colour cannot satisfy.

        SSIM and MSE are both dominated by the low frequencies here: the interior render
        is a little too pale (residual gaps let the white background through) so the bulk
        of the gradient asks it to be more orange, and the thin white membranes -- a small
        share of the pixels -- never get attended to. Measured over three reference
        regimes, interior saturation rose 0.810 -> 0.850 while white voxels stayed at
        34/27/37. Differencing first removes the mean entirely, so this term can only be
        satisfied by putting structure in the right place.
        """
        r = rendering.unsqueeze(0) if rendering.dim() == 3 else rendering
        g = ground_truth.unsqueeze(0) if ground_truth.dim() == 3 else ground_truth
        def grads(t):
            return (t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :])
        rx, ry = grads(r)
        gx, gy = grads(g)
        return F.mse_loss(rx, gx) + F.mse_loss(ry, gy)

    def get_quant_loss(rendering, ground_truth, mask=None):
        """Match the colour distribution over the region, not the colour at each pixel.

        Every per-pixel loss here is minimised, under uncertainty about *where* a feature is, by
        the mean of the possibilities. That is the whole failure on the watermelon. Its seeds sit
        at different angles in every reference and its pith ring at slightly different radii, so
        the per-pixel optimum is flesh-coloured where a seed might be and a red-white mixture
        across the band -- and that is exactly what the trained model contains: not one cell in
        the pith band is white (the released model has 1.5% of them) and not one is dark enough
        to read as a seed, while the band's *mean* colour and saturation match the released
        model's to three decimals. The information is in the target; the loss is discarding it.

        Sorting is the cheapest fix that keeps it. Compare the sorted pixel values of render and
        target per channel: a 1-D Wasserstein distance, zero only when the two distributions
        agree. A uniform mixture cannot satisfy it -- its sorted profile is flat where the
        target's has a white tail and a dark tail -- so white and dark pixels have to exist. And
        sorting discards position, so nothing asks for a seed at the angle this particular
        photograph happened to have one, which is the demand that produced the averaging in the
        first place.
        """
        r = rendering.reshape(3, -1)
        g = ground_truth.reshape(3, -1)
        if mask is not None:
            m = mask.reshape(-1) > 0.5
            if int(m.sum()) < 64:
                return rendering.new_zeros(())
            r, g = r[:, m], g[:, m]
        return F.mse_loss(torch.sort(r, dim=1).values,
                          torch.sort(g, dim=1).values.detach())

    def get_patch_loss(rendering, ground_truth, n=None, size=None, stat_w=None):
        """Score the section in pieces instead of all at once.

        The whole-frame loss is `0.7 (1 - SSIM) + 0.3 MSE` over 512 x 512, and that number is
        dominated by the things that occupy most of the frame: the silhouette, the mean hue,
        the radial layout. A section can match all of them and still be locally flat -- flesh
        with no grain, a pith ring with no fibre -- because the area where the grain would be
        contributes a few percent of the total and moves the number less than a pixel of
        silhouette does. Super-resolution training has the same problem and the same answer:
        supervise crops, so each crop has to stand on its own.

        Crops are drawn from the foreground only. A crop of background is two constant images
        and scores perfectly, so including them dilutes the gradient in proportion to how much
        of the frame the object does not fill.

        `stat_w` adds the band term on each crop: low frequencies compared where they sit, and
        finer octaves compared only in quantity. That is the part that asks for texture without
        asking for it in a particular place, which matters here because the reference is a
        photograph of a different specimen -- its fibres are not ours, and demanding them
        per-pixel is what produces grain rather than structure.
        """
        n = SEC_PATCH_N if n is None else n
        size = SEC_PATCH if size is None else size
        stat_w = SEC_PATCH_STAT if stat_w is None else stat_w
        H, W = rendering.shape[-2:]
        size = min(size, H, W)
        fg = (ground_truth.min(0).values < 0.98) | (rendering.min(0).values < 0.98)
        ys, xs = fg.nonzero(as_tuple=True)
        if ys.numel() < 16:
            return (0.7 * get_ssim_loss(rendering, ground_truth)
                    + 0.3 * torch.nn.functional.mse_loss(rendering, ground_truth))
        pick = torch.randint(0, ys.numel(), (n,), device=ys.device)
        total = 0.0
        for k in range(n):
            y0 = int(ys[pick[k]]) - size // 2
            x0 = int(xs[pick[k]]) - size // 2
            y0 = max(0, min(y0, H - size))
            x0 = max(0, min(x0, W - size))
            r = rendering[:, y0:y0 + size, x0:x0 + size]
            g = ground_truth[:, y0:y0 + size, x0:x0 + size]
            total = total + 0.7 * get_ssim_loss(r, g) \
                          + 0.3 * torch.nn.functional.mse_loss(r, g)
            if stat_w > 0:
                m = ((g.min(0).values < 0.98) | (r.min(0).values < 0.98)).float()[None]
                total = total + stat_w * get_band_loss(r, g, m, w_stat=1.0,
                                                       sig=(0.5, 1.0, 2.0, 4.0))
            if SEC_QUANT > 0:
                m2 = ((g.min(0).values < 0.98) | (r.min(0).values < 0.98)).float()
                total = total + SEC_QUANT * get_quant_loss(r, g, m2)
        return total / n

    def get_ssim_loss(rendering, ground_truth):
        rendering = rendering.unsqueeze(0) 
        ground_truth = ground_truth.unsqueeze(0)
        return 1 - ssim(
            rendering,
            ground_truth, 
            data_range=1,
            size_average=True
        )

    with torch.no_grad():
        if not ALLVOXEL:
            # collapses every primitive to a point; on a lattice that is exactly the
            # periodic dot screen we are trying to avoid, so the all-voxel model keeps
            # the per-cell extent baked in by make_allvoxel.py
            gaussians._scaling.clamp_(max=-16)

    # Only load the samplers if something is going to sample.
    #
    # Under REF_PHOTO the section targets are photographs and refitting one is a resize and a
    # paste, so both pipelines sit on the card doing nothing -- about five gigabytes of it.
    # That was affordable at a million primitives and is not at three: the run reaches the
    # first horizontal plane and dies in the backward pass with 19.9 GB already held.
    # The samplers are needed unless the photograph is the target for the entire run.
    #
    # Testing REF_PHOTO alone was wrong: the photograph hands over to the released path at
    # REF_WARMUP, and past that the sampler is what makes the target, so a run with a finite
    # warmup died at iteration 60 inside `encode_prompt` on a text encoder that had been
    # dropped. What decides this is whether the handover ever happens.
    _warm = int(_os.environ.get("REF_WARMUP", "10000000"))
    _need_sd = not (_os.environ.get("REF_PHOTO", "") and _os.environ.get("REF_PHOTO_V", "")
                    and _warm >= 10000000)
    if _need_sd:
        pipe_v = StableDiffusionDepth2ImgPipeline.from_pretrained(
            SD_MODEL_VERTICAL).to("cuda:0")
        # One copy when both are the same weights, which they are by default. Two pipelines is
        # about five gigabytes and at three million primitives that is the difference between
        # fitting and not.
        pipe_h = (pipe_v if SD_MODEL_HORIZONTAL == SD_MODEL_VERTICAL else
                  StableDiffusionDepth2ImgPipeline.from_pretrained(
                      SD_MODEL_HORIZONTAL).to("cuda:0"))
        if pipe_h is pipe_v:
            print("the two section samplers are the same weights: one copy is loaded")
    else:
        # The depth estimator is still wanted -- `section_depth` conditions on it -- but it is
        # a small monocular network and the samplers around it are not. Load one pipeline,
        # keep the estimator, drop the U-Net, the VAE and the text encoder, and let both
        # branches share it: they would have loaded two copies of the same weights anyway.
        pipe_h = StableDiffusionDepth2ImgPipeline.from_pretrained(SD_MODEL_HORIZONTAL)
        pipe_h.depth_estimator = pipe_h.depth_estimator.to("cuda:0")
        for _part in ("unet", "vae", "text_encoder"):
            if getattr(pipe_h, _part, None) is not None:
                setattr(pipe_h, _part, None)
        torch.cuda.empty_cache()
        pipe_v = pipe_h
        print("section targets are photographs: only the depth estimator is on the card")

    # --- interior diagnostic -------------------------------------------------
    # Uniform init makes every interior Gaussian identical, so its colour variance
    # starts at zero. If SDS is genuinely writing texture into the interior the
    # variance must grow; if it stays flat the interior is not being updated at all.
    # Membership is geometric (radius from the centroid) so it survives densification.
    _diag_path = os.path.join(args.output_path, "interior_diag.csv")
    with open(_diag_path, "w") as _f:
        _f.write("iteration,n_interior,mean_brightness,std_brightness,black_pct,sh_std\n")

    def _log_interior(it):
        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            c = xyz.mean(0)
            r = (xyz - c).norm(dim=1)
            inner = r < 0.6 * r.max()          # interior region, index-free
            if inner.sum() == 0:
                return
            f = gaussians._features_dc.detach().squeeze(1)[inner]
            rgb = (f * 0.28209479177387814 + 0.5)
            b = rgb.clamp(0, 1).mean(dim=1)
            with open(_diag_path, "a") as _f:
                _f.write(f"{it},{int(inner.sum())},{b.mean():.6f},{b.std():.6f},"
                         f"{(rgb<=0).all(dim=1).float().mean()*100:.3f},{f.std():.6f}\n")

    # --- fixed-viewpoint progress render --------------------------------------
    # One extra rasterise every N iterations, always from the same camera so the
    # frames can be read as a time series. Renders the cut-away view (the same
    # `mask` the v-phase trains on) rather than the closed exterior.
    _prog_dir = os.path.join(args.output_path, "progress")
    os.makedirs(_prog_dir, exist_ok=True)

    def _progress_render(it, azimuth=60):
        with torch.no_grad():
            shs_p, op_p, pos_p, cov_p, so_p, om_p, sp_p = preprocess_particles(
                gaussians, pipeline, preprocessing_params, args)
            vc, oc = get_center_view_worldspace_and_observant_coordinate(
                mpm_space_viewpoint_center, mpm_space_vertical_upward_axis,
                rotation_matrices, so_p, om_p)
            cam, raw = get_camera_view(
                model_path, default_camera_index=-1,
                center_view_world_space=vc, observant_coordinates=oc,
                show_hint=False, init_azimuthm=azimuth, init_elevation=0,
                init_radius=camera_params["init_radius"], move_camera=False,
                current_frame=0, delta_a=None, delta_e=None, delta_r=None)
            cov = cov_p / (so_p * so_p)
            cov = apply_inverse_cov_rotations(cov, rotation_matrices)
            plane = generate_plane(raw, filling_params["boundary"])
            m, _ = plane_filter(plane, pos_p, raw, surf_dis=0.006, include_double=True)
            pos_w = apply_inverse_rotations(
                undotransform2origin(undoshift2center111(pos_p), so_p, om_p),
                rotation_matrices)
            rast = initialize_resterize(cam, gaussians, pipeline, background,
                                        image_height=512, image_width=512)
            colors = convert_SH(shs_p[m], cam, gaussians, pos_w[m], None)
            img, _, _, _ = rast(means3D=pos_w[m], means2D=sp_p[m], shs=None,
                                colors_precomp=colors, opacities=op_p[m],
                                scales=None, rotations=None, cov3D_precomp=cov[m])
            arr = img.permute(1, 2, 0).detach().cpu().numpy()
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) * 255
            cv2.imwrite(os.path.join(_prog_dir, f"iter_{it:05d}.png"), arr)

    _log_interior(-1)                          # state before any training
    _progress_render(0)

    for j in range(ABL_ITERS):
        density_and_prune(j)
        print(f"Starting iteration {j}")
        # One iteration's longitudinal targets, kept so the transverse pass can be
        # reconciled against them where the planes meet.
        _v_cache = []
        for i in range(DDP_RANK, N_VPLANES, DDP_WORLD):
            print(f"Starting v{i}/{N_VPLANES}")
            pipe = pipe_v
            el = 0
            torch.cuda.empty_cache()
            init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(_dec_now(gaussians), pipeline, preprocessing_params, args)
            shs_render = init_shs
            opacity_render = init_opacity
            (
                viewpoint_center_worldspace,
                observant_coordinates,
            ) = get_center_view_worldspace_and_observant_coordinate(
                mpm_space_viewpoint_center,
                mpm_space_vertical_upward_axis,
                rotation_matrices,
                scale_origin,
                original_mean_pos,
            )
            cur_camera, raw_camera = get_camera_view(
                model_path,
                default_camera_index=-1,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=False,
                # Cover the angles, and move between them.
                #
                # This was 12*i: ten planes at twelve degrees, reaching 108. A vertical cut at
                # azimuth a and at a+180 is the same plane, so the range that exists is 180
                # degrees and two fifths of it was never supervised at all. The horizontal
                # branch already jitters its depth every iteration, and its own comment says
                # why -- fixed planes can be satisfied without building a volume -- but the
                # vertical branch was left fixed, and it shows: rendered at an angle the run
                # never saw, the segments smear into vertical streaks while the supervised
                # angles look right.
                init_azimuthm=(180.0 / N_VPLANES) * (i + (random.random() - 0.5) * 2.0
                                                     * JITTER),
                init_elevation=el,
                init_radius=camera_params["init_radius"],
                move_camera=False,
                current_frame=0,
                delta_a=None,
                delta_e=None,
                delta_r=None
            )
            torch.cuda.empty_cache()
            pos = mpm_init_pos
            cov3D = mpm_init_cov
            rot = None

            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render

            plane = generate_plane(raw_camera, filling_params["boundary"])
            thickness = 0.006 # not used, can ignore
            mask, _  = plane_filter(plane, pos, raw_camera, surf_dis=thickness, include_double=True)
            mask = mask & ~gaussians.is_outer          # the outside is the exterior's                                         
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            pos_cs = pos[mask]
            shs_cs = shs[mask]
            cov3D_cs = cov3D[mask]
            opacity_cs = opacity[mask]
            init_screen_points_cs = init_screen_points[mask]
            rasterize = initialize_resterize(_cap(cur_camera), gaussians, pipeline, background_sec, image_height=ABL_RES, image_width=ABL_RES
            )
            colors_precomp_cs = convert_SH(shs_cs, cur_camera, gaussians, pos_cs, None)
            rendering, raddi, depth_r, alpha_r = rasterize(
                means3D=pos_cs,
                means2D=init_screen_points_cs,
                shs=None,
                colors_precomp=colors_precomp_cs,
                opacities=opacity_cs,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_cs
            )
            depth_map_tensor_resized = section_depth(rendering, alpha_r, pipe)
            save_img(rendering, args.output_path, 0, f"v{i}_init_")
            set_iter(j, ABL_ITERS)
            if rebuild_ref(j):
                cur_img = Image.open(os.path.join(args.output_path, f"v{i}_init_0.png"))
                set_plane(i, N_VPLANES)
                ref = one_step_sds_orange(cur_img, depth_map_tensor_resized, sds_epochs(j), pipe, "vertical")
                ref.save(os.path.join(args.output_path, f"v{i}_ref.png"))
                log_ref(args.output_path, f"v{i}", j, cur_img, ref)
            else:
                ref = Image.open(os.path.join(args.output_path, f"v{i}_ref.png"))
            if SEC_XCONS_HOLD and j > SEC_XCONS_AT and _held.get(f"v{i}") is not None:
                # Reuse what was reconciled last iteration. Re-deriving from the current render
                # would rebuild the very disagreement the reconciliation removed, which is what
                # made the measured disagreement oscillate instead of fall.
                ground_truth_tensor = _held[f"v{i}"]
            else:
                ground_truth_tensor = (section_target(rendering, np.asarray(ref), alpha_r)
                                       if SECTION_MATCH else transform(ref).to(device))
            if SEC_XCONS > 0 and j >= SEC_XCONS_AT:
                _held[f"v{i}"] = ground_truth_tensor
                # Keep this family's target with the geometry that produced it. The transverse
                # pass runs next and is where the two are reconciled; nothing is changed here,
                # because at this point there is only one family and nothing to disagree with.
                _v_cache.append((plane, cur_camera, ground_truth_tensor))
            if SNAP_INTERVAL == 1:
                save_img(ground_truth_tensor, args.output_path, 0, f"v{i}_tgt_")
            if SEC_PATCH > 0:
                total_loss = get_patch_loss(rendering, ground_truth_tensor)
            else:
                total_loss = 0.7 * get_ssim_loss(rendering, ground_truth_tensor)
                total_loss += 0.3 * torch.nn.functional.mse_loss(rendering, ground_truth_tensor)
            # The residual, per plane, every iteration. Whether a reference has been fitted is
            # the one thing the schedule needs to know -- the paper regenerates "until the
            # reconstruction losses for all slices converge below a threshold" -- and it was
            # not recorded, so a run that regenerated every ten iterations was resampling from
            # a render less than a fifth of the way to its target and nothing said so.
            with torch.no_grad():
                _res = float(torch.nn.functional.mse_loss(rendering, ground_truth_tensor))
            _RESID.append(_res)
            print(f"  fit j={j} h{i} loss={float(total_loss):.5f} resid={_res:.5f}",
                  flush=True)
            # The residual, per plane, every iteration. Whether a reference has been fitted is
            # the one thing the schedule needs to know -- the paper regenerates "until the
            # reconstruction losses for all slices converge below a threshold" -- and it was
            # not recorded, so a run that regenerated every ten iterations was resampling from
            # a render less than a fifth of the way to its target and nothing said so.
            with torch.no_grad():
                _res = float(torch.nn.functional.mse_loss(rendering, ground_truth_tensor))
            _RESID.append(_res)
            print(f"  fit j={j} v{i} loss={float(total_loss):.5f} resid={_res:.5f}",
                  flush=True)
            if HF_W > 0:
                total_loss = total_loss + HF_W * get_hf_loss(rendering, ground_truth_tensor)
            _bw(total_loss)
            output_radii = torch.zeros(pos.shape[0], dtype=torch.int32).to(device)
            output_radii[mask] = raddi
            visibility_filter = output_radii > 0
            training_step(gaussians, total_loss, mask, init_screen_points, visibility_filter)

        pipe = pipe_h
        torch.cuda.empty_cache()
        init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(_dec_now(gaussians), pipeline, preprocessing_params, args)
        shs_render = init_shs
        opacity_render = init_opacity
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )
        cur_camera, raw_camera = get_camera_view(
            model_path,
            default_camera_index=-1,
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=False,
            init_azimuthm=0,
            init_elevation=90,
            init_radius=camera_params["init_radius"],
            move_camera=False,
            current_frame=0,
            delta_a=None,
            delta_e=None,
            delta_r=None,
            # Draw the transverse camera's roll rather than fix it. Looking down the object's
            # own axis, upright is undetermined by the scene; it is also undetermined by the
            # reference, which photographs a different specimen whose segments sit at their own
            # angles. Holding a roll makes the model fit that phase into one position. Drawing
            # one per iteration marginalises over it, and what survives supervision is the part
            # of the section that does not depend on the phase -- the radial structure.
            #
            # This was discovered as a defect: at elevation 90 the look-at construction
            # normalised a zero vector, so the roll came from rounding residue and swung by up
            # to 218 degrees between iterations. The run that carried it scored 101.4 transverse
            # FID against 141.0 for the same configuration with the roll held fixed.
            roll_deg=random.uniform(0, 360) if SEC_ROLL else 0.0,
        )
        torch.cuda.empty_cache()
        pos = mpm_init_pos
        cov3D = mpm_init_cov
        rot = None
        steps = H_STEPS
        _, _, centers, avg_dis = interpolate_along_camera_direction(raw_camera, pos, steps)
        avg_dis = avg_dis.item()
        for i, c in list(enumerate(centers[H_LO:H_HI]))[DDP_RANK::DDP_WORLD]:
            # Jitter the cut depth every iteration. With sixteen fixed planes the model
            # can satisfy those sixteen images without building a volume: measured, the
            # anchor model scores +0.688 on the supervised cross-section and +0.274 on an
            # independent cut of the same object. Moving the planes each time makes the
            # supervised set cover the interior instead of sampling it.
            if JITTER > 0 and len(centers) > 1:
                _step = centers[1] - centers[0]
                c = c + _step * ((random.random() - 0.5) * 2.0 * JITTER)
            print(f"Starting h{i}/{len(centers)}")
            init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(_dec_now(gaussians), pipeline, preprocessing_params, args)
            shs_render = init_shs
            opacity_render = init_opacity
            torch.cuda.empty_cache()
            pos = mpm_init_pos
            cov3D = mpm_init_cov
            rot = None
            opacity = opacity_render
            shs = shs_render
            cov3D = cov3D / (scale_origin * scale_origin)
            plane = generate_plane_center(raw_camera, c)
            mask, mask_suf = plane_filter(plane, pos, raw_camera, surf_dis=avg_dis/2, include_double=True)
            mask = mask & ~gaussians.is_outer
            mask_suf = mask_suf & ~gaussians.is_outer
            _pos_plane = pos[mask_suf].detach()      # still in the frame the plane is stated in
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            pos_cs = pos[mask_suf]
            shs_cs = shs[mask_suf]
            cov3D_cs = cov3D[mask_suf]
            opacity_cs = opacity[mask_suf]
            init_screen_points_cs = init_screen_points[mask_suf]
            colors_precomp_cs = convert_SH(shs_cs, cur_camera, gaussians, pos_cs, None)
            rasterize = initialize_resterize(_cap(cur_camera), gaussians, pipeline, background_sec, image_height=ABL_RES, image_width=ABL_RES
            )
            rendering, raddi, _, alpha_r = rasterize(
                means3D=pos_cs,
                means2D=init_screen_points_cs,
                shs=None,
                colors_precomp=colors_precomp_cs,
                opacities=opacity_cs,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_cs
            )
            depth_map_tensor_resized = section_depth(rendering, alpha_r, pipe)
            save_img(rendering, args.output_path, 0, f"h{i}_init_")
            set_iter(j, ABL_ITERS)
            if rebuild_ref(j):
                cur_img = Image.open(os.path.join(args.output_path, f"h{i}_init_0.png"))
                set_plane(i, len(centers[H_LO:H_HI]))
                ref = one_step_sds_orange(cur_img, depth_map_tensor_resized, sds_epochs(j), pipe, "horizontal")
                ref.save(os.path.join(args.output_path, f"h{i}_ref.png"))
                log_ref(args.output_path, f"h{i}", j, cur_img, ref)
            else:
                ref = Image.open(os.path.join(args.output_path, f"h{i}_ref.png"))
            ground_truth_tensor = (section_target(rendering, np.asarray(ref), alpha_r)
                                   if SECTION_MATCH else transform(ref).to(device))
            if PHASE_ALIGN:
                ground_truth_tensor = _align_phase(rendering, ground_truth_tensor)
            if SEC_XCONS_HOLD and j > SEC_XCONS_AT and _held.get(f"h{i}") is not None:
                ground_truth_tensor = _held[f"h{i}"]
            if SEC_XCONS > 0 and j >= SEC_XCONS_AT and _v_cache:
                # Where this plane meets each longitudinal one, the two targets describe the
                # same cells and were written without reference to each other. Average them
                # along the shared line before either is used, so the model is not asked to be
                # two colours at once and does not answer with their mean over the whole face.
                _tot, _dis = 0, 0.0
                for _vp, _vc, _vt in _v_cache:
                    _n, _d = section_consistency.reconcile(
                        ground_truth_tensor, cur_camera, plane, _vt, _vc, _vp,
                        _pos_plane,
                        to_world=lambda q: apply_inverse_rotations(
                            undotransform2origin(undoshift2center111(q), scale_origin,
                                                 original_mean_pos), rotation_matrices),
                        band=float(avg_dis), weight=SEC_XCONS, mode=SEC_XCONS_MODE)
                    _tot += _n
                    _dis += _d * _n
                if SEC_XCONS_HOLD:
                    _held[f"h{i}"] = ground_truth_tensor
                if _tot and DDP_RANK == 0 and j % 10 == 0:
                    print(f"  cross-section consistency: {_tot:,} px reconciled, "
                          f"mean disagreement {_dis / _tot:.4f}")
            # The target, not the reference. `section_target` remaps the reference onto the
            # component the render actually has, so the two differ wherever the mapping did
            # any work -- and it is the target the gradient is taken against. Keeping only
            # the reference leaves the more important half of the pair unrecorded.
            if SNAP_INTERVAL == 1:
                save_img(ground_truth_tensor, args.output_path, 0, f"h{i}_tgt_")
            if SEC_PATCH > 0:
                total_loss = get_patch_loss(rendering, ground_truth_tensor)
            else:
                total_loss = 0.7 * get_ssim_loss(rendering, ground_truth_tensor)
                total_loss += 0.3 * torch.nn.functional.mse_loss(rendering, ground_truth_tensor)
            if HF_W > 0:
                total_loss = total_loss + HF_W * get_hf_loss(rendering, ground_truth_tensor)
            _bw(total_loss)
            output_radii = torch.zeros(pos.shape[0], dtype=torch.int32).to(device)
            output_radii[mask_suf] = raddi
            visibility_filter = output_radii > 0
            training_step(gaussians, total_loss, mask_suf, init_screen_points, visibility_filter)

        for ttt in range(DDP_RANK, EXT_VIEWS, DDP_WORLD):
            print(f"Starting ori{ttt}/10")
            init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(gaussians_ori, pipeline, preprocessing_params, args)
            shs_render = init_shs
            opacity_render = init_opacity
            torch.cuda.empty_cache()
            (
                viewpoint_center_worldspace,
                observant_coordinates,
            ) = get_center_view_worldspace_and_observant_coordinate(
                mpm_space_viewpoint_center,
                mpm_space_vertical_upward_axis,
                rotation_matrices,
                scale_origin,
                original_mean_pos,
            )
            cur_camera, raw_camera = get_camera_view(
                model_path,
                default_camera_index=-1,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=False,
                init_azimuthm=(_EXT_DIRS[(j * EXT_VIEWS + ttt) % len(_EXT_DIRS)][1]
                               if _EXT_DIRS else
                               EXT_CUBE[ttt][0] if EXT_CUBE else
                               (ttt * (360 // EXT_VIEWS)) if EXT_PROMPT
                               else random.randint(0, 360)),
                init_elevation=(_EXT_DIRS[(j * EXT_VIEWS + ttt) % len(_EXT_DIRS)][2]
                                if _EXT_DIRS else
                                EXT_CUBE[ttt][1] if EXT_CUBE else
                                EXT_ELEV[ttt % len(EXT_ELEV)] if EXT_PROMPT
                                else random.randint(-90, 90)),
                init_radius=camera_params["init_radius"],
                move_camera=False,
                current_frame=0,
                delta_a=None,
                delta_e=None,
                delta_r=None
            )
            pos = mpm_init_pos
            cov3D = mpm_init_cov
            rot = None
            opacity = opacity_render
            shs = shs_render
            cov3D = cov3D / (scale_origin * scale_origin)
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            colors_precomp = convert_SH(shs, cur_camera, gaussians_ori, pos, None)
            rasterize = initialize_resterize(_cap(cur_camera), gaussians_ori, pipeline, background, image_height=ABL_RES, image_width=ABL_RES
            )
            rendering_ori, _, _, _ = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D
            )
            for p in range(EXT_REPEAT if EXT_PROMPT else 1):
                init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(_dec_now(gaussians), pipeline, preprocessing_params, args)
                shs_render = init_shs
                opacity_render = init_opacity
                torch.cuda.empty_cache()
                pos = mpm_init_pos
                cov3D = mpm_init_cov
                opacity = opacity_render
                shs = shs_render
                cov3D = cov3D / (scale_origin * scale_origin)
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                colors_precomp = convert_SH(shs, cur_camera, gaussians, pos, None)
                rasterize = initialize_resterize(_cap(cur_camera), gaussians, pipeline, background, image_height=ABL_RES, image_width=ABL_RES
                )
                rendering, radii, _, _ = rasterize(
                    means3D=pos,
                    means2D=init_screen_points,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D
                )
                save_img(rendering, args.output_path, 0, f"o{ttt}_init_")
                ground_truth_tensor = _ext_gt(rendering, rendering_ori, ttt, j)
                # Structure and colour, as two terms that can be chosen independently. They
                # used to be one switch, and neither setting of it works.
                #
                # `get_hf_loss` matches the magnitude of the spatial gradient. A flat grey
                # render has almost none and the reference's peel has a lot, and the cheapest
                # way to raise a gradient magnitude is not texture but noise: with it on, the
                # exterior comes back as a grey disc covered in saturated red dots, speckle
                # 0.159 against the reference's 0.068. With it off the surface is clean
                # (0.082) and carries the peel's dimpling, but the colour does not move at
                # all -- (0.551,0.557,0.523) after sixty iterations from a start of
                # (0.509,0.512,0.499), because SSIM and MSE against a differently lit
                # reference are dominated by its shading.
                #
                # The colour term is the half that works, and it was only ever reachable
                # through the half that does not. So: SSIM and MSE for structure, the global
                # colour term for hue, and EXT_HF for the gradient term alone.
                # A view speaks for what it faces, not for what it grazes.
                #
                # Seven of the thirty-two directions have the north pole inside their cone and
                # only `up` shows the calyx there: measured on the references, the departure
                # from plain peel at the pole is 0.217 for `up` and 0.002 to 0.015 for the six
                # others. Six votes to one, and the calyx went from 0.135 at initialisation to
                # 0.017 after fifty iterations. That imbalance is not a defect in the six --
                # from 43 degrees away the calyx is a squashed sliver near the silhouette, and
                # regenerating them from a model that has it only took the nearest from 0.006
                # to 0.046 -- it is what an oblique view of a small feature looks like.
                #
                # So weight each pixel by how squarely this camera meets the surface there.
                # For a convex object the obliquity is the radius in the image: the centre of
                # the silhouette faces the camera and the rim is edge-on. `up` then owns the
                # pole and the six others still own their own centres, which is the same rule
                # the initialisation blend uses and the only part of it the loss lacked.
                # Each view supervises only the part of the surface it is nearest to.
                #
                # Weighting does not settle a disagreement between views. Seven directions see
                # the pole and six of them show plain peel, so the loss hears "no calyx" six
                # times per iteration and "calyx" once; scaling the one by six or fifteen left
                # the calyx at 0.004 and 0.012 against 0.027 unweighted, because Adam
                # normalises each parameter's step by its own gradient history and a constant
                # on a loss term mostly divides out. What does not divide out is a term that
                # is not there.
                #
                # So partition the sphere by direction and let a view speak only for its own
                # cell -- the same nearest-direction rule the initialisation blend uses, moved
                # to the supervision. Over one cycle of the rota every part of the surface is
                # supervised exactly once, by the direction that sees it squarely, and the six
                # plain-peel views never mention the pole at all.
                #
                # SSIM has no masked form, so under this the structure term is the masked MSE
                # and the global colour term carries the hue.
                _vmask = None
                if EXT_VORONOI and _EXT_DIRS and _AXES_CACHE[0] is None:
                    # Take each direction's axis from the camera the trainer actually builds for it,
                    # not from a formula. Azimuth and elevation are interpreted in the object's own
                    # frame -- its up axis comes from the physics config and the in-plane axes from
                    # `get_center_view_worldspace_and_observant_coordinate` -- so a nominal
                    # [cos e sin a, sin e, cos e cos a] is a different sphere. Built from the cameras
                    # the convention cannot disagree with itself: the masks came out the right size
                    # and in the wrong place, 0.62 R off centre, and `up` and `down` owned nothing.
                    _ctr0 = pos.mean(0).detach()
                    _ax = []
                    for _n, _az, _el in _EXT_DIRS:
                        _cm, _ = get_camera_view(
                            model_path, default_camera_index=-1,
                            center_view_world_space=viewpoint_center_worldspace,
                            observant_coordinates=observant_coordinates, show_hint=False,
                            init_azimuthm=_az, init_elevation=_el,
                            init_radius=camera_params["init_radius"], move_camera=False,
                            current_frame=0, delta_a=None, delta_e=None, delta_r=None)
                        _v = _cm.camera_center.reshape(3).to(_ctr0.device) - _ctr0
                        _ax.append(_v / _v.norm().clamp_min(1e-9))
                    _AXES_CACHE[0] = torch.stack(_ax)
                    if DDP_RANK == 0:
                        print(f"  voronoi: {len(_ax)} axes taken from their own cameras")
                if EXT_VORONOI and _EXT_DIRS and _AXES_CACHE[0] is not None:
                    _AXES = _AXES_CACHE[0]
                    _my_dir = (j * EXT_VIEWS + ttt) % len(_EXT_DIRS)
                    _obj_centre = pos.mean(0).detach()
                    _iv = torch.inverse(cur_camera.world_view_transform)
                    _ex, _ey = _iv[0, :3], _iv[1, :3]
                    _cc = cur_camera.camera_center.reshape(3)
                    _u = _cc - _obj_centre
                    _u = _u / _u.norm().clamp_min(1e-9)
                    _fg = (rendering.mean(0) < 0.97)
                    _ys, _xs = torch.nonzero(_fg, as_tuple=True)
                    if _ys.numel() > 64:
                        _cy, _cx = _ys.float().mean(), _xs.float().mean()
                        _Y = torch.arange(rendering.shape[1], device=rendering.device
                                          ).reshape(-1, 1).float() - _cy
                        _X = torch.arange(rendering.shape[2], device=rendering.device
                                          ).reshape(1, -1).float() - _cx
                        _rr = torch.hypot(_Y, _X)
                        _R = torch.quantile(_rr[_fg], 0.98).clamp_min(1.0)
                        _a = (_X / _R).clamp(-1, 1)
                        _b = (-_Y / _R).clamp(-1, 1)          # image y runs downward
                        _c = (1.0 - _a * _a - _b * _b).clamp_min(0.0).sqrt()
                        _n = (_c[..., None] * _u + _a[..., None] * _ex + _b[..., None] * _ey)
                        _n = _n / _n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                        _dots = _n @ _AXES.T                  # (H, W, ndirs)
                        # A soft share, not a hard cell.
                        #
                        # Taking the argmax removes the contradiction and adds a discontinuity:
                        # nothing in the loss couples a pixel to the one across a cell boundary,
                        # so the colour is free to step there, and it does -- the calyx came
                        # back (0.011 -> 0.039) and the silhouette picked up a polygonal
                        # patchwork, seam area 0.0% -> 9.2%. The remedy is not to go back to
                        # sharing everything; it is to let the share fall off smoothly. A
                        # softmax over the directions' alignment is 1 at a cell's centre, one
                        # half at a boundary between two, and negligible for a direction that
                        # only grazes -- so the six plain-peel views still say nothing about
                        # the pole, and neighbouring pixels are no longer supervised by
                        # unrelated terms.
                        if EXT_VORONOI_TAU > 0:
                            _sm = torch.softmax(_dots / EXT_VORONOI_TAU, dim=-1)
                            _own = _sm[..., _my_dir]
                        else:
                            _own = (_dots.argmax(-1) == _my_dir).float()
                        _vmask = _own * _fg
                        if float(_vmask.sum()) < 64:
                            _vmask = None
                        elif EXT_VORONOI_DEBUG and DDP_RANK == 0:
                            _my, _mx = torch.nonzero(_vmask > 0.5, as_tuple=True)
                            print(f"  voronoi view {ttt} dir {_EXT_DIRS[_my_dir][0]}: "
                                  f"{100*float(_vmask.sum()/_fg.sum().clamp_min(1)):.1f}% of the "
                                  f"silhouette, centroid offset "
                                  f"{float(((_my.float().mean()-_cy)**2+(_mx.float().mean()-_cx)**2).sqrt()/_R):.2f} R")
                if EXT_FACING > 0:
                    _fg = (rendering.mean(0) < 0.97)
                    _ys, _xs = torch.nonzero(_fg, as_tuple=True)
                    if _ys.numel() > 64:
                        _cy, _cx = _ys.float().mean(), _xs.float().mean()
                        _Y = torch.arange(rendering.shape[1], device=rendering.device
                                          ).reshape(-1, 1).float()
                        _X = torch.arange(rendering.shape[2], device=rendering.device
                                          ).reshape(1, -1).float()
                        _r = torch.hypot(_Y - _cy, _X - _cx)
                        _R = torch.quantile(_r[_fg], 0.98).clamp_min(1.0)
                        # cos of the angle between the normal and the view, for a sphere
                        _w = (1.0 - (_r / _R).clamp(0, 1) ** 2).clamp_min(0.0).sqrt()
                        _w = (_w ** EXT_FACING) * _fg
                        _w = _w / _w.mean().clamp_min(1e-6)
                    else:
                        _w = torch.ones_like(rendering[:1])
                else:
                    _w = None
                if EXT_BAND:
                    # The frequency split, with the partition kept for what it is good at.
                    #
                    # The Voronoi mask exists so a view owns its own share of the surface,
                    # which is the only thing that keeps a feature only one view can see. That
                    # argument is about *where* a colour goes and applies to the alignable,
                    # low-frequency half; the statistical half has no position to own. So the
                    # partition rides on the low frequencies and the texture term is global.
                    _m_in = _inner_mask(rendering, ground_truth_tensor, EXT_ERODE)
                    _m_own = _m_in if _vmask is None else (_m_in * _vmask.reshape(
                        1, *_m_in.shape[-2:]))
                    total_loss = get_band_loss(rendering, ground_truth_tensor, _m_own,
                                               w_stat=EXT_BAND_W)
                    if EXT_VORONOI_MIX > 0 and _vmask is not None:
                        total_loss = total_loss + EXT_VORONOI_MIX * get_band_loss(
                            rendering, ground_truth_tensor, _m_in, w_stat=EXT_BAND_W)
                    if EXT_SSIM_W > 0:
                        # Optimise what is being judged.
                        #
                        # The band loss matches how much texture there is at each scale and
                        # says nothing about where it sits, which is the right shape for
                        # supervision from a photograph of a different orange -- but the
                        # target set for this work is SSIM against those photographs, and
                        # thirty iterations of the band loss alone moved it the wrong way,
                        # 0.689 to 0.670. SSIM rewards agreement in local mean, variance and
                        # covariance over a window, so it is not asking for the dimples to
                        # line up either; it is asking for the large-scale structure to, which
                        # the band loss never mentions.
                        total_loss = total_loss + EXT_SSIM_W * get_ssim_loss(
                            rendering, ground_truth_tensor)
                    _fr = getattr(gaussians, "_features_rest", None)
                    if SH_BAL_W > 0 and _fr is not None and _fr.shape[1] > 0:
                        # Balance the colour across directions, globally.
                        #
                        # The directional terms are what carry a feature only one reference can
                        # see: the stem scar survives because from `up` the cell returns what
                        # `up` photographed, and in the blended mean seven other directions
                        # outvote it. Dropping them evens the shading and takes the calyx with
                        # it, so they stay.
                        #
                        # What has to go is the achromatic part of the variation. A patch that
                        # is brighter head-on and darker at an angle reads as the peel changing
                        # while the object turns, which a peel does not do -- measured on the
                        # trained model, a cell moved 0.0665 in colour between head-on and 45
                        # degrees off, where a real surface moves 0. Splitting each coefficient
                        # into its mean over the three channels and the remainder separates
                        # brightness from colour exactly, so penalise the mean and leave the
                        # remainder to carry the feature.
                        total_loss = total_loss + SH_BAL_W * (_fr.mean(2) ** 2).mean()
                elif _vmask is not None:
                    # Two terms, because the requirements are two.
                    #
                    # The masked term lets a view own its own share of the surface, which is
                    # what keeps a feature only one view can see: the calyx survives at 0.039
                    # under a hard partition against 0.011 without one, and the peel's gradient
                    # goes from 0.083 to 0.199. The masked term alone also leaves the cells
                    # uncoupled, so the colour is free to step across a boundary and does --
                    # 9.2% of the silhouette in polygonal patches. Softening the mask fixes
                    # that and undoes the first: at tau 0.12 the seams are 0.6% and the calyx
                    # is 0.002, because both are governed by the same number.
                    #
                    # A second, unmasked term separates them. It says neighbouring pixels
                    # answer to the same reference wherever two views overlap, which is the
                    # only thing the partition removed and the one thing the seams need.
                    _d2 = (rendering - ground_truth_tensor) ** 2
                    total_loss = (_vmask * _d2).sum() / _vmask.sum().clamp_min(1.0) / 3.0
                    if EXT_VORONOI_MIX > 0:
                        total_loss = total_loss + EXT_VORONOI_MIX * _d2.mean()
                elif EXT_HF:
                    total_loss = get_hf_loss(rendering, ground_truth_tensor)
                elif _w is not None:
                    # SSIM has no per-pixel form here, so the weighting rides on the MSE half
                    # and SSIM keeps the structure term global.
                    total_loss = 0.6 * get_ssim_loss(rendering, ground_truth_tensor)
                    total_loss += 0.4 * (_w * (rendering - ground_truth_tensor) ** 2).mean()
                else:
                    total_loss = 0.6 * get_ssim_loss(rendering, ground_truth_tensor)
                    total_loss += 0.4 * torch.nn.functional.mse_loss(
                        rendering, ground_truth_tensor)
                if EXT_COL_W > 0:
                    # Each mean over its own foreground. Using the reference's mask for both
                    # was wrong whenever the silhouettes differed: the reference covers 72.9%
                    # of the frame and the render 33.4%, so the render's "mean colour" was
                    # mostly white background, and the only way to pull it down to the
                    # target was to drive the object far darker than the target.
                    _fg_g = (ground_truth_tensor.mean(0) < 0.95)
                    _fg_r = (rendering.mean(0) < 0.95)
                    _c_r = (rendering * _fg_r).sum((1, 2)) / _fg_r.sum().clamp_min(1)
                    _c_g = (ground_truth_tensor * _fg_g).sum((1, 2)) / _fg_g.sum().clamp_min(1)
                    total_loss = total_loss + EXT_COL_W * torch.nn.functional.mse_loss(
                        _c_r, _c_g)
                # A named direction counts for more than a filler.
                #
                # The calyx survives no initialisation, however strongly it is put there:
                # 0.135 and 0.172 at initialisation both end at 0.027 and 0.012 after fifty
                # iterations. Seven directions have the pole in view and only `up` shows a
                # calyx, so the loss hears "plain peel" six times for every "calyx", and no
                # weighting inside a frame changes a vote taken between frames. Weighting the
                # frames does. The six named faces were each prompted for what belongs on them
                # and the twenty-six fillers all fell back to the plain-peel prompt, so this
                # is the same distinction the initialisation already makes, applied where the
                # supervision happens.
                if EXT_NAMED_LOSS_W != 1.0 and _EXT_DIRS:
                    _nm2 = _EXT_DIRS[(j * EXT_VIEWS + ttt) % len(_EXT_DIRS)][0]
                    if not _nm2.startswith(("r", "c")):
                        total_loss = total_loss * EXT_NAMED_LOSS_W
                _bw(total_loss)
                visibility_filter = radii > 0
                training_step(gaussians, total_loss, None, init_screen_points, visibility_filter)
        if PHASE_ALIGN and _gacc[1] > 0:
            # Both ranks see half the planes, so the shared template has to be shared
            # across them too, or each would align to its own half and reintroduce exactly
            # the split this removes.
            g_ = _gacc[0] / _gacc[1]
            if DDP_WORLD > 1:
                import torch.distributed as dist
                dist.all_reduce(g_, op=dist.ReduceOp.SUM)
                g_ = g_ / DDP_WORLD
            _gprof[0] = (g_ - g_.mean()) / (g_.std() + 1e-6)
            _gacc[0] = torch.zeros_like(_gacc[0]); _gacc[1] = 0
        if DDP_RANK == 0:
            # One line per iteration, measured on the field rather than on a render.
            # Every metric watched during the closed-loop runs went through a picture, and a
            # picture gets calmer as cells go transparent -- so a hollowing interior read as
            # success on all of them while three quarters of the cells fell below opacity 0.5.
            # Opacity is the one that catches it and it costs nothing: the decoder already
            # produced it.
            with torch.no_grad():
                _xyz = gaussians.get_xyz.detach()
                _r = (_xyz - _xyz.mean(0)).norm(dim=1)
                _in = _r < 0.6 * _r.max()          # same interior region _log_interior uses
                _oi = gaussians.get_opacity.detach().reshape(-1)[_in]
                _f = open(os.path.join(args.output_path, "field.csv"),
                          "a" if j else "w", buffering=1)
                if not j:
                    _f.write("iter,interior_prims,mean_opacity,frac_below_0.5,frac_below_0.1\n")
                _f.write(f"{j},{_oi.shape[0]},{_oi.mean().item():.5f},"
                         f"{(_oi < 0.5).float().mean().item():.5f},"
                         f"{(_oi < 0.1).float().mean().item():.5f}\n")
                _f.close()
        if (j % 10 == 0 or j == ABL_ITERS - 1) and DDP_RANK == 0:
            _log_interior(j)
            _progress_render(j)
        # Close the iteration: hand this iteration's mean residual to the convergence test,
        # which is what decides whether the reference has stopped paying and should be
        # regenerated. One number per iteration, over every plane the iteration touched.
        if _RESID:
            import sds_demo as _sd
            _sd.note_residual(j, float(sum(_RESID)) / len(_RESID))
            _RESID.clear()

        # A checkpoint, so an experiment on the schedule does not mean retraining what came
        # before it. Everything needed to carry on: the cells, the decoder and its features,
        # the optimiser, the iteration, and the references in force at the time -- a run
        # resumed without its references would silently continue against different targets.
        if CKPT_INTERVAL and DDP_RANK == 0 and j > 0 and j % CKPT_INTERVAL == 0:
            import shutil as _sh, glob as _g
            _cd = os.path.join(args.output_path, "ckpt", f"iter_{j:05d}")
            os.makedirs(_cd, exist_ok=True)
            gaussians.save_ply(os.path.join(_cd, "model.ply"))
            if ANCHOR and _dec is not None:
                torch.save({"state": _dec.state_dict(), "feat": _dec.feat.detach().cpu(),
                            "anchor_xyz": _dec.anchor_xyz.detach().cpu(), "K": _dec.K,
                            "iter": j}, os.path.join(_cd, "anchor.pt"))
            try:
                torch.save(gaussians.optimizer.state_dict(), os.path.join(_cd, "opt.pt"))
            except Exception as _e:
                print(f"  checkpoint j={j}: optimiser not saved ({_e})", flush=True)
            for _f in (_g.glob(os.path.join(args.output_path, "[hv]*_ref.png"))):
                _sh.copy2(_f, _cd)
            print(f"  checkpoint j={j} -> {_cd}", flush=True)

        if SNAP_INTERVAL and DDP_RANK == 0 and (j % SNAP_INTERVAL == 0
                                                or j == ABL_ITERS - 1):
            import shutil, glob as _g
            _sd = os.path.join(args.output_path, "snap", f"iter_{j:04d}")
            os.makedirs(_sd, exist_ok=True)
            for _f in (_g.glob(os.path.join(args.output_path, "[hvo]*_init_0.png"))
                       + _g.glob(os.path.join(args.output_path, "[hv]*_tgt_0.png"))
                       + _g.glob(os.path.join(args.output_path, "[hv]*_ref.png"))):
                shutil.copy(_f, _sd)
            # The ply is optional. What makes a snapshot useful for finding a defect is the
            # section images -- a few hundred kilobytes -- and at every twenty iterations the
            # plys are 135 MB each and fill the disk long before the run ends. Keep the
            # pictures often and the weights rarely; CKPT_INTERVAL writes the resumable state.
            if SNAP_PLY:
                gaussians.save_ply(os.path.join(_sd, "model.ply"))
            print(f"snapshot {j} -> {_sd}", flush=True)
        # only the final checkpoint: an anchor model decodes to 2.2M primitives and
        # each intermediate .ply is 145 MB, which the disk no longer has room for
        if j == ABL_ITERS - 1 and DDP_RANK == 0:
            print("Saving epoch")
            gaussians.save_ply(os.path.join(args.output_path, f"orange_demo_epoch_{j}.ply"))
            # Keep the decoder, not only what it decoded. The per-cell feature is the thing
            # the run actually learned -- the ply holds one colour per cell, which is that
            # feature seen through the visual head and cannot be inverted. Anything
            # downstream that wants to group cells by what they are, rather than by what
            # colour they came out, needs the feature.
            if ANCHOR and _dec is not None:
                torch.save({"state": _dec.state_dict(),
                            "feat": _dec.feat.detach().cpu(),
                            "anchor_xyz": _dec.anchor_xyz.detach().cpu(),
                            "K": _dec.K},
                           os.path.join(args.output_path, f"anchor_epoch_{j}.pt"))

