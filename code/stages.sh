#!/usr/bin/env bash
# The stages every object runs, so each object's script is only its parameters.
#
# Sourced, not executed. The caller sets OBJ, SRC, COARSE_DX, CFG, DEMO, REF_H, REF_V,
# ITERS and PROMPT, then calls the stages it needs. Each stage skips itself if its output is
# already there, so a run can be resumed by re-running the same script.
# -e as well as -u: a stage that fails must stop the run. Without it the lattice stage failed,
# training ran anyway against a lattice that was not there, the evaluator ran against a model
# that was not there, and the only traceback in the log was the last one -- which is the stage
# that looks like the problem and is not. This file's own comment warned about that shape of
# failure before the file had the guard.
set -eu
# Both settable, so the same scripts run on the remote box without edits.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${FN_ROOT:?set FN_ROOT to the directory holding prefilled/, config/ and the reference images}
GS_ROOT=${GS_ROOT:-$ROOT/gaussian-splatting}
PY=${FN_PY:?set FN_PY to the python that has torch, taichi, warp and diff_gaussian_rasterization}
export PYTHONPATH="$HERE/src:$HERE/inherited:$HERE/inherited/mpm_solver_warp:$GS_ROOT:${PYTHONPATH:-}"
export GS_ROOT
# Every stage, not only the two that used to set it. GPU=1 was reaching the trainer and the
# evaluator but not the voxeliser, so on a machine whose first card is busy the lattice stage
# went to it and died of out-of-memory -- and because each later stage skips when its input is
# missing, the run then walked to the end and failed at the last one, which is where it looks
# like the problem is.
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# Where this run's outputs go. Defaults to the object's name, so nothing changes for the runs
# the paper reports; setting it keeps a variant beside the baseline instead of on top of it,
# and the train stage begins with `rm -rf` on this directory.
#
# Resolved inside the stages, not here. This file is sourced before run.sh sets OBJ, so writing
# RUN=${RUN:-$OBJ} at the top evaluates $OBJ while it is still unset and `set -u` kills the run.
# It survived a whole afternoon because every invocation that exercised it passed RUN explicitly,
# which is exactly the path that does not evaluate the default.
cd "$ROOT"

say () { echo "=== $(date +%H:%M) $*"; }

# --- 1. the lattice ------------------------------------------------------------------------
# Everything runs on a lattice; this is the only stage that makes one, and what kind of input it
# was given is a property of the file rather than a mode anyone selects.
#
#   a directory holding lattice.pt   already a lattice, use it
#   a mesh (.obj/.stl/.glb)          mesh_to_voxel: a closed mesh states inside-or-outside and
#                                    where its surface is, so no shell recovery and no fill
#   a point cloud (.ply)             voxelize: quantise it as it is
#
# There used to be a fourth route -- recover the surface from a filled model, re-run the paper's
# fill, then quantise -- and it existed because the released watermelon's own skin covers 84.5%
# of its silhouette, so quantising it directly left holes. That was measured on a lattice with
# no skin refinement. Refined, quantising the same model directly covers 97.9%, better than the
# 97.0% the recover-and-refill route reached, with a solid core either way. The fill was the
# slowest stage and the only one that fails silently -- a hollow result trains to a white hole
# through the middle and scores FID 693 -- and it is gone.
#
# refine=2 is the method. The published models were built at refine=1, coarse and fine spacing
# equal, no two-level lattice at all, and that alone cost the watermelon 43 points of FID.
stage_lattice () {
  local dst="build_$OBJ/lattice"
  [ -f "$dst/gs_fill.ply" ] && { say "lattice exists, skipping"; return; }
  mkdir -p "build_$OBJ"
  if [ -d "$SRC" ] && [ -f "$SRC/lattice.pt" ]; then
    say "using the lattice at $SRC"
    ln -sfn "$ROOT/$SRC" "$dst"
    return
  fi
  case "$SRC" in
    *.obj|*.stl|*.glb|*.gltf)
      say "meshing $SRC to a two-level lattice"
      $PY "$HERE/src/mesh_to_voxel.py" "$SRC" "$dst" "${CELLS:-600000}" 2 ;;
    *)
      say "voxelising $SRC at refine=2, coarse dx $COARSE_DX, skin from the occupancy"
      $PY "$HERE/src/voxelize.py" "$SRC" "$dst" 2 0 "$COARSE_DX" ;;
  esac
}

# --- 2. the exterior -----------------------------------------------------------------------
# Where the outside comes from, read off the source rather than selected by a flag, the same way
# stage_lattice reads what kind of input it was given.
#
#   a .ply           the object was captured, and its appearance came in with it: the lattice
#                    already carries the exterior, quantised from the model's own primitives
#   anything else    the shape was generated, so there is no appearance anywhere in it, and the
#                    six references are projected onto the surface by their own cameras
#
# The two routes differ only in where the colour comes from. What counts as the surface is one
# function -- occupancy.surface_cells -- called here and again by the trainer that pins it, so
# neither route can end up with a different idea of where the object's outside is.
stage_exterior () {
  case "$SRC" in
    *.ply) say "exterior came in with $SRC"; return ;;
  esac
  local src="build_$OBJ/lattice" dst="build_$OBJ/skin"
  [ -n "${REFS6:-}" ] || { say "no REFS6 set for $OBJ, leaving the lattice's own exterior"; return; }
  [ -f "$dst/gs_fill.ply" ] && { say "painted shell exists, skipping"; return; }
  say "projecting the six references in $REFS6 onto the shell -> $dst"
  $PY "$HERE/src/skin_project.py" "$src" "$CFG" "$DEMO" "$REFS6" "$dst"
}

# --- 3. the references' phases ---------------------------------------------------------------
# Equation (27), solved once per object before any gradient is taken.
#
# The two families describe one object, and where a transverse plane meets a longitudinal one they
# describe one line. The disagreement on those lines is a function of the phases alone, so it can
# be minimised before training rather than left for training to average away. This solves the
# phases and the assignment together and writes them beside the references; sds_demo reads them
# and falls back to the greedy per-family alignment of (11) if they are not there.
stage_phases () {
  [ -n "${REF_H:-}" ] && [ -n "${REF_V:-}" ] || { say "no reference families for $OBJ"; return; }
  [ -d "$ROOT/$REF_H" ] && [ -d "$ROOT/$REF_V" ] || { say "references are single files, nothing to solve"; return; }
  [ -f "$ROOT/$REF_H/phase_opt.npz" ] && { say "phases already solved, skipping"; return; }
  say "solving the reference phases and assignment against the shared chords"
  $PY "$HERE/src/phaseopt.py" "$REF_H" "$REF_V" || say "phase solve failed; the greedy alignment will be used"
}

# --- 4. train ----------------------------------------------------------------------------
stage_train () {
  local RUN=${RUN:-$OBJ}
  [ -f "$RUN/orange_demo_epoch_$((ITERS-1)).ply" ] && { say "trained model exists, skipping"; return; }
  # Which lattice to train on. The default is the one stage_lattice built; setting LATTICE to a
  # painted shell trains the interior against the cross-sections while SHELL_PIN holds the skin
  # at what the six views put there, so the exterior owes nothing to this run. It is an
  # environment variable rather than an object parameter because it is a property of the
  # experiment, not of the object -- all three take it the same way.
  # The painted shell when stage_exterior made one, the lattice itself otherwise. Both
  # carry the exterior on their surface cells; only the route that put it there differs.
  local LAT=${LATTICE:-}
  [ -n "$LAT" ] || { [ -f "build_$OBJ/skin/gs_fill.ply" ] && LAT="build_$OBJ/skin" || LAT="build_$OBJ/lattice"; }
  say "training $ITERS iterations on $LAT -> $RUN"
  rm -rf "$RUN"; mkdir -p "$RUN"
  env SDS_PROMPT="$PROMPT" \
    ANCHOR=1 ANCHOR_K=1 ANCHOR_DIM=8 ANCHOR_SPLIT=1 ANCHOR_PREFIT=1 SECTION_MATCH=1 \
    SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3 \
    ANCHOR_CHUNK=262144 OPACITY_FREEZE=1.0 EXT_FIT_DISC=1 VOXEL_SMOOTH=1 \
    EXT_VIEWS=${EXT_VIEWS:-0} SEC_SKIP_OUTER=0.10 SHELL_PIN=${SHELL_PIN:-1} \
    POS_FREEZE=${POS_FREEZE:-1} \
    EXT_DIRS="${EXT_DIRS:-}" \
    REF_PHOTO="${REF_PHOTO-$REF_H}" REF_PHOTO_V="${REF_PHOTO_V-$REF_V}" \
    REF_INTERVAL=1 REF_LOG=1 \
    REF_WARMUP="${REF_WARMUP:-10000000}" REF_REFINE_INTERVAL=1 REF_STRENGTH=0.25 \
    REF_CONV_TAU="${REF_CONV_TAU:-0}" \
    ABL_RES=512 ABL_ITERS=$ITERS ABL_INTERVAL=30 ABL_GRID=16 JITTER=0.5 \
    SNAP_INTERVAL=20 SNAP_PLY=0 CKPT_INTERVAL=20 DARK_W=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    $PY -u "$HERE/src/train_voxel.py" --model_path "$DEMO" --output_path "$RUN" \
      --physics_config "$CFG" --gs_path "$LAT/gs_fill.ply" \
      --gs_ori_path "$LAT/gs_fill.ply" --train > "log_$RUN.log" 2>&1
}

# --- 5. evaluate -------------------------------------------------------------------------
# Held-out cuts at depths training never sampled, scored against the photographs. The renderer
# loads the model at the spherical-harmonic degree the file actually carries -- the usual
# loader discards the higher bands, which only our models have, and that alone moved the
# watermelon's FID by 17 points in the wrong direction.
stage_eval () {
  local RUN=${RUN:-$OBJ}
  say "rendering held-out cuts and scoring"
  rm -rf "eval_$RUN"
  HELDOUT_BAND=0.30,0.70 FULL_SH=1 \
    $PY "$HERE/src/random_cuts.py" "$RUN/orange_demo_epoch_$((ITERS-1)).ply" \
       "$CFG" "$DEMO" "eval_$RUN" 12
  # Score against EVAL_REF, which every object here now points at the same photographs training
  # was shown -- so these are fits, not held-out scores, and evaluate/README.md says so per
  # object rather than leaving it to be assumed. The variable stays separate because the
  # distinction is real and was once used: when the watermelon trained against one shared
  # reference, FID here would have been computed against a single image. KID refused outright,
  # which is how that was found, and FID would have returned a number.
  $PY "$HERE/src/fid_eval.py" "${EVAL_REF:-$REF_H}" "eval_$RUN"/rh*_init_0.png
  $PY "$HERE/src/clip_eval.py" "$CLIP_PROMPT" "eval_$RUN"/rh*_init_0.png
}

# --- 5b. evaluate, when there is no distribution to compare against -----------------------
# The doughnut has one reference image per family, and FID and KID both need more than that.
# What it is here to show is not appearance but that the topology survives the whole pipeline,
# so the score is what a section still *is*: every held-out transverse cut of a torus should
# read as an annulus.
stage_eval_topology () {
  local RUN=${RUN:-$OBJ}
  say "rendering held-out cuts and checking each still reads as an annulus"
  rm -rf "eval_$RUN"
  HELDOUT_BAND=0.30,0.70 FULL_SH=1 \
    $PY "$HERE/src/random_cuts.py" "$RUN/orange_demo_epoch_$((ITERS-1)).ply" \
       "$CFG" "$DEMO" "eval_$RUN" 12
  $PY -c "
import sys, glob, cv2, numpy as np
sys.path.insert(0,'$HERE/src'); import section_match as sm
h=[sm._holes(np.abs(cv2.imread(p)[:,:,::-1].astype(np.float32)/255.-1).max(2)>0.06)
   for p in sorted(glob.glob('eval_$RUN/rh*_init_0.png'))]
print(f'  transverse sections reporting a hole: {sum(1 for x in h if x>0)}/{len(h)}')"
}
