#!/usr/bin/env bash
# The stages every object runs, so each object's script is only its parameters.
#
# Sourced, not executed. The caller sets OBJ, SRC, COARSE_DX, SKIN_FRAC, CFG, DEMO, REF_H, REF_V,
# ITERS and PROMPT, then calls the stages it needs. Each stage skips itself if its output is
# already there, so a run can be resumed by re-running the same script.
# -e as well as -u: a stage that fails must stop the run. Without it the lattice stage failed,
# training ran anyway against a lattice that was not there, the evaluator ran against a model
# that was not there, and the only traceback in the log was the last one -- which is the stage
# that looks like the problem and is not. This file's own comment warned about that shape of
# failure before the file had the guard.
set -eu
# Both settable, so the same scripts run on the remote box without edits.
ROOT=${FN_ROOT:-/home/gino/project/FruitNinja_clean}
PY=${FN_PY:-/home/gino/miniconda3/envs/fruitninja/bin/python}
export PYTHONPATH="$ROOT:$ROOT/gaussian-splatting:${PYTHONPATH:-}"
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
      $PY method/common/pipeline/mesh_to_voxel.py "$SRC" "$dst" "${CELLS:-600000}" 2 ;;
    *)
      say "voxelising $SRC at refine=2, coarse dx $COARSE_DX, skin from r/R $SKIN_FRAC"
      $PY method/common/pipeline/voxelize.py "$SRC" "$dst" 2 0 "$COARSE_DX" "$SKIN_FRAC" ;;
  esac
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
  local LAT=${LATTICE:-build_$OBJ/lattice}
  say "training $ITERS iterations on $LAT -> $RUN"
  rm -rf "$RUN"; mkdir -p "$RUN"
  env SDS_PROMPT="$PROMPT" \
    ANCHOR=1 ANCHOR_K=1 ANCHOR_DIM=8 ANCHOR_SPLIT=1 ANCHOR_PREFIT=1 SECTION_MATCH=1 \
    SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3 \
    ANCHOR_CHUNK=262144 OPACITY_FREEZE=1.0 EXT_FIT_DISC=1 VOXEL_SMOOTH=1 \
    EXT_VIEWS=${EXT_VIEWS:-0} SEC_SKIP_OUTER=0.10 SHELL_PIN=${SHELL_PIN:-1} \
    EXT_DIRS="${EXT_DIRS:-}" \
    REF_PHOTO="$REF_H" REF_PHOTO_V="$REF_V" REF_INTERVAL=1 REF_LOG=1 \
    REF_WARMUP=10000000 REF_REFINE_INTERVAL=1 REF_STRENGTH=0.25 REF_CONV_TAU=0 \
    ABL_RES=512 ABL_ITERS=$ITERS ABL_INTERVAL=30 ABL_GRID=16 JITTER=0.5 \
    SNAP_INTERVAL=20 SNAP_PLY=0 CKPT_INTERVAL=20 DARK_W=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    $PY -u train_voxel.py --model_path "$DEMO" --output_path "$RUN" \
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
    $PY method/common/eval/random_cuts.py "$RUN/orange_demo_epoch_$((ITERS-1)).ply" \
       "$CFG" "$DEMO" "eval_$RUN" 12
  # Score against the photographs, not against what training was shown. They are different
  # sets and conflating them is not a small error: the watermelon trains against one shared
  # reference, so FID would have been computed against a single image -- KID refuses outright,
  # which is how this was found, and FID would have returned a number.
  $PY fid_eval.py "${EVAL_REF:-$REF_H}" "eval_$RUN"/rh*_init_0.png
  $PY clip_eval.py "$CLIP_PROMPT" "eval_$RUN"/rh*_init_0.png
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
    $PY method/common/eval/random_cuts.py "$RUN/orange_demo_epoch_$((ITERS-1)).ply" \
       "$CFG" "$DEMO" "eval_$RUN" 12
  $PY -c "
import sys, glob, cv2, numpy as np
sys.path.insert(0,'$ROOT'); import section_match as sm
h=[sm._holes(np.abs(cv2.imread(p)[:,:,::-1].astype(np.float32)/255.-1).max(2)>0.06)
   for p in sorted(glob.glob('eval_$RUN/rh*_init_0.png'))]
print(f'  transverse sections reporting a hole: {sum(1 for x in h if x>0)}/{len(h)}')"
}
