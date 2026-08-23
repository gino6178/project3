#!/usr/bin/env bash
# One object, end to end, on the O-Voxel-native representation.
#
#     bash objrun.sh OBJ GPU [EXTRA_ENV...]
#
# OBJ is the name of an objects/<OBJ>.conf and of a build_<OBJ>/lattice beside it -- the same
# pairing the pipeline uses -- so the only thing this file decides is the order of the three
# steps. Everything else comes from that conf, which is why the orange is run through it too
# rather than kept on the paths it was developed with.
#
# The cameras are rebuilt per object because they are derived from the object's own physics
# config and its own lattice; the state is rebuilt because it is the object's occupancy.
# TAG names the output directory, so the same object can be run twice under different settings --
# `ov_<obj>` for the method and `ov0_<obj>` for the block-rule control it has to be read against.
set -u
OBJ=$1; GPU=$2; shift 2
R=/workspace/rebuild; W=/workspace/ovoxel_native
CONF=$R/project3/code/objects/$OBJ.conf
[ -f "$CONF" ] || { echo "no conf for $OBJ"; exit 1; }
eval "$(grep -E '^(CFG|REF_H|REF_V|REF_H_FLIP|REF_V_FLIP|UP_AXIS)=' "$CONF")"
# Every conf points CFG at the orange's physics file, so the polar axis every object inherits is
# the orange's. It is right for the four released models that stand the same way and 90 degrees
# out for the ones that do not, and an object that needs its own says so in its own conf.
export REF_H_FLIP="${REF_H_FLIP:-}" REF_V_FLIP="${REF_V_FLIP:-}" UP_AXIS="${UP_AXIS:-}"
# The trailing KEY=VALUE arguments are exported here, not only handed to the trainer at the end.
# They used to reach the trainer alone, so a run asking for a different plane count got the old
# cameras -- N_PLANES_TOTAL never reached mvcams and CAMS_SUFFIX never reached this script, and the
# arm silently repeated the one it was meant to differ from. Anything that is not an assignment is
# passed through untouched.
_rest=()
for _a in "$@"; do
  case "$_a" in
    [A-Za-z_]*=*) export "${_a?}" ;;
    *) _rest+=("$_a") ;;
  esac
done
set -- "${_rest[@]+"${_rest[@]}"}"

# A suffix on the camera file, so a run at a different plane count does not overwrite the one
# every other arm was trained against. _v2 is the baseline: the polar axis each conf names, a
# camera far enough back that the object is inside the frame, and the balanced plane split that
# `rs7.sh` was scoring against but not training on. Pass CAMS_SUFFIX= for the older sets.
CS="${CAMS_SUFFIX-_v2}"
LATD=build_$OBJ/lattice
[ -f "$R/worktree/$LATD/lattice.pt" ] || { echo "no lattice at $LATD"; exit 1; }

echo "== $OBJ  cfg=$CFG  refs=$REF_H,$REF_V  lattice=$LATD  gpu=$GPU"

if [ ! -f "$W/cams_$OBJ$CS.npz" ]; then
  # GS_ROOT explicitly: `scene` and `utils` are the gaussian-splatting checkout's, which sits
  # beside the worktree rather than inside it, and mvcams' default guess of FN_ROOT/gaussian-
  # splatting is wrong on this box.
  CUDA_VISIBLE_DEVICES=$GPU LAT=$LATD/gs_fill.ply CFG=$CFG OUT=$W/cams_$OBJ$CS.npz \
    N_PLANES_TOTAL="${N_PLANES_TOTAL:-26}" N_VPLANES="${N_VPLANES:-0}" \
    GS_ROOT=$R/gaussian-splatting bash "$W/runfn.sh" "$W/mvcams.py" || { echo "$OBJ: cameras failed"; exit 1; }
fi
if [ ! -f "$W/state_$OBJ.pt" ]; then
  CUDA_VISIBLE_DEVICES=$GPU ROUTE=1 LATDIR=$R/worktree/$LATD STATE=$W/state_$OBJ.pt \
    bash "$W/run.sh" "$W/build_state.py" || { echo "$OBJ: state failed"; exit 1; }
fi

# `env` and not a bare assignment prefix: an assignment that arrives through "$@" has already
# been through expansion, so the shell reads it as the command word rather than as an assignment
# and reports it as not found. Anything the caller passes therefore has to be handed to env, and
# it lands after these so it wins.
exec env CUDA_VISIBLE_DEVICES=$GPU ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 \
  STATE=$W/state_$OBJ.pt CAMS=$W/cams_$OBJ$CS.npz REF_H=$REF_H REF_V=$REF_V \
  ITERS=${ITERS:-200} ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
  SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 \
  ANCHOR=1 ANCHOR_DIM=8 ANCHOR_PREFIT=1 ANCHOR_CHUNK=262144 ABL_INTERVAL=30 ABL_GRID=16 \
  SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3 \
  OUT=$W/${TAG:-ov}_$OBJ "$@" \
  bash "$W/run.sh" "$W/mvtrain.py"
