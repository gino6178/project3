#!/usr/bin/env bash
# The animation at the top of the page: three planes through an object, the pieces drawn apart.
#
#   bash code/figures/draw_cuts.sh orange out.mp4 [frames]
#
# Reads the object's conf for its physics and demo configs, exactly as run.sh does, and takes
# appearance from the trained model while topology comes from the lattice it was trained on.
# An output name ending in .mp4 writes PNG frames beside it -- there is no encoder in the
# dependency list -- and ffmpeg turns them into the file the page serves:
#
#   ffmpeg -framerate 14 -i out_frames/%04d.png -c:v libx264 -crf 20 -pix_fmt yuv420p out.mp4
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${FN_ROOT:?set FN_ROOT}
GS_ROOT=${GS_ROOT:-$ROOT/gaussian-splatting}
PY=${FN_PY:?set FN_PY}
OBJ=$1; OUT=$2; N=${3:-48}
# shellcheck disable=SC1090
source "$HERE/objects/$OBJ.conf"
export PYTHONPATH="$HERE/figures:$HERE/src:$HERE/inherited:$HERE/inherited/mpm_solver_warp:$GS_ROOT"
export FN_ROOT="$ROOT"
cd "$ROOT"

# where the object was trained, and from what: the same two the trainer used
LAT="build_$OBJ/lattice"; [ -f "build_$OBJ/skin/lattice.pt" ] && LAT="build_$OBJ/skin"
PLY=${MODEL:-$OBJ/orange_demo_epoch_$((ITERS-1)).ply}
[ -f "$PLY" ] || { echo "no model at $PLY -- set MODEL, or train it first"; exit 1; }

# The cake's conf inherits the orange's up axis, which is not the cake model's own, so its demo
# camera looks from above instead. Nothing the method uses changes.
[ "$OBJ" = cake ] && export DEMO_EL=${DEMO_EL:-78}

DEMO_NO_CAPTION=${DEMO_NO_CAPTION:-1} DEMO_SIZE=${DEMO_SIZE:-640} \
DEMO_SS=${DEMO_SS:-3} DEMO_KSAMP=${DEMO_KSAMP:-6} \
  $PY "$HERE/figures/multicut_gif.py" "$LAT" "$PLY" "$CFG" "$DEMO" "$OUT" "$N"
