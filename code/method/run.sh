#!/usr/bin/env bash
# One path, any object.
#
#   bash method/run.sh orange
#   bash method/run.sh watermelon
#   bash method/run.sh doughnut
#   GPU=1 bash method/run.sh orange        # on the second card
#   bash method/run.sh orange eval         # just re-score an existing model
#
# Adding an object is writing `objects/<name>.conf`; nothing here changes. The three that exist
# differ only in that file, and every way they can differ is a parameter rather than a branch in
# the code:
#
#   SRC     where the lattice comes from, and its kind is read off the file: a directory that
#           already holds one, a mesh, or a point cloud. There is no mode to choose.
#
#   SCORE=fid         held-out cuts against the photographs. Needs more than one reference.
#   SCORE=topology    what a section should still be. The doughnut has one reference image per
#                     family, too few for FID, and what it is here to show is that every
#                     held-out transverse section still reads as an annulus.
set -eu
ROOT=${FN_ROOT:-/home/gino/project/FruitNinja_clean}
source "$ROOT/method/common/stages.sh"

OBJ=${1:?usage: run.sh OBJECT [stage]}
ONLY=${2:-all}
CONF="$ROOT/method/objects/$OBJ.conf"
[ -f "$CONF" ] || { echo "no such object: $OBJ  (have: $(ls "$ROOT/method/objects" | sed 's/.conf$//' | tr '\n' ' '))"; exit 1; }
# shellcheck disable=SC1090
source "$CONF"

say "$OBJ: src=$SRC score=$SCORE iters=$ITERS gpu=${GPU:-0}"

if [ "$ONLY" = all ] || [ "$ONLY" = geometry ]; then stage_lattice; fi
if [ "$ONLY" = all ] || [ "$ONLY" = train ]; then stage_train; fi
if [ "$ONLY" = all ] || [ "$ONLY" = eval ]; then
  case "$SCORE" in
    fid)      stage_eval ;;
    topology) stage_eval_topology ;;
    *)        echo "SCORE must be fid or topology"; exit 1 ;;
  esac
fi
say "$OBJ done"
