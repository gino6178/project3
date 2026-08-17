#!/usr/bin/env bash
# Step 3: train the six objects, one per GPU, and wait for all of them.
#
#   bash six/train.sh                 # all six that have a conf and a model
#   bash six/train.sh cake apple      # just these
#   SUFFIX=_w bash six/train.sh       # write to <obj>_w/ instead of <obj>/
#
# The lattice is not rebuilt: background changes do not touch geometry, so an existing
# build_<obj>/lattice is reused. Delete it to force step 3 to rebuild.
set -u
cd "${FN_ROOT:-/workspace/fn_voxel}"

OBJS=${*:-"orange watermelon apple pomegranate bread cake"}
SUFFIX=${SUFFIX:-}
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
g=0

for obj in $OBJS; do
  conf="method/objects/$obj.conf"
  [ -f "$conf" ] || { echo "== $obj: no conf, skipped"; continue; }
  # shellcheck disable=SC1090
  ( . "$conf"; [ -f "$SRC" ] ) || { echo "== $obj: no released model, skipped"; continue; }
  run="${obj}${SUFFIX}"
  echo "== $obj -> $run on gpu $g"
  nohup env -u LD_LIBRARY_PATH -u PYTHONHOME -u PYTHONPATH \
    RUN="$run" LATTICE="build_$obj/lattice" ITERS="${ITERS:-200}" GPU="$g" \
    FN_ROOT="$PWD" FN_PY=/workspace/fn_remote/venv/bin/python \
    bash method/run.sh "$obj" train > "launch_$run.log" 2>&1 &
  g=$(( (g + 1) % NGPU ))
done

echo "waiting"
wait
for obj in $OBJS; do
  run="${obj}${SUFFIX}"
  if [ -f "$run/orange_demo_epoch_$(( ${ITERS:-200} - 1 )).ply" ]; then
    echo "== $run done"
  else
    echo "== $run FAILED, tail of its log:"
    tail -5 "log_$run.log" 2>/dev/null
  fi
done
