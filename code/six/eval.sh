#!/usr/bin/env bash
# Step 4: render both arms for every object and score them, as one batch.
#
#   bash six/eval.sh                  # all six
#   bash six/eval.sh cake             # just one
#   SUFFIX=_w bash six/eval.sh        # score <obj>_w/ instead of <obj>/
#
# Both arms are re-rendered every time, not only ours. The baseline is scored against the same
# EVAL_REF, so if the references changed its old number is not comparable to anything measured now.
# DreamSim drifts by up to 0.006 between render batches, which is why the whole table is one run.
#
# HELDOUT_BAND=0.30,0.70 holds out the plane, not the depth: training covers f in [0.146, 0.813]
# continuously at JITTER=0.5. See six/README.md.
set -u
cd "${FN_ROOT:-/workspace/fn_voxel}"

OBJS=${*:-"orange watermelon apple pomegranate bread cake"}
SUFFIX=${SUFFIX:-}
NCUT=${NCUT:-40}
E="env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME FN_ROOT=$PWD"
P=/workspace/fn_remote/venv/bin/python
A=/usr/bin/python3

for obj in $OBJS; do
  conf="method/objects/$obj.conf"
  [ -f "$conf" ] || continue
  # shellcheck disable=SC1090
  . "$conf"
  model="${obj}${SUFFIX}/orange_demo_epoch_$(( ${ITERS:-200} - 1 )).ply"
  [ -f "$model" ] || { echo "== $obj: not trained"; continue; }
  rm -rf "evalw_$obj" "evalw_${obj}_base"
  for pair in "$model:evalw_$obj" "$SRC:evalw_${obj}_base"; do
    $E HELDOUT_BAND=0.30,0.70 FULL_SH=1 CUDA_VISIBLE_DEVICES="${GPU:-0}" $P \
       method/common/eval/random_cuts.py "${pair%%:*}" "$CFG" "$DEMO" "${pair##*:}" "$NCUT" \
       >/dev/null 2>&1
  done
  no=$(ls "evalw_$obj"/rh*_init_0.png 2>/dev/null | wc -l)
  nb=$(ls "evalw_${obj}_base"/rh*_init_0.png 2>/dev/null | wc -l)
  [ "$no" -gt 0 ] && [ "$no" = "$nb" ] || { echo "== $obj: rendered $no ours, $nb released -- skipped"; continue; }
  echo "== $obj  ($no transverse cuts per arm, references $EVAL_REF)"
  $E CUDA_VISIBLE_DEVICES="${GPU:-0}" $A method/common/eval/realism.py "$EVAL_REF" \
     ours="evalw_$obj" released="evalw_${obj}_base" 2>/dev/null \
     | grep -E "ours|released|photographs"
done
