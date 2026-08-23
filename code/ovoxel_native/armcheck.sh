#!/usr/bin/env bash
# One arm, both numbers and the four panels.
#
#     bash armcheck.sh q_cyc_orange_sp orange_sp
#
# The seam peak is what sees the stripes -- the angular power of a transverse render at the
# longitudinal plane count, 1.0 meaning no peak -- and the four DreamSim columns are the guard
# against removing the stripes by removing the content. Neither alone decides anything.
set -u
RUN=$1; OBJ=$2; W=/workspace/ovoxel_native
echo "=================== $RUN"
PYTHONPATH= CUDA_VISIBLE_DEVICES=${GPU:-0} OBJ="$OBJ" RUN="$RUN" OUT="$W/fig_$RUN.jpg" \
  "$W/env/bin/python" "$W/pairfig.py" 2>&1 | tail -1
PYTHONPATH= CUDA_VISIBLE_DEVICES=${GPU:-0} OBJ="$OBJ" SP_RUN="$RUN" \
  "$W/env/bin/python" "$W/spokes.py" 2>&1 | grep -E "number of longitudinal|strongest"
PYTHONPATH= CUDA_VISIBLE_DEVICES=${GPU:-0} OBJ="$OBJ" SF_RUNS="$RUN" SF_CAMS=_v2 \
  "$W/env/bin/python" "$W/scorefull.py" 2>&1 | grep -E "^  $RUN|shown h"
