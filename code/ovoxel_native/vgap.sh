#!/usr/bin/env bash
# Everything that says whether the two families agree, for one run, in one call.
#
# The task is to close the gap between the transverse and longitudinal families on the watermelon,
# and four numbers decide it: the four DreamSim columns, the field's variation along the axis
# against across it, the gradient's directional concentration on each family's planes, and the
# picture. Running them together keeps the DreamSim model from being loaded once per question.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-watermelon_sp}
RUNS=${RUNS:?list of runs}
cd "$W"
echo "=== scores"
PYTHONPATH= CUDA_VISIBLE_DEVICES=${GPU:-0} OBJ=$OBJ SF_RUNS=$RUNS ./env/bin/python scorefull.py \
  2>&1 | grep -vE "Using|Warning|warn|WeightNorm"
echo "=== field, along the axis against across it"
for r in ${RUNS//,/ }; do
  PYTHONPATH= CUDA_VISIBLE_DEVICES=${GPU:-0} EX_OBJS=$OBJ EX_RUN=${r%_$OBJ} \
    ./env/bin/python extrude.py 2>&1 | grep -E "ratio along" | sed "s|^|  ${r%_$OBJ}  |"
done
echo "=== gradient direction, by family"
for r in ${RUNS//,/ }; do
  echo "  ${r%_$OBJ}"
  PYTHONPATH= CUDA_VISIBLE_DEVICES=${GPU:-0} OBJ=$OBJ AN_RUN=$r ./env/bin/python aniso.py 2>&1 \
    | grep -E "photographed|held out|the photographs" | sed "s|^|  |"
done
