#!/usr/bin/env bash
# The orange with the alignment made differentiable and optimised alongside the volume.
set -u
R=/workspace/rebuild; H=$R/project3/code
export FN_ROOT=$R/worktree GS_ROOT=$R/gaussian-splatting
export FN_PY=$R/mc/envs/fn/bin/python FN_PY_SCORE=$R/mc/envs/score/bin/python
unset PYTHONHOME LD_LIBRARY_PATH; export PYTHONPATH= MPLBACKEND=Agg
cd $FN_ROOT
NAME=${NAME:-or_dalign}
ITERS=${ITERS:-200}
cp $H/objects/orange.conf $H/objects/$NAME.conf
sed -i "1i # the orange with a differentiable Phi, to see whether the optimiser moves the target." \
   $H/objects/$NAME.conf
rm -rf $NAME
LATTICE=build_orange/lattice DIFF_ALIGN=1 DIFF_ALIGN_LR=${DIFF_ALIGN_LR:-0.01} \
  ITERS=$ITERS GPU=${GPU:-1} bash $H/run.sh $NAME train > $R/$NAME.log 2>&1
echo "${NAME}_DONE" >> $R/align.stat
