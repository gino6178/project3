#!/bin/bash
# The finished asset in the paper's own terms: the lifted interior baked into the trained O-Voxel PLY
# (writeback.py: interior cells' colour only, skin untouched) and cut by the paper's closed-form
# operator (figures/draw_cuts.sh -> multicut_gif.py).   $1 object  $2 arm dir (cyl|cart|curv2)  $3 gpu
o=$1; arm=$2; g=$3; D=~/ov/$o; PY=~/env/bin/python; FPY=~/p3build/mc/envs/fn/bin/python
export FN_ROOT=~/p3work GS_ROOT=~/p3build/gaussian-splatting FN_PY=$FPY CUDA_VISIBLE_DEVICES=$g
$PY ~/repro/frames/grid2ov.py $D/grid.pt $D/gsply.json $D/ovgrid.pt $D/$arm/state.pt $D/state_ov_$arm.pt > $D/show_$arm.log 2>&1
$PY ~/repro/code/writeback.py ~/p3work/$o/orange_demo_epoch_199.ply ~/p3work/build_$o/lattice $D/ovgrid.pt $D/state_ov_$arm.pt ~/p3work/$o/lifted_$arm.ply >> $D/show_$arm.log 2>&1
cd ~/p3paper && MODEL=$o/lifted_$arm.ply bash code/figures/draw_cuts.sh $o ~/show/${o}_$arm.gif 48 >> $D/show_$arm.log 2>&1
MODEL=$o/orange_demo_epoch_199.ply bash code/figures/draw_cuts.sh $o ~/show/${o}_carrier.gif 48 >> $D/show_$arm.log 2>&1
echo SHOW_DONE >> $D/show_$arm.log
