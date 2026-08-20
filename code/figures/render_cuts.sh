#!/usr/bin/env bash
# The three cut animations, seen from below the horizon.
#
# The released orange carries its stem scar -- 575 near-black primitives at the top of the
# fruit -- and at the coarse spacing it quantises to a khaki cap that reads as a defect rather
# than as the fruit. Looking up at the object rather than down puts the top pole behind the
# body. Rotating the azimuth instead does not: at 180 degrees the scar is still on the skyline
# and a shading lobe joins it, which measured worse than the view it replaced.
set -u
R=/workspace/rebuild; H=$R/project3/code
export FN_ROOT=$R/worktree GS_ROOT=$R/gaussian-splatting
unset PYTHONHOME LD_LIBRARY_PATH; export PYTHONPATH= MPLBACKEND=Agg
P=$R/mc/envs/fn/bin/python
export PYTHONPATH=$H/src:$H/figures:$H/inherited:$H/inherited/mpm_solver_warp:$GS_ROOT
export CUDA_VISIBLE_DEVICES=2
# Each cell carries its own decoded colour and neighbouring cells are uncorrelated, so the face
# is white noise at the cell scale. At two samples per output pixel that noise survives the
# downsample and reads as grain laid over the fruit; measured, going to eight takes the
# high-frequency residual from 0.0238 to 0.0174.
export DEMO_SS=8
cd $FN_ROOT
for Q in 1 3 5; do
  echo "===== $Q cuts"
  DEMO_AZ=0 DEMO_EL=-15 CUT_Q=$Q $P $H/figures/multicut_gif.py build_orange/lattice \
     orange/orange_demo_epoch_199.ply config/orange_physics.json config/orange_demo \
     $R/newfigs/cutsc$Q.gif 48 360 2>&1 | tail -3
done
echo GIFS2_DONE
