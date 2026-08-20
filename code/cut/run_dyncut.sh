#!/usr/bin/env bash
# Elasticity and cutting in one demo: the object falls under gravity onto a floor, a plane
# arrives while it is moving, the pieces are discovered by labelling the lattice rather than
# declared, and each piece becomes its own solver carrying its particles' current state.
# The material is the two-rule field -- stiff peel, graded interior -- so the pieces do not
# deform like one substance.
set -u
R=/workspace/rebuild; H=$R/project3/code
export FN_ROOT=$R/worktree GS_ROOT=$R/gaussian-splatting
unset PYTHONHOME LD_LIBRARY_PATH; export PYTHONPATH= MPLBACKEND=Agg
export PYTHONPATH=$H/src:$H/cut:$H/figures:$H/inherited:$H/inherited/mpm_solver_warp:$GS_ROOT
export CUDA_VISIBLE_DEVICES=1
export MATERIAL=orange CELL_LEVEL=$FN_ROOT/build_orange/lattice/cell_level.pt
rm -rf $R/newfigs/dyncut
cd $FN_ROOT
$R/mc/envs/fn/bin/python $H/cut/mkflag.py orange/orange_demo_epoch_199.ply $R/newfigs/latflag.pt
$R/mc/envs/fn/bin/python $H/cut/dynamic_cut.py \
   orange/orange_demo_epoch_199.ply $R/newfigs/dyncut 210 $R/newfigs/latflag.pt 5.0e-5 180
