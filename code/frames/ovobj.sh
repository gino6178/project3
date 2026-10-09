#!/bin/bash
# One object through the PAPER's carrier with its priors trained ALONGSIDE it (they read only the
# photographs), then every lift that applies, then evaluation.   $1 object  $2 gpu
o=$1; g=$2; D=~/ov/$o; PY=~/env/bin/python; FPY=~/p3build/mc/envs/fn/bin/python
export FN_ROOT=~/p3work GS_ROOT=~/p3build/gaussian-splatting FN_PY=$FPY GPU=$g CUDA_VISIBLE_DEVICES=$g
cd ~/repro/code
( [ -f $D/u_long/model.pt ] || env FAM=long MULT=1,2 PDIR=$D/spl_long STEPS=30000 OUT=$D/u_long DEV=cuda:0 GRID=$D/grid.pt LR=5e-4 PMIX=1.0 MASKED=0 BS=4 $PY sd3d_train.py > $D/u_long.log 2>&1
  [ -f $D/p_trans/model.pt ] || env PDIR=$D/polar_spl_trans OUT=$D/p_trans DEV=cuda:0 STEPS=4000 $PY polar_train.py > $D/p_trans.log 2>&1
  if [ -f $D/planes.json ]; then cd ~/repro/frames; [ -f $D/p_cart/model.pt ] || PDIR=$D/spl_trans OUT=$D/p_cart KIND=cart STEPS=30000 MULT=1,2 DEV=cuda:0 $PY train_prior.py > $D/p_cart.log 2>&1; fi ) &
mkdir -p ~/p3work/prefilled/new
[ -f ~/p3work/prefilled/new/$o.ply ] || $PY ~/ov/make_gsply.py $D/grid.pt ~/p3work/prefilled/new/$o.ply $D/gsply.json > $D/gsply.log 2>&1
rm -rf ~/p3work/new_${o}_h ~/p3work/new_${o}_v; cp -r $D/spl_trans ~/p3work/new_${o}_h; cp -r $D/spl_long ~/p3work/new_${o}_v
DX=$(python3 -c "import json;print(round(json.load(open('$D/gsply.json'))['coarse_dx'],5))")
printf 'SRC=prefilled/new/%s.ply\nCOARSE_DX=%s\nCFG=config/orange_physics.json\nDEMO=config/sphere_demo\nREF_H=new_%s_h\nREF_V=new_%s_v\nITERS=${ITERS:-200}\nSCORE=fid\nEVAL_REF=new_%s_h\nEVAL_REF_V=new_%s_v\nPROMPT="the {view_cut} cross-sectional view of %s"\nCLIP_PROMPT="a cross-section of %s"\n' $o $DX $o $o $o $o $o $o > ~/p3paper/code/objects/$o.conf
cd ~/p3paper; for st in geometry exterior phases train; do bash code/run.sh $o $st >> $D/ov_run.log 2>&1; done
PLY=~/p3work/$o/orange_demo_epoch_199.ply; [ -f $PLY ] || { echo NO_CARRIER >> $D/ov_run.log; exit 1; }
cd ~/repro/code
[ -f $D/carrier.pt ] || { PLY=$PLY META=~/p3work/build_$o/lattice N=128 DEV=cuda:0 OUT=$D/ovgrid.pt $PY voxelize_ov.py > $D/voxov.log 2>&1; $PY ~/ov/ov2grid.py $D/grid.pt $D/gsply.json $D/ovgrid.pt $D/carrier.pt > $D/ov2grid.log 2>&1; }
wait
[ -f $D/cyl/state.pt ] || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long/model.pt CKH=$D/p_trans/model.pt OUT=$D/cyl $PY x3dcyl.py > $D/cyl.log 2>&1
if [ -f $D/planes.json ]; then
  cd ~/repro/frames; ARMS="cylinder=$D/cyl/state.pt"
  [ -f $D/cart/state.pt ] || env T0=0.3 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long/model.pt CKH=$D/p_cart/model.pt OUT=$D/cart $PY x3dcart.py > $D/cart.log 2>&1
  ARMS="$ARMS cartesian=$D/cart/state.pt"
  if [ -f $D/centerline.json ]; then
    [ -f $D/curv/state.pt ]  || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long/model.pt CKH=$D/p_trans/model.pt CENTERLINE=$D/centerline.json OUT=$D/curv $PY x3dcurv.py > $D/curv.log 2>&1
    [ -f $D/curv2/state.pt ] || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long/model.pt CKH=$D/p_trans/model.pt CENTERLINE=$D/centerline.json OUT=$D/curv2 $PY x3dcurv2.py > $D/curv2.log 2>&1
    ARMS="$ARMS curvilinear_shift=$D/curv/state.pt curvilinear=$D/curv2/state.pt"
  fi
  $PY eval_gt.py $D $ARMS > $D/eval.log 2>&1
  [ -d $D/hld_trans ] && $PY eval_cross.py $D $ARMS > $D/eval_cross.log 2>&1
else
  OBJDIR=$D $PY dsscore.py $D/carrier.pt $D/cyl/state.pt > $D/score.log 2>&1
  DEV=cuda:0 OBJDIR=$D $PY dsrun.py 2>&1 | grep -v Warning >> $D/score.log
fi
echo OBJ_DONE >> $D/ov_run.log
