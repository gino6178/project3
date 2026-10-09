#!/bin/bash
# One generated object through the paper's program on the GPU machine (p3-train-4g), end to end.
#   bash remote_object.sh <object> <gpu>
# Expects ~/ov/<object>/ from local_object.sh: grid.pt, meta.json, spl_*/hld_*, polar_spl_trans,
# and for a curved object centerline.json + spl_long_straight.  Every stage skips itself when its output exists.
#
# Lessons built in, each of which cost a run:
#   - the carrier is converted at the paper's 128^3 (voxelize_ov) and resampled to the object's grid by
#     ov2grid: converted finer than the lattice's interior cells, every other voxel stays empty (white)
#   - the priors read only photographs, so they train alongside the carrier, one at a time per GPU
#   - a curved object's lengthwise prior for the curvilinear lattices is trained on straightened photographs
set -u
o=$1; G=$2; D=~/ov/$o; PY=~/env/bin/python; FPY=~/p3build/mc/envs/fn/bin/python; FR=~/repro/frames
export FN_ROOT=~/p3work GS_ROOT=~/p3build/gaussian-splatting FN_PY=$FPY GPU=$G CUDA_VISIBLE_DEVICES=$G
KIND=$(python3 -c "import json;print(json.load(open('$D/meta.json')).get('kind','revolve'))")
CURVED=0; [ -f $D/centerline.json ] && [ -d $D/spl_long_straight ] && CURVED=1
say(){ echo "$(date +%H:%M) $o: $*" >> $D/pipeline.log; }
say "start (kind $KIND, curved $CURVED, gpu $G)"
# ---- the priors, alongside the carrier ----
cd ~/repro/code
( [ -f $D/u_long/model.pt ]  || env FAM=long MULT=1,2 PDIR=$D/spl_long STEPS=30000 OUT=$D/u_long DEV=cuda:0 GRID=$D/grid.pt LR=5e-4 PMIX=1.0 MASKED=0 BS=4 $PY sd3d_train.py > $D/u_long.log 2>&1
  [ $CURVED = 1 ] && { [ -f $D/u_long_s/model.pt ] || env FAM=long MULT=1,2 PDIR=$D/spl_long_straight STEPS=30000 OUT=$D/u_long_s DEV=cuda:0 GRID=$D/grid.pt LR=5e-4 PMIX=1.0 MASKED=0 BS=4 $PY sd3d_train.py > $D/u_long_s.log 2>&1; }
  [ -f $D/p_trans/model.pt ] || env PDIR=$D/polar_spl_trans OUT=$D/p_trans DEV=cuda:0 STEPS=4000 $PY polar_train.py > $D/p_trans.log 2>&1
  cd $FR; [ -f $D/p_cart/model.pt ] || PDIR=$D/spl_trans OUT=$D/p_cart KIND=cart STEPS=30000 MULT=1,2 DEV=cuda:0 $PY train_prior.py > $D/p_cart.log 2>&1 ) &
# ---- the carrier: the paper's run.sh, first route ----
mkdir -p ~/p3work/prefilled/new
[ -f ~/p3work/prefilled/new/$o.ply ] || $PY $FR/make_gsply.py $D/grid.pt ~/p3work/prefilled/new/$o.ply $D/gsply.json > $D/gsply.log 2>&1
rm -rf ~/p3work/new_${o}_h ~/p3work/new_${o}_v; cp -r $D/spl_trans ~/p3work/new_${o}_h; cp -r $D/spl_long ~/p3work/new_${o}_v
DX=$(python3 -c "import json;print(round(json.load(open('$D/gsply.json'))['coarse_dx'],5))")
printf 'SRC=prefilled/new/%s.ply\nCOARSE_DX=%s\nCFG=config/orange_physics.json\nDEMO=config/sphere_demo\nREF_H=new_%s_h\nREF_V=new_%s_v\nITERS=${ITERS:-200}\nSCORE=fid\nEVAL_REF=new_%s_h\nEVAL_REF_V=new_%s_v\nPROMPT="the {view_cut} cross-sectional view of %s"\nCLIP_PROMPT="a cross-section of %s"\n' $o $DX $o $o $o $o $o $o > ~/p3paper/code/objects/$o.conf
cd ~/p3paper; for st in geometry exterior phases train; do bash code/run.sh $o $st >> $D/ov_run.log 2>&1; done
PLY=~/p3work/$o/orange_demo_epoch_199.ply; [ -f $PLY ] || { say "NO CARRIER"; exit 1; }
say "carrier trained"
cd ~/repro/code
[ -f $D/carrier.pt ] || { PLY=$PLY META=~/p3work/build_$o/lattice N=128 DEV=cuda:0 OUT=$D/ovgrid.pt $PY voxelize_ov.py > $D/voxov.log 2>&1
                          $PY $FR/ov2grid.py $D/grid.pt $D/gsply.json $D/ovgrid.pt $D/carrier.pt > $D/ov2grid.log 2>&1; }
wait; say "priors trained"
# ---- the lifts ----
[ -f $D/cyl/state.pt ] || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long/model.pt CKH=$D/p_trans/model.pt OUT=$D/cyl $PY x3dcyl.py > $D/cyl.log 2>&1
cd $FR
[ -f $D/cart/state.pt ] || env T0=0.3 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long/model.pt CKH=$D/p_cart/model.pt OUT=$D/cart $PY x3dcart.py > $D/cart.log 2>&1
ARMS="cylinder=$D/cyl/state.pt cartesian=$D/cart/state.pt"; SHOW=cyl; [ $KIND = box ] && SHOW=cart
if [ $CURVED = 1 ]; then
  [ -f $D/curv/state.pt ]  || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long_s/model.pt CKH=$D/p_trans/model.pt CENTERLINE=$D/centerline.json OUT=$D/curv  $PY x3dcurv.py  > $D/curv.log 2>&1
  [ -f $D/curv2/state.pt ] || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=cuda:0 GRID=$D/carrier.pt CKV=$D/u_long_s/model.pt CKH=$D/p_trans/model.pt CENTERLINE=$D/centerline.json OUT=$D/curv2 $PY x3dcurv2.py > $D/curv2.log 2>&1
  ARMS="$ARMS curvilinear_shift=$D/curv/state.pt curvilinear=$D/curv2/state.pt"; SHOW=curv2
fi
say "lifts done"
# ---- held-out DreamSim: the paper's flat protocol; a curved object also cut along its centreline ----
cd ~/repro/code
[ -f $D/score.log ] || { STATES=""; for a in $ARMS; do STATES="$STATES ${a#*=}"; done
  OBJDIR=$D $PY dsscore.py $D/carrier.pt $STATES > $D/score.log 2>&1; DEV=cuda:0 OBJDIR=$D $PY dsrun.py 2>&1 | grep -v Warning >> $D/score.log; }
[ $CURVED = 1 ] && [ ! -f $D/curved_scores.json ] && { cd $FR; $PY eval_curved.py $D $ARMS > $D/eval_curved.log 2>&1; }
say "scored"
# ---- the asset: baked into the O-Voxel and cut in closed form ----
mkdir -p ~/show
[ -f ~/show/${o}_$SHOW.gif ] || bash $FR/showcase.sh $o $SHOW $G > /dev/null 2>&1
say "DONE (showcase arm $SHOW)"
