#!/bin/bash
# One generated object through both frames, on one GPU.  $1 = object directory (grid.pt, spl_*,
# polar_spl_trans, meta.json).  The carrier is fitted once and shared; the longitudinal prior is
# trained once and shared; each frame gets its own transverse prior (polar strips for the
# cylinder at the paper's 4k, canonical squares for the Cartesian at the longitudinal recipe's 30k).
set -eu
D=$(cd "$1" && pwd); H=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}; DEV=${DEV:-cuda:0}
cd "$H"
[ -f $D/carrier.pt ] || GRID=$D/grid.pt OBJDIR=$D OUT=$D/carrier.pt DEV=$DEV $PY carrier.py > $D/carrier.log 2>&1
[ -f $D/p_long/model.pt ]  || PDIR=$D/spl_long        OUT=$D/p_long  KIND=long  STEPS=30000 MULT=1,2   DEV=$DEV $PY train_prior.py > $D/p_long.log 2>&1 &
[ -f $D/p_cart/model.pt ]  || PDIR=$D/spl_trans       OUT=$D/p_cart  KIND=cart  STEPS=30000 MULT=1,2   DEV=$DEV $PY train_prior.py > $D/p_cart.log 2>&1 &
[ -f $D/p_polar/model.pt ] || PDIR=$D/polar_spl_trans OUT=$D/p_polar KIND=polar STEPS=4000  MULT=1,2,4 DEV=$DEV $PY train_prior.py > $D/p_polar.log 2>&1 &
wait
[ -f $D/cyl/state.pt ]  || env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=$DEV GRID=$D/carrier.pt CKV=$D/p_long/model.pt CKH=$D/p_polar/model.pt OUT=$D/cyl  $PY ../slicefill/x3dcyl.py > $D/cyl.log 2>&1
[ -f $D/cart/state.pt ] || env T0=0.3 NSTEP=100 DEV=$DEV GRID=$D/carrier.pt CKV=$D/p_long/model.pt CKH=$D/p_cart/model.pt OUT=$D/cart $PY x3dcart.py > $D/cart.log 2>&1
echo OBJ_DONE > $D/DONE
