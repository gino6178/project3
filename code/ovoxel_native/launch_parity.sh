#!/usr/bin/env bash
# The two mechanisms stages.sh sets that this build did not have, isolated and together.
# Baseline for all of these is r1_pin = 0.0504 (whole-frame L1, no voxel smoothing).
cd /workspace/ovoxel_native
BASE="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 \
ANCHOR=1 ANCHOR_DIM=8 ANCHOR_PREFIT=1 ANCHOR_CHUNK=262144 ABL_INTERVAL=30 ABL_GRID=16"
PATCH="SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3"

go () {  # go GPU TAG EXTRA...
  local g=$1 tag=$2; shift 2
  setsid env $BASE "$@" OUT=/workspace/ovoxel_native/$tag CUDA_VISIBLE_DEVICES=$g \
    nohup bash run.sh mvtrain.py > /workspace/ovoxel_native/$tag.log 2>&1 < /dev/null &
  disown; sleep 4
}

# SEC_PATCH alone
go 1 r1_pin_patch  ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=0 $PATCH
# VOXEL_SMOOTH alone
go 1 r1_pin_vs     ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 SEC_PATCH=0
# both: full parity with stage_train
go 2 r1_pin_full   ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 $PATCH
go 3 r2_pin_full   ROUTE=2 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 $PATCH
echo LAUNCHED_PARITY
