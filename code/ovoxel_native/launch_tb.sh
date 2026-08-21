#!/usr/bin/env bash
# The continuous depth assignment given to the transverse family as well as the longitudinal one.
#
# r1_pin_full is the control and already exists (REF_TRANS_BLEND unset = 0). These two are that
# arm with nothing else changed:
#
#   r1_tb2   the same photographs on a common disc, still nearest-neighbour: what the re-framing
#            alone does, so an improvement in tb1 can be attributed
#   r1_tb1   the two photographs either side of the depth, mixed at the fractional part
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

go 0 r1_tb2 ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 REF_TRANS_BLEND=2 $PATCH
go 1 r1_tb1 ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 REF_TRANS_BLEND=1 $PATCH
echo LAUNCHED_TB
