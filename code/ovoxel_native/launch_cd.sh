#!/usr/bin/env bash
# Chamfer between the two patch distributions, at two weights.
#
# The control is r1_tb1, not r1_pin_full: the depth blend is the method now, so these carry it and
# the only difference is the term. sw is not repeated -- measured on the references themselves, the
# six photographs' patch distributions are 35x further apart under sw and 60x under JS than one
# photograph is from itself, against 10.7x for MSE, so both make the target less consistent than
# the pixel loss it would replace. Chamfer is 2.44, the only one below MSE, because it asks whether
# the vocabulary of patches matches and not in what proportions -- and the proportions are the part
# that belongs to whichever orange was photographed.
#
# The weight is derived rather than guessed. Chamfer runs 0.16 within one photograph and 0.39
# between two, so its useful range is about 0.23, while the pixel term after SEC_DIST_MIX is
# 0.3 * 0.15 = 0.045. W = 0.2 makes them comparable; W = 1.0 is the same arm five times louder,
# so the weight is measured here too.
cd /workspace/ovoxel_native
BASE="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 REF_TRANS_BLEND=1 POS_FREEZE=1 \
ANCHOR=1 ANCHOR_DIM=8 ANCHOR_PREFIT=1 ANCHOR_CHUNK=262144 ABL_INTERVAL=30 ABL_GRID=16"
PATCH="SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3"
DIST="SEC_DIST=chamfer SEC_DIST_START=0.5 SEC_DIST_MIX=0.3"

go () {  # go GPU TAG EXTRA...
  local g=$1 tag=$2; shift 2
  setsid env $BASE "$@" OUT=/workspace/ovoxel_native/$tag CUDA_VISIBLE_DEVICES=$g \
    nohup bash run.sh mvtrain.py > /workspace/ovoxel_native/$tag.log 2>&1 < /dev/null &
  disown; sleep 4
}

go 0 r1_cd02 ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 $PATCH $DIST SEC_DIST_W=0.2
go 1 r1_cd10 ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 $PATCH $DIST SEC_DIST_W=1.0
echo LAUNCHED_CD
