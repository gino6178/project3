#!/usr/bin/env bash
# Four arms: both routes crossed with SHELL_PIN, decoder on, the full multi-view schedule.
# SHELL_PIN is crossed on route 2 as well because the route-1 finding rests on a kNN resample of
# a released model, and route 2 has no released model to resample.
cd /workspace/ovoxel_native
COMMON="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 \
ANCHOR=1 ANCHOR_DIM=8 ANCHOR_PREFIT=1 ANCHOR_CHUNK=262144 VOXEL_SMOOTH=0"

go () {  # go ROUTE PIN EXTV GPU TAG
  setsid env $COMMON ROUTE=$1 SHELL_PIN=$2 EXT_VIEWS=$3 \
    OUT=/workspace/ovoxel_native/$5 CUDA_VISIBLE_DEVICES=$4 \
    nohup bash run.sh mvtrain.py > /workspace/ovoxel_native/$5.log 2>&1 < /dev/null &
  disown
  sleep 4
}

go 1 0 6 1 r1_free
go 1 1 0 2 r1_pin
go 2 0 6 3 r2_free
go 2 1 0 3 r2_pin
echo LAUNCHED4
