#!/usr/bin/env bash
# Route 1, image-initialised only: exterior from the quantised ply as before, interior flat.
cd /workspace/ovoxel_native
COMMON="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 \
ANCHOR=1 ANCHOR_DIM=8 ANCHOR_PREFIT=1 ANCHOR_CHUNK=262144 VOXEL_SMOOTH=0 FLAT_INIT=0.5"
go () {
  setsid env $COMMON ROUTE=1 SHELL_PIN=$1 EXT_VIEWS=$2 \
    OUT=/workspace/ovoxel_native/$4 CUDA_VISIBLE_DEVICES=$3 \
    nohup bash run.sh mvtrain.py > /workspace/ovoxel_native/$4.log 2>&1 < /dev/null &
  disown; sleep 4
}
go 0 6 2 r1flat_free
go 1 0 3 r1flat_pin
echo LAUNCHED_FLAT
