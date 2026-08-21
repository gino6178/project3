#!/usr/bin/env bash
# The cut face scored as a distribution of patches, in its second stage.
#
# r1_pin_full is the control. This is that arm with the sliced Wasserstein distance between the
# render's patches and the photograph's added from half way, keeping 0.3 of the pixel term --
# nothing else changed. The pixel term is kept because no distributional distance can say where
# anything belongs; all three are invariant to position by construction.
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

go 3 r1_sw ROUTE=1 SHELL_PIN=1 EXT_VIEWS=0 VOXEL_SMOOTH=1 $PATCH \
  SEC_DIST=sw SEC_DIST_W=1.0 SEC_DIST_START=0.5 SEC_DIST_MIX=0.3
echo LAUNCHED_SW
