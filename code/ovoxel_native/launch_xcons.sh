#!/usr/bin/env bash
# The cross-family reconciliation, on top of full parity, both modes, both routes.
#
# Full parity is the baseline it has to beat: SEC_PATCH on with the pipeline's own section loss
# (0.7(1-SSIM)+0.3MSE on crops plus the band term), VOXEL_SMOOTH on, exterior pinned. Those are
# r1_pin_full and r2_pin_full, already measured at rh 0.0599 / rv 0.2315 and rh 0.1018 /
# rv 0.2731 on the plane filter.
#
# AT=100 rather than 0: section_target re-derives every target from the current render each
# pass, so reconciling from iteration zero removes a contradiction that is rebuilt before the
# next one. HOLD=1 keeps the reconciled longitudinal targets instead.
cd /workspace/ovoxel_native
COMMON="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 \
ANCHOR=1 ANCHOR_DIM=8 ANCHOR_PREFIT=1 ANCHOR_CHUNK=262144 VOXEL_SMOOTH=1 \
SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3 \
SEC_XCONS=1 SEC_XCONS_AT=100 SEC_XCONS_HOLD=1"

go () {  # go ROUTE MODE GPU TAG
  setsid env $COMMON ROUTE=$1 SEC_XCONS_MODE=$2 SHELL_PIN=1 EXT_VIEWS=0 \
    OUT=/workspace/ovoxel_native/$4 CUDA_VISIBLE_DEVICES=$3 \
    nohup bash run.sh mvtrain.py > /workspace/ovoxel_native/$4.log 2>&1 < /dev/null &
  disown
  sleep 5
}

go 1 copy 2 r1_xc_copy
go 2 copy 2 r2_xc_copy
echo LAUNCHED_XCONS
