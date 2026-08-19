#!/usr/bin/env bash
# The two parity runs: everything stage_train sets, and the one thing being tested.
#
#   mv_free   the exterior learns too, from the six exterior views -- what the extra supervision
#             is worth when nothing is held.
#   mv_pin    SHELL_PIN=1, the pipeline's own setting: the exterior comes from the released ply
#             and is not trained, so only the interior beneath it learns.  EXT_VIEWS=0 to match
#             stage_train, which sets it because a pinned exterior has nothing to learn from them.
cd /workspace/ovoxel_native
COMMON="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 SEC_PATCH=128 SEC_PATCH_N=6 SEC_PATCH_STAT=0.3 \
REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 OPACITY_FREEZE=1.0"

setsid env $COMMON SHELL_PIN=0 EXT_VIEWS=6 OUT=/workspace/ovoxel_native/mv_free \
  CUDA_VISIBLE_DEVICES=2 nohup bash run.sh mvtrain.py \
  > /workspace/ovoxel_native/mv_free.log 2>&1 < /dev/null &
disown
sleep 5
setsid env $COMMON SHELL_PIN=1 EXT_VIEWS=0 OUT=/workspace/ovoxel_native/mv_pin \
  CUDA_VISIBLE_DEVICES=3 nohup bash run.sh mvtrain.py \
  > /workspace/ovoxel_native/mv_pin.log 2>&1 < /dev/null &
disown
echo LAUNCHED
