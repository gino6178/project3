cd /workspace/ovoxel_native
COMMON="ITERS=200 ABL_RES=512 JITTER=0.5 SEC_SKIP_OUTER=0.10 SHELL_PIN_LAYERS=2 \
SECTION_MATCH=1 REF_PHASE_MODE=solve REF_DEPTH_BLEND=1 POS_FREEZE=1 \
VOXEL_SMOOTH=1 ABL_INTERVAL=30 ABL_GRID=16"
setsid env $COMMON SHELL_PIN=0 EXT_VIEWS=6 OUT=/workspace/ovoxel_native/mv_free_v2 \
  CUDA_VISIBLE_DEVICES=2 nohup bash run.sh mvtrain.py \
  > /workspace/ovoxel_native/mv_free_v2.log 2>&1 < /dev/null &
disown
sleep 5
setsid env $COMMON SHELL_PIN=1 EXT_VIEWS=0 OUT=/workspace/ovoxel_native/mv_pin_v2 \
  CUDA_VISIBLE_DEVICES=3 nohup bash run.sh mvtrain.py \
  > /workspace/ovoxel_native/mv_pin_v2.log 2>&1 < /dev/null &
disown
echo LAUNCHED2
