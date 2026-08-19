#!/usr/bin/env bash
cd /workspace/ovoxel_native
exec env -u LD_LIBRARY_PATH -u PYTHONHOME PYTHONPATH= \
  CUDA_HOME=/workspace/rebuild/mc/envs/cu118 \
  PATH=/workspace/rebuild/mc/envs/cu118/bin:/usr/bin:/bin \
  CC=$(ls /workspace/rebuild/mc/envs/cu118/bin/*-gcc|head -1) \
  CXX=$(ls /workspace/rebuild/mc/envs/cu118/bin/*-g++|head -1) \
  TORCH_CUDA_ARCH_LIST=8.9 MPLBACKEND=Agg \
  TRELLIS2_ROOT=/workspace/rebuild/TRELLIS.2 \
  FN_ROOT=/workspace/rebuild/worktree \
  /workspace/ovoxel_native/env/bin/python "$@"
