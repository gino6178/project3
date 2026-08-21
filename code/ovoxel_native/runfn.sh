#!/usr/bin/env bash
# The pipeline's interpreter, for the one step here that needs the gaussian-splatting extensions.
#
# mvcams reads the released model through scene.gaussian_model, which imports simple_knn: a CUDA
# extension built into mc/envs/fn and not into this build's own venv. Nothing in the training
# needs it, so the step that does is run over there rather than the extension duplicated here.
cd /workspace/ovoxel_native
# The order matters and is the pipeline's own: `utils` is a package in BOTH code/inherited and
# gaussian-splatting, and the one that has camera_view_utils is inherited's, so inherited has to
# come first or the import resolves into the wrong package and fails on a name that is there.
H=/workspace/rebuild/project3/code
exec env -u LD_LIBRARY_PATH -u PYTHONHOME \
  PYTHONPATH="$H/src:$H/figures:$H/inherited:$H/inherited/mpm_solver_warp:${GS_ROOT:-/workspace/rebuild/gaussian-splatting}" \
  CUDA_HOME=/workspace/rebuild/mc/envs/cu118 \
  PATH=/workspace/rebuild/mc/envs/cu118/bin:/usr/bin:/bin \
  MPLBACKEND=Agg \
  FN_ROOT="${FN_ROOT:-/workspace/rebuild/worktree}" \
  GS_ROOT="${GS_ROOT:-/workspace/rebuild/gaussian-splatting}" \
  /workspace/rebuild/mc/envs/fn/bin/python "$@"
