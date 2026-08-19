#!/usr/bin/env bash
set -eux
R=/workspace/rebuild
W=/workspace/ovoxel_native
export PATH=$R/mc/bin:$PATH
unset PYTHONHOME LD_LIBRARY_PATH || true
export PYTHONPATH=
export CONDA_NO_PLUGINS=true CONDA_SOLVER=classic
mkdir -p $W
date -u +%H:%M:%S
if [ ! -x $W/env/bin/python ]; then
  conda create -y -q -p $W/env --clone $R/mc/envs/score
fi
date -u +%H:%M:%S
PY=$W/env/bin/python
$PY -c "import torch;print('torch',torch.__version__,torch.version.cuda)"

export CUDA_HOME=$R/mc/envs/cu118
export PATH=$CUDA_HOME/bin:$PATH
export CC=$(ls $CUDA_HOME/bin/*-gcc | head -1)
export CXX=$(ls $CUDA_HOME/bin/*-g++ | head -1)
export TORCH_CUDA_ARCH_LIST="8.9"
nvcc --version | tail -2

$PY -m pip install -q ninja imageio matplotlib
$PY -c "import nvdiffrast" 2>/dev/null || \
  $PY -m pip install -q --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast"
$PY -c "import nvdiffrast, os; print('nvdiffrast at', os.path.dirname(nvdiffrast.__file__))"
echo SETUP_OK
date -u +%H:%M:%S
