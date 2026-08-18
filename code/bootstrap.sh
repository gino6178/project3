#!/usr/bin/env bash
# Build the interpreter and the CUDA extensions the pipeline needs, from nothing.
#
#   bash code/bootstrap.sh /path/to/build
#   export FN_PY=/path/to/build/mc/envs/fn/bin/python
#   export GS_ROOT=/path/to/build/gaussian-splatting
#
# README says FN_PY needs torch, taichi and warp and GS_ROOT needs a built gaussian-splatting,
# and leaves the rest to the reader. This is that rest. Every line below was arrived at by a
# failure on a machine that had only the base image, and each one says which.
#
# Re-runnable: every step skips itself if its output is already there.
set -eu
HERE_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gaussian_model.patch
R=$(mkdir -p "${1:?usage: bash code/bootstrap.sh DIR}" && cd "$1" && pwd)
MC=$R/mc
export PATH=$MC/bin:$PATH
# The image ships a system torch for python 3.12 and puts it on PYTHONPATH. Anything
# built --no-build-isolation picks that up instead of the one in the env and fails on a
# symbol from the wrong interpreter. Clear the three variables REMOTE_ACCESS.md names.
unset PYTHONHOME LD_LIBRARY_PATH || true
export PYTHONPATH=
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic

step () { echo; echo "=== $* ==="; date -u +%H:%M:%S; }

if [ ! -x "$MC/bin/conda" ]; then
  step "miniconda"
  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC"
fi

if [ ! -d "$MC/envs/fn" ]; then
  step "python 3.10"
  conda create -y -q -n fn python=3.10
fi
PY=$MC/envs/fn/bin/python

step "torch 2.0.1+cu118"
$PY -c "import torch" 2>/dev/null || \
  $PY -m pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

step "the rest of what the tree imports"
$PY -m pip install -q \
  taichi==1.5.0 warp-lang==0.10.1 \
  numpy==1.24.4 scipy opencv-python-headless pillow plyfile tqdm h5py zstandard \
  trimesh scikit-image PyMCubes pytorch_msssim lpips torchmetrics \
  "diffusers==0.19.3" "transformers==4.30.2" "huggingface_hub==0.16.4" safetensors \
  imageio imageio-ffmpeg matplotlib ninja \
  "setuptools<70" wheel      # the rasteriser's setup.py imports pkg_resources, which
                             # setuptools dropped in 81

if [ ! -d "$MC/envs/cu118" ]; then
  step "cuda 11.8 toolkit and gcc 11 -- 11.7 has no sm_89, and the L40 is Ada"
  conda create -y -q -n cu118 -c "nvidia/label/cuda-11.8.0" -c conda-forge \
    cuda-toolkit=11.8 gxx_linux-64=11.4 gcc_linux-64=11.4
fi

step "gaussian-splatting"
# Two checkouts, because GS_ROOT and the CUDA extensions come from different places and the
# README's one line does not say so. FruitNinja's fork supplies scene/ and gaussian_renderer/ --
# the pipeline calls GaussianModel.load_ply_zero_sh, which is only there -- and it carries no
# submodules, so the two extensions come from upstream, whose rasteriser API the fork's renderer
# is written against.
cd $R
[ -d gaussian-splatting ] || {
  git clone -q https://github.com/fanguw/gaussian-splatting.git
  # The three per-primitive flags the interior needs, which the fork does not have. `trained`
  # records whether a primitive has ever been covered by a section mask, `is_interior` whether
  # it is one of the lattice points, and `lattice_pure` stops densification from splitting
  # those -- without it the children land off the lattice, the parent is pruned, and the cell
  # is gone. This has always been required and was never in the repository: the pipeline calls
  # get_trained() on the first training step and stops there without it.
  git -C gaussian-splatting apply "$HERE_PATCH"
}
[ -d gs-upstream ] || \
  git clone -q --recursive https://github.com/graphdeco-inria/gaussian-splatting.git gs-upstream

export CUDA_HOME=$MC/envs/cu118
export PATH=$CUDA_HOME/bin:$PATH
export CC=$(ls $CUDA_HOME/bin/*-gcc | head -1)
export CXX=$(ls $CUDA_HOME/bin/*-g++ | head -1)
export TORCH_CUDA_ARCH_LIST="8.9"          # L40 is Ada; 11.7 cannot target it at all
nvcc --version | tail -2

# The rasteriser is not upstream's. FruitNinja's own code and the fork's renderer both unpack
# four values -- `rendering, radii, depth, alpha` -- and no graphdeco version returns four:
# the original returns two, and the one that added `antialiasing` returns three and takes a
# field the fork does not pass. ashawkey's fork returns exactly those four. FruitNinja's
# requirements.txt names graphdeco unpinned, which cannot have worked as written; this is what
# their code is against.
step "diff-gaussian-rasterization"
[ -d $R/dgr ] || git clone -q --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git $R/dgr
$PY -c "import diff_gaussian_rasterization" 2>/dev/null || \
  $PY -m pip install -q --no-build-isolation $R/dgr

step "simple-knn"
$PY -c "import simple_knn" 2>/dev/null || \
  $PY -m pip install -q --no-build-isolation $R/gs-upstream/submodules/simple-knn

step "check"
$PY - <<'PYEOF'
import torch, taichi, warp, diff_gaussian_rasterization, cv2, scipy, trimesh
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "gpus")
print("taichi", taichi.__version__, "warp", warp.config.version)
print("rasteriser ok")
PYEOF
echo; echo "BUILD_OK"; date -u +%H:%M:%S
