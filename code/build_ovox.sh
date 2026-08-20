#!/usr/bin/env bash
# Build O-Voxel against the newer interpreter, because its source needs one.
#
# src/hash/hash.cu uses torch::kUInt32, which does not exist before torch 2.3; the fn environment
# runs 2.0.1 and cannot be moved, since the trainer and the rasteriser are built against it. The
# dual-grid conversion is an offline step, so it does not have to share that interpreter -- it
# only has to hand back a mesh.
set -u
R=/workspace/rebuild
unset PYTHONHOME LD_LIBRARY_PATH; export PYTHONPATH=
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
P=/opt/venv/bin/python
cd $R/TRELLIS.2/o-voxel
$P -c "import torch;print('building against torch', torch.__version__, 'cuda', torch.version.cuda)"
rm -rf build o_voxel/_C.*.so
$P setup.py build_ext --inplace 2>&1 | grep -vE "^\s*$" | tail -20
echo "--- import check"
$P -c "
import glob, importlib.util
so=glob.glob('o_voxel/_C.*.so'); print('so:', so)
spec=importlib.util.spec_from_file_location('o_voxel._C', so[0])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('imported ok')
"
echo OVOX2_DONE >> $R/ovoxbuild.stat
