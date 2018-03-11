#!/usr/bin/env bash
source ~/.bashrc

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PYENV="${SCRIPT_DIR}/../py_env"
source ${PYENV}/bin/activate

#CUDNNPATH="/lustre/shared/caffe_shared/cuda_stuff/cudnn-5.1_for_cuda8.0rc"
CUDNNPATH="/lustre/shared/caffe_shared/cuda_stuff/cudnnn-6.0_for_cuda8"
export LD_LIBRARY_PATH=${CUDNNPATH}/lib64:$LD_LIBRARY_PATH
CUDAPATH="/lustre/shared/caffe_shared/cuda_stuff/cuda-8.0.27.1_RC"
export LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH

python "${SCRIPT_DIR}/../train_pascal_mixture.py"
