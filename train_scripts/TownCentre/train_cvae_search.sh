#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../../"
PYENV="${PROJECT_DIR}/py_env"

source "${PROJECT_DIR}/scripts/dbash.sh" || exit 1

CUDNNPATH="/lustre/shared/caffe_shared/cuda_stuff/cudnn-5.1_for_cuda8.0rc"
CUDAPATH="/lustre/shared/caffe_shared/cuda_stuff/cuda-8.0.27.1_RC"
export PATH=${CUDAPATH}/bin:$PATH
export CPATH=${CUDAPATH}/include:$CPATH
export PATH='/lustre/shared/caffe_shared/cuda_stuff/cudnn-5.1_for_cuda8.0rc/bin':$PATH
export LD_LIBRARY_PATH=${CUDNNPATH}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH='/lustre/shared/caffe_shared/cuda_stuff/cudnn-5.1_for_cuda8.0rc/lib64':$LIBRARY_PATH
export LIBRARY_PATH=${CUDAPATH}/lib64:$LIBRARY_PATH

source ${PYENV}/bin/activate

cd ${PROJECT_DIR}

python "${PROJECT_DIR}/train.py" "${PROJECT_DIR}/train_configs/TownCentre/train_cvae_towncentre.yml"