#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../../../"
PYENV="${PROJECT_DIR}/py_env"

source "${PROJECT_DIR}/scripts/dbash.sh" || exit 1
dbash::cluster_cuda

source ${PYENV}/bin/activate

python "${PROJECT_DIR}/train_cvae.py" "${PROJECT_DIR}/train_configs/IDIAP/pan/train_cvae_idiap_best.yml"