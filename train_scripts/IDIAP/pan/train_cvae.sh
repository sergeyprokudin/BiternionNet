#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
PYENV="${SCRIPT_DIR}/../py_env"

source ${PYENV}/bin/activate

python "${PROJECT_DIR}/train_cvae.py" "${PROJECT_DIR}/train_configs/IDIAP/pan/train_cvae_idiap_best.yml"