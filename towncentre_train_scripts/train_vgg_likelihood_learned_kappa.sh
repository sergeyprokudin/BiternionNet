#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
PYENV="${SCRIPT_DIR}/../py_env"

source ${PYENV}/bin/activate

python train_vgg_towncentre.py "${PROJECT_DIR}/train_configs/train_vgg_towncentre_likelihood_learned_kappa.yml"