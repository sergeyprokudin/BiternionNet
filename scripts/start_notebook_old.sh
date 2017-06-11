#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PYENV="${SCRIPT_DIR}/../py_env_old"
source ${PYENV}/bin/activate

jupyter notebook