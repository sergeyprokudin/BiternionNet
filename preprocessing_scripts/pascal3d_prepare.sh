#!/usr/bin/env bash

#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PYENV="${SCRIPT_DIR}/../py_env"
source ${PYENV}/bin/activate

LPATH="/usr/local/cudnn-5.1/lib64"
if [[ -d ${LPATH} ]];then
    dbash::pp "cudnn cluster ${LPATH}"
    export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
fi

python ../utils/pascal3d.py