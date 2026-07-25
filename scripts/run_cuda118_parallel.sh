#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_DIR="${VENV_DIR:-.venv-cu118}"
CUDA_HOME="${CUDA_HOME:-${HOME}/private/cuda-11.8}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "${VENV_DIR} does not exist. Run bash scripts/setup_cuda118.sh first." >&2
  exit 1
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA 11.8 nvcc was not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Fail before starting workers if an interrupted install left incompatible
# Transformers or torchvision packages in the environment.
python - <<'PY'
from importlib.metadata import version

import torch
import torchvision
from transformers import PreTrainedModel

expected = {
    "torch": "2.6.0+cu118",
    "torchvision": "0.21.0+cu118",
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
}
actual = {name: version(name) for name in expected}
bad = {name: (actual[name], wanted) for name, wanted in expected.items()
       if actual[name] != wanted}
if bad:
    raise RuntimeError(f"Incompatible CUDA 11.8 environment: {bad}")
if torch.version.cuda != "11.8":
    raise RuntimeError(f"Expected torch CUDA 11.8, found {torch.version.cuda}")
print("[preflight]", actual)
PY

exec bash scripts/run_experiment_parallel.sh "${1:-math500}" "${2:-6}" "${3:-7}"
