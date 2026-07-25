#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_DIR="${1:-.venv-cu118}"
CUDA_HOME="${CUDA_HOME:-${HOME}/private/cuda-11.8}"
TEST_GPU="${CUDA_TEST_DEVICE:-7}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10 or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA 11.8 nvcc was not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found; a user-space CUDA Toolkit cannot replace the driver." >&2
  exit 1
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "[driver]"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo "[toolkit]"
"${CUDA_HOME}/bin/nvcc" --version

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install -r requirements-cu118.txt
python -m pip check

CUDA_VISIBLE_DEVICES="${TEST_GPU}" python - <<'PY'
from importlib.metadata import version

import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable with the CUDA 11.8 environment")
print("visible GPU:", torch.cuda.get_device_name(0))
print("vLLM:", version("vllm"))
print("GPTQModel:", version("gptqmodel"))
print("Transformers:", version("transformers"))

if torch.version.cuda != "11.8":
    raise RuntimeError(f"Expected PyTorch cu118, found CUDA {torch.version.cuda}")
if not version("vllm").startswith("0.8.5"):
    raise RuntimeError(f"Expected vLLM 0.8.5, found {version('vllm')}")
if not version("gptqmodel").startswith("3.0"):
    raise RuntimeError(f"Expected GPTQModel 3.0, found {version('gptqmodel')}")
PY

echo
echo "[done] CUDA 11.8 environment created at ${VENV_DIR}"
echo "Run: bash scripts/run_cuda118_parallel.sh math500 6 7"
