#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No virtual environment is active." >&2
  echo "Create and activate one before running this installer:" >&2
  echo "  python3 -m venv qwen-cu130" >&2
  echo "  source qwen-cu130/bin/activate" >&2
  exit 1
fi

VENV_DIR="${1:-${VIRTUAL_ENV}}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment does not exist: ${VENV_DIR}" >&2
  exit 1
fi
if [[ "$(readlink -f "${VENV_DIR}")" != "$(readlink -f "${VIRTUAL_ENV}")" ]]; then
  echo "Active virtual environment does not match VENV_DIR." >&2
  echo "VIRTUAL_ENV=${VIRTUAL_ENV}" >&2
  echo "VENV_DIR=${VENV_DIR}" >&2
  exit 1
fi

TEST_GPU="${CUDA_TEST_DEVICE:-0}"
PYTHON_BIN="${VENV_DIR}/bin/python"
if ! "${PYTHON_BIN}" -c \
  'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))'; then
  echo "vLLM 0.20.2 requires Python 3.10-3.13; found $(${PYTHON_BIN} -V)." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found." >&2
  exit 1
fi

# Prebuilt PyTorch/vLLM wheels carry their CUDA 13.0 user-space runtime.  A
# local CUDA toolkit is not required.  Do not accidentally compile extensions
# against the old CUDA 11.8 tree inherited from the previous server.
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 13\.'; then
    echo "[warning] ignoring non-CUDA-13 CUDA_HOME=${CUDA_HOME}"
    unset CUDA_HOME
  fi
fi
export MAX_JOBS="${MAX_JOBS:-4}"

echo "[driver]"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
if command -v nvcc >/dev/null 2>&1; then
  echo "[optional local toolkit]"
  nvcc --version
else
  echo "[toolkit] nvcc not found; prebuilt wheels do not require it."
fi

echo "[venv] installing into active environment: ${VENV_DIR}"

"${PYTHON_BIN}" -m pip install --upgrade pip uv
# uv selects the official CUDA 13.0 PyTorch index and keeps vLLM's exact torch
# dependency intact.  flash-attn/torchvision/torchaudio are not used here.
uv pip install --python "${VENV_DIR}/bin/python" \
  --torch-backend=cu130 \
  -r requirements-cu130.txt
uv pip check --python "${VENV_DIR}/bin/python"

CUDA_VISIBLE_DEVICES="${TEST_GPU}" "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

import torch
from transformers import PreTrainedModel
from gptqmodel import BACKEND, GPTQModel
from vllm import LLM, SamplingParams

versions = {
    "torch": version("torch"),
    "vllm": version("vllm"),
    "gptqmodel": version("gptqmodel"),
    "transformers": version("transformers"),
}
print("versions:", versions)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the cu130 environment")
print("visible GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))

if not versions["torch"].startswith("2.11.0"):
    raise RuntimeError(f"Expected PyTorch 2.11.0, found {versions['torch']}")
if torch.version.cuda != "13.0":
    raise RuntimeError(f"Expected PyTorch cu130, found CUDA {torch.version.cuda}")
if versions["vllm"] != "0.20.2":
    raise RuntimeError(f"Expected vLLM 0.20.2, found {versions['vllm']}")
if versions["gptqmodel"] != "7.2.0":
    raise RuntimeError(
        f"Expected GPTQModel 7.2.0, found {versions['gptqmodel']}"
    )
if torch.cuda.get_device_capability(0) < (8, 0):
    raise RuntimeError("GPTQ-Marlin requires an Ampere-or-newer NVIDIA GPU")
if not hasattr(BACKEND, "GPTQ_TORCH"):
    raise RuntimeError("GPTQModel does not expose BACKEND.GPTQ_TORCH")
PY

echo
echo "[done] CUDA 13.0 environment created at ${VENV_DIR}"
echo "Run: bash scripts/run_shared_storage_four_gpu_cuda130.sh"
