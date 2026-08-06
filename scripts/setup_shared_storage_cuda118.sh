#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/shared_storage_env.sh"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwen}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
  elif [[ -x /usr/local/cuda-11.8/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda-11.8
  else
    echo "CUDA 11.8 nvcc was not found." >&2
    echo "Load the server's CUDA 11.8 module or set CUDA_HOME explicitly." >&2
    exit 1
  fi
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc was not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
if ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 11\.8'; then
  echo "This reproducible environment requires the CUDA 11.8 toolkit." >&2
  "${CUDA_HOME}/bin/nvcc" --version >&2
  exit 1
fi

export CUDA_HOME
export VENV_DIR
echo "[storage] ${STORAGE_ROOT}"
echo "[venv] ${VENV_DIR}"
echo "[cuda] ${CUDA_HOME}"

exec bash "${SCRIPT_DIR}/setup_cuda118.sh" "${VENV_DIR}"
