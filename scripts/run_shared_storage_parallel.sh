#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/shared_storage_env.sh"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwen}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}" >&2
  echo "Run: bash scripts/setup_shared_storage_cuda118.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [[ -n "${CUDA_HOME:-}" ]]; then
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Invalid CUDA_HOME: ${CUDA_HOME}" >&2
    exit 1
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[storage] ${STORAGE_ROOT}"
echo "[repo] ${PROJECT_ROOT}"
echo "[venv] ${VENV_DIR}"
echo "[HF cache] ${HF_HOME}"
echo "[results] ${PROJECT_ROOT}/results_39k_v2"

python - <<'PY'
from importlib.metadata import version

import torch
from transformers import PreTrainedModel

expected = {
    "torch": "2.6.0+cu118",
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
}
actual = {name: version(name) for name in expected}
bad = {name: (actual[name], wanted) for name, wanted in expected.items()
       if actual[name] != wanted}
if bad:
    raise RuntimeError(f"Incompatible experiment environment: {bad}")
if torch.version.cuda != "11.8" or not torch.cuda.is_available():
    raise RuntimeError(
        f"Expected an available PyTorch CUDA 11.8 runtime, found {torch.version.cuda}"
    )
print("[preflight]", actual)
print("[GPU count]", torch.cuda.device_count())
PY

exec bash "${SCRIPT_DIR}/run_experiment_parallel.sh" \
  "${1:-math500}" "${2:-6}" "${3:-7}"
