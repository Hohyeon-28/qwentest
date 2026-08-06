#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_SHARED_ROOT="${HOME}/shared/hdd_ext/nvme4000/${USER}"

export RUNTIME_PROFILE=cu130
export VENV_DIR="${VENV_DIR:-${VIRTUAL_ENV:-${PROJECT_ROOT}/qwen-cu130}}"
export CONFIG="${CONFIG:-configs/experiment_cuda130.yaml}"
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_SHARED_ROOT}}"
export EXPERIMENT_CACHE_ROOT="${EXPERIMENT_CACHE_ROOT:-${STORAGE_ROOT}}"
export EXPERIMENT_OUTPUT_ROOT="${EXPERIMENT_OUTPUT_ROOT:-${STORAGE_ROOT}/qwentest_results}"
if [[ ! -d "${STORAGE_ROOT}" ]]; then
  echo "Shared storage root does not exist: ${STORAGE_ROOT}" >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/run_shared_storage_four_gpu.sh"
