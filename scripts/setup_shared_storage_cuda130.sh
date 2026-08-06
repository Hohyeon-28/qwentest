#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_SHARED_ROOT="${HOME}/shared/hdd_ext/nvme4000/${USER}"
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_SHARED_ROOT}}"
export EXPERIMENT_CACHE_ROOT="${EXPERIMENT_CACHE_ROOT:-${STORAGE_ROOT}}"
export EXPERIMENT_OUTPUT_ROOT="${EXPERIMENT_OUTPUT_ROOT:-${STORAGE_ROOT}/qwentest_results}"
if [[ ! -d "${STORAGE_ROOT}" ]]; then
  echo "Shared storage root does not exist: ${STORAGE_ROOT}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/shared_storage_env.sh"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwen-cu130}"
export VENV_DIR

echo "[storage] ${STORAGE_ROOT}"
echo "[model/cache] ${EXPERIMENT_CACHE_ROOT}"
echo "[results] ${EXPERIMENT_OUTPUT_ROOT}"
echo "[venv] ${VENV_DIR}"
echo "[runtime] prebuilt CUDA 13.0 wheels (no local CUDA_HOME required)"

exec bash "${SCRIPT_DIR}/setup_cuda130.sh" "${VENV_DIR}"
