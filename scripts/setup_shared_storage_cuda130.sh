#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/shared_storage_env.sh"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwen-cu130}"
export VENV_DIR

echo "[storage] ${STORAGE_ROOT}"
echo "[venv] ${VENV_DIR}"
echo "[runtime] prebuilt CUDA 13.0 wheels (no local CUDA_HOME required)"

exec bash "${SCRIPT_DIR}/setup_cuda130.sh" "${VENV_DIR}"
