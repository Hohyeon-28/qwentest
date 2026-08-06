#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export RUNTIME_PROFILE=cu130
export VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwen-cu130}"
export CONFIG="${CONFIG:-configs/experiment_cuda130.yaml}"

exec bash "${SCRIPT_DIR}/run_shared_storage_four_gpu.sh"
