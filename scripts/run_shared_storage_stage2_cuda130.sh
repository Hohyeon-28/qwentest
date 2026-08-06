#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Remaining suite: three code datasets x three conditions = nine queued jobs.
export DATASETS="${DATASETS:-humaneval mbpp livecodebench}"

exec bash "${SCRIPT_DIR}/run_shared_storage_four_gpu_cuda130.sh"
