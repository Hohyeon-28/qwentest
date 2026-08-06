#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Pilot: run one fully scored dataset under all three conditions, concurrently.
export DATASETS="${DATASETS:-gsm8k}"
if [[ -z "${GPU_IDS:-}" ]]; then
  visible="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  IFS=',' read -r -a visible_array <<< "${visible}"
  if [[ "${#visible_array[@]}" -lt 3 ]]; then
    echo "Stage 1 requires at least three visible GPUs." >&2
    exit 2
  fi
  export GPU_IDS="${visible_array[0]} ${visible_array[1]} ${visible_array[2]}"
fi

exec bash "${SCRIPT_DIR}/run_shared_storage_four_gpu_cuda130.sh"
