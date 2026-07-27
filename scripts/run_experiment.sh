#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${1:-math500}"
GPU_ID="${2:-7}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
OUTPUT_ROOT="$(
  python -c \
    'import sys; from src.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_dir"])' \
    "${CONFIG}"
)"

if [[ "${DATASET}" != "gsm8k" && "${DATASET}" != "math500" ]]; then
  echo "Usage: bash scripts/run_experiment.sh [gsm8k|math500] [physical_gpu_id]" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[config] dataset=${DATASET}"
echo "[config] physical GPU=${GPU_ID} (visible inside Python as cuda:0)"
echo "[config] config=${CONFIG}"

python scripts/preflight_experiment.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}"

python scripts/prepare_fake_quant.py \
  --config "${CONFIG}" \
  --verify-checkpoint

python scripts/run_bf16.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}"

python scripts/run_fake_quant.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --device cuda

python scripts/run_vllm_marlin.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

python scripts/evaluate_answers.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --all

python scripts/validate_results.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}"

python scripts/compare_results.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}"

python scripts/plot_results.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}"

echo "[done] results are under ${OUTPUT_ROOT}/${DATASET}/"
