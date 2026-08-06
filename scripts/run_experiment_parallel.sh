#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${1:-math500}"
BF16_GPU="${2:-6}"
QUANT_GPU="${3:-7}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
OUTPUT_ROOT="$(
  python -c \
    'import sys; from src.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_dir"])' \
    "${CONFIG}"
)"
LOG_DIR="${OUTPUT_ROOT}/${DATASET}/runner_logs"

case "${DATASET}" in
  gsm8k|math500|humaneval|mbpp|livecodebench) ;;
  *)
  echo "Usage: bash scripts/run_experiment_parallel.sh [gsm8k|math500|humaneval|mbpp|livecodebench] [bf16_gpu] [quant_gpu]" >&2
  exit 2
  ;;
esac

if [[ "${BF16_GPU}" == "${QUANT_GPU}" ]]; then
  echo "BF16 and quantized workers must use different physical GPU IDs." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

run_logged() {
  local label="$1"
  shift
  local log_path="${LOG_DIR}/${label}.log"
  echo "[start] ${label}; log=${log_path}"
  "$@" 2>&1 | tee "${log_path}"
  echo "[done] ${label}"
}

echo "[config] dataset=${DATASET}"
echo "[config] BF16 physical GPU=${BF16_GPU}"
echo "[config] Fake/Real physical GPU=${QUANT_GPU}"
echo "[config] config=${CONFIG}"
echo "[config] vLLM V0 + spawn + eager execution (CUDA graph disabled)"

python scripts/preflight_experiment.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}"

# Verify the shared (q,s,z,g) checkpoint before launching either worker.
python scripts/prepare_fake_quant.py \
  --config "${CONFIG}" \
  --verify-checkpoint

(
  export CUDA_VISIBLE_DEVICES="${BF16_GPU}"
  run_logged bf16 \
    python scripts/run_bf16.py \
      --config "${CONFIG}" \
      --dataset "${DATASET}"
) &
BF16_PID=$!

(
  export CUDA_VISIBLE_DEVICES="${QUANT_GPU}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  run_logged fake_quant \
    python scripts/run_fake_quant.py \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --device cuda

  run_logged real_quant_marlin \
    python scripts/run_vllm_marlin.py \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
) &
QUANT_PID=$!

BF16_STATUS=0
QUANT_STATUS=0
wait "${BF16_PID}" || BF16_STATUS=$?
wait "${QUANT_PID}" || QUANT_STATUS=$?

if (( BF16_STATUS != 0 || QUANT_STATUS != 0 )); then
  echo "A worker failed: BF16=${BF16_STATUS}, Fake/Real=${QUANT_STATUS}" >&2
  echo "Inspect ${LOG_DIR}/ for details." >&2
  exit 1
fi

if [[ "${DATASET}" == "humaneval" || "${DATASET}" == "mbpp" || "${DATASET}" == "livecodebench" ]]; then
  python scripts/export_code_eval.py \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --all
  echo "[done] generations exported for the official code harness."
  echo "[pending] run code evaluation in an isolated sandbox, then import pass/fail results."
  exit 0
fi

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
