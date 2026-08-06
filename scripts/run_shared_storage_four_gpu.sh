#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/shared_storage_env.sh"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwen}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
DATASETS="${DATASETS:-livecodebench mbpp humaneval gsm8k}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}" >&2
  echo "Run: bash scripts/setup_shared_storage_cuda118.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

if ! command -v flock >/dev/null 2>&1; then
  echo "flock (util-linux) is required for the dynamic four-GPU queue." >&2
  exit 1
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Invalid CUDA_HOME: ${CUDA_HOME}" >&2
    exit 1
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# GPU_IDS always denotes physical nvidia-smi indices.
unset CUDA_VISIBLE_DEVICES || true

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four physical GPU IDs." >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU_IDS contains duplicate GPU IDs: ${GPU_IDS}" >&2
  exit 2
fi

for gpu_id in "${GPU_ARRAY[@]}"; do
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID: ${gpu_id}" >&2
    exit 2
  fi
done

read -r -a DATASET_ARRAY <<< "${DATASETS}"
for dataset in "${DATASET_ARRAY[@]}"; do
  case "${dataset}" in
    gsm8k|math500|humaneval|mbpp|livecodebench) ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 2
      ;;
  esac
done

OUTPUT_ROOT="$(
  python -c \
    'import sys; from src.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_dir"])' \
    "${CONFIG}"
)"
SUITE_LOG_DIR="${OUTPUT_ROOT}/four_gpu_runner_logs"
mkdir -p "${SUITE_LOG_DIR}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
QUEUE_FILE="${SUITE_LOG_DIR}/queue-${RUN_ID}.tsv"
LOCK_FILE="${SUITE_LOG_DIR}/queue-${RUN_ID}.lock"
FAILED_FILE="${SUITE_LOG_DIR}/failed-${RUN_ID}.tsv"

# Long datasets go first. Within each dataset, the slower dense paths are queued
# before Real so that all four GPUs remain occupied as long as possible.
for dataset in "${DATASET_ARRAY[@]}"; do
  printf '%s\t%s\n' "${dataset}" bf16 >> "${QUEUE_FILE}"
  printf '%s\t%s\n' "${dataset}" fake_quant >> "${QUEUE_FILE}"
  printf '%s\t%s\n' "${dataset}" real_quant_marlin >> "${QUEUE_FILE}"
done

echo "[storage] ${STORAGE_ROOT}"
echo "[repo] ${PROJECT_ROOT}"
echo "[venv] ${VENV_DIR}"
echo "[datasets] ${DATASETS}"
echo "[physical GPUs] ${GPU_IDS}"
echo "[queue] ${QUEUE_FILE}"

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
print("[visible GPUs]", torch.cuda.device_count())
PY

for dataset in "${DATASET_ARRAY[@]}"; do
  python scripts/preflight_experiment.py \
    --config "${CONFIG}" \
    --dataset "${dataset}"
done

# Download and fingerprint the shared quantized checkpoint once before the four
# workers start, avoiding simultaneous first-download races.
python scripts/prepare_fake_quant.py \
  --config "${CONFIG}" \
  --verify-checkpoint

claim_job() {
  local worker_id="$1"
  local claimed=""
  local temporary="${QUEUE_FILE}.tmp.${worker_id}"
  {
    flock 9
    if [[ -s "${QUEUE_FILE}" ]]; then
      IFS= read -r claimed < "${QUEUE_FILE}" || true
      tail -n +2 "${QUEUE_FILE}" > "${temporary}"
      mv "${temporary}" "${QUEUE_FILE}"
    fi
  } 9>"${LOCK_FILE}"
  printf '%s' "${claimed}"
}

run_condition() {
  local dataset="$1"
  local condition="$2"
  local gpu_id="$3"
  local log_dir="${OUTPUT_ROOT}/${dataset}/runner_logs"
  local log_path="${log_dir}/${condition}_gpu${gpu_id}_${RUN_ID}.log"
  mkdir -p "${log_dir}"
  echo "[start] dataset=${dataset} condition=${condition} physical_gpu=${gpu_id}"
  case "${condition}" in
    bf16)
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python scripts/run_bf16.py \
          --config "${CONFIG}" \
          --dataset "${dataset}" \
          2>&1 | tee "${log_path}"
      ;;
    fake_quant)
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python scripts/run_fake_quant.py \
          --config "${CONFIG}" \
          --dataset "${dataset}" \
          --device cuda \
          2>&1 | tee "${log_path}"
      ;;
    real_quant_marlin)
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
      VLLM_USE_V1="${VLLM_USE_V1:-0}" \
      TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}" \
      TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        python scripts/run_vllm_marlin.py \
          --config "${CONFIG}" \
          --dataset "${dataset}" \
          --tensor-parallel-size 1 \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          2>&1 | tee "${log_path}"
      ;;
    *)
      echo "Unknown condition: ${condition}" >&2
      return 2
      ;;
  esac
  echo "[done] dataset=${dataset} condition=${condition} physical_gpu=${gpu_id}"
}

worker() {
  local worker_id="$1"
  local gpu_id="$2"
  local job=""
  local dataset=""
  local condition=""
  while true; do
    job="$(claim_job "${worker_id}")"
    if [[ -z "${job}" ]]; then
      return 0
    fi
    IFS=$'\t' read -r dataset condition <<< "${job}"
    if ! run_condition "${dataset}" "${condition}" "${gpu_id}"; then
      {
        flock 8
        printf '%s\t%s\t%s\n' "${dataset}" "${condition}" "${gpu_id}" \
          >> "${FAILED_FILE}"
      } 8>"${FAILED_FILE}.lock"
      return 1
    fi
  done
}

WORKER_PIDS=()
for index in "${!GPU_ARRAY[@]}"; do
  worker "${index}" "${GPU_ARRAY[$index]}" &
  WORKER_PIDS+=("$!")
done

WORKER_STATUS=0
for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}" || WORKER_STATUS=1
done

if [[ "${WORKER_STATUS}" -ne 0 || -s "${FAILED_FILE}" ]]; then
  echo "At least one four-GPU worker failed." >&2
  if [[ -s "${FAILED_FILE}" ]]; then
    echo "dataset  condition  physical_gpu" >&2
    tr '\t' ' ' < "${FAILED_FILE}" >&2
  fi
  echo "Inspect logs under ${OUTPUT_ROOT}/<dataset>/runner_logs/." >&2
  exit 1
fi

for dataset in "${DATASET_ARRAY[@]}"; do
  if [[ "${dataset}" == "humaneval" || "${dataset}" == "mbpp" || "${dataset}" == "livecodebench" ]]; then
    python scripts/export_code_eval.py \
      --config "${CONFIG}" \
      --dataset "${dataset}" \
      --all
  else
    python scripts/evaluate_answers.py \
      --config "${CONFIG}" \
      --dataset "${dataset}" \
      --all
    python scripts/validate_results.py \
      --config "${CONFIG}" \
      --dataset "${dataset}"
    python scripts/compare_results.py \
      --config "${CONFIG}" \
      --dataset "${dataset}"
    python scripts/plot_results.py \
      --config "${CONFIG}" \
      --dataset "${dataset}"
  fi
done

echo "[done] four-GPU suite completed under ${PROJECT_ROOT}/${OUTPUT_ROOT}/"
echo "[pending] code datasets require official sandbox execution and result import."
