#!/usr/bin/env bash

# Source this file. It intentionally has no `set -e` so that the caller keeps
# control of shell options.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
STORAGE_ROOT="${STORAGE_ROOT:-$(cd -- "${PROJECT_ROOT}/.." && pwd)}"
EXPERIMENT_CACHE_ROOT="${EXPERIMENT_CACHE_ROOT:-${STORAGE_ROOT}}"

export STORAGE_ROOT
export EXPERIMENT_CACHE_ROOT
export HF_HOME="${EXPERIMENT_CACHE_ROOT}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${EXPERIMENT_CACHE_ROOT}/cache"
export PIP_CACHE_DIR="${EXPERIMENT_CACHE_ROOT}/pip_cache"
export TORCH_HOME="${EXPERIMENT_CACHE_ROOT}/torch_cache"
export TORCH_EXTENSIONS_DIR="${EXPERIMENT_CACHE_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${EXPERIMENT_CACHE_ROOT}/triton_cache"
export NUMBA_CACHE_DIR="${EXPERIMENT_CACHE_ROOT}/numba_cache"
export VLLM_CACHE_ROOT="${EXPERIMENT_CACHE_ROOT}/vllm_cache"
export TMPDIR="${EXPERIMENT_CACHE_ROOT}/tmp"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${TMPDIR}"
