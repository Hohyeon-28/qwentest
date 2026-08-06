# CUDA 13 server setup

This profile is for the shared-storage server whose NVIDIA 590 driver reports
CUDA 13.1 support. `nvidia-smi` reports the maximum CUDA API supported by the
driver; the experiment intentionally uses vLLM/PyTorch's official CUDA 13.0
wheels.

Pinned runtime:

- PyTorch 2.11.0 with CUDA 13.0
- vLLM 0.20.2
- GPTQModel 7.2.0
- Python 3.10-3.13

The CUDA 11.8 profile remains available only for reproducing the old-server
run. Do not export its `CUDA_HOME` on this server. The prebuilt CUDA 13.0
wheels do not need a local CUDA toolkit, and `flash-attn` is not installed
separately because vLLM supplies its own attention kernels.

## One-time setup

```bash
cd ~/private/qwentest
git pull
unset CUDA_HOME
python3.12 -m venv qwen-cu130
source qwen-cu130/bin/activate
bash scripts/setup_shared_storage_cuda130.sh
```

The setup script checks the installed package versions, CUDA availability,
GPU name, and compute capability. It stops instead of leaving a partially
compatible experiment runtime.

## Four-GPU run

```bash
cd ~/private/qwentest
tmux new -s qwen-cuda130
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_shared_storage_four_gpu_cuda130.sh
```

The dynamic queue runs BF16, Fake INT4, and Real GPTQ-Marlin jobs across all
four GPUs. Code and the virtual environment remain under `~/private/qwentest`.
Downloaded models/datasets and runtime caches are stored under
`~/shared/hdd_ext/nvme4000/hohyeon/`, while results are isolated under
`~/shared/hdd_ext/nvme4000/hohyeon/qwentest_results/results_cuda130_39k_v1/`.
They never overwrite `results_39k_v2/` from the CUDA 11.8 run.

To select datasets explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATASETS="gsm8k humaneval mbpp livecodebench" \
  bash scripts/run_shared_storage_four_gpu_cuda130.sh
```

The main quality protocol deliberately keeps generation batch size at one.
Consequently, its per-request token/s is not a maximum-throughput benchmark.
Use `scripts/benchmark_throughput.py` and the separate batch-size sweep in the
config when reporting peak serving throughput.
