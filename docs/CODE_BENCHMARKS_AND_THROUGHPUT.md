# Code benchmarks and Real-Marlin throughput

## What was added

The generation runners now accept `gsm8k`, `math500`, `humaneval`, `mbpp`, and
`livecodebench`. GSM8K was already implemented before this change. The three code
benchmarks use the same Qwen3 thinking protocol and the same BF16/Fake/Real
quantization boundary as the math runs.

Code generations are deliberately **not** scored by string matching. After all
three conditions finish, the shell runner writes an `official_eval_input` file in
each condition directory. Execute those files with the benchmark's official test
harness in an isolated container or sandbox, then import the pass/fail output:

```bash
python scripts/import_code_eval_results.py \
  --dataset humaneval \
  --condition real_quant_marlin \
  --results /path/to/humaneval_results.jsonl
```

The importer accepts:

- HumanEval-style JSONL containing `task_id` and `passed`;
- LiveCodeBench JSON containing `question_id` and `graded_list` or `pass@1`;
- MBPP JSON/JSONL containing `task_id` and `passed`.

After importing all three conditions, build the code-specific comparison (pass@1,
Fake/Real correctness disagreement, exact-code agreement, McNemar test, and BF16
reasoning-length quintiles):

```bash
python scripts/compare_code_results.py --dataset humaneval
```

For code, the primary disagreement is **pass/fail disagreement**, not literal code
text disagreement. Two different programs can both be correct, so exact-code
agreement is reported only as a diagnostic.

Never run model-generated code directly on the experiment host. HumanEval itself
warns that its evaluator executes untrusted model output. Keep network disabled and
apply process, memory, filesystem, and time limits.

For LiveCodeBench, the exported JSON is accepted by its official custom evaluator:

```bash
python -m lcb_runner.runner.custom_evaluator \
  --scenario codegeneration \
  --release_version release_v6 \
  --custom_output_file /path/to/official_eval_input.json
```

Import the generated `*_eval_all.json` file with
`scripts/import_code_eval_results.py`.

The primary code metric in this experiment is deterministic **pass@1**, one sample
per task. It is an execution-gap comparison, not a reproduction of every public
leaderboard setting. LiveCodeBench is pinned to `release_v6`; changing to
`release_latest` would silently change the test set over time.

## Running a code dataset

The existing two-GPU runner can be used directly:

```bash
bash scripts/run_experiment_parallel.sh humaneval 6 7
bash scripts/run_experiment_parallel.sh mbpp 6 7
bash scripts/run_experiment_parallel.sh livecodebench 6 7
```

It generates BF16 on GPU 6 and Fake then Real on GPU 7, resumes by sample ID, and
stops after exporting official-harness inputs. Accuracy remains `null` until the
external code-execution results are imported.

### Shared-storage server

When the repository is cloned under a large shared-storage directory, keep the
virtual environment, model/dataset caches, compiler caches, temporary files, and
results under the same storage root:

```bash
STORAGE_ROOT="$PWD" bash qwentest/scripts/setup_shared_storage_cuda118.sh
STORAGE_ROOT="$PWD" bash qwentest/scripts/run_shared_storage_parallel.sh humaneval 6 7
```

The setup wrapper requires a CUDA 11.8 toolkit from `CUDA_HOME`, the current
`nvcc`, or `/usr/local/cuda-11.8`. It refuses a different toolkit rather than
silently changing the software stack between servers. Results remain under
`qwentest/results_39k_v2`; all other caches are direct children of `STORAGE_ROOT`.

## Why 27.48 tok/s is not evidence of a slow Marlin kernel

The quality protocol uses `generation.batch_size=1` and `enforce_eager=true`.
Therefore only one sequence is decoded at a time and CUDA graph capture is disabled.
The legacy summary field `tokens_per_second` is generated tokens divided by the sum
of per-request latencies. It is a single-stream/request-latency metric, not vLLM
offline aggregate throughput.

The updated summaries additionally report
`aggregate_generation_tokens_per_second`, defined as total generated tokens divided
by unique batch wall time. For batch 1 the two rates are close; with concurrent
requests only the aggregate field answers an offline-throughput question.

The copied experiment artifacts support this diagnosis:

- the 39K run used an RTX A5000, vLLM 0.8.5, batch size 1, and
  `enforce_eager=true`, and reported 27.484 tok/s;
- every Real prediction records `batch_size_used=1`;
- generated-length quintiles were nearly flat at 27.20, 27.45, 27.47, 27.30,
  and 27.59 tok/s, so a progressive long-context slowdown is not the main cause;
- the earlier 4K run on the same GPU/runtime captured 35 CUDA-graph shapes and
  reached 86.53 tok/s.

The strongest observed explanation is therefore the switch from CUDA-graph hybrid
execution to forced eager execution, compounded by zero request concurrency. The
longer 39K budget greatly increases total experiment time, but the recorded
per-token rate itself stayed nearly constant across output-length quintiles.

Run the separate benchmark on GPU 0 as follows:

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

python scripts/benchmark_real_throughput.py \
  --config configs/experiment.yaml \
  --dataset gsm8k \
  --batch-sizes 1,2,4,8,16 \
  --max-samples 64 \
  --max-new-tokens 512
```

This benchmark does not overwrite quality predictions. Its report is written to
`results_39k_v2/gsm8k/throughput_benchmark/real_marlin.json`. The batch-1 row is the
single-stream baseline; the later rows show whether concurrency raises aggregate
throughput. The 100 tok/s target is meaningful only after the target is specified
with GPU, model, input/output lengths, concurrency, backend, and throughput
definition.

Do not enable CUDA graphs merely to reach a target number on the current CUDA 11.8
stack. `--enable-cuda-graph` is an explicit opt-in diagnostic because this server
previously failed in TorchDynamo/CUDA-graph initialization. If it fails, the valid
comparison is the eager result with that limitation reported.
