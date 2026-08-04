# Controlled first-divergence audit

The original Fake and Real generations use different HF and vLLM runtimes. This
diagnostic replays only the already observed first-divergence prefixes inside a
common GPTQModel/Transformers graph. It compares a dense BF16 dequantized
reference with GPTQModel's Marlin Linear backend, with KV cache disabled. It
does not overwrite the original prediction files.

## Run

```bash
TAG=math500_first_divergence_v1

python scripts/analyze_first_divergence.py select \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG"

CUDA_VISIBLE_DEVICES=0 python scripts/analyze_first_divergence.py capture \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG" \
  --condition fake_dense \
  --device cuda

CUDA_VISIBLE_DEVICES=0 GPTQMODEL_MARLIN_USE_FP32=1 \
python scripts/analyze_first_divergence.py capture \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG" \
  --condition controlled_marlin \
  --device cuda

python scripts/analyze_first_divergence.py compare \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG"
```

Use a new tag for every independent run. Existing selections and captures fail
closed unless `--overwrite` is explicitly supplied.

## Outputs

```text
results_39k_v2/math500/first_divergence/<TAG>/
|-- all_divergences.jsonl
|-- candidates.jsonl
|-- selection_summary.json
|-- fake_dense_capture.json
|-- fake_dense_logits.pt
|-- controlled_marlin_capture.json
|-- controlled_marlin_logits.pt
|-- controlled_comparison.jsonl
`-- controlled_summary.json
```

The selector includes every observed correctness flip and an equal number of
deterministically matched non-flip controls. Each backend repeats the same prefix
forward pass twice. `controlled_summary.json` reports:

- same-backend repeatability;
- exact reproduction of the old free-generation token pair;
- controlled top-token flips;
- pairwise candidate-gap zero crossings;
- top-token margins and EOS margins;
- full-vocabulary logit RMS and maximum differences.

## Interpretation boundary

This audit removes the original HF-vs-vLLM runtime confound, but its controlled
Marlin path is GPTQModel's integration rather than vLLM's wrapper. A positive
result motivates exact vLLM operator replay. A negative result means the old
free-generation divergence cannot be attributed to Marlin from the current
experiment.

The selected prefix is the first divergence observed in the old cross-runtime
free generations. If the controlled run does not reproduce that decision, the
script does not search for a new first divergence; it records the non-reproduction
as evidence against attributing the old split to Marlin alone.

## Actual vLLM deployment-prefix replay

After the controlled captures above exist, replay the same 44 prefixes through
the exact vLLM GPTQ-Marlin deployment configuration used by the original Real
run:

```bash
TAG=math500_first_divergence_v1

CUDA_VISIBLE_DEVICES=0 python scripts/analyze_first_divergence.py capture-vllm \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG" \
  --repeats 2

python scripts/analyze_first_divergence.py compare-vllm \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG"
```

The public vLLM API does not expose a raw full-vocabulary logit tensor. The
legacy CUDA 11.8/vLLM stack also fails inside its rank calculation when prompt
log-probabilities are requested for these prefixes. The replay therefore uses
two logprob-free one-token requests: an ordinary greedy request records the
exact full-vocabulary top-1, and a request constrained with
`allowed_token_ids=[fake_token, real_token]` records the exact preference
between the old candidates. Repeats detect nondeterministic token decisions.
The numerical magnitude of the vLLM candidate gap is intentionally not
reported.

Additional outputs are:

```text
vllm_deployment_replay.jsonl
vllm_deployment_capture.json
vllm_deployment_comparison.jsonl
vllm_deployment_summary.json
```

This comparison intentionally includes vLLM attention, scheduling, numerical
ordering, and Marlin. It tests whether the old split reappears at a fixed
prefix; it does not attribute a positive result to Marlin alone.

## Forced-token counterfactual continuation

Run the primary causal check on `math500-00135` after prefix replay:

```bash
TAG=math500_first_divergence_v1

CUDA_VISIBLE_DEVICES=0 python scripts/analyze_first_divergence.py branch-vllm \
  --config configs/experiment.yaml \
  --dataset math500 \
  --analysis-tag "$TAG" \
  --branch-tag math500_00135_v1 \
  --sample-id math500-00135
```

Both branches use the same vLLM GPTQ-Marlin backend after the forced token.
The default continuation budget is

```text
original max_new_tokens - common generated prefix length - 1 forced token
```

so the two branches retain the original total generation cap. This is required
when testing a completion-versus-truncation outcome; granting a fresh 38,912
tokens at the branch would move the budget boundary and answer a different
question.

Results are written without touching the original predictions:

```text
counterfactual/math500_00135_v1/branch_outputs.jsonl
counterfactual/math500_00135_v1/branch_summary.json
```

The result estimates the conditional effect of the selected token at the fixed
prefix. It cannot identify which upstream runtime component originally made
that token preferable. As a negative control, the same command can be run with
a new branch tag and `--sample-id math500-00420`, where the old token split was
reproduced even though the original final outcome did not flip.
