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
