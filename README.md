# Qwen3-8B shared-quantization CoT experiment

Qwen3-8B의 긴 Chain-of-Thought에서 BF16, Fake INT4, GPTQ-Marlin INT4를
비교합니다. 가장 중요한 통제 조건은 Fake와 Real이 동일한 논리적 양자화 결과
`(q,s,z,g)`를 사용한다는 것입니다.

## 정확한 pipeline 정의

```text
W_FP16
  -> GPTQ quantization algorithm
  -> (q, s, z, g)
  -> packing / permutation / interleave
  -> W_packed
  -> kernel
  -> Y
```

- `q`: 4-bit integer weight
- `s`: scale
- `z`: zero-point
- `g`: group information 또는 `g_idx`
- `W_packed`: kernel 실행용 물리적 저장 형식

Fake와 Real은 반드시 같은 standard GPTQ checkpoint를 읽습니다.

```text
(q_fake, s_fake, z_fake, g_fake)
  =
(q_real, s_real, z_real, g_real)
```

각 checkpoint tensor의 dtype, shape, raw bytes를 SHA-256으로 계산합니다.
`qweight`, `scales`, `qzeros`, `g_idx` 중 하나라도 없거나 한 bit라도 다르면
비교를 중단합니다.

standard GPTQ의 `qweight/qzeros`도 파일 안에서는 int32에 bit-pack되어 있지만,
이는 lossless한 canonical checkpoint serialization이며 Marlin 전용
`W_packed`와는 구분합니다. 고정된 format/bits에서 이 serialization의 동일
fingerprint는 동일한 논리적 `q`와 `z`를 뜻합니다. Fake는 이를 unpack/dequantize하고,
Real은 이를 별도의 Marlin 물리 layout으로 repack합니다.

### Fake Quant

```text
shared standard-GPTQ (q,s,z,g)
  -> W_fake = s[g] * (q - z[g])
  -> BF16 torch.nn.Linear
  -> dense BF16 GEMM
```

`GPTQModel`의 pure-Torch GPTQ decoder로 checkpoint를 로드하고, 각 quantized
Linear를 동일 tuple에서 복원한 dense `torch.nn.Linear`로 교체합니다. Fake 경로에는
INT4 packing이나 Marlin kernel이 없습니다.

### Real Quant

```text
same shared standard-GPTQ (q,s,z,g)
  -> vLLM runtime Marlin repack/layout
  -> W_packed
  -> GPTQ-Marlin kernel
```

따라서 Fake–Real 차이는 quantization algorithm 차이가 아닙니다. 동일
`(q,s,z,g)` 이후의 packing, dequantization 위치/정밀도, accumulation order,
kernel 및 runtime 구현 차이를 포함하는 shared-quant execution gap입니다.

## 지원 checkpoint

`configs/experiment.yaml`의 `models.real_gptq`에 다음 조건의 checkpoint를
지정해야 합니다.

```text
format = standard GPTQ 또는 GPTQ-v2
bits = 4
group_size = 128
sym = true
desc_act = false
serialization = safetensors
qweight/scales/qzeros/g_idx가 모든 quantized layer에 존재
```

이미 Marlin layout으로 저장된 checkpoint는 허용하지 않습니다. 이 경우 논리적
`(q,s,z,g)`와 물리적 `W_packed`의 경계가 backend 독립적으로 보장되지 않기
때문입니다. standard GPTQ checkpoint를 입력하고 vLLM이 실행 시 Marlin layout으로
repack하게 해야 합니다.

## 설치

vLLM이 지원하는 Linux/CUDA 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## checkpoint와 tuple 먼저 검증

```bash
python scripts/prepare_fake_quant.py \
  --config configs/experiment.yaml \
  --verify-checkpoint
```

출력되는 `tuple_sha256`이 실험 전체의 `(q,s,z,g)` 식별자입니다.

내부 fingerprint fail-closed 동작을 확인하려면:

```bash
pytest -q
python scripts/prepare_fake_quant.py \
  --config configs/experiment.yaml \
  --self-test
```

## GSM8K 20-sample 검증

```bash
python scripts/run_bf16.py \
  --config configs/experiment.yaml --dataset gsm8k --max-samples 20

python scripts/run_fake_quant.py \
  --config configs/experiment.yaml --dataset gsm8k --max-samples 20 \
  --device cuda

python scripts/run_vllm_marlin.py \
  --config configs/experiment.yaml --dataset gsm8k --max-samples 20

python scripts/evaluate_answers.py \
  --config configs/experiment.yaml --dataset gsm8k --all

python scripts/compare_results.py \
  --config configs/experiment.yaml --dataset gsm8k

python scripts/plot_results.py \
  --config configs/experiment.yaml --dataset gsm8k
```

비교 스크립트는 각 sample마다 다음 조건을 모두 검사합니다.

1. 세 조건의 prompt token hash가 동일함
2. Fake와 Real의 source checkpoint가 동일함
3. Fake와 Real의 `(q,s,z,g)` SHA-256 fingerprint가 동일함

조건이 하나라도 깨지면 결과 표를 만들지 않고 오류로 종료합니다.

기존 raw-RTN Fake 결과가 남아 있다면 반드시 세 실행을 `--overwrite`로 다시
실행하십시오. 이전 prediction에는 shared tuple fingerprint가 없어 비교 단계에서
의도적으로 거부됩니다.

## 전체 데이터셋

위 명령에서 `--max-samples 20`을 제거하면 GSM8K 전체 test set을 실행합니다.
동일하게 `--dataset math500`으로 MATH-500을 실행합니다. 중단 후 같은 명령을
다시 실행하면 저장된 sample ID 이후부터 resume합니다.

## 두 종류의 풀이 길이

```text
S_i^gold = GSM8K 정답 풀이의 계산 step 수
L_i,m^gen = 모델 m이 실제 생성한 reasoning token 수
```

- `S_i^gold`는 GSM8K gold rationale의 `<<expression=result>>` annotation
  개수이며 `gold_calculation_step_count`에 저장합니다.
- `L_i,m^gen`은 `</think>` 이전 실제 생성 token 수이며
  `generated_reasoning_token_count`에 저장합니다.
- MATH-500은 동일한 annotation 규약이 없어 `S_i^gold=null`입니다.

BF16 reasoning 길이를 공통 bucket으로 사용하는 분석과 각 모델 자체 생성 길이
bucket 분석을 모두 만듭니다. GSM8K에서는 gold-step별 정확도와 평균
`L_i,m^gen`도 추가로 계산합니다.

## Teacher-forced logprob 비교

```bash
python scripts/analyze_logits.py capture-hf --condition bf16 \
  --config configs/experiment.yaml --dataset gsm8k
python scripts/analyze_logits.py capture-hf --condition fake_quant \
  --config configs/experiment.yaml --dataset gsm8k --device cuda
python scripts/analyze_logits.py capture-vllm-real \
  --config configs/experiment.yaml --dataset gsm8k
python scripts/analyze_logits.py compare \
  --config configs/experiment.yaml --dataset gsm8k
```

vLLM 공개 API가 full Marlin logits/hidden states를 반환하지 않으므로 공통으로
관찰 가능한 top-k prompt logprobs를 사용해 top-1 agreement, top-k overlap,
truncated-union 근사 KL, reference-token logprob 차이를 계산합니다.

## 출력

```text
results/<dataset>/
├── config.json
├── environment.json
├── bf16/
│   ├── predictions.jsonl
│   └── summary.json
├── fake_quant/
│   ├── predictions.jsonl
│   ├── summary.json
│   ├── quantization_manifest.json
│   └── dequantization_report.jsonl
├── real_quant_marlin/
│   ├── predictions.jsonl
│   ├── summary.json
│   ├── quantization_manifest.json
│   └── startup.log
├── comparisons/
│   ├── sample_comparison.jsonl
│   ├── length_bucket.csv
│   ├── length_bucket_own.csv
│   ├── accuracy_by_gold_steps.csv
│   ├── error_cases.jsonl
│   └── summary.json
└── plots/
    ├── accuracy_by_precision.png
    ├── accuracy_drop_by_length.png
    ├── agreement_by_length.png
    ├── latency_by_precision.png
    ├── token_length_distribution.png
    ├── accuracy_by_gold_steps.png
    └── generated_reasoning_tokens_by_gold_steps.png
```

## 해석상 주의

- Qwen3 공식 권고는 thinking mode에서 greedy decoding을 피하는 것이지만,
  본 실험은 backend 간 결정론적 비교를 위해 명세대로 greedy를 기본 사용합니다.
- Fake latency는 dense BF16 GEMM이므로 Real INT4 kernel의 속도 비교 대상으로
  해석하지 않습니다.
- shared tuple이 동일해도 HF dense runtime과 vLLM runtime에는 attention kernel,
  scheduler 등 weight GEMM 외 차이가 남습니다. 따라서 Fake–Real gap을 오직
  Marlin GEMM 하나의 효과라고 단정하지 않습니다.
